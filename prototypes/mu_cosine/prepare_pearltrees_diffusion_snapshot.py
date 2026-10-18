#!/usr/bin/env python3
"""Prepare a deterministic, privacy-preserving Pearltrees diffusion snapshot.

This is a raw-evidence compiler, not a harvester or graph-repair tool. Detailed
outputs are local-only. It performs no diffusion solve and consumes no labels,
judge outputs, embeddings, or filing outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unicodedata
from urllib.parse import quote, unquote, urlsplit
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from privacy import is_private_title  # noqa: E402


SCHEMA_VERSION = 1
ALGORITHM = "pearltrees-diffusion-snapshot-v1"
SOURCE_SPEC_SCHEMA = "pearltrees-diffusion-source-spec-v1"
POLICY_SCHEMA = "pearltrees-diffusion-relation-policy-v1"
RELATIONS = ("collection", "ref", "path", "alias", "shortcut", "cross_link")
PRIVACY_RELATIONS = frozenset(("collection", "ref", "path"))
SOURCE_KINDS = frozenset(("rdf", "api_sqlite", "api_json_dir", "path_jsonl"))
ARTIFACT_NAMES = (
    "sources.json",
    "nodes.jsonl",
    "visibility_evidence.jsonl",
    "exclusions.jsonl",
    "conflicts.jsonl",
    "edge_evidence.jsonl",
    "physical_edges.tsv",
    "adjacency.jsonl",
    "components.jsonl",
    "anchor_eligibility.jsonl",
    "scrub_manifest.json",
    "legacy_parity.json",
    "aggregate_release_candidate.json",
)
ALL_RUN_FILES = frozenset(ARTIFACT_NAMES + ("manifest.json", "LOCAL_ONLY_DO_NOT_PUBLISH"))
MAX_SOURCE_BYTES = 2_000_000_000
MAX_JSON_DEPTH = 128
MAX_TEXT_CHARS = 100_000
ID_URI_RE = re.compile(r"(?:^|/)id([1-9][0-9]*)(?:$|[/?#])")
RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NAMESPACE = "http://www.w3.org/2000/01/rdf-schema#"
SIOC_NAMESPACE = "http://rdfs.org/sioc/ns#"
DCTERMS_NAMESPACES = frozenset(
    ("http://purl.org/dc/terms/", "http://purl.org/dc/elements/1.1/")
)
PEARLTREES_NAMESPACES = frozenset(
    (
        "http://www.pearltrees.com/rdf/0.1/#",
        "https://www.pearltrees.com/rdf/0.1/#",
        "http://www.pearltrees.com/xmlns/pearl-trees#",
        "https://www.pearltrees.com/xmlns/pearl-trees#",
    )
)
FOLDER_CONTENT_RELATIONS = {2: "collection", 5: "shortcut", 6: "cross_link"}
MISSING = object()


class SnapshotError(ValueError):
    """Fail-closed snapshot preparation or verification error."""


def _duplicate_checked_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SnapshotError("duplicate JSON key")
        value[key] = item
    return value


def _reject_nonfinite(value):
    raise SnapshotError("non-finite JSON constant")


def _strict_json_bytes(data, label):
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid UTF-8 JSON in {label}") from exc
    _check_json_depth(value, 0)
    return value


def _strict_json_file(path, label):
    return _strict_json_bytes(_read_file(path), label)


def _check_json_depth(value, depth):
    if depth > MAX_JSON_DEPTH:
        raise SnapshotError("JSON nesting exceeds the frozen limit")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SnapshotError("JSON object key is not a string")
            _check_json_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_json_depth(item, depth + 1)


def _canonical_json(value):
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SnapshotError("value is not canonical finite JSON") from exc
    return (text + "\n").encode("utf-8")


def _jsonl_bytes(records):
    return b"".join(_canonical_json(record) for record in records)


def _content_record(data):
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _read_file(path):
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SnapshotError("declared source is unavailable") from exc
    if not path.is_file() or path.is_symlink():
        raise SnapshotError("declared file source must be a regular non-symlink file")
    if size > MAX_SOURCE_BYTES:
        raise SnapshotError("declared source exceeds the frozen size limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SnapshotError("declared source could not be read") from exc


def _file_record(path):
    return _content_record(_read_file(path))


def _safe_text(value, label):
    if value is None:
        return None
    if not isinstance(value, str):
        raise SnapshotError(f"{label} must be text")
    text = unicodedata.normalize("NFC", value.strip())
    if len(text) > MAX_TEXT_CHARS:
        raise SnapshotError(f"{label} exceeds the frozen length limit")
    return text or None


def _strict_id(value, label="node ID"):
    if isinstance(value, bool):
        raise SnapshotError(f"{label} must be a positive decimal integer")
    if isinstance(value, int):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise SnapshotError(f"{label} must be a positive decimal integer")
    if not raw or not raw.isascii() or not raw.isdigit():
        raise SnapshotError(f"{label} must be a positive decimal integer")
    if raw[0] == "0" or int(raw) < 1:
        raise SnapshotError(f"{label} must use canonical positive decimal form")
    return f"pt:{raw}"


def _pearltrees_uri_parts(value, label):
    if not isinstance(value, str):
        raise SnapshotError(f"{label} must contain a Pearltrees URI")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise SnapshotError(f"{label} must contain a valid Pearltrees URI") from exc
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme not in {"http", "https"}
        or (host != "pearltrees.com" and not host.endswith(".pearltrees.com"))
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
    ):
        raise SnapshotError(f"{label} must remain inside the Pearltrees URI boundary")
    return parsed


def _id_from_uri(value, label):
    parsed = _pearltrees_uri_parts(value, label)
    matches = [
        match.group(1)
        for segment in parsed.path.split("/")
        if (match := re.fullmatch(r"id([1-9][0-9]*)", segment)) is not None
    ]
    if not matches:
        raise SnapshotError(f"{label} must contain a numeric Pearltrees URI")
    if len(matches) != 1:
        raise SnapshotError(f"{label} must contain exactly one numeric Pearltrees ID")
    return _strict_id(matches[0], label)


def _account_root_scope_from_uri(value, label, *, allowed_fragment=None):
    parsed = _pearltrees_uri_parts(value, label)
    if (
        parsed.query
        or (parsed.fragment and parsed.fragment != allowed_fragment)
        or ID_URI_RE.search(parsed.path)
    ):
        raise SnapshotError(f"{label} is not a canonical account-root URI")
    segments = [unquote(part) for part in parsed.path.split("/") if part]
    if len(segments) == 2 and segments[0] == "t":
        team_account = _safe_text(segments[1], "team-space account-root name")
        if (
            team_account is None
            or "/" in team_account
            or team_account.startswith("account:")
        ):
            raise SnapshotError(f"{label} has an invalid account-root name")
        return "groups", tuple(segments)
    if len(segments) != 1:
        raise SnapshotError(f"{label} is not a canonical account-root URI")
    account = _safe_text(segments[0], "account-root name")
    if account is None or "/" in account or account.startswith("account:"):
        raise SnapshotError(f"{label} has an invalid account-root name")
    return account, tuple(segments)


def _account_root_from_uri(value, label):
    account, _ = _account_root_scope_from_uri(value, label)
    return account


def _local_name(name):
    return name.rsplit("}", 1)[-1] if "}" in name else name.split(":")[-1]


def _namespace(name):
    if isinstance(name, str) and name.startswith("{") and "}" in name:
        return name[1:].split("}", 1)[0]
    return None


def _validate_rdf_namespaces(root):
    if root.tag != f"{{{RDF_NAMESPACE}}}RDF":
        raise SnapshotError("RDF root namespace is not allowlisted")
    entity_names = {
        "AliasPearl",
        "NotePearl",
        "PagePearl",
        "RefPearl",
        "RootPearl",
        "SectionPearl",
        "Tree",
        "UserAccount",
    }
    child_namespaces = {
        "parentTree": PEARLTREES_NAMESPACES,
        "privacy": PEARLTREES_NAMESPACES,
        "rootTree": PEARLTREES_NAMESPACES,
        "seeAlso": PEARLTREES_NAMESPACES | {RDFS_NAMESPACE},
        "title": PEARLTREES_NAMESPACES | DCTERMS_NAMESPACES,
        "treeId": PEARLTREES_NAMESPACES,
    }
    for element in root.iter():
        local = _local_name(element.tag)
        namespace = _namespace(element.tag)
        if local == "UserAccount":
            if namespace not in PEARLTREES_NAMESPACES | {SIOC_NAMESPACE}:
                raise SnapshotError("Pearltrees UserAccount namespace is not allowlisted")
        elif (local in entity_names or local.endswith("Pearl")) and namespace not in PEARLTREES_NAMESPACES:
            raise SnapshotError("Pearltrees RDF entity namespace is not allowlisted")
        if local in child_namespaces and namespace not in child_namespaces[local]:
            raise SnapshotError("Pearltrees RDF field namespace is not allowlisted")
        for attribute in element.attrib:
            attr_local = _local_name(attribute)
            if attr_local in {"about", "resource"} and _namespace(attribute) != RDF_NAMESPACE:
                raise SnapshotError("RDF identity attribute namespace is not allowlisted")


def _attribute(element, local):
    matches = [value for key, value in element.attrib.items() if _local_name(key) == local]
    if len(matches) > 1:
        raise SnapshotError("ambiguous XML attribute")
    return matches[0] if matches else None


def _child_text(element, local):
    matches = []
    for child in element:
        if _local_name(child.tag) == local:
            matches.append((child.text or "").strip())
    if len(matches) > 1:
        raise SnapshotError("ambiguous XML field")
    return matches[0] if matches else None


def _child_resource(element, local):
    matches = []
    for child in element:
        if _local_name(child.tag) == local:
            matches.append(_attribute(child, "resource") or (child.text or "").strip())
    if len(matches) > 1:
        raise SnapshotError("ambiguous XML resource field")
    return matches[0] if matches else None


def _visibility(value=MISSING):
    if value is MISSING or value is None or (isinstance(value, str) and not value.strip()):
        return "unknown"
    if isinstance(value, bool) or isinstance(value, float):
        raise SnapshotError("visibility must be a canonical nonnegative integer")
    raw = str(value)
    if not raw.isascii() or not raw.isdigit() or (len(raw) > 1 and raw[0] == "0"):
        raise SnapshotError("visibility must be a canonical nonnegative integer")
    return "public" if int(raw) == 0 else "private"


def _optional_bool(value, label):
    """Accept an absent flag or an actual JSON/SQLite Boolean, never truthiness."""
    if value is MISSING or value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise SnapshotError(f"{label} must be Boolean")


def _coalesced_alias(mapping, primary, alias, label, default=MISSING):
    has_primary = primary in mapping
    has_alias = alias in mapping
    if has_primary and has_alias:
        if _canonical_json(mapping[primary]) != _canonical_json(mapping[alias]):
            raise SnapshotError(f"{label} aliases disagree")
        return mapping[primary]
    if has_primary:
        return mapping[primary]
    if has_alias:
        return mapping[alias]
    return default


def _canonical_content_type(value, label):
    if isinstance(value, bool):
        raise SnapshotError(f"{label} must be a canonical integer")
    if isinstance(value, int):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise SnapshotError(f"{label} must be a canonical integer")
    if (
        not raw
        or not raw.isascii()
        or not raw.isdigit()
        or (len(raw) > 1 and raw[0] == "0")
    ):
        raise SnapshotError(f"{label} must be a canonical integer")
    content_type = int(raw)
    if content_type not in {1, 2, 4, 5, 6, 7}:
        raise SnapshotError(f"unsupported {label}")
    return content_type


def _account_marker(value, label):
    if not isinstance(value, str) or not value.startswith("account:"):
        raise SnapshotError(f"{label} must be an account root marker")
    account = _safe_text(value[len("account:") :], label)
    if account is None or re.fullmatch(r"[A-Za-z0-9._-]+", account) is None:
        raise SnapshotError(f"{label} is not canonical")
    if value != f"account:{account}":
        raise SnapshotError(f"{label} is not canonical")
    return account


def _stable_key(node_id):
    prefix, number = node_id.split(":", 1)
    return prefix, int(number)


def _record_key(*parts):
    return hashlib.sha256(_canonical_json(list(parts))).hexdigest()


def _new_state():
    return {
        "nodes": {},
        "edges": [],
        "roots": [],
        "self_edges": 0,
    }


def _ensure_node(state, node_id):
    return state["nodes"].setdefault(
        node_id,
        {
            "titles": set(),
            "accounts": set(),
            "visibilities": set(),
            "visibility_evidence": set(),
            "sources": set(),
            "root_evidence": set(),
            "private_title": False,
        },
    )


def _observe_node(
    state,
    node_id,
    *,
    source_id,
    record_key,
    title=None,
    visibility=MISSING,
    account=None,
    root_evidence=None,
):
    if not isinstance(record_key, str) or re.fullmatch(r"[0-9a-f]{64}", record_key) is None:
        raise SnapshotError("node observation key is malformed")
    node = _ensure_node(state, node_id)
    node["sources"].add(source_id)
    clean_title = _safe_text(title, "title")
    if clean_title is not None:
        node["titles"].add(clean_title)
        if is_private_title(clean_title):
            node["private_title"] = True
    clean_account = _safe_text(account, "account")
    if clean_account is not None:
        node["accounts"].add(clean_account)
    visibility_claim = _visibility(visibility)
    node["visibility_evidence"].add((source_id, record_key, visibility_claim))
    if visibility_claim != "unknown":
        node["visibilities"].add(visibility_claim)
    if root_evidence is not None:
        node["root_evidence"].add(root_evidence)


def _add_edge(state, child, parent, relation, subtype, source_id, record_key):
    if relation not in RELATIONS:
        raise SnapshotError("unmapped source relation")
    _observe_node(
        state,
        child,
        source_id=source_id,
        record_key=_record_key(record_key, "edge-child"),
    )
    _observe_node(
        state,
        parent,
        source_id=source_id,
        record_key=_record_key(record_key, "edge-parent"),
    )
    if child == parent:
        raise SnapshotError("self relation is not a physical or privacy edge")
    state["edges"].append(
        {
            "child": child,
            "parent": parent,
            "relation": relation,
            "subtype": subtype,
            "source_id": source_id,
            "record_key": record_key,
        }
    )


def _parse_rdf(path, source, state):
    data = _read_file(path)
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise SnapshotError("XML document type or entity declarations are forbidden")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise SnapshotError("malformed RDF XML") from exc
    _validate_rdf_namespaces(root)

    account_default = source.get("account")
    if account_default is not None:
        account_default = _safe_text(account_default, "source account")

    recognized_pearls = {
        "AliasPearl",
        "NotePearl",
        "PagePearl",
        "RefPearl",
        "RootPearl",
        "SectionPearl",
    }
    for index, element in enumerate(root.iter()):
        kind = _local_name(element.tag)
        about = _attribute(element, "about")
        record = _record_key(source["source_id"], "rdf", kind, about or "", index)
        if kind == "Tree":
            uri_id = _id_from_uri(about, "Tree rdf:about") if about else None
            text_id = _child_text(element, "treeId")
            child_id = _strict_id(text_id, "Tree treeId") if text_id else None
            if uri_id is not None and child_id is not None and uri_id != child_id:
                raise SnapshotError("Tree URI and treeId disagree")
            node_id = uri_id or child_id
            if node_id is None:
                raise SnapshotError("Tree is missing a canonical ID")
            _observe_node(
                state,
                node_id,
                source_id=source["source_id"],
                record_key=record,
                title=_child_text(element, "title"),
                visibility=_child_text(element, "privacy"),
                account=account_default,
            )
        elif kind in ("RefPearl", "AliasPearl"):
            parent_value = _child_resource(element, "parentTree")
            child_value = _child_resource(element, "seeAlso")
            if not parent_value or not child_value:
                raise SnapshotError("RDF relation is missing an endpoint")
            child = _id_from_uri(child_value, "seeAlso")
            _observe_node(
                state,
                child,
                source_id=source["source_id"],
                record_key=record,
                title=_child_text(element, "title"),
                visibility=_child_text(element, "privacy"),
                account=account_default,
            )
            relation = "ref" if kind == "RefPearl" else "alias"
            parent_parts = _pearltrees_uri_parts(parent_value, "parentTree")
            if ID_URI_RE.search(parent_parts.path):
                parent = _id_from_uri(parent_value, "parentTree")
                _add_edge(
                    state,
                    child,
                    parent,
                    relation,
                    f"rdf_{kind}",
                    source["source_id"],
                    record,
                )
            else:
                root_account = _account_root_from_uri(parent_value, "parentTree")
                if account_default is not None and root_account != account_default:
                    raise SnapshotError("RDF account root disagrees with declared account")
                node = _ensure_node(state, child)
                node["accounts"].add(root_account)
                node["root_evidence"].add(
                    _record_key(source["source_id"], "rdf_account_root", record)
                )
        elif kind == "UserAccount":
            if not about:
                raise SnapshotError("RDF UserAccount is missing rdf:about")
            subject_account, subject_segments = _account_root_scope_from_uri(
                about, "UserAccount rdf:about", allowed_fragment="sioc"
            )
            if account_default is not None and subject_account != account_default:
                raise SnapshotError(
                    "RDF UserAccount subject disagrees with declared account"
                )
            root_value = _child_resource(element, "rootTree")
            if root_value:
                root_parts = _pearltrees_uri_parts(root_value, "rootTree")
                if root_parts.query or root_parts.fragment:
                    raise SnapshotError("RDF rootTree URI is not canonical")
                root_segments = tuple(
                    unquote(part) for part in root_parts.path.split("/") if part
                )
                if root_segments[: len(subject_segments)] != subject_segments:
                    raise SnapshotError(
                        "RDF rootTree disagrees with UserAccount subject"
                    )
                root_id = _id_from_uri(root_value, "rootTree")
                root_record = _record_key(source["source_id"], "rdf_root", root_id)
                _observe_node(
                    state,
                    root_id,
                    source_id=source["source_id"],
                    record_key=root_record,
                    account=subject_account,
                    root_evidence=root_record,
                )
        elif kind.endswith("Pearl") and kind not in recognized_pearls:
            raise SnapshotError("unsupported RDF pearl type")


def _api_content(payload):
    if not isinstance(payload, dict):
        raise SnapshotError("API JSON record must be an object")
    wrappers = [key for key in ("api_response", "response") if key in payload]
    if len(wrappers) > 1:
        raise SnapshotError("API record has ambiguous response wrappers")
    response = payload[wrappers[0]] if wrappers else payload
    if not isinstance(response, dict):
        raise SnapshotError("API response must be an object")
    info = response.get("info", {})
    tree = response.get("tree", {})
    if info is None:
        info = {}
    if tree is None:
        tree = {}
    if not isinstance(info, dict) or not isinstance(tree, dict):
        raise SnapshotError("API info/tree must be objects")
    return response, info, tree


def _parse_api_payload(payload, source_id, state, locator):
    response, info, tree = _api_content(payload)
    candidate_ids = [
        value
        for value in (
            info.get("id"),
            tree.get("id"),
            payload.get("tree_id"),
            response.get("treeId"),
        )
        if value is not None
    ]
    if not candidate_ids:
        raise SnapshotError("API record is missing a tree ID")
    tree_ids = {_strict_id(value, "API tree ID") for value in candidate_ids}
    if len(tree_ids) != 1:
        raise SnapshotError("API tree ID fields disagree")
    tree_id = next(iter(tree_ids))
    account = payload.get("account", response.get("account"))
    info_root = _optional_bool(info.get("isUserRoot", MISSING), "API info root flag")
    tree_root = _optional_bool(tree.get("isUserRoot", MISSING), "API tree root flag")
    root_evidence = (
        _record_key(source_id, locator, "api_root", tree_id)
        if info_root or tree_root
        else None
    )
    # Preserve both visibility/title claims. A private claim must not be hidden by
    # precedence between the API's info and tree objects.
    _observe_node(
        state,
        tree_id,
        source_id=source_id,
        record_key=_record_key(source_id, locator, "info", tree_id),
        title=info.get("title"),
        visibility=info["visibility"] if "visibility" in info else MISSING,
        account=account,
        root_evidence=root_evidence,
    )
    _observe_node(
        state,
        tree_id,
        source_id=source_id,
        record_key=_record_key(source_id, locator, "tree", tree_id),
        title=tree.get("title"),
        visibility=tree["visibility"] if "visibility" in tree else MISSING,
        account=account,
        root_evidence=root_evidence,
    )

    parent = info.get("parentTree", MISSING)
    if parent is None or parent == {}:
        if parent is not MISSING:
            _ensure_node(state, tree_id)["root_evidence"].add(
                _record_key(source_id, locator, "api_absent_parent_root", tree_id)
            )
    elif parent is not MISSING:
        if not isinstance(parent, dict) or "id" not in parent:
            raise SnapshotError("API parentTree must be an object with an ID")
        parent_id = _strict_id(parent["id"], "API parent tree ID")
        parent_visibility = parent["visibility"] if "visibility" in parent else MISSING
        parent_record = _record_key(source_id, locator, "parent", tree_id, parent_id)
        _observe_node(
            state,
            parent_id,
            source_id=source_id,
            record_key=parent_record,
            title=parent.get("title"),
            visibility=parent_visibility,
            account=account,
            root_evidence=(
                _record_key(source_id, locator, "api_parent_root", parent_id)
                if _optional_bool(parent.get("isUserRoot", MISSING), "API parent root flag")
                else None
            ),
        )
        _add_edge(
            state,
            tree_id,
            parent_id,
            "collection",
            "api_parent",
            source_id,
            parent_record,
        )

    pearl_containers = [
        mapping["pearls"]
        for mapping in (tree, response, payload)
        if "pearls" in mapping
    ]
    if pearl_containers:
        canonical_pearls = _canonical_json(pearl_containers[0])
        if any(
            _canonical_json(candidate) != canonical_pearls
            for candidate in pearl_containers[1:]
        ):
            raise SnapshotError("API pearl containers disagree")
        pearls = pearl_containers[0]
    else:
        pearls = []
    if pearls is None:
        pearls = []
    if not isinstance(pearls, list):
        raise SnapshotError("API pearls must be an array")
    for position, pearl in enumerate(pearls):
        if not isinstance(pearl, dict):
            raise SnapshotError("API pearl must be an object")
        raw_type = _coalesced_alias(
            pearl, "contentType", "content_type", "API content type"
        )
        content_type = _canonical_content_type(raw_type, "API content type")
        if content_type not in FOLDER_CONTENT_RELATIONS:
            continue
        content_tree = _coalesced_alias(
            pearl, "contentTree", "content_tree", "API contentTree"
        )
        if not isinstance(content_tree, dict) or "id" not in content_tree:
            raise SnapshotError("folder API pearl is missing contentTree")
        child = _strict_id(content_tree["id"], "API content tree ID")
        parent_value = _coalesced_alias(
            pearl,
            "treeId",
            "tree_id",
            "API pearl parent",
            default=tree_id.split(":", 1)[1],
        )
        parent_id = _strict_id(parent_value, "API pearl parent ID")
        if parent_id != tree_id:
            raise SnapshotError("API pearl parent disagrees with enclosing tree")
        relation = FOLDER_CONTENT_RELATIONS[content_type]
        pearl_record = _record_key(
            source_id, locator, "pearl", position, child, parent_id, relation
        )
        _observe_node(
            state,
            child,
            source_id=source_id,
            record_key=pearl_record,
            title=content_tree.get("title"),
            visibility=content_tree["visibility"] if "visibility" in content_tree else MISSING,
            account=account,
        )
        if "content_tree_title" in pearl:
            _observe_node(
                state,
                child,
                source_id=source_id,
                record_key=_record_key(pearl_record, "flat-title"),
                title=pearl["content_tree_title"],
                account=account,
            )
        _add_edge(
            state,
            child,
            parent_id,
            relation,
            {
                "collection": "api_collection",
                "shortcut": "api_shortcut",
                "cross_link": "api_content_type_6",
            }[relation],
            source_id,
            pearl_record,
        )


def _json_cell_bytes(value, label):
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes):
        return value
    raise SnapshotError(f"{label} must be UTF-8 text or bytes")


def _validate_sqlite_raw_relation(raw_content, content_type, parent):
    if not isinstance(raw_content, dict):
        return
    raw_type = _coalesced_alias(
        raw_content,
        "contentType",
        "content_type",
        "SQLite raw content type",
        default=MISSING,
    )
    if (
        raw_type is not MISSING
        and _canonical_content_type(raw_type, "SQLite raw content type")
        != content_type
    ):
        raise SnapshotError("SQLite raw content type disagrees with columns")
    raw_parent = _coalesced_alias(
        raw_content,
        "treeId",
        "tree_id",
        "SQLite raw parent ID",
        default=MISSING,
    )
    if (
        raw_parent is not MISSING
        and _strict_id(raw_parent, "SQLite raw parent ID") != parent
    ):
        raise SnapshotError("SQLite raw parent ID disagrees with columns")


def _sqlite_columns(connection, table):
    return {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table})")
    }


def _parse_api_sqlite(path, source, state):
    if Path(str(path) + "-wal").exists() or Path(str(path) + "-shm").exists():
        raise SnapshotError("SQLite WAL or shared-memory sidecar is present")
    if Path(str(path) + "-journal").exists():
        raise SnapshotError("SQLite rollback journal sidecar is present")
    uri = "file:" + quote(str(path.resolve())) + "?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        if not journal_mode or str(journal_mode[0]).lower() == "wal":
            raise SnapshotError("WAL-mode SQLite sources are not accepted")
        quick_check = connection.execute("PRAGMA quick_check").fetchall()
        if quick_check != [("ok",)]:
            raise SnapshotError("SQLite quick_check did not pass")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        normalized = {"trees", "pearls"}.issubset(tables)
        response_store = "api_responses" in tables
        if normalized and response_store:
            raise SnapshotError("ambiguous API SQLite schema")
        if normalized:
            tree_columns = _sqlite_columns(connection, "trees")
            pearl_columns = _sqlite_columns(connection, "pearls")
            required_trees = {"id", "title", "account", "visibility"}
            required_pearls = {"tree_id", "content_type", "content_tree_id"}
            if not required_trees.issubset(tree_columns) or not required_pearls.issubset(pearl_columns):
                raise SnapshotError("normalized API SQLite schema is incomplete")
            tree_query = "SELECT id,title,account,visibility"
            tree_query += " FROM trees ORDER BY CAST(id AS TEXT), rowid"
            for row_index, row in enumerate(connection.execute(tree_query)):
                tree_id = _strict_id(row[0], "SQLite tree ID")
                _observe_node(
                    state,
                    tree_id,
                    source_id=source["source_id"],
                    record_key=_record_key(
                        source["source_id"], "sqlite-tree", row_index, tree_id
                    ),
                    title=row[1],
                    account=row[2],
                    visibility=row[3],
                )
            optional = {
                "content_tree_title": "content_tree_title",
                "raw_json": "raw_json",
                "id": "id",
                "left_index": "left_index",
            }
            selected = ["tree_id", "content_type", "content_tree_id"]
            for column in optional:
                if column in pearl_columns:
                    selected.append(column)
            query = "SELECT " + ",".join(selected) + " FROM pearls ORDER BY CAST(tree_id AS TEXT)"
            if "left_index" in pearl_columns:
                query += ",left_index"
            if "id" in pearl_columns:
                query += ",id"
            query += ",rowid"
            for row_index, row in enumerate(connection.execute(query)):
                values = dict(zip(selected, row))
                content_type = _canonical_content_type(
                    values["content_type"], "SQLite content type"
                )
                parent = _strict_id(values["tree_id"], "SQLite parent ID")
                raw_content = None
                if values.get("raw_json") is not None:
                    raw_content = _strict_json_bytes(
                        _json_cell_bytes(
                            values["raw_json"], "SQLite pearl raw_json"
                        ),
                        "SQLite pearl raw_json",
                    )
                if raw_content is not None and not isinstance(raw_content, dict):
                    raise SnapshotError("SQLite pearl raw_json must be an object")
                _validate_sqlite_raw_relation(raw_content, content_type, parent)
                if content_type not in FOLDER_CONTENT_RELATIONS:
                    continue
                if values["content_tree_id"] is None:
                    raise SnapshotError("folder SQLite pearl is missing child ID")
                child = _strict_id(values["content_tree_id"], "SQLite child ID")
                relation = FOLDER_CONTENT_RELATIONS[content_type]
                pearl_record = _record_key(
                    source["source_id"],
                    "sqlite-pearl",
                    row_index,
                    child,
                    parent,
                    relation,
                )
                _observe_node(
                    state,
                    child,
                    source_id=source["source_id"],
                    record_key=_record_key(pearl_record, "normalized-columns"),
                    title=values.get("content_tree_title"),
                )
                if isinstance(raw_content, dict):
                    content_tree = _coalesced_alias(
                        raw_content,
                        "contentTree",
                        "content_tree",
                        "SQLite raw contentTree",
                        default={},
                    )
                    if not isinstance(content_tree, dict):
                        raise SnapshotError("SQLite raw contentTree must be an object")
                    if "id" in content_tree and _strict_id(
                        content_tree["id"], "SQLite raw child ID"
                    ) != child:
                        raise SnapshotError("SQLite raw child ID disagrees with columns")
                    _observe_node(
                        state,
                        child,
                        source_id=source["source_id"],
                        record_key=_record_key(pearl_record, "raw-content-tree"),
                        title=content_tree.get("title"),
                        visibility=(
                            content_tree["visibility"]
                            if "visibility" in content_tree
                            else MISSING
                        ),
                    )
                _add_edge(
                    state,
                    child,
                    parent,
                    relation,
                    {
                        "collection": "sqlite_collection",
                        "shortcut": "sqlite_shortcut",
                        "cross_link": "sqlite_content_type_6",
                    }[relation],
                    source["source_id"],
                    pearl_record,
                )
        elif response_store:
            columns = _sqlite_columns(connection, "api_responses")
            if not {"tree_id", "response_json"}.issubset(columns):
                raise SnapshotError("API response SQLite schema is incomplete")
            query = (
                "SELECT tree_id,response_json FROM api_responses "
                "ORDER BY CAST(tree_id AS TEXT),rowid"
            )
            for row_index, (tree_id, raw_json) in enumerate(connection.execute(query)):
                payload = _strict_json_bytes(
                    _json_cell_bytes(raw_json, "API response JSON"), "API response JSON"
                )
                if isinstance(payload, dict) and "tree_id" not in payload:
                    payload = dict(payload)
                    payload["tree_id"] = tree_id
                _parse_api_payload(
                    payload, source["source_id"], state, f"sqlite-response-{row_index}"
                )
        else:
            raise SnapshotError("unrecognized API SQLite schema")
    except sqlite3.Error as exc:
        raise SnapshotError("SQLite source could not be read coherently") from exc
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise SnapshotError("SQLite sidecar appeared during preparation")


def _directory_inventory(path):
    if not path.is_dir() or path.is_symlink():
        raise SnapshotError("API JSON source must be a non-symlink directory")
    entries = list(path.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise SnapshotError("API JSON directory must contain only regular non-symlink files")
    files = entries
    if not files:
        raise SnapshotError("API JSON directory is empty")
    records = []
    for file_path in files:
        data = _read_file(file_path)
        records.append((_content_record(data), file_path))
    records.sort(key=lambda item: (item[0]["sha256"], item[0]["size_bytes"]))
    return records


def _declared_source_size(path, kind):
    path = Path(path)
    if kind == "api_json_dir":
        if not path.is_dir() or path.is_symlink():
            raise SnapshotError("API JSON source must be a non-symlink directory")
        entries = list(path.iterdir())
        if not entries or any(item.is_symlink() or not item.is_file() for item in entries):
            raise SnapshotError("API JSON directory inventory is invalid")
        sizes = [item.stat().st_size for item in entries]
    else:
        if not path.is_file() or path.is_symlink():
            raise SnapshotError("declared source must be a regular non-symlink file")
        sizes = [path.stat().st_size]
    if any(size < 0 or size > MAX_SOURCE_BYTES for size in sizes):
        raise SnapshotError("declared source exceeds the frozen size limit")
    return sum(sizes)


def _parse_api_json_dir(path, source, state):
    parsed_tree_ids = []
    for index, (record, file_path) in enumerate(_directory_inventory(path)):
        data = _read_file(file_path)
        if _content_record(data) != record:
            raise SnapshotError("API JSON directory member changed during parsing")
        payload = _strict_json_bytes(data, "API JSON directory member")
        before_nodes = set(state["nodes"])
        _parse_api_payload(payload, source["source_id"], state, f"api-file-{index}")
        new_ids = sorted(set(state["nodes"]) - before_nodes, key=_stable_key)
        parsed_tree_ids.extend(new_ids)
    return parsed_tree_ids


def _parse_path_jsonl(path, source, state):
    data = _read_file(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SnapshotError("invalid UTF-8 path JSONL") from exc
    for line_index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise SnapshotError("blank path JSONL record")
        record = _strict_json_bytes(line.encode("utf-8"), "path JSONL record")
        if not isinstance(record, dict) or "tree_id" not in record:
            raise SnapshotError("path record must be an object with tree_id")
        tree_id = _strict_id(record["tree_id"], "path tree ID")
        line_record = _record_key(source["source_id"], "path-record", line_index, tree_id)
        declared_account = _safe_text(record.get("account"), "path account")
        _observe_node(
            state,
            tree_id,
            source_id=source["source_id"],
            record_key=line_record,
            title=record.get("title"),
            visibility=MISSING,
            account=declared_account,
        )
        raw_path = record.get("path_ids", [])
        if raw_path is None:
            raw_path = []
        if not isinstance(raw_path, list):
            raise SnapshotError("path_ids must be an array")
        clean = []
        root_account = None
        for raw_id in raw_path:
            if isinstance(raw_id, str) and raw_id.startswith("account:"):
                if clean or root_account is not None:
                    raise SnapshotError("account root marker must appear once before numeric path IDs")
                root_account = _account_marker(raw_id, "path account root")
                continue
            clean.append(_strict_id(raw_id, "path node ID"))
        if clean and clean[-1] != tree_id:
            raise SnapshotError("path endpoint does not match tree_id")
        if root_account is not None:
            if declared_account is not None and root_account != declared_account:
                raise SnapshotError("path account root disagrees with account field")
            root_node = clean[0] if clean else tree_id
            node = _ensure_node(state, root_node)
            node["accounts"].add(root_account)
            node["root_evidence"].add(
                _record_key(source["source_id"], "path_root", line_index, root_node)
            )
        for position in range(1, len(clean)):
            _add_edge(
                state,
                clean[position],
                clean[position - 1],
                "path",
                "path_chain",
                source["source_id"],
                _record_key(source["source_id"], "path-chain", line_index, position),
            )
        parent_value = record.get("parent_tree_id")
        if parent_value is not None:
            if isinstance(parent_value, str) and parent_value.startswith("account:"):
                parent_account = _account_marker(parent_value, "path parent account")
                if root_account is not None and parent_account != root_account:
                    raise SnapshotError("path parent account disagrees with path_ids")
                if declared_account is not None and parent_account != declared_account:
                    raise SnapshotError("path parent account disagrees with account field")
                node = _ensure_node(state, tree_id)
                node["accounts"].add(parent_account)
                node["root_evidence"].add(
                    _record_key(source["source_id"], "path_parent_root", line_index, tree_id)
                )
            else:
                parent = _strict_id(parent_value, "path parent ID")
                if len(clean) >= 2 and clean[-2] != parent:
                    raise SnapshotError("path parent does not match path_ids")
                _add_edge(
                    state,
                    tree_id,
                    parent,
                    "path",
                    "path_parent",
                    source["source_id"],
                    _record_key(source["source_id"], "path-parent", line_index),
                )


def _resolve_source_paths(spec_path, spec):
    expected = {"schema", "snapshot_label", "sources"}
    if set(spec) - (expected | {"legacy_check"}):
        raise SnapshotError("source spec has unknown fields")
    if spec.get("schema") != SOURCE_SPEC_SCHEMA:
        raise SnapshotError("source spec schema mismatch")
    if not isinstance(spec.get("snapshot_label"), str) or not spec["snapshot_label"].strip():
        raise SnapshotError("snapshot_label must be nonempty text")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SnapshotError("source spec must declare a nonempty sources array")
    base = spec_path.parent
    resolved = []
    seen_ids = set()
    for entry in sources:
        if not isinstance(entry, dict):
            raise SnapshotError("source entry must be an object")
        allowed = {"kind", "source_id", "path", "account"}
        if set(entry) - allowed or not {"kind", "source_id", "path"}.issubset(entry):
            raise SnapshotError("source entry fields mismatch")
        kind = entry["kind"]
        source_id = entry["source_id"]
        if kind not in SOURCE_KINDS:
            raise SnapshotError("unsupported source kind")
        if not isinstance(source_id, str) or not source_id or source_id in seen_ids:
            raise SnapshotError("source_id must be unique nonempty text")
        if len(source_id) > 200:
            raise SnapshotError("source_id exceeds the frozen length limit")
        path_value = entry["path"]
        if not isinstance(path_value, str) or not path_value:
            raise SnapshotError("source path must be nonempty text")
        declared_path = base / path_value if not Path(path_value).is_absolute() else Path(path_value)
        if declared_path.is_symlink():
            raise SnapshotError("declared source cannot be a symlink")
        path = declared_path.resolve()
        item = dict(entry)
        item["path"] = path
        resolved.append(item)
        seen_ids.add(source_id)
    resolved.sort(key=lambda item: item["source_id"])

    legacy = spec.get("legacy_check")
    legacy_resolved = None
    if legacy is not None:
        if not isinstance(legacy, dict) or set(legacy) != {"dag_path"}:
            raise SnapshotError("legacy_check must contain only dag_path")
        value = legacy["dag_path"]
        if not isinstance(value, str) or not value:
            raise SnapshotError("legacy dag_path must be nonempty text")
        declared_legacy = base / value if not Path(value).is_absolute() else Path(value)
        if declared_legacy.is_symlink():
            raise SnapshotError("legacy parity input cannot be a symlink")
        legacy_resolved = declared_legacy.resolve()
    return resolved, legacy_resolved


def _validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {"schema", "physical_edges"}:
        raise SnapshotError("relation policy fields mismatch")
    if policy["schema"] != POLICY_SCHEMA:
        raise SnapshotError("relation policy schema mismatch")
    edges = policy["physical_edges"]
    if not isinstance(edges, dict) or set(edges) != set(RELATIONS):
        raise SnapshotError("relation policy must name every relation")
    if any(not isinstance(edges[key], bool) for key in RELATIONS):
        raise SnapshotError("relation policy decisions must be Boolean")
    if not any(edges.values()):
        raise SnapshotError("relation policy cannot exclude every relation")
    return {"schema": POLICY_SCHEMA, "physical_edges": {key: edges[key] for key in RELATIONS}}


def _load_policy(path, data=None):
    policy = (
        _strict_json_file(path, "relation policy")
        if data is None
        else _strict_json_bytes(data, "relation policy")
    )
    return _validate_policy(policy)


def _expected_privacy_policy():
    return {
        "private_title_rule": "privacy.py:word-boundary-private",
        "propagation_relations": sorted(PRIVACY_RELATIONS),
        "unknown_visibility": "retained_locally_ineligible_and_uncertified",
    }


def _parse_sources(spec_path, sources):
    state = _new_state()
    source_records = []
    resolved_inputs = []
    for source in sources:
        path = source["path"]
        resolved_inputs.append(path)
        if source["kind"] == "api_json_dir":
            before = _directory_inventory(path)
            before_records = [item[0] for item in before]
        else:
            before_records = [_file_record(path)]
        if source["kind"] == "rdf":
            _parse_rdf(path, source, state)
        elif source["kind"] == "api_sqlite":
            _parse_api_sqlite(path, source, state)
        elif source["kind"] == "api_json_dir":
            _parse_api_json_dir(path, source, state)
        elif source["kind"] == "path_jsonl":
            _parse_path_jsonl(path, source, state)
        if source["kind"] == "api_json_dir":
            after_records = [item[0] for item in _directory_inventory(path)]
        else:
            after_records = [_file_record(path)]
        if before_records != after_records:
            raise SnapshotError("declared source changed during preparation")
        source_records.append(
            {
                "kind": source["kind"],
                "source_id": source["source_id"],
                "content_records": before_records,
                "account_declared": bool(source.get("account")),
            }
        )
    return state, source_records, resolved_inputs


def _privacy_and_graph(state, policy):
    edges = sorted(
        {
            (
                edge["child"],
                edge["parent"],
                edge["relation"],
                edge["subtype"],
                edge["source_id"],
                edge["record_key"],
            )
            for edge in state["edges"]
        },
        key=lambda row: (
            _stable_key(row[0]),
            _stable_key(row[1]),
            row[2],
            row[3],
            row[4],
            row[5],
        ),
    )
    visibility_records = [
        {
            "node_id": node_id,
            "record_key": record_key,
            "source_id": source_id,
            "visibility": visibility,
        }
        for node_id, node in state["nodes"].items()
        for source_id, record_key, visibility in node["visibility_evidence"]
    ]
    visibility_records.sort(
        key=lambda row: (
            _stable_key(row["node_id"]),
            row["source_id"],
            row["record_key"],
            row["visibility"],
        )
    )
    direct_private = set()
    visibility_final = {}
    conflicts = []
    for node_id, node in state["nodes"].items():
        states = set(node["visibilities"])
        if "private" in states or node["private_title"]:
            final = "private"
            direct_private.add(node_id)
        elif not states or "unknown" in states:
            final = "unknown"
        else:
            final = "public"
        visibility_final[node_id] = final
        if len(node["titles"]) > 1:
            conflicts.append({"kind": "title", "node_id": node_id, "value_count": len(node["titles"])})
        if len(node["accounts"]) > 1:
            conflicts.append({"kind": "account", "node_id": node_id, "value_count": len(node["accounts"])})
        if len(states) > 1:
            conflicts.append({"kind": "visibility", "node_id": node_id, "values": sorted(states)})

    children = defaultdict(set)
    for child, parent, relation, _subtype, _source, _record in edges:
        if relation in PRIVACY_RELATIONS:
            children[parent].add(child)
    private = set(direct_private)
    queue = deque(sorted(private, key=_stable_key))
    while queue:
        parent = queue.popleft()
        for child in sorted(children.get(parent, ()), key=_stable_key):
            if child not in private:
                private.add(child)
                queue.append(child)

    retained = set(state["nodes"]) - private
    evidence_records = []
    preprivacy_physical = set()
    physical = set()
    for child, parent, relation, subtype, source_id, record_key in edges:
        pair = tuple(sorted((child, parent), key=_stable_key))
        included = policy["physical_edges"][relation]
        excluded_endpoint = child in private or parent in private
        if included:
            preprivacy_physical.add(pair)
            if not excluded_endpoint:
                physical.add(pair)
        evidence_records.append(
            {
                "child": child,
                "parent": parent,
                "physical_policy_included": included,
                "privacy_endpoint_excluded": excluded_endpoint,
                "record_key": record_key,
                "relation": relation,
                "source_id": source_id,
                "subtype": subtype,
            }
        )

    adjacency = {node_id: set() for node_id in retained}
    for left, right in physical:
        adjacency[left].add(right)
        adjacency[right].add(left)

    components = []
    unseen = set(retained)
    while unseen:
        seed = min(unseen, key=_stable_key)
        members = []
        frontier = [seed]
        unseen.remove(seed)
        while frontier:
            node = frontier.pop()
            members.append(node)
            for neighbor in sorted(adjacency[node], key=_stable_key, reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    frontier.append(neighbor)
        members.sort(key=_stable_key)
        member_set = set(members)
        edge_count = sum(
            1 for left, right in physical if left in member_set and right in member_set
        )
        fingerprint = hashlib.sha256(_canonical_json(members)).hexdigest()
        components.append(
            {
                "component_id": fingerprint,
                "edge_count": edge_count,
                "node_count": len(members),
                "nodes": members,
            }
        )
    components.sort(key=lambda item: (-item["node_count"], -item["edge_count"], item["component_id"]))
    largest_id = components[0]["component_id"] if components else None
    component_of = {
        node: component["component_id"]
        for component in components
        for node in component["nodes"]
    }

    node_records = []
    exclusion_records = []
    eligibility = []
    eligible_nodes = []
    for node_id in sorted(state["nodes"], key=_stable_key):
        node = state["nodes"][node_id]
        titles = sorted(node["titles"])
        accounts = sorted(node["accounts"])
        excluded = node_id in private
        node_records.append(
            {
                "accounts": accounts,
                "excluded": excluded,
                "node_id": node_id,
                "root_evidence": sorted(node["root_evidence"]),
                "sources": sorted(node["sources"]),
                "titles": titles,
                "visibility": visibility_final[node_id],
            }
        )
        if excluded:
            reason = "direct_private" if node_id in direct_private else "private_descendant"
            exclusion_records.append({"node_id": node_id, "reason": reason})
            eligibility.append({"eligible": False, "node_id": node_id, "reason": reason})
        else:
            degree = len(adjacency[node_id])
            if visibility_final[node_id] != "public":
                reason = "unknown_visibility"
                ok = False
            elif component_of.get(node_id) != largest_id:
                reason = "outside_largest_component"
                ok = False
            elif degree == 0:
                reason = "isolated"
                ok = False
            else:
                reason = "eligible"
                ok = True
                eligible_nodes.append(node_id)
            eligibility.append(
                {
                    "component_id": component_of.get(node_id),
                    "degree": degree,
                    "eligible": ok,
                    "node_id": node_id,
                    "reason": reason,
                }
            )

    adjacency_records = [
        {
            "neighbors": sorted(adjacency[node_id], key=_stable_key),
            "node_id": node_id,
        }
        for node_id in sorted(adjacency, key=_stable_key)
    ]
    conflicts.sort(key=lambda item: (item["kind"], _stable_key(item["node_id"])))
    retained_ordered = sorted(retained, key=_stable_key)
    study_universe_sha256 = hashlib.sha256(_canonical_json(retained_ordered)).hexdigest()
    empty_membership_sha256 = hashlib.sha256(_canonical_json([])).hexdigest()
    privacy_certified = all(
        visibility_final[node_id] == "public" for node_id in retained
    )
    return {
        "nodes": node_records,
        "visibility_evidence": visibility_records,
        "exclusions": exclusion_records,
        "conflicts": conflicts,
        "evidence": evidence_records,
        "physical": sorted(physical, key=lambda pair: (_stable_key(pair[0]), _stable_key(pair[1]))),
        "preprivacy_physical": preprivacy_physical,
        "adjacency": adjacency_records,
        "components": components,
        "eligibility": eligibility,
        "eligible_nodes": eligible_nodes,
        "largest_component_id": largest_id,
        "largest_component_sha256": largest_id or empty_membership_sha256,
        "study_universe_sha256": study_universe_sha256,
        "privacy_certified": privacy_certified,
        "direct_private_count": len(direct_private),
        "private_count": len(private),
        "retained_count": len(retained),
        "unknown_retained_count": sum(
            1 for node_id in retained if visibility_final[node_id] == "unknown"
        ),
    }


def _legacy_parity(path, authoritative_pairs):
    if path is None:
        return {"status": "not_run"}
    data = _read_file(path)
    pairs = set()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SnapshotError("legacy parity file is not UTF-8") from exc
    for line in text.splitlines():
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise SnapshotError("legacy parity row must have two tab-separated IDs")
        child = _strict_id(fields[0], "legacy child ID")
        parent = _strict_id(fields[1], "legacy parent ID")
        if child == parent:
            raise SnapshotError("legacy parity contains a self edge")
        pairs.add(tuple(sorted((child, parent), key=_stable_key)))
    missing = authoritative_pairs - pairs
    extra = pairs - authoritative_pairs
    return {
        "extra_edge_count": len(extra),
        "legacy_record": _content_record(data),
        "missing_edge_count": len(missing),
        "status": "match" if not missing and not extra else "mismatch",
    }


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        value = "unavailable"
    return value


def _path_is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_git_worktree_marker(path):
    """Recognize an actual Git marker, not an unrelated empty `.git` path."""

    try:
        if path.is_symlink():
            return True
        if path.is_file():
            with open(path, "rb") as stream:
                return stream.read(256).lstrip().startswith(b"gitdir:")
        return path.is_dir() and (path / "HEAD").is_file()
    except OSError:
        # An unreadable candidate marker is conservatively treated as Git.
        return True


def _validate_output_paths(run_dir, local_root, inputs):
    run_input = Path(run_dir)
    local_input = Path(local_root)
    if local_input.is_symlink() or run_input.is_symlink():
        raise SnapshotError("output paths cannot be symlinks")
    local_root = local_input.resolve()
    run_dir = run_input.resolve()
    if not local_root.is_dir():
        raise SnapshotError("local root must be an existing non-symlink directory")
    if not _path_is_within(run_dir, local_root) or run_dir == local_root:
        raise SnapshotError("run directory must be a child of the approved local root")
    if _path_is_within(run_dir, REPO_ROOT.resolve()):
        raise SnapshotError("run directory cannot be inside the Git repository")
    if any(
        _is_git_worktree_marker(ancestor / ".git")
        for ancestor in (run_dir.parent, *run_dir.parents)
    ):
        raise SnapshotError("run directory cannot be inside a Git worktree")
    if run_dir.exists():
        raise SnapshotError("run directory already exists")
    if not run_dir.parent.is_dir():
        raise SnapshotError("run-directory parent must already exist")
    for source in inputs:
        source = source.resolve()
        if (
            source == run_dir
            or _path_is_within(source, run_dir)
            or _path_is_within(run_dir, source)
        ):
            raise SnapshotError("input and output paths overlap")
        try:
            if source.stat().st_dev == local_root.stat().st_dev and source.stat().st_ino == local_root.stat().st_ino:
                raise SnapshotError("local root aliases an input")
        except OSError as exc:
            raise SnapshotError("input path could not be validated") from exc
    return run_dir, local_root


def _rename_directory_noreplace(source, target):
    """Atomically install a directory without replacing any concurrent target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SnapshotError("atomic no-replace rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SnapshotError("run directory appeared during atomic installation")
    raise SnapshotError("atomic no-replace installation failed") from OSError(
        error_number, os.strerror(error_number)
    )


def _artifact_payloads(compiled, source_records, parity, policy):
    sources_payload = {
        "schema_version": SCHEMA_VERSION,
        "sources": source_records,
    }
    edge_lines = ["# left\tright\tconductance\n"]
    for left, right in compiled["physical"]:
        edge_lines.append(f"{left}\t{right}\t1\n")
    scrub = {
        "direct_private_count": compiled["direct_private_count"],
        "excluded_private_count": compiled["private_count"],
        "privacy_certified": compiled["privacy_certified"],
        "retained_node_count": compiled["retained_count"],
        "unknown_retained_count": compiled["unknown_retained_count"],
    }
    aggregate = {
        "artifact_schema": "pearltrees-diffusion-aggregate-v1",
        "eligible_anchor_count": len(compiled["eligible_nodes"]),
        "largest_component_node_count": (
            compiled["components"][0]["node_count"] if compiled["components"] else 0
        ),
        "physical_edge_count": len(compiled["physical"]),
        "privacy_certified": compiled["privacy_certified"],
        "publishable": False,
        "requires_manual_approval": True,
        "retained_node_count": compiled["retained_count"],
    }
    return {
        "sources.json": _canonical_json(sources_payload),
        "nodes.jsonl": _jsonl_bytes(compiled["nodes"]),
        "visibility_evidence.jsonl": _jsonl_bytes(compiled["visibility_evidence"]),
        "exclusions.jsonl": _jsonl_bytes(compiled["exclusions"]),
        "conflicts.jsonl": _jsonl_bytes(compiled["conflicts"]),
        "edge_evidence.jsonl": _jsonl_bytes(compiled["evidence"]),
        "physical_edges.tsv": "".join(edge_lines).encode("utf-8"),
        "adjacency.jsonl": _jsonl_bytes(compiled["adjacency"]),
        "components.jsonl": _jsonl_bytes(compiled["components"]),
        "anchor_eligibility.jsonl": _jsonl_bytes(compiled["eligibility"]),
        "scrub_manifest.json": _canonical_json(scrub),
        "legacy_parity.json": _canonical_json(parity),
        "aggregate_release_candidate.json": _canonical_json(aggregate),
    }


def _write_bytes(path, data):
    with open(path, "xb") as stream:
        os.chmod(path, 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _load_jsonl(data, label):
    records = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SnapshotError(f"{label} is not UTF-8") from exc
    if data and not text.endswith("\n"):
        raise SnapshotError(f"{label} is not newline terminated")
    for line in text.splitlines():
        record = _strict_json_bytes(line.encode("utf-8"), label)
        if not isinstance(record, dict):
            raise SnapshotError(f"{label} record must be an object")
        records.append(record)
    if _jsonl_bytes(records) != data:
        raise SnapshotError(f"{label} is not canonical JSONL")
    return records


def _verify_invariants(run_dir, manifest):
    def artifact_id(value, label):
        if not isinstance(value, str) or not value.startswith("pt:"):
            raise SnapshotError(f"{label} is not a typed Pearltrees ID")
        try:
            canonical = _strict_id(value.split(":", 1)[1], label)
        except (SnapshotError, ValueError, AttributeError) as exc:
            raise SnapshotError(f"{label} is not a typed Pearltrees ID") from exc
        if canonical != value:
            raise SnapshotError(f"{label} is not canonical")
        return value

    def content_record_ok(value):
        return (
            isinstance(value, dict)
            and set(value) == {"sha256", "size_bytes"}
            and isinstance(value["size_bytes"], int)
            and not isinstance(value["size_bytes"], bool)
            and value["size_bytes"] >= 0
            and isinstance(value["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
        )

    sources_payload = _strict_json_bytes(
        (run_dir / "sources.json").read_bytes(), "sources"
    )
    if (
        not isinstance(sources_payload, dict)
        or set(sources_payload) != {"schema_version", "sources"}
        or sources_payload["schema_version"] != SCHEMA_VERSION
        or not isinstance(sources_payload["sources"], list)
    ):
        raise SnapshotError("sources artifact schema mismatch")
    source_records = sources_payload["sources"]
    if source_records != manifest["fingerprint_core"].get("source_records"):
        raise SnapshotError("source records disagree with fingerprint core")
    source_ids = []
    for source in source_records:
        if (
            not isinstance(source, dict)
            or set(source)
            != {"account_declared", "content_records", "kind", "source_id"}
            or source.get("kind") not in SOURCE_KINDS
            or not isinstance(source.get("source_id"), str)
            or not source["source_id"]
            or not isinstance(source.get("account_declared"), bool)
            or not isinstance(source.get("content_records"), list)
            or not source["content_records"]
            or any(not content_record_ok(item) for item in source["content_records"])
        ):
            raise SnapshotError("source record is malformed")
        source_ids.append(source["source_id"])
    if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
        raise SnapshotError("source records are not uniquely sorted")
    source_id_set = set(source_ids)

    nodes = _load_jsonl((run_dir / "nodes.jsonl").read_bytes(), "nodes")
    visibility_evidence = _load_jsonl(
        (run_dir / "visibility_evidence.jsonl").read_bytes(), "visibility evidence"
    )
    exclusions = _load_jsonl(
        (run_dir / "exclusions.jsonl").read_bytes(), "exclusions"
    )
    conflicts = _load_jsonl((run_dir / "conflicts.jsonl").read_bytes(), "conflicts")
    evidence = _load_jsonl(
        (run_dir / "edge_evidence.jsonl").read_bytes(), "edge evidence"
    )
    adjacency_rows = _load_jsonl(
        (run_dir / "adjacency.jsonl").read_bytes(), "adjacency"
    )
    components = _load_jsonl(
        (run_dir / "components.jsonl").read_bytes(), "components"
    )
    eligibility = _load_jsonl(
        (run_dir / "anchor_eligibility.jsonl").read_bytes(), "anchor eligibility"
    )

    node_map = {}
    for row in nodes:
        if set(row) != {
            "accounts",
            "excluded",
            "node_id",
            "root_evidence",
            "sources",
            "titles",
            "visibility",
        }:
            raise SnapshotError("nodes artifact record fields mismatch")
        node_id = artifact_id(row["node_id"], "node ID")
        if node_id in node_map:
            raise SnapshotError("nodes artifact has duplicate IDs")
        if not isinstance(row["excluded"], bool) or row["visibility"] not in {
            "public",
            "private",
            "unknown",
        }:
            raise SnapshotError("node privacy fields are malformed")
        for key in ("accounts", "root_evidence", "sources", "titles"):
            values = row[key]
            if (
                not isinstance(values, list)
                or any(not isinstance(item, str) for item in values)
                or values != sorted(set(values))
            ):
                raise SnapshotError("node list field is not canonical")
        if not set(row["sources"]).issubset(source_id_set):
            raise SnapshotError("node cites an unknown source")
        node_map[node_id] = row
    ordered_node_ids = sorted(node_map, key=_stable_key)
    if [row["node_id"] for row in nodes] != ordered_node_ids:
        raise SnapshotError("nodes artifact is not canonically ordered")

    visibility_by_node = defaultdict(list)
    expected_visibility_order = sorted(
        visibility_evidence,
        key=lambda row: (
            _stable_key(row.get("node_id", "pt:0")),
            row.get("source_id", ""),
            row.get("record_key", ""),
            row.get("visibility", ""),
        ),
    )
    if visibility_evidence != expected_visibility_order:
        raise SnapshotError("visibility evidence is not canonically ordered")
    seen_visibility_rows = set()
    for row in visibility_evidence:
        if set(row) != {"node_id", "record_key", "source_id", "visibility"}:
            raise SnapshotError("visibility evidence fields mismatch")
        node_id = artifact_id(row["node_id"], "visibility node ID")
        if node_id not in node_map or row["source_id"] not in source_id_set:
            raise SnapshotError("visibility evidence cites an unknown node or source")
        if (
            not isinstance(row["record_key"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["record_key"]) is None
            or row["visibility"] not in {"public", "private", "unknown"}
        ):
            raise SnapshotError("visibility evidence is malformed")
        identity = (
            node_id,
            row["source_id"],
            row["record_key"],
            row["visibility"],
        )
        if identity in seen_visibility_rows:
            raise SnapshotError("visibility evidence has a duplicate row")
        seen_visibility_rows.add(identity)
        visibility_by_node[node_id].append(row)

    for node_id, node in node_map.items():
        observations = visibility_by_node.get(node_id, [])
        if not observations:
            raise SnapshotError("node has no visibility evidence")
        observed_sources = sorted({row["source_id"] for row in observations})
        if node["sources"] != observed_sources:
            raise SnapshotError("node sources disagree with visibility evidence")
        known = {row["visibility"] for row in observations} - {"unknown"}
        if "private" in known or any(is_private_title(title) for title in node["titles"]):
            expected_visibility = "private"
        elif "public" in known:
            expected_visibility = "public"
        else:
            expected_visibility = "unknown"
        if node["visibility"] != expected_visibility:
            raise SnapshotError("node visibility disagrees with source evidence")

    direct_private = {
        node_id
        for node_id, row in node_map.items()
        if row["visibility"] == "private"
        or any(is_private_title(title) for title in row["titles"])
    }
    policy = manifest["relation_policy"]
    expected_evidence_order = sorted(
        evidence,
        key=lambda row: (
            _stable_key(row.get("child", "pt:0")),
            _stable_key(row.get("parent", "pt:0")),
            row.get("relation", ""),
            row.get("subtype", ""),
            row.get("source_id", ""),
            row.get("record_key", ""),
        ),
    )
    if evidence != expected_evidence_order:
        raise SnapshotError("edge evidence is not canonically ordered")
    privacy_children = defaultdict(set)
    for row in evidence:
        if set(row) != {
            "child",
            "parent",
            "physical_policy_included",
            "privacy_endpoint_excluded",
            "record_key",
            "relation",
            "source_id",
            "subtype",
        }:
            raise SnapshotError("edge evidence fields mismatch")
        child = artifact_id(row["child"], "edge child")
        parent = artifact_id(row["parent"], "edge parent")
        if child == parent or child not in node_map or parent not in node_map:
            raise SnapshotError("edge evidence has invalid endpoints")
        if row["relation"] not in RELATIONS or row["source_id"] not in source_id_set:
            raise SnapshotError("edge evidence relation or source is unknown")
        if (
            not isinstance(row["subtype"], str)
            or not isinstance(row["record_key"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["record_key"]) is None
            or not isinstance(row["physical_policy_included"], bool)
            or not isinstance(row["privacy_endpoint_excluded"], bool)
        ):
            raise SnapshotError("edge evidence metadata is malformed")
        if row["physical_policy_included"] is not policy["physical_edges"][row["relation"]]:
            raise SnapshotError("edge evidence disagrees with relation policy")
        if row["relation"] in PRIVACY_RELATIONS:
            privacy_children[parent].add(child)

    expected_private = set(direct_private)
    queue = deque(sorted(expected_private, key=_stable_key))
    while queue:
        parent = queue.popleft()
        for child in sorted(privacy_children.get(parent, ()), key=_stable_key):
            if child not in expected_private:
                expected_private.add(child)
                queue.append(child)
    declared_excluded = {node_id for node_id, row in node_map.items() if row["excluded"]}
    if declared_excluded != expected_private:
        raise SnapshotError("privacy closure does not match node exclusions")

    exclusion_map = {}
    for row in exclusions:
        if set(row) != {"node_id", "reason"}:
            raise SnapshotError("exclusion record fields mismatch")
        node_id = artifact_id(row["node_id"], "excluded node ID")
        if node_id in exclusion_map or row["reason"] not in {
            "direct_private",
            "private_descendant",
        }:
            raise SnapshotError("exclusion record is malformed or duplicated")
        exclusion_map[node_id] = row["reason"]
    expected_exclusions = {
        node_id: "direct_private" if node_id in direct_private else "private_descendant"
        for node_id in expected_private
    }
    if exclusion_map != expected_exclusions or [row["node_id"] for row in exclusions] != sorted(
        expected_private, key=_stable_key
    ):
        raise SnapshotError("exclusion ledger disagrees with privacy closure")

    for row in evidence:
        should_exclude = row["child"] in expected_private or row["parent"] in expected_private
        if row["privacy_endpoint_excluded"] is not should_exclude:
            raise SnapshotError("edge evidence privacy flag is inconsistent")

    edge_data = (run_dir / "physical_edges.tsv").read_bytes()
    try:
        edge_text = edge_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SnapshotError("physical edge artifact is not UTF-8") from exc
    if not edge_text.endswith("\n") or not edge_text.startswith(
        "# left\tright\tconductance\n"
    ):
        raise SnapshotError("physical edge header or termination mismatch")
    physical = []
    for line in edge_text.splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) != 3 or fields[2] != "1":
            raise SnapshotError("physical edge row is malformed")
        left = artifact_id(fields[0], "physical left node")
        right = artifact_id(fields[1], "physical right node")
        pair = tuple(sorted((left, right), key=_stable_key))
        if left == right or pair != (left, right) or left in expected_private or right in expected_private:
            raise SnapshotError("physical edge is noncanonical or privacy-excluded")
        physical.append(pair)
    if physical != sorted(set(physical), key=lambda pair: (_stable_key(pair[0]), _stable_key(pair[1]))):
        raise SnapshotError("physical edges are not uniquely sorted")
    expected_physical = sorted(
        {
            tuple(sorted((row["child"], row["parent"]), key=_stable_key))
            for row in evidence
            if row["physical_policy_included"] and not row["privacy_endpoint_excluded"]
        },
        key=lambda pair: (_stable_key(pair[0]), _stable_key(pair[1])),
    )
    if physical != expected_physical:
        raise SnapshotError("physical edges disagree with evidence and policy")

    retained = set(node_map) - expected_private
    adjacency = {}
    for row in adjacency_rows:
        if set(row) != {"neighbors", "node_id"}:
            raise SnapshotError("adjacency record fields mismatch")
        node_id = artifact_id(row["node_id"], "adjacency node ID")
        neighbors = row["neighbors"]
        if node_id not in retained or node_id in adjacency or not isinstance(neighbors, list):
            raise SnapshotError("adjacency row has invalid node or neighbors")
        checked = [artifact_id(item, "adjacency neighbor ID") for item in neighbors]
        if checked != sorted(set(checked), key=_stable_key) or node_id in checked:
            raise SnapshotError("adjacency neighbors are not canonical")
        if any(item not in retained for item in checked):
            raise SnapshotError("adjacency cites a non-retained node")
        adjacency[node_id] = checked
    if [row["node_id"] for row in adjacency_rows] != sorted(retained, key=_stable_key):
        raise SnapshotError("adjacency does not cover retained nodes canonically")
    for node_id, neighbors in adjacency.items():
        for neighbor in neighbors:
            if node_id not in adjacency.get(neighbor, []):
                raise SnapshotError("adjacency is not reciprocal")
    adjacency_edges = sorted(
        {
            tuple(sorted((node_id, neighbor), key=_stable_key))
            for node_id, neighbors in adjacency.items()
            for neighbor in neighbors
        },
        key=lambda pair: (_stable_key(pair[0]), _stable_key(pair[1])),
    )
    if adjacency_edges != physical:
        raise SnapshotError("adjacency edge set disagrees with physical edges")

    expected_components = []
    unseen = set(retained)
    physical_set = set(physical)
    while unseen:
        seed = min(unseen, key=_stable_key)
        unseen.remove(seed)
        frontier = [seed]
        members = []
        while frontier:
            node_id = frontier.pop()
            members.append(node_id)
            for neighbor in sorted(adjacency[node_id], key=_stable_key, reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    frontier.append(neighbor)
        members.sort(key=_stable_key)
        member_set = set(members)
        component_id = hashlib.sha256(_canonical_json(members)).hexdigest()
        expected_components.append(
            {
                "component_id": component_id,
                "edge_count": sum(
                    1 for left, right in physical_set if left in member_set and right in member_set
                ),
                "node_count": len(members),
                "nodes": members,
            }
        )
    expected_components.sort(
        key=lambda row: (-row["node_count"], -row["edge_count"], row["component_id"])
    )
    if components != expected_components:
        raise SnapshotError("component ledger does not match adjacency")
    largest_id = expected_components[0]["component_id"] if expected_components else None
    component_of = {
        node_id: component["component_id"]
        for component in expected_components
        for node_id in component["nodes"]
    }

    expected_eligibility = []
    for node_id in ordered_node_ids:
        row = node_map[node_id]
        if node_id in expected_private:
            reason = "direct_private" if node_id in direct_private else "private_descendant"
            expected_eligibility.append({"eligible": False, "node_id": node_id, "reason": reason})
            continue
        degree = len(adjacency[node_id])
        if row["visibility"] != "public":
            reason, eligible_flag = "unknown_visibility", False
        elif component_of.get(node_id) != largest_id:
            reason, eligible_flag = "outside_largest_component", False
        elif degree == 0:
            reason, eligible_flag = "isolated", False
        else:
            reason, eligible_flag = "eligible", True
        expected_eligibility.append(
            {
                "component_id": component_of.get(node_id),
                "degree": degree,
                "eligible": eligible_flag,
                "node_id": node_id,
                "reason": reason,
            }
        )
    if eligibility != expected_eligibility:
        raise SnapshotError("anchor eligibility does not match privacy and components")
    eligible_count = sum(row["eligible"] is True for row in expected_eligibility)

    expected_scrub = {
        "direct_private_count": len(direct_private),
        "excluded_private_count": len(expected_private),
        "privacy_certified": all(node_map[node_id]["visibility"] == "public" for node_id in retained),
        "retained_node_count": len(retained),
        "unknown_retained_count": sum(
            node_map[node_id]["visibility"] == "unknown" for node_id in retained
        ),
    }
    scrub = _strict_json_bytes(
        (run_dir / "scrub_manifest.json").read_bytes(), "scrub manifest"
    )
    if scrub != expected_scrub:
        raise SnapshotError("scrub manifest disagrees with verified privacy state")

    study_universe_sha256 = hashlib.sha256(
        _canonical_json(sorted(retained, key=_stable_key))
    ).hexdigest()
    empty_hash = hashlib.sha256(_canonical_json([])).hexdigest()
    largest_component_sha256 = largest_id or empty_hash
    core = manifest["fingerprint_core"]
    if core.get("study_universe_sha256") != study_universe_sha256:
        raise SnapshotError("study-universe hash mismatch")
    if core.get("largest_component_sha256") != largest_component_sha256:
        raise SnapshotError("largest-component hash mismatch")

    aggregate_file = _strict_json_bytes(
        (run_dir / "aggregate_release_candidate.json").read_bytes(), "aggregate"
    )
    expected_aggregate_file = {
        "artifact_schema": "pearltrees-diffusion-aggregate-v1",
        "eligible_anchor_count": eligible_count,
        "largest_component_node_count": (
            expected_components[0]["node_count"] if expected_components else 0
        ),
        "physical_edge_count": len(physical),
        "privacy_certified": expected_scrub["privacy_certified"],
        "publishable": False,
        "requires_manual_approval": True,
        "retained_node_count": len(retained),
    }
    if aggregate_file != expected_aggregate_file:
        raise SnapshotError("aggregate release candidate mismatch")
    aggregate = manifest["aggregate"]
    for key, value in expected_aggregate_file.items():
        if aggregate.get(key) != value:
            raise SnapshotError("manifest aggregate disagrees with verified artifacts")
    if aggregate.get("eligible_anchor_count") != eligible_count:
        raise SnapshotError("eligible anchor count mismatch")
    if aggregate.get("study_universe_sha256") != study_universe_sha256:
        raise SnapshotError("manifest study-universe hash mismatch")
    if aggregate.get("largest_component_sha256") != largest_component_sha256:
        raise SnapshotError("manifest largest-component hash mismatch")
    minimum = aggregate.get("minimum_anchor_count")
    coverage = aggregate.get("minimum_anchor_coverage_pass")
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or minimum < 1
        or coverage is not (eligible_count >= minimum)
    ):
        raise SnapshotError("minimum-anchor coverage decision mismatch")
    expected_ready = expected_scrub["privacy_certified"] and coverage
    if manifest.get("graph_asset_ready") is not expected_ready:
        raise SnapshotError("graph-asset readiness decision mismatch")

    parity = _strict_json_bytes(
        (run_dir / "legacy_parity.json").read_bytes(), "legacy parity"
    )
    if not isinstance(parity, dict) or parity.get("status") not in {
        "not_run",
        "match",
        "mismatch",
    }:
        raise SnapshotError("legacy parity artifact is malformed")
    if aggregate.get("legacy_parity_status") != parity["status"]:
        raise SnapshotError("legacy parity status mismatch")

    # Conflicts are local diagnostics, but still require canonical, typed records.
    for row in conflicts:
        if not isinstance(row, dict) or row.get("kind") not in {
            "account",
            "title",
            "visibility",
        }:
            raise SnapshotError("conflict ledger is malformed")
        artifact_id(row.get("node_id"), "conflict node ID")


def verify_snapshot(run_dir):
    run_input = Path(run_dir)
    if run_input.is_symlink():
        raise SnapshotError("snapshot run directory cannot be a symlink")
    run_dir = run_input.resolve()
    if not run_dir.is_dir():
        raise SnapshotError("snapshot run directory is unavailable")
    entries = list(run_dir.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise SnapshotError("snapshot directory contains a nonregular entry")
    observed_files = {item.name for item in entries}
    if observed_files != ALL_RUN_FILES:
        raise SnapshotError("snapshot file set mismatch")
    if stat.S_IMODE(run_dir.stat().st_mode) != 0o700:
        raise SnapshotError("snapshot directory mode is not 0700")
    if any(stat.S_IMODE(item.stat().st_mode) != 0o600 for item in entries):
        raise SnapshotError("snapshot artifact mode is not 0600")
    marker = run_dir / "LOCAL_ONLY_DO_NOT_PUBLISH"
    if marker.read_bytes() != b"LOCAL ONLY - DO NOT PUBLISH NODE-LEVEL ARTIFACTS\n":
        raise SnapshotError("local-only marker mismatch")
    manifest_data = (run_dir / "manifest.json").read_bytes()
    manifest = _strict_json_bytes(manifest_data, "manifest")
    if _canonical_json(manifest) != manifest_data:
        raise SnapshotError("manifest is not canonical JSON")
    expected_keys = {
        "aggregate",
        "algorithm",
        "artifact_records",
        "fingerprint_core",
        "privacy_policy",
        "relation_policy",
        "repository_commit",
        "schema_version",
        "snapshot_fingerprint",
        "snapshot_label_hash",
        "graph_asset_ready",
        "input_records",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise SnapshotError("manifest fields mismatch")
    if manifest["algorithm"] != ALGORITHM or manifest["schema_version"] != SCHEMA_VERSION:
        raise SnapshotError("manifest schema mismatch")
    if (
        re.fullmatch(r"[0-9a-f]{40}", manifest["repository_commit"]) is None
        or manifest["repository_commit"]
        != manifest["fingerprint_core"].get("repository_commit")
    ):
        raise SnapshotError("repository commit provenance is malformed or unbound")
    for name in ARTIFACT_NAMES:
        data = (run_dir / name).read_bytes()
        if manifest["artifact_records"].get(name) != _content_record(data):
            raise SnapshotError("artifact content record mismatch")
        if name.endswith(".json"):
            value = _strict_json_bytes(data, name)
            if _canonical_json(value) != data:
                raise SnapshotError(f"{name} is not canonical JSON")
        elif name.endswith(".jsonl"):
            _load_jsonl(data, name)
    if set(manifest["artifact_records"]) != set(ARTIFACT_NAMES):
        raise SnapshotError("manifest artifact inventory mismatch")
    core = manifest["fingerprint_core"]
    expected_core_keys = {
        "algorithm",
        "artifact_records",
        "authoritative_artifact_set_sha256",
        "implementation_records",
        "largest_component_sha256",
        "numeric_contract",
        "observed_contract_bytes",
        "privacy_policy",
        "relation_policy",
        "repository_commit",
        "resource_ceiling_bytes",
        "schema_version",
        "source_records",
        "study_universe_sha256",
    }
    if not isinstance(core, dict) or set(core) != expected_core_keys:
        raise SnapshotError("fingerprint core fields mismatch")
    scientific_records = {
        name: record
        for name, record in manifest["artifact_records"].items()
        if name != "legacy_parity.json"
    }
    if core["artifact_records"] != scientific_records:
        raise SnapshotError("scientific artifact records mismatch")
    if core["authoritative_artifact_set_sha256"] != hashlib.sha256(
        _canonical_json(scientific_records)
    ).hexdigest():
        raise SnapshotError("authoritative artifact-set hash mismatch")
    current_implementation = {
        "preparer": _file_record(Path(__file__).resolve()),
        "privacy_rule": _file_record(HERE / "privacy.py"),
    }
    if core["implementation_records"] != current_implementation:
        raise SnapshotError("snapshot implementation record is stale")
    expected_numeric = {
        "downstream_decision_dtype": "float64",
        "preparer_arithmetic": "exact_integer_graph",
        "preparer_threads": 1,
    }
    if core["numeric_contract"] != expected_numeric:
        raise SnapshotError("numeric contract mismatch")
    input_records = manifest["input_records"]
    if (
        not isinstance(input_records, dict)
        or set(input_records) != {"source_spec", "relation_policy"}
        or any(
            not isinstance(record, dict)
            or set(record) != {"sha256", "size_bytes"}
            or re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")) is None
            or not isinstance(record.get("size_bytes"), int)
            or isinstance(record.get("size_bytes"), bool)
            or record["size_bytes"] < 0
            for record in input_records.values()
        )
    ):
        raise SnapshotError("snapshot input content record is malformed")
    expected_privacy_policy = _expected_privacy_policy()
    if (
        manifest["privacy_policy"] != expected_privacy_policy
        or core["privacy_policy"] != expected_privacy_policy
    ):
        raise SnapshotError("privacy policy mismatch")
    validated_relation_policy = _validate_policy(manifest["relation_policy"])
    if (
        manifest["relation_policy"] != validated_relation_policy
        or core["relation_policy"] != validated_relation_policy
    ):
        raise SnapshotError("relation policy mismatch")
    if core["algorithm"] != ALGORITHM or core["schema_version"] != SCHEMA_VERSION:
        raise SnapshotError("fingerprint core schema mismatch")
    declared_source_bytes = sum(
        record["size_bytes"]
        for source in core["source_records"]
        for record in source["content_records"]
    )
    expected_observed = declared_source_bytes + sum(
        record["size_bytes"] for record in scientific_records.values()
    )
    if core["observed_contract_bytes"] != expected_observed:
        raise SnapshotError("observed byte contract mismatch")
    if (
        not isinstance(core["resource_ceiling_bytes"], int)
        or isinstance(core["resource_ceiling_bytes"], bool)
        or expected_observed > core["resource_ceiling_bytes"]
        or manifest["aggregate"].get("observed_contract_bytes") != expected_observed
    ):
        raise SnapshotError("resource byte contract mismatch")
    expected_fingerprint = hashlib.sha256(
        _canonical_json(manifest["fingerprint_core"])
    ).hexdigest()
    if manifest["snapshot_fingerprint"] != expected_fingerprint:
        raise SnapshotError("snapshot fingerprint mismatch")
    _verify_invariants(run_dir, manifest)
    return manifest


def prepare_snapshot(
    source_spec,
    relation_policy,
    run_dir,
    local_root,
    *,
    minimum_anchors=128,
    resource_ceiling_bytes,
):
    if isinstance(minimum_anchors, bool) or minimum_anchors < 1:
        raise SnapshotError("minimum_anchors must be positive")
    if isinstance(resource_ceiling_bytes, bool) or resource_ceiling_bytes < 1:
        raise SnapshotError("resource ceiling must be positive")

    spec_input = Path(source_spec)
    policy_input = Path(relation_policy)
    spec_bytes = _read_file(spec_input)
    policy_bytes = _read_file(policy_input)
    spec_before = _content_record(spec_bytes)
    policy_before = _content_record(policy_bytes)
    spec_path = spec_input.resolve()
    policy_path = policy_input.resolve()
    spec = _strict_json_bytes(spec_bytes, "source spec")
    if not isinstance(spec, dict):
        raise SnapshotError("source spec must be an object")
    sources, legacy_path = _resolve_source_paths(spec_path, spec)
    policy = _load_policy(policy_path, policy_bytes)
    inputs = [spec_path, policy_path] + [source["path"] for source in sources]
    if legacy_path is not None:
        inputs.append(legacy_path)
    run_dir, local_root = _validate_output_paths(Path(run_dir), Path(local_root), inputs)
    preflight_source_bytes = sum(
        _declared_source_size(source["path"], source["kind"]) for source in sources
    )
    if preflight_source_bytes > resource_ceiling_bytes:
        raise SnapshotError("frozen preparer byte ceiling exceeded")

    state, source_records, _resolved_inputs = _parse_sources(spec_path, sources)
    if spec_before != _file_record(spec_path) or policy_before != _file_record(policy_path):
        raise SnapshotError("source spec or policy changed during preparation")

    compiled = _privacy_and_graph(state, policy)
    parity = _legacy_parity(legacy_path, compiled["preprivacy_physical"])
    payloads = _artifact_payloads(compiled, source_records, parity, policy)
    artifact_records = {name: _content_record(data) for name, data in payloads.items()}
    scientific_artifact_records = {
        name: record
        for name, record in artifact_records.items()
        if name != "legacy_parity.json"
    }
    authoritative_artifact_set_sha256 = hashlib.sha256(
        _canonical_json(scientific_artifact_records)
    ).hexdigest()
    declared_source_bytes = sum(
        record["size_bytes"]
        for source_record in source_records
        for record in source_record["content_records"]
    )
    scientific_artifact_bytes = sum(
        len(payloads[name]) for name in scientific_artifact_records
    )
    if declared_source_bytes != preflight_source_bytes:
        raise SnapshotError("declared source byte count changed during preparation")
    observed_contract_bytes = declared_source_bytes + scientific_artifact_bytes
    if observed_contract_bytes > resource_ceiling_bytes:
        raise SnapshotError("frozen preparer byte ceiling exceeded")
    aggregate = _strict_json_bytes(
        payloads["aggregate_release_candidate.json"], "aggregate"
    )
    implementation_records = {
        "preparer": _file_record(Path(__file__).resolve()),
        "privacy_rule": _file_record(HERE / "privacy.py"),
    }
    privacy_policy = _expected_privacy_policy()
    numeric_contract = {
        "downstream_decision_dtype": "float64",
        "preparer_arithmetic": "exact_integer_graph",
        "preparer_threads": 1,
    }
    repository_commit = _git_commit()
    if re.fullmatch(r"[0-9a-f]{40}", repository_commit) is None:
        raise SnapshotError("repository commit provenance is unavailable")
    fingerprint_core = {
        "algorithm": ALGORITHM,
        "artifact_records": scientific_artifact_records,
        "authoritative_artifact_set_sha256": authoritative_artifact_set_sha256,
        "implementation_records": implementation_records,
        "largest_component_sha256": compiled["largest_component_sha256"],
        "numeric_contract": numeric_contract,
        "observed_contract_bytes": observed_contract_bytes,
        "privacy_policy": privacy_policy,
        "relation_policy": policy,
        "repository_commit": repository_commit,
        "resource_ceiling_bytes": resource_ceiling_bytes,
        "schema_version": SCHEMA_VERSION,
        "source_records": source_records,
        "study_universe_sha256": compiled["study_universe_sha256"],
    }
    fingerprint = hashlib.sha256(_canonical_json(fingerprint_core)).hexdigest()
    coverage_ok = len(compiled["eligible_nodes"]) >= minimum_anchors
    graph_asset_ready = compiled["privacy_certified"] and coverage_ok
    manifest = {
        "aggregate": {
            **aggregate,
            "largest_component_sha256": compiled["largest_component_sha256"],
            "legacy_parity_status": parity["status"],
            "minimum_anchor_count": minimum_anchors,
            "minimum_anchor_coverage_pass": coverage_ok,
            "observed_contract_bytes": observed_contract_bytes,
            "study_universe_sha256": compiled["study_universe_sha256"],
        },
        "algorithm": ALGORITHM,
        "artifact_records": artifact_records,
        "fingerprint_core": fingerprint_core,
        "privacy_policy": privacy_policy,
        "relation_policy": policy,
        "repository_commit": repository_commit,
        "schema_version": SCHEMA_VERSION,
        "snapshot_fingerprint": fingerprint,
        "snapshot_label_hash": hashlib.sha256(
            spec["snapshot_label"].encode("utf-8")
        ).hexdigest(),
        "graph_asset_ready": graph_asset_ready,
        "input_records": {
            "relation_policy": policy_before,
            "source_spec": spec_before,
        },
    }
    manifest_bytes = _canonical_json(manifest)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}.", dir=run_dir.parent)
    )
    os.chmod(temporary, 0o700)
    installed = False
    try:
        for name in ARTIFACT_NAMES:
            _write_bytes(temporary / name, payloads[name])
        _write_bytes(
            temporary / "LOCAL_ONLY_DO_NOT_PUBLISH",
            b"LOCAL ONLY - DO NOT PUBLISH NODE-LEVEL ARTIFACTS\n",
        )
        _write_bytes(temporary / "manifest.json", manifest_bytes)
        verify_snapshot(temporary)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_directory_noreplace(temporary, run_dir)
        installed = True
        try:
            parent_fd = os.open(run_dir.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError as exc:
            # Promotion already committed. Do not roll back or overwrite it; a
            # caller may verify the installed run before deciding on recovery.
            raise SnapshotError(
                "snapshot installed but parent-directory durability is unconfirmed"
            ) from exc
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)
    return verify_snapshot(run_dir)


def snapshot_status(run_dir):
    manifest = verify_snapshot(run_dir)
    return {
        "eligible_anchor_count": manifest["aggregate"]["eligible_anchor_count"],
        "minimum_anchor_coverage_pass": manifest["aggregate"][
            "minimum_anchor_coverage_pass"
        ],
        "physical_edge_count": manifest["aggregate"]["physical_edge_count"],
        "privacy_certified": manifest["aggregate"]["privacy_certified"],
        "legacy_parity_status": manifest["aggregate"]["legacy_parity_status"],
        "snapshot_fingerprint": manifest["snapshot_fingerprint"],
        "graph_asset_ready": manifest["graph_asset_ready"],
    }


def _positive_int(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-spec", required=True)
    prepare.add_argument("--relation-policy", required=True)
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--local-root", required=True)
    prepare.add_argument("--local-only", action="store_true", required=True)
    prepare.add_argument("--minimum-anchors", type=_positive_int, default=128)
    prepare.add_argument("--resource-ceiling-bytes", type=_positive_int, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-dir", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", required=True)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_snapshot(
                args.source_spec,
                args.relation_policy,
                args.run_dir,
                args.local_root,
                minimum_anchors=args.minimum_anchors,
                resource_ceiling_bytes=args.resource_ceiling_bytes,
            )
            output = {
                "privacy_certified": manifest["aggregate"]["privacy_certified"],
                "snapshot_fingerprint": manifest["snapshot_fingerprint"],
                "graph_asset_ready": manifest["graph_asset_ready"],
            }
        elif args.command == "verify":
            manifest = verify_snapshot(args.run_dir)
            output = {
                "snapshot_fingerprint": manifest["snapshot_fingerprint"],
                "graph_asset_ready": manifest["graph_asset_ready"],
                "verified": True,
            }
        else:
            output = snapshot_status(args.run_dir)
        print(json.dumps(output, sort_keys=True))
        if args.command == "prepare" and not output["graph_asset_ready"]:
            return 2
        return 0
    except Exception:
        print(json.dumps({"error": "snapshot preparation failed closed"}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
