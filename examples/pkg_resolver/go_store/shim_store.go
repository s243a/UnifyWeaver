// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// shim_store.go -- EDGE of the Go-WAM compiled uw-resolve P2 STORE adapter.
//
// WHAT IS IN HERE, exhaustively: conversion between JSON env/requests and
// WAM terms, plus driving NewWamState / Run. There is NO resolver logic and
// NO catalog: package/depends/conflicts/revdep/provides facts come from the
// D43 indexed seek stores compiled in (registerIndexedSeekFact2). The machine-
// local environment stays a term. Mirrors examples/pkg_resolver/wamjs_store/
// resolver_store.mjs for the Go lane.

package wam

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

func st(name string, args ...Value) *Structure {
	return &Structure{
		Functor: fmt.Sprintf("%s/%d", name, len(args)),
		Arity:   len(args),
		Args:    args,
	}
}

func atom(s string) *Atom { return InternAtom(s) }

func i64(n int64) *Integer { return &Integer{Val: n} }

func nilList() Value { return InternAtom("[]") }

func listOf(items []Value) Value {
	if len(items) == 0 {
		return nilList()
	}
	return &List{Elements: items}
}

func vTerm(ver interface{}) Value { return verTerm(ver) }

func verTerm(ver interface{}) Value {
	if m, ok := ver.(map[string]interface{}); ok {
		if d, has := m["deb"]; has {
			arr := asArray(d)
			epoch := int64(0)
			var up, rev interface{}
			if len(arr) > 0 {
				epoch = asInt(arr[0])
			}
			if len(arr) > 1 {
				up = arr[1]
			}
			if len(arr) > 2 {
				rev = arr[2]
			}
			return st("deb", i64(epoch), segsTerm(up), segsTerm(rev))
		}
	}
	a := asArray(ver)
	return st("v", i64(asInt(a[0])), i64(asInt(a[1])), i64(asInt(a[2])))
}

func segsTerm(v interface{}) Value {
	var items []Value
	for _, seg := range asArray(v) {
		s := asArray(seg)
		order := ""
		num := int64(0)
		if len(s) > 0 {
			order = asString(s[0])
		}
		if len(s) > 1 {
			num = asInt(s[1])
		}
		var codes []Value
		for _, r := range order {
			codes = append(codes, i64(int64(r)))
		}
		items = append(items, st("s", listOf(codes), i64(num)))
	}
	return listOf(items)
}

func pairTerm(name string, ver interface{}) *Structure {
	return st("-", atom(name), verTerm(ver))
}

func constraintTerm(c interface{}) Value {
	if c == nil {
		return atom("any")
	}
	switch t := c.(type) {
	case string:
		if t == "any" {
			return atom("any")
		}
	case map[string]interface{}:
		op, _ := t["op"].(string)
		switch op {
		case "eq", "gte", "lt", "lte", "gt":
			return st(op, verTerm(t["v"]))
		case "range":
			return st("range", verTerm(t["lo"]), verTerm(t["hi"]))
		}
	}
	panic(fmt.Sprintf("shim: unknown constraint %v", c))
}

func holdTerm(row []interface{}) Value {
	name := asString(row[0])
	ver := row[1]
	if len(row) >= 3 {
		return st("base", pairTerm(name, ver), atom(asString(row[2])))
	}
	return pairTerm(name, ver)
}

func layerTerm(row map[string]interface{}) Value {
	name := asString(row["name"])
	var pkgs []Value
	for _, p := range asArray(row["packages"]) {
		pkgs = append(pkgs, holdTerm(asArray(p)))
	}
	return st("layer", atom(name), listOf(pkgs))
}

func aliasTerm(row []interface{}) Value {
	return st("alias", atom(asString(row[0])), atom(asString(row[1])))
}

func requestTerm(req interface{}) Value {
	if m, ok := req.(map[string]interface{}); ok {
		if r, ok := m["req"]; ok {
			return st("req", atom(asString(r)), constraintTerm(m["constraint"]))
		}
	}
	return atom(asString(req))
}

// envToTerm builds env(CatId, Base, Installed, Requested, Layers, Excluded,
// Aliases). No catalog rows: those live in the compiled-in seek stores.
func envToTerm(env map[string]interface{}) Value {
	catID := "default"
	if v, ok := env["catalog_id"]; ok && v != nil {
		catID = asString(v)
	}
	base := mapList(env["base"], func(x interface{}) Value { return holdTerm(asArray(x)) })
	inst := mapList(env["installed"], func(x interface{}) Value {
		a := asArray(x)
		return pairTerm(asString(a[0]), a[1])
	})
	req := mapList(env["requested"], func(x interface{}) Value { return atom(asString(x)) })
	var layers []Value
	for _, l := range asArray(env["layers"]) {
		layers = append(layers, layerTerm(asMap(l)))
	}
	excl := mapList(env["excluded"], func(x interface{}) Value { return atom(asString(x)) })
	var aliases []Value
	for _, a := range asArray(env["aliases"]) {
		aliases = append(aliases, aliasTerm(asArray(a)))
	}
	return st("env", atom(catID), listOf(base), listOf(inst), listOf(req),
		listOf(layers), listOf(excl), listOf(aliases))
}

func envOf(row map[string]interface{}) map[string]interface{} {
	if e, ok := row["env"].(map[string]interface{}); ok {
		env := map[string]interface{}{}
		for k, v := range e {
			env[k] = v
		}
		if _, has := env["catalog_id"]; !has {
			if cid, ok := row["catalog_id"]; ok {
				env["catalog_id"] = cid
			}
		}
		return env
	}
	c := asMap(row["catalog"])
	cid := "default"
	if v, ok := row["catalog_id"]; ok && v != nil {
		cid = asString(v)
	}
	return map[string]interface{}{
		"catalog_id": cid,
		"base":       c["base"], "installed": c["installed"], "requested": c["requested"],
		"layers": c["layers"], "excluded": c["excluded"], "aliases": c["aliases"],
	}
}

func mapList(v interface{}, f func(interface{}) Value) []Value {
	arr := asArray(v)
	out := make([]Value, 0, len(arr))
	for _, x := range arr {
		out = append(out, f(x))
	}
	return out
}

func runPred(predArity string, argTerms []Value) (ok bool, vm *WamState, saved []Value) {
	pc, found := sharedWamLabels[predArity]
	if !found {
		panic("unknown predicate: " + predArity)
	}
	vm = NewWamState(sharedWamCode, sharedWamLabels)
	setupSharedForeignPredicates(vm)
	slash := strings.LastIndex(predArity, "/")
	arity := 0
	if slash >= 0 {
		arity, _ = strconv.Atoi(predArity[slash+1:])
	}
	saved = make([]Value, arity)
	for i := 0; i < arity; i++ {
		var t Value
		if i < len(argTerms) && argTerms[i] != nil {
			t = argTerms[i]
		} else {
			t = &Unbound{Name: "Out", Idx: vm.allocVarId()}
		}
		vm.Regs[i] = t
		saved[i] = t
	}
	vm.PC = pc
	ok = vm.Run()
	return ok, vm, saved
}

func readSaved(vm *WamState, saved []Value, n int) interface{} {
	return termToJS(vm, saved[n-1])
}

func termToJS(vm *WamState, term0 Value) interface{} {
	term := vm.Deref(term0)
	if term == nil {
		return nil
	}
	switch t := term.(type) {
	case *Integer:
		return t.Val
	case *Float:
		if t.Val == float64(int64(t.Val)) {
			return int64(t.Val)
		}
		return t.Val
	case *Atom:
		switch t.Name {
		case "[]":
			return []interface{}{}
		case "any", "true", "false":
			if t.Name == "true" {
				return true
			}
			if t.Name == "false" {
				return false
			}
			return t.Name
		default:
			return t.Name
		}
	case *Unbound:
		return nil
	}
	if items, ok := vm.listToSlice(term); ok {
		out := make([]interface{}, 0, len(items))
		for _, it := range items {
			out = append(out, termToJS(vm, it))
		}
		return out
	}
	var name string
	var args []Value
	switch t := term.(type) {
	case *Structure:
		name = parseFunctorName(t.Functor)
		args = t.Args
	case *Compound:
		name = parseFunctorName(t.Functor)
		args = t.Args
	default:
		panic(fmt.Sprintf("shim: unhandled term %T %v", term, term))
	}
	jsArgs := make([]interface{}, len(args))
	for i, a := range args {
		jsArgs[i] = termToJS(vm, a)
	}
	switch name {
	case "v":
		if len(jsArgs) == 3 {
			return jsArgs
		}
	case "s":
		if len(jsArgs) == 2 {
			order := ""
			if codes, ok := jsArgs[0].([]interface{}); ok {
				rs := make([]rune, 0, len(codes))
				for _, c := range codes {
					rs = append(rs, rune(asInt(c)))
				}
				order = string(rs)
			}
			return []interface{}{order, jsArgs[1]}
		}
	case "deb":
		if len(jsArgs) == 3 {
			return map[string]interface{}{"deb": jsArgs}
		}
	case "-":
		if len(jsArgs) == 2 {
			return []interface{}{jsArgs[0], jsArgs[1]}
		}
	case "blocked":
		if len(jsArgs) == 1 {
			if a, ok := jsArgs[0].([]interface{}); ok && len(a) == 2 && a[0] == "alternatives" {
				return map[string]interface{}{"alternatives": a[1]}
			}
		}
		if len(jsArgs) >= 3 {
			needs := jsArgs[1]
			if a, ok := needs.([]interface{}); ok && len(a) == 2 && a[0] == "needs" {
				needs = a[1]
			}
			third := jsArgs[2]
			if a, ok := third.([]interface{}); ok && len(a) == 2 && a[0] == "providers" {
				return map[string]interface{}{"name": jsArgs[0], "needs": needs, "providers": a[1]}
			}
			bh := third
			if a, ok := third.([]interface{}); ok && len(a) == 2 && a[0] == "base_has" {
				bh = a[1]
			}
			return map[string]interface{}{"name": jsArgs[0], "needs": needs, "base_has": bh}
		}
	case "alt":
		if len(jsArgs) == 2 {
			return map[string]interface{}{"dep": jsArgs[0], "reason": jsArgs[1]}
		}
	case "safe":
		cost := jsArgs[0]
		if a, ok := cost.([]interface{}); ok && len(a) == 2 && a[0] == "cost" {
			cost = a[1]
		}
		return map[string]interface{}{"cost": cost, "verdict": "safe"}
	case "coordinated":
		return map[string]interface{}{"set": jsArgs[0], "verdict": "coordinated"}
	case "unsafe":
		return map[string]interface{}{"reason": jsArgs[0], "verdict": "unsafe"}
	case "audit":
		return normalizeAuditTerm(jsArgs[0], jsArgs[1])
	case "ok":
		return map[string]interface{}{"__ok_set": jsArgs[0]}
	case "needs", "base_has", "eq", "gte", "lt", "lte", "gt", "cost", "held", "suggest", "providers", "alternatives":
		return []interface{}{name, jsArgs[0]}
	case "range":
		return map[string]interface{}{"op": "range", "lo": jsArgs[0], "hi": jsArgs[1]}
	}
	out := []interface{}{name}
	return append(out, jsArgs...)
}

func normalizeConstraint(c interface{}) interface{} {
	if c == "any" {
		return "any"
	}
	if a, ok := c.([]interface{}); ok && len(a) == 2 {
		if a[0] == "gte" || a[0] == "eq" || a[0] == "lt" || a[0] == "lte" || a[0] == "gt" {
			return map[string]interface{}{"op": a[0], "v": a[1]}
		}
		if a[0] == "range" {
			return map[string]interface{}{"op": "range", "lo": a[1], "hi": a[2]}
		}
	}
	if m, ok := c.(map[string]interface{}); ok {
		if _, has := m["op"]; has {
			return c
		}
	}
	return c
}

func normalizeAuditTerm(name, payload interface{}) map[string]interface{} {
	if payload == "over_frozen" {
		return map[string]interface{}{"kind": "over_frozen", "name": name}
	}
	if a, ok := payload.([]interface{}); ok && len(a) == 2 {
		if a[0] == "suggest" {
			return map[string]interface{}{"kind": "suggest", "name": name, "reason": a[1]}
		}
		if a[0] == "held" {
			return map[string]interface{}{"kind": "held", "name": name, "reason": a[1]}
		}
	}
	if m, ok := payload.(map[string]interface{}); ok {
		if _, has := m["kind"]; has {
			return m
		}
	}
	return map[string]interface{}{"kind": "held", "name": name, "reason": payload}
}

func normalizeVerdict(v interface{}) interface{} {
	if v == "no_candidate" {
		return map[string]interface{}{"verdict": "no_candidate"}
	}
	if m, ok := v.(map[string]interface{}); ok {
		if _, has := m["verdict"]; has {
			return v
		}
	}
	return v
}

func normalizeUpgrade(r interface{}) map[string]interface{} {
	if r == "no_candidate" {
		return map[string]interface{}{"fail": true}
	}
	if m, ok := r.(map[string]interface{}); ok {
		if set, has := m["__ok_set"]; has {
			return map[string]interface{}{"ok": set}
		}
		if _, has := m["name"]; has {
			if _, has2 := m["base_has"]; has2 {
				return map[string]interface{}{"ok": map[string]interface{}{"blocked": normalizeBlocked(m)}}
			}
			if _, has2 := m["providers"]; has2 {
				return map[string]interface{}{"ok": map[string]interface{}{"blocked": normalizeBlocked(m)}}
			}
		}
	}
	if _, ok := r.([]interface{}); ok {
		return map[string]interface{}{"ok": r}
	}
	return map[string]interface{}{"ok": r}
}

func normalizeBlocked(b interface{}) interface{} {
	m, ok := b.(map[string]interface{})
	if !ok {
		return b
	}
	if _, has := m["alternatives"]; has {
		return map[string]interface{}{"alternatives": m["alternatives"]}
	}
	if _, has := m["providers"]; has {
		var nested []interface{}
		for _, p := range asArray(m["providers"]) {
			nested = append(nested, normalizeBlocked(p))
		}
		return map[string]interface{}{
			"name":      m["name"],
			"needs":     normalizeConstraint(m["needs"]),
			"providers": nested,
		}
	}
	if _, has := m["name"]; has {
		return map[string]interface{}{
			"base_has": m["base_has"],
			"name":     m["name"],
			"needs":    normalizeConstraint(m["needs"]),
		}
	}
	return b
}

func runCase(row map[string]interface{}) map[string]interface{} {
	env := envOf(row)
	q := asString(row["query"])
	args := row["args"]
	switch q {
	case "resolve":
		return resolveQ(env, asArray(args))
	case "resolve_layered":
		return resolveLayeredQ(env, asArray(args))
	case "explain_blocked":
		return explainBlockedQ(env, args)
	case "layer_closure":
		return layerClosureQ(env, args)
	case "removal_orphans":
		return removalOrphansQ(env, asString(args))
	case "safe_upgrade":
		a := asArray(args)
		return safeUpgradeQ(env, asString(a[0]), a[1])
	case "upgrade_set":
		a := asArray(args)
		return upgradeSetQ(env, asString(a[0]), a[1])
	case "freeze_audit":
		return freezeAuditQ(env)
	case "dependents":
		return dependentsQ(env, asString(args))
	case "dependents_installed":
		return dependentsInstalledQ(env, asString(args))
	default:
		panic("unknown query " + q)
	}
}

func resolveQ(env map[string]interface{}, reqs []interface{}) map[string]interface{} {
	ok, vm, saved := runPred("resolve_store/3", []Value{envToTerm(env), listOf(mapList(reqs, requestTerm)), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func resolveLayeredQ(env map[string]interface{}, reqs []interface{}) map[string]interface{} {
	ok, vm, saved := runPred("resolve_layered_store/3", []Value{envToTerm(env), listOf(mapList(reqs, requestTerm)), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func explainBlockedQ(env map[string]interface{}, req interface{}) map[string]interface{} {
	ok, vm, saved := runPred("explain_blocked_list_store/3", []Value{envToTerm(env), requestTerm(req), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	list, _ := readSaved(vm, saved, 3).([]interface{})
	out := make([]interface{}, 0, len(list))
	for _, b := range list {
		out = append(out, normalizeBlocked(b))
	}
	return map[string]interface{}{"ok": out}
}

func layerClosureQ(env map[string]interface{}, req interface{}) map[string]interface{} {
	ok, vm, saved := runPred("layer_closure_store/3", []Value{envToTerm(env), requestTerm(req), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func removalOrphansQ(env map[string]interface{}, pkg string) map[string]interface{} {
	ok, vm, saved := runPred("removal_orphans_store/3", []Value{envToTerm(env), atom(pkg), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func safeUpgradeQ(env map[string]interface{}, pkg string, ver interface{}) map[string]interface{} {
	ok, vm, saved := runPred("safe_upgrade_store/4", []Value{envToTerm(env), atom(pkg), verTerm(ver), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": normalizeVerdict(readSaved(vm, saved, 4))}
}

func upgradeSetQ(env map[string]interface{}, pkg string, ver interface{}) map[string]interface{} {
	ok, vm, saved := runPred("upgrade_set_result_store/4", []Value{envToTerm(env), atom(pkg), verTerm(ver), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return normalizeUpgrade(readSaved(vm, saved, 4))
}

func freezeAuditQ(env map[string]interface{}) map[string]interface{} {
	ok, vm, saved := runPred("freeze_audit_store/2", []Value{envToTerm(env), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	list := readSaved(vm, saved, 2)
	if list == nil {
		list = []interface{}{}
	}
	return map[string]interface{}{"ok": list}
}

func dependentsQ(env map[string]interface{}, pkg string) map[string]interface{} {
	ok, vm, saved := runPred("dependents_store/3", []Value{envToTerm(env), atom(pkg), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func dependentsInstalledQ(env map[string]interface{}, pkg string) map[string]interface{} {
	ok, vm, saved := runPred("dependents_installed_store/3", []Value{envToTerm(env), atom(pkg), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func asString(v interface{}) string {
	switch t := v.(type) {
	case string:
		return t
	case json.Number:
		return t.String()
	case fmt.Stringer:
		return t.String()
	default:
		return fmt.Sprint(v)
	}
}

func asMap(v interface{}) map[string]interface{} {
	if v == nil {
		return map[string]interface{}{}
	}
	if m, ok := v.(map[string]interface{}); ok {
		return m
	}
	return map[string]interface{}{}
}

func asArray(v interface{}) []interface{} {
	if v == nil {
		return nil
	}
	if a, ok := v.([]interface{}); ok {
		return a
	}
	return []interface{}{v}
}

func asInt(v interface{}) int64 {
	switch t := v.(type) {
	case float64:
		return int64(t)
	case json.Number:
		n, _ := t.Int64()
		return n
	case int64:
		return t
	case int:
		return int64(t)
	default:
		n, _ := strconv.ParseInt(fmt.Sprint(t), 10, 64)
		return n
	}
}

func stableStringify(x interface{}) string {
	if x == nil {
		return "null"
	}
	switch t := x.(type) {
	case bool:
		if t {
			return "true"
		}
		return "false"
	case string:
		b, _ := json.Marshal(t)
		return string(b)
	case float64:
		if t == float64(int64(t)) {
			return strconv.FormatInt(int64(t), 10)
		}
		return strconv.FormatFloat(t, 'g', -1, 64)
	case int64:
		return strconv.FormatInt(t, 10)
	case int:
		return strconv.Itoa(t)
	case json.Number:
		return t.String()
	case []interface{}:
		parts := make([]string, len(t))
		for i, e := range t {
			parts[i] = stableStringify(e)
		}
		return "[" + strings.Join(parts, ",") + "]"
	case map[string]interface{}:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			kb, _ := json.Marshal(k)
			parts = append(parts, string(kb)+":"+stableStringify(t[k]))
		}
		return "{" + strings.Join(parts, ",") + "}"
	default:
		b, _ := json.Marshal(t)
		return string(b)
	}
}

// CLI is the JSONL driver. Default: read cases from stdin, write one result
// object per line. `--corpus` also compares against `expected`.
// `--scale-probe DIR` times a bound resolve_layered_store on the 5k store and
// reports the D43 bytes-read proof against total store size.
func CLI(args []string) int {
	if len(args) > 0 && args[0] == "--scale-probe" {
		dir := "."
		if len(args) > 1 {
			dir = args[1]
		}
		return runScaleProbe(dir)
	}
	corpus := false
	in := os.Stdin
	var err error
	rest := args
	if len(rest) > 0 && rest[0] == "--corpus" {
		corpus = true
		rest = rest[1:]
	}
	if len(rest) > 0 && rest[0] != "-" {
		in, err = os.Open(rest[0])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 2
		}
		defer in.Close()
	}
	return runJSONL(in, os.Stdout, corpus)
}

func runJSONL(in io.Reader, out io.Writer, corpus bool) int {
	sc := bufio.NewScanner(in)
	buf := make([]byte, 0, 1024*1024)
	sc.Buffer(buf, 32*1024*1024)
	var n, divergences int
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		var row map[string]interface{}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			fmt.Fprintf(os.Stderr, "json: %v\n", err)
			return 2
		}
		n++
		var got map[string]interface{}
		func() {
			defer func() {
				if r := recover(); r != nil {
					got = map[string]interface{}{"crash": fmt.Sprint(r)}
				}
			}()
			got = runCase(row)
		}()
		id := row["id"]
		result := map[string]interface{}{"id": id}
		for k, v := range got {
			result[k] = v
		}
		enc := json.NewEncoder(out)
		enc.SetEscapeHTML(false)
		_ = enc.Encode(result)
		if corpus {
			exp, _ := row["expected"].(map[string]interface{})
			if stableStringify(got) != stableStringify(exp) {
				divergences++
				fmt.Fprintf(os.Stderr, "DIVERGE %v\n  expected %s\n  got      %s\n",
					id, stableStringify(exp), stableStringify(got))
			} else {
				fmt.Fprintf(os.Stderr, "ok %v\n", id)
			}
		}
	}
	if err := sc.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	if corpus {
		if divergences != 0 {
			fmt.Fprintf(os.Stderr, "corpus-under-go-store: %d divergences / %d\n", divergences, n)
			return 1
		}
		fmt.Fprintf(os.Stderr, "corpus-under-go-store: %d/%d matched SWI\n", n, n)
	}
	return 0
}

// runScaleProbe times ONE bound resolve_layered_store on the 5k store, then
// prints the D43 proof: bytes read off disk vs total store size, plus wall
// time. There is no catalog load — the store is read on demand by seek.
func runScaleProbe(dir string) int {
	probePath := dir + "/probe.json"
	pf, err := os.Open(probePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	defer pf.Close()
	var probe map[string]interface{}
	if err := json.NewDecoder(pf).Decode(&probe); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	env := asMap(probe["env"])
	args := asArray(probe["args"])
	// Total store size = sum of the five .data blobs (what a term-catalog
	// load would have to read in full).
	var totalStore int64
	for _, name := range []string{"pkg", "dep", "conflict", "revdep", "provide"} {
		if fi, ferr := os.Stat(dir + "/" + name + ".data"); ferr == nil {
			totalStore += fi.Size()
		}
	}
	ResetFactIO()
	t0 := time.Now()
	got := resolveLayeredQ(env, args)
	resolve := time.Since(t0)
	bytesRead := FactIOBytes()
	reads := FactIOReads()
	fmt.Printf("go_store_resolve_s %.3f\n", resolve.Seconds())
	fmt.Printf("go_store_total_s %.3f\n", resolve.Seconds())
	fmt.Printf("go_store_bytes_read %d\n", bytesRead)
	fmt.Printf("go_store_n_reads %d\n", reads)
	fmt.Printf("go_store_total_store_bytes %d\n", totalStore)
	if totalStore > 0 {
		fmt.Printf("go_store_read_fraction %.6f\n", float64(bytesRead)/float64(totalStore))
	}
	b, _ := json.Marshal(got)
	fmt.Printf("go_store_result %s\n", string(b))
	return 0
}
