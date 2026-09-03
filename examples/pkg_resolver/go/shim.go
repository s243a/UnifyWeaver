// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// shim.go -- EDGE of the Go-WAM compiled uw-resolve P0.5.
//
// WHAT IS IN HERE, exhaustively: conversion between JSON catalogs/requests
// and WAM terms, plus driving NewWamState / Run. There is NO resolver
// logic — no candidate order, no constraint arithmetic, no layer walk.
// Those live in the generated WAM tables compiled from resolver.pl.

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

func vTerm(triple []int64) *Structure {
	return st("v", i64(triple[0]), i64(triple[1]), i64(triple[2]))
}

func pairTerm(name string, ver []int64) *Structure {
	return st("-", atom(name), vTerm(ver))
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
		case "eq":
			return st("eq", vTerm(asVer(t["v"])))
		case "gte":
			return st("gte", vTerm(asVer(t["v"])))
		case "lt":
			return st("lt", vTerm(asVer(t["v"])))
		case "range":
			return st("range", vTerm(asVer(t["lo"])), vTerm(asVer(t["hi"])))
		}
	}
	panic(fmt.Sprintf("shim: unknown constraint %v", c))
}

func holdTerm(row []interface{}) Value {
	name := asString(row[0])
	ver := asVer(row[1])
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

func pkgTerm(row []interface{}) Value {
	return st("package", atom(asString(row[0])), vTerm(asVer(row[1])))
}

func depTerm(row []interface{}) Value {
	return st("depends",
		atom(asString(row[0])), vTerm(asVer(row[1])),
		atom(asString(row[2])), constraintTerm(row[3]))
}

func confTerm(row []interface{}) Value {
	return st("conflicts",
		atom(asString(row[0])), vTerm(asVer(row[1])), atom(asString(row[2])))
}

func requestTerm(req interface{}) Value {
	if m, ok := req.(map[string]interface{}); ok {
		if r, ok := m["req"]; ok {
			return st("req", atom(asString(r)), constraintTerm(m["constraint"]))
		}
	}
	return atom(asString(req))
}

func catalogToTerm(cat map[string]interface{}) Value {
	if cat == nil {
		cat = map[string]interface{}{}
	}
	pkgs := mapList(cat["packages"], func(x interface{}) Value { return pkgTerm(asArray(x)) })
	deps := mapList(cat["depends"], func(x interface{}) Value { return depTerm(asArray(x)) })
	confs := mapList(cat["conflicts"], func(x interface{}) Value { return confTerm(asArray(x)) })
	base := mapList(cat["base"], func(x interface{}) Value { return holdTerm(asArray(x)) })
	inst := mapList(cat["installed"], func(x interface{}) Value {
		a := asArray(x)
		return pairTerm(asString(a[0]), asVer(a[1]))
	})
	req := mapList(cat["requested"], func(x interface{}) Value { return atom(asString(x)) })
	core := []Value{listOf(pkgs), listOf(deps), listOf(confs), listOf(base), listOf(inst), listOf(req)}
	layersIn := asArray(cat["layers"])
	exclIn := asArray(cat["excluded"])
	aliasIn := asArray(cat["aliases"])
	if len(layersIn) == 0 && len(exclIn) == 0 && len(aliasIn) == 0 {
		return st("catalog", core...)
	}
	var layers []Value
	for _, l := range layersIn {
		layers = append(layers, layerTerm(asMap(l)))
	}
	excl := mapList(cat["excluded"], func(x interface{}) Value { return atom(asString(x)) })
	var aliases []Value
	for _, a := range aliasIn {
		aliases = append(aliases, aliasTerm(asArray(a)))
	}
	return st("catalog", append(core, listOf(layers), listOf(excl), listOf(aliases))...)
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
	case "-":
		if len(jsArgs) == 2 {
			return []interface{}{jsArgs[0], jsArgs[1]}
		}
	case "blocked":
		needs := jsArgs[1]
		if a, ok := needs.([]interface{}); ok && len(a) == 2 && a[0] == "needs" {
			needs = a[1]
		}
		bh := jsArgs[2]
		if a, ok := bh.([]interface{}); ok && len(a) == 2 && a[0] == "base_has" {
			bh = a[1]
		}
		return map[string]interface{}{"name": jsArgs[0], "needs": needs, "base_has": bh}
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
	case "needs", "base_has", "eq", "gte", "lt", "cost", "held", "suggest":
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
		if a[0] == "gte" || a[0] == "eq" || a[0] == "lt" {
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
	cat := asMap(row["catalog"])
	q := asString(row["query"])
	args := row["args"]
	switch q {
	case "resolve":
		return resolveQ(cat, asArray(args))
	case "resolve_layered":
		return resolveLayeredQ(cat, asArray(args))
	case "explain_blocked":
		return explainBlockedQ(cat, args)
	case "layer_closure":
		return layerClosureQ(cat, args)
	case "removal_orphans":
		return removalOrphansQ(cat, asString(args))
	case "safe_upgrade":
		a := asArray(args)
		return safeUpgradeQ(cat, asString(a[0]), asVer(a[1]))
	case "upgrade_set":
		a := asArray(args)
		return upgradeSetQ(cat, asString(a[0]), asVer(a[1]))
	case "freeze_audit":
		return freezeAuditQ(cat)
	case "dependents":
		return dependentsQ(cat, asString(args))
	case "dependents_installed":
		return dependentsInstalledQ(cat, asString(args))
	default:
		panic("unknown query " + q)
	}
}

func resolveQ(cat map[string]interface{}, reqs []interface{}) map[string]interface{} {
	ok, vm, saved := runPred("resolve/3", []Value{catalogToTerm(cat), listOf(mapList(reqs, requestTerm)), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func resolveLayeredQ(cat map[string]interface{}, reqs []interface{}) map[string]interface{} {
	ok, vm, saved := runPred("resolve_layered/3", []Value{catalogToTerm(cat), listOf(mapList(reqs, requestTerm)), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func explainBlockedQ(cat map[string]interface{}, req interface{}) map[string]interface{} {
	ok, vm, saved := runPred("explain_blocked_list/3", []Value{catalogToTerm(cat), requestTerm(req), nil})
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

func layerClosureQ(cat map[string]interface{}, req interface{}) map[string]interface{} {
	ok, vm, saved := runPred("layer_closure/3", []Value{catalogToTerm(cat), requestTerm(req), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func removalOrphansQ(cat map[string]interface{}, pkg string) map[string]interface{} {
	ok, vm, saved := runPred("removal_orphans/3", []Value{catalogToTerm(cat), atom(pkg), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func safeUpgradeQ(cat map[string]interface{}, pkg string, ver []int64) map[string]interface{} {
	ok, vm, saved := runPred("safe_upgrade/4", []Value{catalogToTerm(cat), atom(pkg), vTerm(ver), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": normalizeVerdict(readSaved(vm, saved, 4))}
}

func upgradeSetQ(cat map[string]interface{}, pkg string, ver []int64) map[string]interface{} {
	ok, vm, saved := runPred("upgrade_set_result/4", []Value{catalogToTerm(cat), atom(pkg), vTerm(ver), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return normalizeUpgrade(readSaved(vm, saved, 4))
}

func freezeAuditQ(cat map[string]interface{}) map[string]interface{} {
	ok, vm, saved := runPred("freeze_audit/2", []Value{catalogToTerm(cat), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	list := readSaved(vm, saved, 2)
	if list == nil {
		list = []interface{}{}
	}
	return map[string]interface{}{"ok": list}
}

func dependentsQ(cat map[string]interface{}, pkg string) map[string]interface{} {
	ok, vm, saved := runPred("dependents/3", []Value{catalogToTerm(cat), atom(pkg), nil})
	if !ok {
		return map[string]interface{}{"fail": true}
	}
	return map[string]interface{}{"ok": readSaved(vm, saved, 3)}
}

func dependentsInstalledQ(cat map[string]interface{}, pkg string) map[string]interface{} {
	ok, vm, saved := runPred("dependents_installed/3", []Value{catalogToTerm(cat), atom(pkg), nil})
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
	// A single string args value (explain_blocked, dependents, ...).
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

func asVer(v interface{}) []int64 {
	a := asArray(v)
	return []int64{asInt(a[0]), asInt(a[1]), asInt(a[2])}
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

func stripID(m map[string]interface{}) map[string]interface{} {
	out := map[string]interface{}{}
	for k, v := range m {
		if k == "id" {
			continue
		}
		out[k] = v
	}
	return out
}

// CLI is the JSONL driver. Default: read cases from stdin, write one
// result object per line. `--corpus` also compares against `expected`.
// `--scale-probe DIR` times resolve_layered on the 5k rich.jsonl dump.
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
		id, _ := row["id"]
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
			fmt.Fprintf(os.Stderr, "corpus-under-go: %d divergences / %d\n", divergences, n)
			return 1
		}
		fmt.Fprintf(os.Stderr, "corpus-under-go: %d/%d matched SWI\n", n, n)
	}
	return 0
}

func runScaleProbe(dir string) int {
	richPath := dir + "/rich.jsonl"
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
	t0 := time.Now()
	cat, err := loadRichJSONL(richPath, probe)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	load := time.Since(t0)
	args := asArray(probe["args"])
	t1 := time.Now()
	got := resolveLayeredQ(cat, args)
	resolve := time.Since(t1)
	fmt.Printf("go_term_load_s %.3f\n", load.Seconds())
	fmt.Printf("go_term_resolve_s %.3f\n", resolve.Seconds())
	fmt.Printf("go_term_total_s %.3f\n", (load + resolve).Seconds())
	b, _ := json.Marshal(got)
	fmt.Printf("go_term_result %s\n", string(b))
	return 0
}

func loadRichJSONL(path string, probe map[string]interface{}) (map[string]interface{}, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var pkgs, deps, confs []interface{}
	sc := bufio.NewScanner(f)
	buf := make([]byte, 0, 1024*1024)
	sc.Buffer(buf, 32*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		var row map[string]interface{}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, err
		}
		switch asString(row["kind"]) {
		case "package":
			pkgs = append(pkgs, []interface{}{row["name"], row["ver"]})
		case "depends":
			deps = append(deps, []interface{}{row["name"], row["ver"], row["dep"], row["constraint"]})
		case "conflicts":
			confs = append(confs, []interface{}{row["name"], row["ver"], row["other"]})
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	env := asMap(probe["env"])
	base := env["base"]
	if base == nil {
		base = []interface{}{[]interface{}{"p0", []interface{}{float64(0), float64(0), float64(0)}}}
	}
	return map[string]interface{}{
		"packages": pkgs, "depends": deps, "conflicts": confs,
		"base": base, "installed": env["installed"], "requested": env["requested"],
		"layers": env["layers"], "excluded": env["excluded"], "aliases": env["aliases"],
	}, nil
}
