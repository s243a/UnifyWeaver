package wam

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
)

type TrailEntry struct {
	Addr   int
	Old    Value
	HadOld bool
	// Register-restore entry: when RegIdx >= 0, this entry records a
	// register-alias rewrite done by bindUnbound (Regs[RegIdx] was RegOld,
	// an Unbound, before being rewritten to the bound value). Backtrack
	// restores Regs[RegIdx] = RegOld so the register goes back to sharing
	// the (now-unbound) variable cell -- otherwise a register that aliased
	// a bound var keeps the stale value across backtracking and a later
	// rebind of the same var doesn't reach it. RegIdx == -1 for the normal
	// Bindings-restore entry.
	RegIdx int
	RegOld Value
}

type AtomPair struct {
	Left  string
	Right string
}

type AtomFact2Source interface {
	Scan() []AtomPair
	LookupArg1(left string) []AtomPair
}

type WeightedEdgeTriple struct {
	Left   string
	Right  string
	Weight float64
}

type SelectSolution struct {
	Elem Value
	Rest Value
}

type StreamHandle struct {
	ID     int
	File   *os.File
	Reader *bufio.Reader
	Mode   string
	Closed bool
}

func (v *StreamHandle) valueTag() {}
func (v *StreamHandle) String() string { return fmt.Sprintf("stream(%d)", v.ID) }
func (v *StreamHandle) Equals(other Value) bool {
	o, ok := other.(*StreamHandle)
	return ok && v == o
}

type ChoicePoint struct {
	NextPC    int
	ResumePC  int
	CP        int
	// E captures vm.E (env-frame stack index) at choicepoint push.
	// Restored at backtrack so the env-trimming-aware Deallocate
	// continues from the right frame after the failed branch's
	// changes are undone.
	E         int
	// StackLen captures len(vm.Stack) at choicepoint push. Backtrack
	// truncates the stack to this length; everything pushed during
	// the failed attempt is dropped. With env trimming in place the
	// truncation is safe — env frames pushed after the CP are dead
	// (their PrevE chain has already been walked back), and frames
	// alive at CP push time are guaranteed to be at indices < StackLen
	// because Deallocate doesn't physically pop frames whose B0 < the
	// youngest live CP. Replaces the prior `Stack: copyStack(vm.Stack)`
	// per-push allocation, which was ~19% of CPU at scale-300.
	StackLen  int
	SavedRegs []Value
	HeapTop   int
	TrailMark int
	IndexedClausePCs []int
	ForeignPredKey   string
	ForeignResultRegs []int
	ForeignResults   []Value
	MemberTail       Value
	SelectResults    []SelectSolution
	BetweenActive    bool
	BetweenCurrent   int64
	BetweenHigh      int64
	BetweenReg       int
	YSaveLen         int
	// PendingB0 / CutB0Stack / YSaves snapshot the call-frame regime
	// (docs/WAM_BACKEND_CONVENTIONS.md §9). Restore on backtrack so a
	// Proceed that already popped does not extra-pop on the retry, and
	// so a neck-cut still sees the B0 that was live at this CP.
	PendingB0    int
	CutB0Stack   []int
	YSaves       [][100]Value
	// Levels: if-then-else barrier levels captured by `get_level Yn`,
	// keyed by the Y-register index the shared emitter named. nil until
	// a barrier is actually recorded (most choice points never carry one).
	//
	// `compile_if_then_else/7` in wam_target.pl reserves a *permanent* (Y)
	// register for the ITE barrier AFTER it has decided whether the clause
	// needs an environment, so under `ite_use_y_level(true)` — which every
	// Go compile passes — it emits `get_level Y1` … `cut Y1` into clauses
	// with NO `allocate`. `sat(V, gte(G)) :- \+ lt(V, G)` (the resolver's
	// satisfies/2 shape) is exactly that clause. Y registers are global
	// slots here (Regs[200..299]), so writing the level into Regs[200]
	// scribbled on whatever Y1 the *caller* was holding — the
	// WAM_FLEET_GAPS A2 hazard in its frameless-Y-write form.
	//
	// Keeping the level on the if-then-else's own choice point instead
	// (the wam_rust `ChoicePoint::levels` / wam_python `ChoicePoint.levels`
	// model) means it never touches a register at all: it is per-activation
	// for free, and backtracking discards it with the choice point that
	// owns it.
	Levels map[int]int
}

type EnvFrame struct {
	CP int
	B0 int
	// PrevE is the WamState.E (env-frame stack index) at the moment
	// this frame was Allocated. Deallocate restores vm.E to PrevE
	// for the LIFO walk back through nested activations. The "logical
	// pop only" discipline lets env frames stay physically on the
	// stack while a younger choicepoint still references them, so
	// `pushChoicePoint` no longer has to deep-copy the stack — it
	// just records `len(vm.Stack)` as a mark, and backtrack truncates
	// to that mark. Stack frames whose B0 makes them "dead" stay
	// inert until the truncation rolls past them.
	PrevE int
	// SavedYRegs is a snapshot of Regs[200..299] (the Y-register range)
	// taken at Allocate time and restored at Deallocate. Y-regs are
	// global slots in this runtime, so without per-call save/restore
	// nested predicates collide on the same slot — symptom: in the
	// effective-distance bench, power_sum_bound reserved X203 for the
	// per-iteration Hops accumulator, but the inner category_ancestor's
	// clause 2 reuses X203 for the parent atom (also via PutVariable),
	// so when category_ancestor returned, Regs[203] held an atom
	// instead of the integer the wrapper expected, and the next is/2
	// in power_sum_bound failed. Restoring at Deallocate gives the
	// caller's Y-regs back. Bindings-based deref still propagates any
	// genuine variable binding the callee made on a caller-passed
	// Unbound — Bindings[Idx] is global and survives the Y-reg
	// restore, so reading the restored Y-reg dereferences through the
	// (still-bound) Unbound to the bound value.
	SavedYRegs [100]Value
	// CutB0: the predicate's call-time choicepoint height (copied from
	// vm.PendingB0 at Allocate). A plain cut truncates choicepoints back
	// to CutB0. Kept separate from B0 (allocate-time) because B0 drives
	// Deallocate's physical-pop heuristic and must not change meaning.
	CutB0 int
}

type UnifyCtx struct {
	Args []Value
}

// WriteCtx is one frame of the write-mode argument stack: N counts the
// argument slots still to be filled in the term being built.
//
// Struct/List name that term. They exist because write contexts *nest*:
// a head like `p(C, A, [tk(C)|A])` compiles to
//
//	get_list A3
//	unify_variable X3      % cons head
//	get_structure tk/1, X3 % nested term, pushes its own context
//	unify_value X1         % fills tk's argument, pops back
//	unify_value X2         % must fill the *cons tail*
//
// The runtime used to track the term under construction in the single
// vm.CurrentStruct / vm.CurrentList slots, which the nested
// get_structure overwrote and then cleared on pop. The final
// unify_value had no target left, so the cons tail was never filled and
// `[tk(c)]` came out as a two-element list with an empty second slot.
// Keeping the target on the frame lets popStackRestoringWriteTarget
// restore the enclosing term.
type WriteCtx struct {
	N      int
	Struct *Structure
	List   *List
}

type StackEntry interface {
	stackTag()
}

func (e *EnvFrame) stackTag() {}
func (e *UnifyCtx) stackTag() {}
func (e *WriteCtx) stackTag() {}

// WamContext holds cold, shared read-only data across parallel seeds.
type WamContext struct {
	Code         []Instruction
	Labels       map[string]int
	ForeignNativeKinds        map[string]string
	ForeignResultLayouts      map[string]string
	ForeignResultModes        map[string]string
	ForeignStringConfigs      map[string]map[string]string
	ForeignUsizeConfigs       map[string]map[string]int
	AtomFact2Sources          map[string]AtomFact2Source
	IndexedAtomFactPairs      map[string][]AtomPair
	IndexedWeightedEdgeTriples map[string][]WeightedEdgeTriple
	AtomIntern                map[string]int
	InternedFacts             map[string][][]int
	InternedWeightedFacts     map[string][]InternedWeightedEdge
}

// InternedWeightedEdge stores interned atom IDs + weight for weighted edges.
type InternedWeightedEdge struct {
	Left   int
	Right  int
	Weight float64
}

// WamState holds hot, per-seed mutable state.
type WamState struct {
	Ctx          *WamContext
	PC           int
	CP           int
	// PendingB0: choicepoint-stack height captured by Call/Execute just
	// before transferring to a user predicate (BEFORE its try_me_else
	// pushes the clause-chain CP). This is the WAM B0 / JS cut_barrier.
	// Allocate copies it into EnvFrame.CutB0; !/0 truncates to
	// PendingB0 (not the caller's env), so a no-Allocate neck-cut
	// still leaves the caller's alternatives.
	PendingB0    int
	// CutB0Stack is the saved-B0 stack (JS cut_stack). Call pushes the
	// caller's PendingB0; Proceed pops it. Execute rebases PendingB0
	// WITHOUT pushing — LCO reuses the caller's slot.
	CutB0Stack   []int
	// Regs holds A/X/Y registers in a single flat array.
	//   * A1..A8: indices 0..7 (argument registers)
	//   * X-regs: 8..199  (clause-local temporaries)
	//   * Y-regs: 200..299 (env-frame permanents; saved/restored at
	//     Allocate/Deallocate via EnvFrame.SavedYRegs)
	// Beyond 299 is unused space. The size used to be 512 but profiling
	// at scale-300 showed `copy(vm.Regs[:len(cp.SavedRegs)],
	// cp.SavedRegs)` was ~37% of CPU; trimming to 320 cuts that copy
	// (and snapshotAllRegs's matching alloc) by ~37% per choicepoint
	// without changing the addressable-slot range.
	Regs         [320]Value
	// Bindings is a dense slice indexed by Unbound.Idx. nil means
	// unbound; non-nil is the bound value. The earlier rep was
	// `map[int]Value`; with allocVarId starting at 1000 and
	// monotonically incrementing, Idx values pack densely, so a
	// slice access (~5ns) wins over a map access (~30ns) once the
	// per-CP work is small enough that map ops show up. The
	// `setBinding` helper grows the slice on demand (doubling)
	// for the rare out-of-range write.
	Bindings []Value
	Heap         []Value
	HeapLen      int
	Stack        []StackEntry
	Trail        []TrailEntry
	TrailLen     int
	ChoicePoints []ChoicePoint
	// YSaves: caller Y-register snapshots pushed by Call (not Execute)
	// and popped by Proceed. Y slots are global (Regs[200..299]), so a
	// callee that writes Y without Allocate (GetVariable Y1) would
	// otherwise clobber the caller's live Y. This is the Allocate-less
	// half of A2; X101≡Y1 aliasing is unchanged.
	//
	// It only protects entries made through the WAM `Call` instruction.
	// A predicate entered any other way — a shim / lowered.go doing
	// `vm.PC = labels["p/4"]; vm.Run()`, which pushes no Y save — is
	// NOT covered, which is why the if-then-else barrier does not live
	// in a register at all any more (see ChoicePoint.Levels).
	YSaves       [][100]Value
	// PendingLevel* park the barrier level captured by a `get_level Yn`
	// that sits immediately before an if-then-else `try_me_else`, so the
	// choice point that try_me_else is about to push can own it. The
	// other emission shape (`get_level Yn` immediately AFTER the
	// try_me_else, used when the condition holds a top-level `!`) needs
	// no parking: its guard choice point already exists.
	// The zero value means "nothing parked", so no constructor has to
	// initialise these.
	PendingLevelSet bool
	PendingLevelReg int
	PendingLevelVal int
	Halted       bool
	CurrentStruct *Structure
	CurrentList   *List
	// NextVarId is a monotonically-increasing counter used to give every
	// new Unbound a globally-unique Idx. Earlier code reused i.Xn (the
	// X-register slot index) as Idx, which made the Bindings map collide
	// across activations: a recursive call's PutVariable X207 reused
	// Idx=207, and the outer call's Unbound{Idx:207} would inherit the
	// inner call's binding via the shared Bindings[207] entry. Symptom
	// at depth-3 of category_ancestor: outer-level X206 (which held an
	// Unbound{Idx:207} carried in via the caller's PutVariable X207)
	// would deref to the inner-level's bound integer, and the
	// `OuterHops is RecursiveHops + 1` is/2 call would fail with
	// `1 is 2`. Start above 1000 so the counter never collides with
	// the register-slot range that was used pre-fix.
	NextVarId    int
	// MaxYReg is the highest Y-register index ever written + 1 (i.e.,
	// the count of slots the program actually uses in the Y range
	// 200..299). snapshotAllRegs uses this as the upper bound on the
	// Y-reg copy: most programs use only a handful of Y-regs (the
	// effective-distance bench uses ~10 of the 100 available),
	// shrinking the per-CP snapshot from 108 elements to 8 + actual.
	// Monotonically grows; never resets. Starts at 200 so 0 means
	// "no Y-regs used yet" via the formula `MaxYReg - 200`.
	MaxYReg int
	// MaxAReg is the highest A-register index ever written + 1.
	//
	// A-registers occupy Regs[0..99] (go_reg_index maps A(N) to N-1),
	// so a predicate of arity > 8 uses Regs[8..] — which snapshotAllRegs
	// used to skip as "the X-register range". X actually starts at
	// Regs[100], so arguments A9 and up were never saved at a
	// choicepoint and never restored on backtracking. Predicates like
	// the portable parser's parse_op_loop/10 silently lost arguments on
	// their second clause. Tracking the high-water mark keeps the
	// snapshot small for the common arity<=8 case while staying correct
	// for wider predicates.
	MaxAReg int
	// E is the stack index of the currently-active EnvFrame, or -1
	// when no frame is allocated (e.g., right after NewWamState before
	// any Allocate). With env trimming, Deallocate moves E back along
	// the EnvFrame.PrevE chain rather than physically popping; the
	// frame stays on the stack until either (a) physical pop is safe
	// (no younger CP referencing it) or (b) backtrack truncates the
	// stack down past it. peekEnvFrame is now O(1) via this index
	// instead of an O(stack) reverse scan.
	E int
	// Input is the default input stream used by get_char/1, peek_char/1,
	// and get_code/1. File-backed stream handles carry their own readers.
	Input *bufio.Reader
	NextStreamID int
	// UncaughtBall holds the term of an ISO error that escaped every
	// catch/3. Run recovers it and reports failure, so an uncaught
	// error does not take the process down; callers that care can
	// inspect this after Run returns false.
	UncaughtBall Value
}

// allocVarId returns a fresh, globally-unique Idx for a new logical
// variable. The 1000 floor reserves the X-register-slot range below
// (which earlier code used as Idx values) so existing snapshots /
// trail entries that happened to pre-date this change can't collide.
//
// Drivers (the uw-resolve JSON shim, probe mains) mint output
// Unbounds at Idx 10000+arity. A long query used to walk NextVarId
// from 1000 through that window (~9000 PutVariable/SetVariable
// cells) and alias Bindings[10002] with resolve_layered's Selection
// — sort/2 then unified a 10-package Acc with [] and failed. Jump
// over 10000..10999 so those driver-minted cells stay unique.
func (vm *WamState) allocVarId() int {
	if vm.NextVarId < 1000 {
		vm.NextVarId = 1000
	}
	if vm.NextVarId >= 10000 && vm.NextVarId < 11000 {
		vm.NextVarId = 11000
	}
	id := vm.NextVarId
	vm.NextVarId++
	return id
}

func NewWamContext(code []Instruction, labels map[string]int) *WamContext {
	return &WamContext{
		Code:   code,
		Labels: labels,
		ForeignNativeKinds:   make(map[string]string),
		ForeignResultLayouts: make(map[string]string),
		ForeignResultModes:   make(map[string]string),
		ForeignStringConfigs: make(map[string]map[string]string),
		ForeignUsizeConfigs:  make(map[string]map[string]int),
		AtomFact2Sources:     make(map[string]AtomFact2Source),
		IndexedAtomFactPairs: make(map[string][]AtomPair),
		IndexedWeightedEdgeTriples: make(map[string][]WeightedEdgeTriple),
		AtomIntern:           make(map[string]int),
		InternedFacts:        make(map[string][][]int),
		InternedWeightedFacts: make(map[string][]InternedWeightedEdge),
	}
}

func NewWamState(code []Instruction, labels map[string]int) *WamState {
	ctx := NewWamContext(code, labels)
	return &WamState{Ctx: ctx, Bindings: make([]Value, 4096), E: -1, Input: bufio.NewReader(os.Stdin), NextStreamID: 1}
}

func NewWamStateFromCtx(ctx *WamContext) *WamState {
	return &WamState{Ctx: ctx, Bindings: make([]Value, 4096), E: -1, Input: bufio.NewReader(os.Stdin), NextStreamID: 1}
}

// RunParallel executes multiple seeds in parallel, each with their own
// WamState sharing the read-only WamContext.
func RunParallel(ctx *WamContext, seeds [][]Value, maxWorkers int) [][]Value {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}

	results := make([][]Value, len(seeds))
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	for idx, seed := range seeds {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, seedArgs []Value) {
			defer wg.Done()
			defer func() { <-sem }()

			state := NewWamStateFromCtx(ctx)
			for j, arg := range seedArgs {
				state.Regs[j] = arg
			}
			if state.Run() {
				results[i] = state.CollectResults()
			}
		}(idx, seed)
	}

	wg.Wait()
	return results
}

func (vm *WamState) putReg(idx int, val Value) {
	vm.Regs[idx] = val
	// Track the high-water mark of Y-reg writes so snapshotAllRegs
	// can size its Y-range copy to actually-used slots only. The
	// extra branch + compare is cheap relative to the snapshot
	// shrink it enables (per-CP Y-range copy goes from 100
	// elements to typically ~10).
	if idx >= 200 {
		if idx >= vm.MaxYReg {
			vm.MaxYReg = idx + 1
		}
	} else if idx < 100 && idx >= vm.MaxAReg {
		vm.MaxAReg = idx + 1
	}
}

func (vm *WamState) getReg(idx int) Value {
	return vm.Regs[idx]
}

// getBinding reads vm.Bindings[addr] safely — returns nil if addr is
// past the slice's current length (which the runtime treats the same
// as "unbound"). Hot path; keep tiny.
func (vm *WamState) getBinding(addr int) Value {
	if addr < 0 || addr >= len(vm.Bindings) {
		return nil
	}
	return vm.Bindings[addr]
}

// setBinding writes val to vm.Bindings[addr], growing the slice if
// addr is past the current length. Doubles capacity (with floor 64)
// to amortise allocs across the dense 1000+ Idx range allocVarId
// hands out during a query.
func (vm *WamState) setBinding(addr int, val Value) {
	if addr >= len(vm.Bindings) {
		newCap := len(vm.Bindings)
		if newCap < 64 {
			newCap = 64
		}
		for newCap <= addr {
			newCap *= 2
		}
		newBindings := make([]Value, newCap)
		copy(newBindings, vm.Bindings)
		vm.Bindings = newBindings
	}
	vm.Bindings[addr] = val
}

func (vm *WamState) trailBinding(addr int) {
	old := vm.getBinding(addr)
	vm.Trail = append(vm.Trail, TrailEntry{Addr: addr, Old: old, HadOld: old != nil, RegIdx: -1})
	vm.TrailLen++
}

// bindUnbound binds an unbound variable u to val, recording the change
// on the trail so backtrack can undo it, and rewriting every register
// slot that aliases u so subsequent reads see the bound value.
//
// The previous implementation also did `vm.putReg(u.Idx, val)` as an
// unconditional fallback. That was load-bearing only when u.Idx was a
// valid register index that happened not to be caught by the alias
// rewrite — but for any caller that constructs an Unbound without
// setting Idx (e.g. a benchmark driver doing `&Unbound{Name: "weight"}`
// to ask for an output register), Idx defaults to zero and the line
// silently overwrites Regs[0] = A1 with the bound value, corrupting
// the input atom. The for-loop above already handles every register
// that genuinely aliases u, so the putReg call is redundant in the
// happy path and actively harmful when callers don't fill in Idx.
func (vm *WamState) bindUnbound(u *Unbound, val Value) {
	vm.trailBinding(u.Idx)
	vm.setBinding(u.Idx, val)
	for idx, reg := range vm.Regs {
		if reg == u {
			// Trail the rewrite so backtrack restores this register to the
			// (now-unbound) variable cell. Without this, a register that
			// aliased u keeps the stale bound value after backtrack and a
			// later rebind of u (e.g. a generator yielding its next
			// solution) never reaches it -- which made forall over a
			// generator see only the first element.
			vm.Trail = append(vm.Trail, TrailEntry{RegIdx: idx, RegOld: reg})
			vm.TrailLen++
			vm.Regs[idx] = val
		}
	}
}

func (vm *WamState) heapPush(v Value) int {
	addr := vm.HeapLen
	vm.Heap = append(vm.Heap, v)
	vm.HeapLen++
	return addr
}

func (vm *WamState) heapTrimTo(mark int) {
	vm.Heap = vm.Heap[:mark]
	vm.HeapLen = mark
}

func (vm *WamState) trailTrimTo(mark int) {
	vm.Trail = vm.Trail[:mark]
	vm.TrailLen = mark
}

// snapshotAllRegs captures the used A-register range plus the used
// portion of the Y-register range (Regs[200..vm.MaxYReg]). The
// X-register range (Regs[100..199]) is intentionally skipped — X-regs
// are clause-local in the codegen, so the next clause's head writes
// whatever X-regs it needs and stale leftovers from the failed clause
// never get touched. Y-regs *do* need saving because env trimming
// stores Y-regs at *Allocate* time (caller's outer Y-regs) — at
// TryMeElse time the caller's *current* Y-regs only exist in the CP
// snapshot.
//
// The A range is sized by vm.MaxAReg, floored at 8. It used to be a
// hard Regs[0..7]: A-registers live at Regs[0..99] (A(N) -> N-1), so
// everything from A9 up was silently outside the snapshot and was
// never restored on backtracking. A predicate of arity > 8 — the
// portable parser's parse_op_loop/10, for instance — lost arguments
// the moment it tried its second clause. Sizing by the high-water mark
// keeps the common arity<=8 case exactly as cheap as before.
//
// Layout in the returned slice:
//   - saved[0]                   := acount (as *Integer)
//   - saved[1 .. 1+acount)       := vm.Regs[0..acount)        (A regs)
//   - saved[1+acount .. ]        := vm.Regs[200..200+ycount)  (Y regs)
//
// where acount := max(8, vm.MaxAReg) and ycount := max(0, MaxYReg-200).
// acount is stored rather than implied because restoreSavedRegs has to
// split the slice, and MaxAReg may have grown since the push.
func (vm *WamState) snapshotAllRegs() []Value {
	acount := vm.MaxAReg
	if acount < 8 {
		acount = 8
	}
	ycount := vm.MaxYReg - 200
	if ycount < 0 {
		ycount = 0
	}
	saved := make([]Value, 1+acount+ycount)
	saved[0] = &Integer{Val: int64(acount)}
	copy(saved[1:1+acount], vm.Regs[:acount])
	if ycount > 0 {
		copy(saved[1+acount:], vm.Regs[200:200+ycount])
	}
	return saved
}

// restoreSavedRegs is the dual of snapshotAllRegs. The snapshot
// captures up to MaxYReg-at-push-time; vm.MaxYReg may have grown
// since (the failed clause wrote new Y-regs). Those "post-push"
// Y-regs were nil at push time and need to be cleared back to nil
// here, otherwise they'd leak failed-clause values into the next
// clause's body. Without the clear, the bench had Velocity at
// 1.602159 vs the post-Phase-D 1.601079 — a Y-reg from a longer
// recursion path was bleeding into a shorter one. The clear loop
// is bounded by MaxYReg-since-push (typically 0-2 slots), so it's
// much cheaper than restoring the full 100-element Y range.
func (vm *WamState) restoreSavedRegs(saved []Value) {
	if len(saved) < 1 {
		return
	}
	marker, ok := saved[0].(*Integer)
	if !ok {
		return
	}
	acount := int(marker.Val)
	if acount < 0 || 1+acount > len(saved) {
		return
	}
	copy(vm.Regs[:acount], saved[1:1+acount])
	ycount := len(saved) - 1 - acount
	if ycount > 0 {
		copy(vm.Regs[200:200+ycount], saved[1+acount:])
	}
	// Clear A-regs written between snapshot and now but not captured
	// (they were outside the snapshot's range, so nil at push time).
	for i := acount; i < vm.MaxAReg; i++ {
		vm.Regs[i] = nil
	}
	// Same for Y-regs.
	pushMaxY := 200 + ycount
	for i := pushMaxY; i < vm.MaxYReg; i++ {
		vm.Regs[i] = nil
	}
}

// ClauseSnapshot is the clause-entry state a T4 (multi_clause_n) lowered
// method restores between clause attempts. The lowered method enumerates
// EVERY clause itself (first-solution / deterministic-prefix), so the
// interpreter is never entered for the predicate. Mirrors the ChoicePoint
// snapshot, minus the resume PC; NextVarId is monotonic (unique ids) and is
// intentionally not restored.
type ClauseSnapshot struct {
	savedRegs []Value
	trailMark int
	heapTop   int
	stackLen  int
	cp        int
	e         int
}

// LoClauseSnapshot captures the clause-entry state for LoRestoreClause.
func (vm *WamState) LoClauseSnapshot() ClauseSnapshot {
	return ClauseSnapshot{
		savedRegs: vm.snapshotAllRegs(),
		trailMark: vm.TrailLen,
		heapTop:   vm.HeapLen,
		stackLen:  len(vm.Stack),
		cp:        vm.CP,
		e:         vm.E,
	}
}

// LoRestoreClause restores vm to a clause-entry snapshot before the next
// clause attempt (mirrors backtrack's WamCP restore, minus the PC).
func (vm *WamState) LoRestoreClause(snap ClauseSnapshot) {
	vm.unwindTrailTo(snap.trailMark)
	vm.restoreSavedRegs(snap.savedRegs)
	if snap.stackLen <= len(vm.Stack) {
		vm.Stack = vm.Stack[:snap.stackLen]
	}
	vm.E = snap.e
	if snap.heapTop >= 0 && snap.heapTop <= vm.HeapLen {
		vm.heapTrimTo(snap.heapTop)
	}
	vm.CP = snap.cp
	vm.CurrentStruct = nil
	vm.CurrentList = nil
}

func (vm *WamState) pushChoicePoint(nextPC int, arity int) {
	_ = arity
	cp := ChoicePoint{
		NextPC:    nextPC,
		CP:        vm.CP,
		E:         vm.E,
		StackLen:  len(vm.Stack),
		SavedRegs: vm.snapshotAllRegs(),
		HeapTop:   vm.HeapLen,
		TrailMark: vm.TrailLen,
	}
	vm.fillBarrier(&cp)
	vm.ChoicePoints = append(vm.ChoicePoints, cp)
}

func (vm *WamState) pushIndexedChoicePoint(pcs []int, arity int) {
	_ = arity
	cp := ChoicePoint{
		CP:        vm.CP,
		E:         vm.E,
		StackLen:  len(vm.Stack),
		SavedRegs: vm.snapshotAllRegs(),
		HeapTop:   vm.HeapLen,
		TrailMark: vm.TrailLen,
		IndexedClausePCs: append([]int(nil), pcs...),
	}
	vm.fillBarrier(&cp)
	vm.ChoicePoints = append(vm.ChoicePoints, cp)
}

func (vm *WamState) copyCutB0Stack() []int {
	if len(vm.CutB0Stack) == 0 {
		return nil
	}
	out := make([]int, len(vm.CutB0Stack))
	copy(out, vm.CutB0Stack)
	return out
}

func (vm *WamState) copyYSaveStack() [][100]Value {
	if len(vm.YSaves) == 0 {
		return nil
	}
	out := make([][100]Value, len(vm.YSaves))
	copy(out, vm.YSaves)
	return out
}

func (vm *WamState) fillBarrier(cp *ChoicePoint) {
	cp.PendingB0 = vm.PendingB0
	cp.CutB0Stack = vm.copyCutB0Stack()
	cp.YSaves = vm.copyYSaveStack()
	cp.YSaveLen = len(vm.YSaves)
	// A `get_level Yn` sitting immediately before this try_me_else parked
	// its barrier level; this is the guard choice point that owns it.
	if vm.PendingLevelSet {
		if cp.Levels == nil {
			cp.Levels = make(map[int]int, 2)
		}
		cp.Levels[vm.PendingLevelReg] = vm.PendingLevelVal
		vm.PendingLevelSet = false
	}
}

func (vm *WamState) restoreBarrier(cp *ChoicePoint) {
	vm.PendingB0 = cp.PendingB0
	// A parked-but-unconsumed barrier level cannot outlive a backtrack:
	// the try_me_else that was going to claim it never ran.
	vm.PendingLevelSet = false
	if cp.CutB0Stack != nil {
		vm.CutB0Stack = make([]int, len(cp.CutB0Stack))
		copy(vm.CutB0Stack, cp.CutB0Stack)
	} else {
		vm.CutB0Stack = nil
	}
	if cp.YSaves != nil {
		vm.YSaves = make([][100]Value, len(cp.YSaves))
		copy(vm.YSaves, cp.YSaves)
	} else {
		vm.YSaves = nil
	}
}

// recordIteLevel implements `get_level Yn`: snapshot the cut level for an
// if-then-else / negation barrier WITHOUT writing register Yn.
//
// The level itself is unchanged from the register-based version —
// len(vm.ChoicePoints) at this instant — only where it is kept changes.
// Two emission shapes come out of compile_if_then_else/7:
//
//  1. `get_level BarrierReg` immediately BEFORE the ITE `try_me_else`:
//     park it, and let that try_me_else record it on the guard choice
//     point it pushes (fillBarrier). The level equals the index the guard
//     will occupy, so `cut BarrierReg` prunes the guard and everything the
//     branch pushed above it.
//  2. `get_level CondBarrierReg` immediately AFTER the try_me_else — only
//     when the condition holds a top-level `!`. Its guard choice point
//     already exists, so attach the level to it directly; the level then
//     includes the guard, and a cut in the condition prunes only the
//     condition's own choice points.
//
// Shape 1 must park rather than write the top choice point: the clause may
// be entered with NO choice points at all (`sat/2` reached from a shim or
// from lowered.go), and there would be nowhere to record.
func (vm *WamState) recordIteLevel(reg int) {
	level := len(vm.ChoicePoints)
	if vm.PC+1 < len(vm.Ctx.Code) {
		switch vm.Ctx.Code[vm.PC+1].(type) {
		case *TryMeElse, *TryMeElsePc:
			vm.PendingLevelSet = true
			vm.PendingLevelReg = reg
			vm.PendingLevelVal = level
			return
		}
	}
	if n := len(vm.ChoicePoints); n > 0 {
		cp := &vm.ChoicePoints[n-1]
		if cp.Levels == nil {
			cp.Levels = make(map[int]int, 2)
		}
		cp.Levels[reg] = level
	}
}

// lookupIteLevel implements the read half of `cut Yn`: find the level that
// `get_level Yn` recorded, innermost choice point first. Searching the
// choice-point stack rather than a register makes the barrier
// per-activation for free — two live activations of the same clause each
// carry their own — and a callee can never clobber a caller's.
//
// `false` means the guard was already cut away by an inner commit, in which
// case the stack is at or below the level anyway and cutting is a no-op.
func (vm *WamState) lookupIteLevel(reg int) (int, bool) {
	for i := len(vm.ChoicePoints) - 1; i >= 0; i-- {
		if lvl, ok := vm.ChoicePoints[i].Levels[reg]; ok {
			return lvl, true
		}
	}
	return 0, false
}

func (vm *WamState) pushYSave() {
	var s [100]Value
	copy(s[:], vm.Regs[200:300])
	vm.YSaves = append(vm.YSaves, s)
}

func (vm *WamState) popYSave() {
	n := len(vm.YSaves)
	if n == 0 {
		return
	}
	s := vm.YSaves[n-1]
	vm.YSaves = vm.YSaves[:n-1]
	copy(vm.Regs[200:300], s[:])
}

// pushCallFrame is WAM Call: snapshot caller Ys and B0, then B0 <- B.
func (vm *WamState) pushCallFrame() {
	vm.CutB0Stack = append(vm.CutB0Stack, vm.PendingB0)
	vm.pushYSave()
	vm.PendingB0 = len(vm.ChoicePoints)
}

// enterExecute is WAM Execute: B0 <- B without pushing. The callee
// reuses the caller's cut_stack/Y-save slot (its Proceed pops the
// Call that entered this activation) but gets its own barrier, so a
// neck-cut cannot wipe the caller's clause alternatives.
func (vm *WamState) enterExecute() {
	vm.PendingB0 = len(vm.ChoicePoints)
}

func (vm *WamState) popCallFrame() {
	vm.popYSave()
	n := len(vm.CutB0Stack)
	if n == 0 {
		vm.PendingB0 = 0
		return
	}
	vm.PendingB0 = vm.CutB0Stack[n-1]
	vm.CutB0Stack = vm.CutB0Stack[:n-1]
}

func (vm *WamState) popEnvFrame() *EnvFrame {
	for i := len(vm.Stack) - 1; i >= 0; i-- {
		if f, ok := vm.Stack[i].(*EnvFrame); ok {
			vm.Stack = vm.Stack[:i]
			return f
		}
	}
	return nil
}

func (vm *WamState) peekUnifyCtx() *UnifyCtx {
	if len(vm.Stack) == 0 { return nil }
	if ctx, ok := vm.Stack[len(vm.Stack)-1].(*UnifyCtx); ok {
		return ctx
	}
	return nil
}

func (vm *WamState) peekWriteCtx() *WriteCtx {
	if len(vm.Stack) == 0 { return nil }
	if ctx, ok := vm.Stack[len(vm.Stack)-1].(*WriteCtx); ok {
		return ctx
	}
	return nil
}

func (vm *WamState) peekEnvFrame() *EnvFrame {
	if vm.E < 0 || vm.E >= len(vm.Stack) {
		return nil
	}
	if f, ok := vm.Stack[vm.E].(*EnvFrame); ok {
		return f
	}
	return nil
}

func (vm *WamState) popStack() {
	if len(vm.Stack) > 0 {
		vm.Stack = vm.Stack[:len(vm.Stack)-1]
	}
}

// popStackRestoringWriteTarget pops a finished write context and points
// vm.CurrentStruct / vm.CurrentList back at the enclosing term, so the
// unify_* instructions that follow a nested get_structure / get_list
// fill the *outer* term's remaining slots. With no enclosing write
// context both slots clear, which is the old unconditional behaviour.
func (vm *WamState) popStackRestoringWriteTarget() {
	vm.popStack()
	if wctx := vm.peekWriteCtx(); wctx != nil {
		vm.CurrentStruct = wctx.Struct
		vm.CurrentList = wctx.List
		return
	}
	vm.CurrentStruct = nil
	vm.CurrentList = nil
}

func (vm *WamState) heapSubargs(start, n int) []Value {
	if start+n > vm.HeapLen { return nil }
	return vm.Heap[start : start+n]
}

func (vm *WamState) Clone() *WamState {
	newState := &WamState{
		Ctx:          vm.Ctx,
		PC:           vm.PC,
		CP:           vm.CP,
		Bindings:     make([]Value, len(vm.Bindings)),
		Heap:         make([]Value, vm.HeapLen),
		HeapLen:      vm.HeapLen,
		Stack:        make([]StackEntry, len(vm.Stack)),
		Trail:        make([]TrailEntry, vm.TrailLen),
		TrailLen:     vm.TrailLen,
		ChoicePoints: make([]ChoicePoint, len(vm.ChoicePoints)),
		Halted:       vm.Halted,
		CurrentStruct: vm.CurrentStruct,
		CurrentList:   vm.CurrentList,
		// Carry over the Idx counter so the cloned VM allocates IDs
		// strictly above what the parent has issued. Without this the
		// sub-VM (used by executeAggregate) starts at 1000 and reuses
		// Idx values the parent already bound — its "fresh" Unbounds
		// then deref through the parent's bindings and silently drop
		// every solution.
		NextVarId:    vm.NextVarId,
		E:            vm.E,
		Input:        vm.Input,
		NextStreamID: vm.NextStreamID,
		PendingB0:    vm.PendingB0,
		CutB0Stack:   vm.copyCutB0Stack(),
		// Carry MaxYReg so the sub-VM's snapshotAllRegs sees the
		// same Y-reg high-water mark as the parent (sub.Regs is a
		// value-copy of parent.Regs and therefore *contains* those
		// Y-reg values; without copying MaxYReg the sub-VM thinks
		// it has 0 Y-regs in use, the snapshot omits them, and
		// backtrack inside the sub-VM doesn't restore values that
		// the failed clause overwrote).
		MaxYReg:      vm.MaxYReg,
		MaxAReg:      vm.MaxAReg,
	}
	newState.Regs = vm.Regs
	copy(newState.Bindings, vm.Bindings)
	copy(newState.Heap, vm.Heap)
	copy(newState.Stack, vm.Stack)
	copy(newState.Trail, vm.Trail)
	copy(newState.ChoicePoints, vm.ChoicePoints)
	if len(vm.YSaves) > 0 {
		newState.YSaves = append([][100]Value(nil), vm.YSaves...)
	}
	if len(vm.CutB0Stack) > 0 {
		newState.CutB0Stack = append([]int(nil), vm.CutB0Stack...)
	}
	return newState
}

// freezeTerm copies v with every Unbound/Ref chased through this VM's
// Bindings and Heap. findall/bagof run in a Clone(); a shallow deref of
// the template leaves nested Unbounds whose bindings live only on the
// sub-VM. After the clone is discarded those slots read as unbound in
// the parent (inst_walk's findall(D-DV, ...) yielded [null-v, null-v]).
func (vm *WamState) freezeTerm(v Value) Value {
	if v == nil {
		return nil
	}
	v = vm.deref(v)
	switch t := v.(type) {
	case *Structure:
		args := make([]Value, len(t.Args))
		for i, a := range t.Args {
			args[i] = vm.freezeTerm(a)
		}
		return &Structure{Functor: t.Functor, Arity: t.Arity, Args: args}
	case *Compound:
		args := make([]Value, len(t.Args))
		for i, a := range t.Args {
			args[i] = vm.freezeTerm(a)
		}
		return &Compound{Functor: t.Functor, Args: args}
	case *List:
		elems := make([]Value, len(t.Elements))
		for i, a := range t.Elements {
			elems[i] = vm.freezeTerm(a)
		}
		return listFromItems(elems)
	default:
		return v
	}
}

func (vm *WamState) ForkAtChoicePoint() *WamState {
	if len(vm.ChoicePoints) == 0 {
		return nil
	}
	clone := vm.Clone()
	if clone.backtrack() {
		vm.ChoicePoints = vm.ChoicePoints[:len(vm.ChoicePoints)-1]
		return clone
	}
	return nil
}

func copyStack(stack []StackEntry) []StackEntry {
	if len(stack) == 0 {
		return nil
	}
	stack2 := make([]StackEntry, len(stack))
	copy(stack2, stack)
	return stack2
}

func parseFunctorArity(f string) int {
	// Handle str(f/n) format
	if strings.HasPrefix(f, "str(") && strings.HasSuffix(f, ")") {
		f = f[4 : len(f)-1]
	}
	parts := strings.Split(f, "/")
	if len(parts) >= 2 {
		arity, _ := strconv.Atoi(parts[len(parts)-1])
		return arity
	}
	return 0
}

func parseFunctorName(f string) string {
	if strings.HasPrefix(f, "str(") && strings.HasSuffix(f, ")") {
		f = f[4 : len(f)-1]
	}
	parts := strings.Split(f, "/")
	if len(parts) >= 2 {
		return strings.Join(parts[:len(parts)-1], "/")
	}
	return f
}

// isConsName reports whether a bare functor name (arity stripped) is one of
// the interchangeable cons-cell spellings WAM lists use.
func isConsName(n string) bool {
	return n == "[|]" || n == "."
}

// consHeadTail exposes head/tail for the cons-cell spellings that arrive as
// a Compound/Structure (e.g. "[|]/2", built by put_structure for list tail
// cells) rather than as a *List. Without this, GetList recognised only the
// outer *List cell and failed on the inner "[|]/2" tail cells, so recursive
// list predicates (member/reverse) mis-traversed the tail.
func consHeadTail(v Value) (Value, Value, bool) {
	switch t := v.(type) {
	case *Compound:
		if isConsName(parseFunctorName(t.Functor)) && len(t.Args) == 2 {
			return t.Args[0], t.Args[1], true
		}
	case *Structure:
		if isConsName(parseFunctorName(t.Functor)) && len(t.Args) == 2 {
			return t.Args[0], t.Args[1], true
		}
	}
	return nil, nil, false
}

// compareValues returns <0, 0, or >0 for sort ordering of atoms/integers.
func compareValues(a, b Value) int {
	// Type ordering: Atom < Integer < Float < other
	typeRank := func(v Value) int {
		switch v.(type) {
		case *Atom:    return 0
		case *Integer: return 1
		case *Float:   return 2
		default:       return 3
		}
	}
	ra, rb := typeRank(a), typeRank(b)
	if ra != rb { return ra - rb }
	switch ta := a.(type) {
	case *Atom:
		tb := b.(*Atom)
		if ta.Name < tb.Name { return -1 }
		if ta.Name > tb.Name { return 1 }
		return 0
	case *Integer:
		tb := b.(*Integer)
		if ta.Val < tb.Val { return -1 }
		if ta.Val > tb.Val { return 1 }
		return 0
	case *Float:
		tb := b.(*Float)
		if ta.Val < tb.Val { return -1 }
		if ta.Val > tb.Val { return 1 }
		return 0
	}
	return 0
}

// compareTerms is the order sort/2, msort/2, @</2 and compare/3 use.
// compareValues only ranks atoms/numbers and treats every compound as
// equal, so sort([a-1,b-1,c-1], L) uniquely collapsed to one pair —
// uw-resolve's sort(Acc, Selection) kept only the last pick.
func (vm *WamState) compareTerms(a, b Value) int {
	a = vm.deref(a)
	b = vm.deref(b)
	if isEmptyListValue(a) || isEmptyListValue(b) {
		if isEmptyListValue(a) && isEmptyListValue(b) {
			return 0
		}
		if isEmptyListValue(a) {
			return -1
		}
		return 1
	}
	ha, ta, oka := vm.valueListHeadTail(a)
	hb, tb, okb := vm.valueListHeadTail(b)
	if oka || okb {
		if !oka {
			return 1
		}
		if !okb {
			return -1
		}
		if c := vm.compareTerms(ha, hb); c != 0 {
			return c
		}
		return vm.compareTerms(ta, tb)
	}
	rank := func(v Value) int {
		switch v.(type) {
		case *Unbound:
			return 0
		case *Atom:
			return 1
		case *Integer:
			return 2
		case *Float:
			return 3
		default:
			return 4
		}
	}
	ra, rb := rank(a), rank(b)
	if ra != rb {
		return ra - rb
	}
	switch ta := a.(type) {
	case *Unbound:
		tb := b.(*Unbound)
		if ta.Idx < tb.Idx {
			return -1
		}
		if ta.Idx > tb.Idx {
			return 1
		}
		return 0
	case *Atom, *Integer, *Float:
		return compareValues(a, b)
	}
	fa, argsa := decompose(a)
	fb, argsb := decompose(b)
	if len(argsa) != len(argsb) {
		return len(argsa) - len(argsb)
	}
	if fa < fb {
		return -1
	}
	if fa > fb {
		return 1
	}
	for i := range argsa {
		if c := vm.compareTerms(argsa[i], argsb[i]); c != 0 {
			return c
		}
	}
	return 0
}

func isEmptyListValue(v Value) bool {
	if atom, ok := v.(*Atom); ok {
		return atom.Name == "[]"
	}
	if list, ok := v.(*List); ok {
		return len(list.Elements) == 0
	}
	return false
}

// listFromItems is the canonical Go→Prolog list constructor: empty
// becomes the interned [] atom (so get_constant [] matches), non-empty
// becomes a flat *List. Builtins that rebuild lists (append/sort/reverse)
// must go through this rather than `&List{Elements: items}` with a
// possibly-empty slice.
func listFromItems(items []Value) Value {
	if len(items) == 0 {
		return emptyListAtom
	}
	return &List{Elements: items}
}

// emptyListAtom is a process-wide singleton for the `[]` terminator,
// initialised once via internAtom("[]") and reused everywhere a list
// tail terminator is needed. Avoids the per-call internAtom map
// lookup from rawListHeadTail / listHeadTail; those run once per list
// cell in the bench's recursion-heavy paths.
var emptyListAtom = internAtom("[]")

func rawListHeadTail(list *List) (Value, Value, bool) {
	switch len(list.Elements) {
	case 0:
		return nil, nil, false
	case 1:
		return list.Elements[0], emptyListAtom, true
	case 2:
		tail := list.Elements[1]
		if _, ok := tail.(*List); ok {
			return list.Elements[0], tail, true
		}
		if isEmptyListValue(tail) {
			return list.Elements[0], emptyListAtom, true
		}
	}
	return list.Elements[0], &List{Elements: list.Elements[1:]}, true
}

func (vm *WamState) heapConsAfterUnbound(v *Unbound) (Value, bool) {
	for i := 0; i < vm.HeapLen; i++ {
		cell := vm.Heap[i]
		if cell == v && i+1 < vm.HeapLen {
			next := vm.deref(vm.Heap[i+1])
			switch t := next.(type) {
			case *Compound:
				if isConsFunctor(t.Functor) && len(t.Args) == 2 {
					return next, true
				}
			case *Structure:
				if isConsFunctor(t.Functor) && len(t.Args) == 2 {
					return next, true
				}
			}
		}
	}
	return nil, false
}

func (vm *WamState) heapListAfterUnbound(v *Unbound) (Value, bool) {
	found := false
	for i := 0; i < vm.HeapLen; i++ {
		cell := vm.Heap[i]
		if cell == v {
			found = true
			continue
		}
		if !found {
			continue
		}
		next := vm.deref(cell)
		switch t := next.(type) {
		case *List:
			return next, true
		case *Compound:
			if isConsFunctor(t.Functor) && len(t.Args) == 2 {
				return next, true
			}
		case *Structure:
			if isConsFunctor(t.Functor) && len(t.Args) == 2 {
				return next, true
			}
		}
	}
	return nil, false
}

func (vm *WamState) listHeadTail(list *List) (Value, Value, bool) {
	switch len(list.Elements) {
	case 0:
		return nil, nil, false
	case 1:
		return list.Elements[0], emptyListAtom, true
	case 2:
		tail := vm.deref(list.Elements[1])
		if _, ok := tail.(*List); ok {
			return list.Elements[0], tail, true
		}
		if isEmptyListValue(tail) {
			return list.Elements[0], emptyListAtom, true
		}
		if u, ok := tail.(*Unbound); ok {
			if next, ok := vm.heapConsAfterUnbound(u); ok {
				return list.Elements[0], next, true
			}
		}
		switch t := tail.(type) {
		case *Compound:
			if isConsFunctor(t.Functor) && len(t.Args) == 2 {
				return list.Elements[0], tail, true
			}
		case *Structure:
			if isConsFunctor(t.Functor) && len(t.Args) == 2 {
				return list.Elements[0], tail, true
			}
		}
	}
	return list.Elements[0], &List{Elements: list.Elements[1:]}, true
}

func isConsFunctor(functor string) bool {
	name := parseFunctorName(functor)
	return name == "." || name == "[|]"
}

func (vm *WamState) valueListHeadTail(v Value) (Value, Value, bool) {
	v = vm.deref(v)
	switch t := v.(type) {
	case *List:
		return vm.listHeadTail(t)
	case *Compound:
		if isConsFunctor(t.Functor) && len(t.Args) == 2 {
			tail := vm.deref(t.Args[1])
			if u, ok := tail.(*Unbound); ok {
				if next, ok := vm.heapConsAfterUnbound(u); ok {
					return t.Args[0], next, true
				}
			}
			return t.Args[0], t.Args[1], true
		}
	case *Structure:
		if isConsFunctor(t.Functor) && len(t.Args) == 2 {
			tail := vm.deref(t.Args[1])
			if u, ok := tail.(*Unbound); ok {
				if next, ok := vm.heapConsAfterUnbound(u); ok {
					return t.Args[0], next, true
				}
			}
			return t.Args[0], t.Args[1], true
		}
	}
	return nil, nil, false
}

// termIdentical implements ==/2 (and, negated, \==/2) with list-representation
// normalization. findall/aggregate build flat *List values while list literals
// and put_list/put_structure build heap-linked cons cells, so a shallow
// valueEquals sees the two representations of the same logical list as unequal
// (e.g. *List{[a,b,c]} vs *List{[a, _heapTail]}). Mirror Unify's
// valueListHeadTail walk — but without binding anything — so the two
// representations of one logical list compare equal under ==/2. Atomics,
// numbers and variables fall back to valueEquals (variable identity by Idx).
func (vm *WamState) termIdentical(a, b Value) bool {
	a = vm.deref(a)
	b = vm.deref(b)
	if valueEquals(a, b) {
		return true
	}
	ha, ta, oka := vm.valueListHeadTail(a)
	hb, tb, okb := vm.valueListHeadTail(b)
	if oka || okb {
		return oka && okb && vm.termIdentical(ha, hb) && vm.termIdentical(ta, tb)
	}
	fa, argsa := decompose(a)
	fb, argsb := decompose(b)
	if fa != "" && fa == fb && len(argsa) == len(argsb) {
		for i := range argsa {
			if !vm.termIdentical(argsa[i], argsb[i]) {
				return false
			}
		}
		return true
	}
	return false
}

// listToSlice walks a Prolog cons list and collects its elements
// into a Go slice. Used by `length/2`, `member/2`, `append/3`, etc.
// Iterative implementation: the previous recursive version did
// `append([]Value{head}, rest...)` which is O(N²) in allocations
// (each frame allocates a fresh slice and copies the rest); for
// the bench's recursion-heavy `\+ member(M, Visited)` calls on
// 10-deep Visited lists, that was ~4% of post-Phase-G CPU.
func (vm *WamState) listToSlice(v Value) ([]Value, bool) {
	v = vm.deref(v)
	if isEmptyListValue(v) {
		return []Value{}, true
	}
	// Pre-allocate with a small initial capacity; the depth of the
	// bench's Visited list is bounded by max_depth(10) so 16 covers
	// the common case without an early grow.
	out := make([]Value, 0, 16)
	for {
		if isEmptyListValue(v) {
			return out, true
		}
		head, tail, ok := vm.valueListHeadTail(v)
		if !ok {
			return nil, false
		}
		out = append(out, head)
		v = vm.deref(tail)
	}
}

func (vm *WamState) selectSolutions(items []Value) []SelectSolution {
	solutions := make([]SelectSolution, 0, len(items))
	for idx, item := range items {
		rest := make([]Value, 0, len(items)-1)
		rest = append(rest, items[:idx]...)
		rest = append(rest, items[idx+1:]...)
		solutions = append(solutions, SelectSolution{
			Elem: item,
			Rest: &List{Elements: rest},
		})
	}
	return solutions
}

func (vm *WamState) formatTemplate(template string, args []Value) (string, bool) {
	var b strings.Builder
	argIdx := 0
	for i := 0; i < len(template); i++ {
		ch := template[i]
		if ch != '~' || i+1 >= len(template) {
			b.WriteByte(ch)
			continue
		}
		i++
		switch template[i] {
		case 'n':
			b.WriteByte('\n')
		case '~':
			b.WriteByte('~')
		// ~w and ~p render like write/1 (unquoted, arguments included);
		// ~q renders like write_canonical/1 (quoted). All three used to
		// call Value.String(), which printed a compound as "functor/arity"
		// and dropped its arguments.
		case 'w', 'p':
			if argIdx >= len(args) {
				return "", false
			}
			b.WriteString(vm.writeTermString(args[argIdx]))
			argIdx++
		case 'q':
			if argIdx >= len(args) {
				return "", false
			}
			b.WriteString(vm.canonicalTermString(args[argIdx]))
			argIdx++
		case 'a':
			if argIdx >= len(args) {
				return "", false
			}
			value := vm.deref(args[argIdx])
			if atom, ok := value.(*Atom); ok {
				b.WriteString(atom.Name)
			} else {
				b.WriteString(vm.writeTermString(value))
			}
			argIdx++
		case 'd':
			if argIdx >= len(args) {
				return "", false
			}
			value := vm.deref(args[argIdx])
			if integer, ok := value.(*Integer); ok {
				b.WriteString(strconv.FormatInt(integer.Val, 10))
			} else {
				b.WriteString(value.String())
			}
			argIdx++
		case 's':
			if argIdx >= len(args) {
				return "", false
			}
			text, ok := vm.formatStringDirective(vm.deref(args[argIdx]))
			if !ok {
				return "", false
			}
			b.WriteString(text)
			argIdx++
		default:
			b.WriteByte('~')
			b.WriteByte(template[i])
		}
	}
	return b.String(), true
}

func (vm *WamState) formatStringDirective(value Value) (string, bool) {
	if isEmptyListValue(value) {
		return "", true
	}
	if atom, ok := value.(*Atom); ok {
		return atom.Name, true
	}
	items, ok := vm.listToSlice(value)
	if !ok {
		return "", false
	}
	var b strings.Builder
	for _, item := range items {
		code, ok := vm.deref(item).(*Integer)
		if !ok || code.Val < 0 || code.Val > 0x10ffff {
			return "", false
		}
		b.WriteRune(rune(code.Val))
	}
	return b.String(), true
}

func outputAtomCharText(value Value) (string, bool) {
	atom, ok := value.(*Atom)
	if !ok {
		return "", false
	}
	if len([]rune(atom.Name)) != 1 {
		return "", false
	}
	return atom.Name, true
}

func outputCodeText(value Value) (string, bool) {
	integer, ok := value.(*Integer)
	if !ok || integer.Val < 0 || integer.Val > int64(unicode.MaxRune) {
		return "", false
	}
	return string(rune(integer.Val)), true
}

func streamPathText(value Value) (string, bool) {
	atom, ok := value.(*Atom)
	if !ok {
		return "", false
	}
	return strings.TrimPrefix(atom.Name, quotedNumericAtomPrefix), true
}

func streamOpenMode(value Value) (string, bool) {
	atom, ok := value.(*Atom)
	if !ok {
		return "", false
	}
	switch atom.Name {
	case "read", "write", "append":
		return atom.Name, true
	default:
		return "", false
	}
}

func (vm *WamState) openStreamHandle(path string, mode string) (*StreamHandle, bool) {
	var file *os.File
	var err error
	switch mode {
	case "read":
		file, err = os.Open(path)
	case "write":
		file, err = os.Create(path)
	case "append":
		file, err = os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	default:
		return nil, false
	}
	if err != nil {
		return nil, false
	}
	handle := &StreamHandle{ID: vm.NextStreamID, File: file, Mode: mode}
	vm.NextStreamID++
	if mode == "read" {
		handle.Reader = bufio.NewReader(file)
	}
	return handle, true
}

func streamHandle(value Value) (*StreamHandle, bool) {
	handle, ok := value.(*StreamHandle)
	if !ok || handle == nil || handle.Closed || handle.File == nil {
		return nil, false
	}
	return handle, true
}

func readStreamRune(value Value) (rune, bool, bool) {
	handle, ok := streamHandle(value)
	if !ok || handle.Reader == nil {
		return 0, false, false
	}
	r, _, err := handle.Reader.ReadRune()
	if err == io.EOF {
		return 0, true, true
	}
	if err != nil {
		return 0, false, false
	}
	return r, false, true
}

func peekStreamRune(value Value) (rune, bool, bool) {
	handle, ok := streamHandle(value)
	if !ok || handle.Reader == nil {
		return 0, false, false
	}
	r, _, err := handle.Reader.ReadRune()
	if err == io.EOF {
		return 0, true, true
	}
	if err != nil {
		return 0, false, false
	}
	if err := handle.Reader.UnreadRune(); err != nil {
		return 0, false, false
	}
	return r, false, true
}

func readLineStreamValue(value Value) (Value, bool) {
	handle, ok := streamHandle(value)
	if !ok || handle.Reader == nil {
		return nil, false
	}
	var b strings.Builder
	for {
		r, _, err := handle.Reader.ReadRune()
		if err == io.EOF {
			if b.Len() == 0 {
				return internAtom("end_of_file"), true
			}
			return internAtom(b.String()), true
		}
		if err != nil {
			return nil, false
		}
		if r == '\n' {
			return internAtom(b.String()), true
		}
		if r == '\r' {
			next, _, nextErr := handle.Reader.ReadRune()
			if nextErr != nil && nextErr != io.EOF {
				return nil, false
			}
			if nextErr == nil && next != '\n' {
				if err := handle.Reader.UnreadRune(); err != nil {
					return nil, false
				}
			}
			return internAtom(b.String()), true
		}
		b.WriteRune(r)
	}
}

func readStringStreamText(value Value, length int64) (string, bool) {
	if length < 0 {
		return "", false
	}
	handle, ok := streamHandle(value)
	if !ok || handle.Reader == nil {
		return "", false
	}
	var b strings.Builder
	for i := int64(0); i < length; i++ {
		r, _, err := handle.Reader.ReadRune()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", false
		}
		b.WriteRune(r)
	}
	return b.String(), true
}

func writeStreamText(value Value, text string) bool {
	handle, ok := streamHandle(value)
	if !ok || handle.Mode == "read" {
		return false
	}
	_, err := handle.File.WriteString(text)
	return err == nil
}

func (vm *WamState) inputReader() *bufio.Reader {
	if vm.Input == nil {
		vm.Input = bufio.NewReader(os.Stdin)
	}
	return vm.Input
}

func (vm *WamState) readDefaultInputRune() (rune, bool, bool) {
	r, _, err := vm.inputReader().ReadRune()
	if err == io.EOF {
		return 0, true, true
	}
	if err != nil {
		return 0, false, false
	}
	return r, false, true
}

func (vm *WamState) peekDefaultInputRune() (rune, bool, bool) {
	reader := vm.inputReader()
	r, _, err := reader.ReadRune()
	if err == io.EOF {
		return 0, true, true
	}
	if err != nil {
		return 0, false, false
	}
	if err := reader.UnreadRune(); err != nil {
		return 0, false, false
	}
	return r, false, true
}

func inputCharValue(r rune, eof bool) Value {
	if eof {
		return internAtom("end_of_file")
	}
	return internAtom(string(r))
}

func inputCodeValue(r rune, eof bool) Value {
	if eof {
		return &Integer{Val: -1}
	}
	return &Integer{Val: int64(r)}
}

// writeTermString renders a term the way write/1 does: recursively, with
// nested arguments dereferenced, and without the quoting that
// write_canonical/1 applies.
//
// The write-family builtins used to call Value.String() directly, which
// renders a compound as bare "functor/arity" (Structure.String) and drops
// every argument — write(foo(a,b)) printed "foo/2". Reuse the canonical
// walker's structure here so writes carry their arguments; the only
// difference is atom quoting.
// WriteTerm is the exported form of writeTermString, for callers that
// need to render a term the runtime handed back — most often
// vm.UncaughtBall after Run reports failure.
func (vm *WamState) WriteTerm(value Value) string {
	if value == nil {
		return "-"
	}
	return vm.writeTermString(value)
}

func (vm *WamState) writeTermString(value Value) string {
	if value == nil {
		return "_"
	}
	value = vm.deref(value)
	if value == nil {
		return "_"
	}
	if isEmptyListValue(value) {
		return "[]"
	}
	if items, ok := vm.listToSlice(value); ok {
		parts := make([]string, 0, len(items))
		for _, item := range items {
			parts = append(parts, vm.writeTermString(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	}
	switch t := value.(type) {
	case *Atom:
		return t.Name
	case *Integer:
		return strconv.FormatInt(t.Val, 10)
	case *Float:
		return strconv.FormatFloat(t.Val, 'g', -1, 64)
	case *Compound:
		return vm.writeCompoundString(parseFunctorName(t.Functor), t.Args)
	case *Structure:
		return vm.writeCompoundString(parseFunctorName(t.Functor), t.Args)
	default:
		return value.String()
	}
}

func (vm *WamState) writeCompoundString(name string, args []Value) string {
	if len(args) == 0 {
		return name
	}
	parts := make([]string, 0, len(args))
	for _, arg := range args {
		parts = append(parts, vm.writeTermString(arg))
	}
	return name + "(" + strings.Join(parts, ", ") + ")"
}

func (vm *WamState) canonicalTermString(value Value) string {
	// An argument slot can still be nil (e.g. a structure built by
	// put_structure whose set_* instructions have not all run yet);
	// render it as an anonymous variable rather than panicking.
	if value == nil {
		return "_"
	}
	value = vm.deref(value)
	if value == nil {
		return "_"
	}
	if isEmptyListValue(value) {
		return "[]"
	}
	if items, ok := vm.listToSlice(value); ok {
		parts := make([]string, 0, len(items))
		for _, item := range items {
			parts = append(parts, vm.canonicalTermString(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	}
	switch t := value.(type) {
	case *Atom:
		return canonicalAtomText(t.Name)
	case *Integer:
		return strconv.FormatInt(t.Val, 10)
	case *Float:
		return strconv.FormatFloat(t.Val, 'g', -1, 64)
	case *Compound:
		return vm.canonicalCompoundString(parseFunctorName(t.Functor), t.Args)
	case *Structure:
		return vm.canonicalCompoundString(parseFunctorName(t.Functor), t.Args)
	case *Unbound:
		return "_"
	case *Ref:
		return fmt.Sprintf("ref(%d)", t.Addr)
	default:
		return value.String()
	}
}

func (vm *WamState) canonicalCompoundString(name string, args []Value) string {
	if len(args) == 0 {
		return canonicalAtomText(name)
	}
	parts := make([]string, 0, len(args))
	for _, arg := range args {
		parts = append(parts, vm.canonicalTermString(arg))
	}
	return canonicalAtomText(name) + "(" + strings.Join(parts, ", ") + ")"
}

func canonicalAtomText(name string) string {
	if !canonicalAtomNeedsQuote(name) {
		return name
	}
	var b strings.Builder
	b.WriteByte('\'')
	for _, r := range name {
		if r == '\'' || r == '\\' {
			b.WriteByte('\\')
		}
		b.WriteRune(r)
	}
	b.WriteByte('\'')
	return b.String()
}

func canonicalAtomNeedsQuote(name string) bool {
	if name == "[]" {
		return false
	}
	if name == "" {
		return true
	}
	if _, err := strconv.ParseInt(name, 10, 64); err == nil {
		return true
	}
	if _, err := strconv.ParseFloat(name, 64); err == nil {
		return true
	}
	first := true
	for _, r := range name {
		if first {
			if !unicode.IsLower(r) {
				return true
			}
			first = false
		}
		if !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_') {
			return true
		}
	}
	return false
}

func formatTemplateText(v Value) (string, bool) {
	switch t := v.(type) {
	case *Atom:
		return t.Name, true
	case *Integer:
		return strconv.FormatInt(t.Val, 10), true
	default:
		return "", false
	}
}

func (vm *WamState) applySelectSolution(solution SelectSolution) bool {
	mark := vm.TrailLen
	if !vm.Unify(vm.getReg(0), solution.Elem) {
		vm.unwindTrailTo(mark)
		return false
	}
	if !vm.Unify(vm.getReg(2), solution.Rest) {
		vm.unwindTrailTo(mark)
		return false
	}
	return true
}

func decompose(v Value) (string, []Value) {
	switch t := v.(type) {
	case *Atom:
		return t.Name, nil
	case *Compound:
		return parseFunctorName(t.Functor), t.Args
	case *Structure:
		return parseFunctorName(t.Functor), t.Args
	case *List:
		if head, tail, ok := rawListHeadTail(t); ok {
			return ".", []Value{head, tail}
		}
		return "[]", nil
	}
	return "", nil
}

func makeStructureValue(name string, args []Value) Value {
	return &Structure{
		Functor: fmt.Sprintf("%s/%d", name, len(args)),
		Arity:   len(args),
		Args:    append([]Value(nil), args...),
	}
}

func (vm *WamState) termFunctorArity(v Value) (Value, int64, bool) {
	v = vm.deref(v)
	switch t := v.(type) {
	case *Atom:
		return t, 0, true
	case *Integer:
		return t, 0, true
	case *Float:
		return t, 0, true
	case *Compound:
		return internAtom(parseFunctorName(t.Functor)), int64(len(t.Args)), true
	case *Structure:
		return internAtom(parseFunctorName(t.Functor)), int64(t.Arity), true
	case *List:
		if isEmptyListValue(t) {
			return emptyListAtom, 0, true
		}
		return internAtom("."), 2, true
	default:
		return nil, 0, false
	}
}

func (vm *WamState) termToUnivList(v Value) (Value, bool) {
	v = vm.deref(v)
	switch t := v.(type) {
	case *Atom, *Integer, *Float:
		return &List{Elements: []Value{t}}, true
	case *Compound:
		items := make([]Value, 0, len(t.Args)+1)
		items = append(items, internAtom(parseFunctorName(t.Functor)))
		items = append(items, t.Args...)
		return &List{Elements: items}, true
	case *Structure:
		items := make([]Value, 0, len(t.Args)+1)
		items = append(items, internAtom(parseFunctorName(t.Functor)))
		items = append(items, t.Args...)
		return &List{Elements: items}, true
	case *List:
		if isEmptyListValue(t) {
			return &List{Elements: []Value{emptyListAtom}}, true
		}
		head, tail, ok := vm.listHeadTail(t)
		if !ok {
			return nil, false
		}
		return &List{Elements: []Value{internAtom("."), head, tail}}, true
	default:
		return nil, false
	}
}

func (vm *WamState) copyTermValue(v Value, seen map[int]*Unbound) Value {
	v = vm.deref(v)
	switch t := v.(type) {
	case *Unbound:
		if existing, ok := seen[t.Idx]; ok {
			return existing
		}
		fresh := &Unbound{Name: fmt.Sprintf("_C%d", len(seen)), Idx: vm.allocVarId()}
		seen[t.Idx] = fresh
		return fresh
	case *Compound:
		args := make([]Value, len(t.Args))
		for i, arg := range t.Args {
			args[i] = vm.copyTermValue(arg, seen)
		}
		return &Compound{Functor: t.Functor, Args: args}
	case *Structure:
		args := make([]Value, len(t.Args))
		for i, arg := range t.Args {
			args[i] = vm.copyTermValue(arg, seen)
		}
		return &Structure{Functor: t.Functor, Arity: t.Arity, Args: args}
	case *List:
		items := make([]Value, len(t.Elements))
		for i, item := range t.Elements {
			items[i] = vm.copyTermValue(item, seen)
		}
		return &List{Elements: items}
	default:
		return t
	}
}

func (vm *WamState) groundTerm(v Value, seen map[Value]bool) bool {
	v = vm.deref(v)
	switch t := v.(type) {
	case *Unbound:
		if next, ok := vm.heapConsAfterUnbound(t); ok {
			return vm.groundTerm(next, seen)
		}
		if next, ok := vm.heapListAfterUnbound(t); ok {
			return vm.groundTerm(next, seen)
		}
		return false
	case *Atom, *Integer, *Float:
		return true
	case *StreamHandle:
		return true
	case *Compound:
		if seen[t] {
			return true
		}
		seen[t] = true
		for _, arg := range t.Args {
			if !vm.groundTerm(arg, seen) {
				return false
			}
		}
		return true
	case *Structure:
		if seen[t] {
			return true
		}
		seen[t] = true
		for _, arg := range t.Args {
			if !vm.groundTerm(arg, seen) {
				return false
			}
		}
		return true
	case *List:
		if isEmptyListValue(t) {
			return true
		}
		if seen[t] {
			return true
		}
		seen[t] = true
		items, ok := vm.listToSlice(t)
		if !ok {
			return false
		}
		for _, item := range items {
			if !vm.groundTerm(item, seen) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

const quotedNumericAtomPrefix = "\x01"

func atomNumberText(v Value) (string, bool) {
	switch t := v.(type) {
	case *Atom:
		return strings.TrimPrefix(t.Name, quotedNumericAtomPrefix), true
	case *Integer:
		return strconv.FormatInt(t.Val, 10), true
	case *Float:
		return strconv.FormatFloat(t.Val, 'g', -1, 64), true
	default:
		return "", false
	}
}

func parseAtomNumberText(text string) (Value, bool) {
	if strings.Contains(text, ".") {
		val, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return nil, false
		}
		return &Float{Val: val}, true
	}
	val, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		return nil, false
	}
	return &Integer{Val: val}, true
}

func atomNumberValue(v Value) (Value, bool) {
	switch t := v.(type) {
	case *Integer, *Float:
		return t, true
	case *Atom:
		return parseAtomNumberText(strings.TrimPrefix(t.Name, quotedNumericAtomPrefix))
	default:
		return nil, false
	}
}

func atomCodeList(text string) *List {
	items := make([]Value, 0, len([]rune(text)))
	for _, r := range text {
		items = append(items, &Integer{Val: int64(r)})
	}
	return &List{Elements: items}
}

func atomCharList(text string) *List {
	items := make([]Value, 0, len([]rune(text)))
	for _, r := range text {
		items = append(items, internAtom(string(r)))
	}
	return &List{Elements: items}
}

func codePointValue(v Value) (rune, bool) {
	switch t := v.(type) {
	case *Integer:
		if t.Val < 0 || t.Val > 65535 {
			return 0, false
		}
		return rune(t.Val), true
	case *Atom:
		parsed, ok := parseAtomNumberText(strings.TrimPrefix(t.Name, quotedNumericAtomPrefix))
		if !ok {
			return 0, false
		}
		integer, ok := parsed.(*Integer)
		if !ok || integer.Val < 0 || integer.Val > 65535 {
			return 0, false
		}
		return rune(integer.Val), true
	default:
		return 0, false
	}
}

func (vm *WamState) codeListText(v Value) (string, bool) {
	items, ok := vm.listToSlice(v)
	if !ok {
		return "", false
	}
	runes := make([]rune, 0, len(items))
	for _, item := range items {
		code, ok := codePointValue(vm.deref(item))
		if !ok {
			return "", false
		}
		runes = append(runes, code)
	}
	return string(runes), true
}

func (vm *WamState) charListText(v Value) (string, bool) {
	items, ok := vm.listToSlice(v)
	if !ok {
		return "", false
	}
	runes := make([]rune, 0, len(items))
	for _, item := range items {
		atom, ok := vm.deref(item).(*Atom)
		if !ok {
			return "", false
		}
		chars := []rune(atom.Name)
		if len(chars) != 1 {
			return "", false
		}
		runes = append(runes, chars[0])
	}
	return string(runes), true
}

func (vm *WamState) Deref(v Value) Value {
	return vm.deref(v)
}

func (vm *WamState) deref(v Value) Value {
	for {
		switch t := v.(type) {
		case *Ref:
			next := vm.Heap[t.Addr]
			if next == v { return v }
			v = next
		case *Unbound:
			if bound := vm.getBinding(t.Idx); bound != nil && bound != v {
				v = bound
			} else {
				return v
			}
		default:
			return v
		}
	}
}

func (vm *WamState) evalArithmetic(v Value) (float64, bool) {
	v = vm.deref(v)
	switch t := v.(type) {
	case *Integer:
		return float64(t.Val), true
	case *Float:
		return t.Val, true
	}

	f, args := decompose(v)
	if len(args) == 1 {
		a, okA := vm.evalArithmetic(args[0])
		if !okA { return 0, false }
		switch f {
		case "-":
			return -a, true
		case "abs":
			return math.Abs(a), true
		case "sign":
			if a > 0 { return 1, true }
			if a < 0 { return -1, true }
			return 0, true
		case "float":
			return a, true
		case "truncate", "integer":
			return float64(int64(a)), true
		case "float_integer_part":
			return math.Trunc(a), true
		case "sqrt":
			r := math.Sqrt(a)
			if math.IsNaN(r) || math.IsInf(r, 0) { return 0, false }
			return r, true
		case "sin":
			return math.Sin(a), true
		case "cos":
			return math.Cos(a), true
		case "tan":
			return math.Tan(a), true
		case "asin":
			r := math.Asin(a)
			if math.IsNaN(r) || math.IsInf(r, 0) { return 0, false }
			return r, true
		case "acos":
			r := math.Acos(a)
			if math.IsNaN(r) || math.IsInf(r, 0) { return 0, false }
			return r, true
		case "atan":
			return math.Atan(a), true
		case "floor":
			return math.Floor(a), true
		case "ceiling":
			return math.Ceil(a), true
		case "round":
			return math.Round(a), true
		}
	}
	if len(args) == 2 {
		a, okA := vm.evalArithmetic(args[0])
		b, okB := vm.evalArithmetic(args[1])
		if !okA || !okB {
			return 0, false
		}
		switch f {
		case "+":
			return a + b, true
		case "-":
			return a - b, true
		case "*":
			return a * b, true
		case "/":
			if b == 0 { return 0, false }
			return a / b, true
		case "//":
			if b == 0 { return 0, false }
			return float64(int64(a) / int64(b)), true
		case "mod":
			if b == 0 { return 0, false }
			return float64(int64(a) % int64(b)), true
		case "abs":
			return math.Abs(a), true
		case "max":
			if a > b { return a, true }
			return b, true
		case "min":
			if a < b { return a, true }
			return b, true
		case "**", "^":
			return math.Pow(a, b), true
		case "/\\":
			return float64(int64(a) & int64(b)), true
		case "\\/":
			return float64(int64(a) | int64(b)), true
		case "xor":
			return float64(int64(a) ^ int64(b)), true
		case ">>":
			return float64(int64(a) >> uint(int64(b))), true
		case "<<":
			return float64(int64(a) << uint(int64(b))), true
		}
	}
	return 0, false
}

// ============================================================================
// ISO error handling
// ============================================================================
//
// Three-form dispatch, matching the F#/Haskell/C++/Elixir contract in
// docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md:
//
//   is/2      the *default* key. The Prolog side rewrites it to the ISO
//             or lax key per predicate before codegen, so what reaches
//             the runtime is already resolved.
//   is_iso/2  throws error(instantiation_error, _),
//             error(type_error(evaluable, F/N), _) or
//             error(evaluation_error(zero_divisor), _).
//   is_lax/2  the historical silent-failure behaviour: fails instead of
//             throwing. Delegates to the plain key.
//
// A throw travels as a Go panic carrying prologBall; catch/3 recovers
// it, and Run recovers any ball that escapes so an uncaught error
// surfaces as failure plus vm.UncaughtBall rather than a process crash.

// prologBall carries a thrown Prolog term through panic/recover.
type prologBall struct {
	Ball Value
}

func makeErrorTerm(formal Value, context Value) Value {
	return &Structure{Functor: "error/2", Arity: 2, Args: []Value{formal, context}}
}

func makeInstantiationError(context string) Value {
	return makeErrorTerm(internAtom("instantiation_error"), internAtom(context))
}

func makeTypeError(expected string, culprit Value, context string) Value {
	formal := &Structure{
		Functor: "type_error/2",
		Arity:   2,
		Args:    []Value{internAtom(expected), culprit},
	}
	return makeErrorTerm(formal, internAtom(context))
}

func makeEvaluationError(what string, context string) Value {
	formal := &Structure{
		Functor: "evaluation_error/1",
		Arity:   1,
		Args:    []Value{internAtom(what)},
	}
	return makeErrorTerm(formal, internAtom(context))
}

// throwTerm raises a Prolog term as a ball.
func (vm *WamState) throwTerm(ball Value) {
	panic(prologBall{Ball: vm.resolveTerm(ball)})
}

// resolveTerm walks a term dereferencing every nested slot, so a ball
// keeps its bindings after the goal's trail is unwound.
func (vm *WamState) resolveTerm(value Value) Value {
	if value == nil {
		return nil
	}
	value = vm.deref(value)
	switch t := value.(type) {
	case *Structure:
		args := make([]Value, len(t.Args))
		for i, a := range t.Args {
			args[i] = vm.resolveTerm(a)
		}
		return &Structure{Functor: t.Functor, Arity: t.Arity, Args: args}
	case *Compound:
		args := make([]Value, len(t.Args))
		for i, a := range t.Args {
			args[i] = vm.resolveTerm(a)
		}
		return &Compound{Functor: t.Functor, Args: args}
	case *List:
		items := make([]Value, len(t.Elements))
		for i, e := range t.Elements {
			items[i] = vm.resolveTerm(e)
		}
		return &List{Elements: items}
	default:
		return value
	}
}

// hasUnboundDeep reports whether a term contains any unbound variable,
// which is what distinguishes instantiation_error from type_error.
func (vm *WamState) hasUnboundDeep(value Value) bool {
	if value == nil {
		return true
	}
	value = vm.deref(value)
	if isUnbound(value) {
		return true
	}
	switch t := value.(type) {
	case *Structure:
		for _, a := range t.Args {
			if vm.hasUnboundDeep(a) {
				return true
			}
		}
	case *Compound:
		for _, a := range t.Args {
			if vm.hasUnboundDeep(a) {
				return true
			}
		}
	case *List:
		for _, e := range t.Elements {
			if vm.hasUnboundDeep(e) {
				return true
			}
		}
	}
	return false
}

// arithCulprit names the offending evaluable for type_error(evaluable, F/N).
func (vm *WamState) arithCulprit(expr Value) Value {
	expr = vm.deref(expr)
	name, args := decompose(expr)
	if name == "" {
		return expr
	}
	return &Structure{
		Functor: "//2",
		Arity:   2,
		Args:    []Value{internAtom(parseFunctorName(name)), &Integer{Val: int64(len(args))}},
	}
}

// isoDivisionByZero reports whether expr contains an integer division or
// modulo by zero. Float division by zero is *not* an error: it yields an
// IEEE-754 infinity or NaN, which is what the lax form returns.
func (vm *WamState) isoDivisionByZero(expr Value) bool {
	expr = vm.deref(expr)
	name, args := decompose(expr)
	if name == "" {
		return false
	}
	base := parseFunctorName(name)
	if len(args) == 2 && (base == "//" || base == "mod" || base == "rem" || base == "/") {
		_, leftIsInt := vm.deref(args[0]).(*Integer)
		right := vm.deref(args[1])
		if rInt, ok := right.(*Integer); ok && rInt.Val == 0 {
			if base != "/" || leftIsInt {
				return true
			}
		}
	}
	for _, a := range args {
		if vm.isoDivisionByZero(a) {
			return true
		}
	}
	return false
}

// evalArithIso evaluates an arithmetic expression, throwing the ISO
// error the expression warrants instead of failing silently.
func (vm *WamState) evalArithIso(expr Value, context string) float64 {
	if vm.hasUnboundDeep(expr) {
		vm.throwTerm(makeInstantiationError(context))
	}
	if vm.isoDivisionByZero(expr) {
		vm.throwTerm(makeEvaluationError("zero_divisor", context))
	}
	val, ok := vm.evalArithmetic(expr)
	if !ok {
		vm.throwTerm(makeTypeError("evaluable", vm.arithCulprit(expr), context))
	}
	return val
}

// arithResultValue mirrors is/2's int-vs-float result heuristic.
func arithResultValue(val float64) Value {
	if !math.IsInf(val, 0) && !math.IsNaN(val) && val == math.Trunc(val) {
		return &Integer{Val: int64(val)}
	}
	return &Float{Val: val}
}

// isoCompare applies one of the six arithmetic comparisons.
func isoCompare(op string, v1 float64, v2 float64) bool {
	switch op {
	case "<":
		return v1 < v2
	case ">":
		return v1 > v2
	case "=<":
		return v1 <= v2
	case ">=":
		return v1 >= v2
	case "=:=":
		return v1 == v2
	case "=\\=":
		return v1 != v2
	}
	return false
}

// isoComparisonOp maps an _iso/_lax comparison key back to its operator,
// e.g. "<_iso/2" -> "<". Returns "" when op is not a comparison key.
func isoComparisonOp(op string, suffix string) string {
	if !strings.HasSuffix(op, suffix) {
		return ""
	}
	base := strings.TrimSuffix(op, suffix)
	switch base {
	case "<", ">", "=<", ">=", "=:=", "=\\=":
		return base
	}
	return ""
}

// callGoalTerm meta-calls a goal held as a term. Used by catch/3; the
// same decompose-then-dispatch shape \+/1 already uses.
//
// Bindings made by a builtin goal propagate directly (it runs in this
// VM). A user-predicate goal runs in a sub-VM, and its bindings are
// copied back on success — catch/3 is therefore deterministic: it
// commits to the goal's first solution and does not leave a
// choicepoint for backtracking into it.
func (vm *WamState) callGoalTerm(goal Value) bool {
	goal = vm.deref(goal)
	name, args := decompose(goal)
	if name == "" {
		return false
	}
	opName := fmt.Sprintf("%s/%d", parseFunctorName(name), len(args))

	savedRegs := vm.snapshotAllRegs()
	for idx := range args {
		if idx < len(vm.Regs) {
			vm.Regs[idx] = args[idx]
		}
	}
	if pc, ok := vm.Ctx.Labels[opName]; ok {
		vm.restoreSavedRegs(savedRegs)
		return vm.runGoalPropagating(pc, args)
	}
	res := vm.executeBuiltin(opName, len(args))
	if !res {
		vm.restoreSavedRegs(savedRegs)
	}
	return res
}

// runGoalPropagating runs a user predicate in a sub-VM and copies the
// resulting bindings back on success.
func (vm *WamState) runGoalPropagating(targetPC int, args []Value) bool {
	sub := vm.Clone()
	for idx := range sub.Regs {
		sub.Regs[idx] = nil
	}
	for idx, arg := range args {
		if idx >= len(sub.Regs) {
			break
		}
		sub.Regs[idx] = arg
	}
	sub.PC = targetPC
	sub.CP = 0
	sub.E = -1
	sub.Stack = nil
	sub.Trail = nil
	sub.TrailLen = 0
	sub.ChoicePoints = nil
	sub.Halted = false
	sub.CurrentStruct = nil
	sub.CurrentList = nil
	if !sub.Run() {
		// Run recovers balls at its own top level, so a throw from
		// inside the sub-VM lands in sub.UncaughtBall rather than
		// unwinding into our caller. Re-raise it here so an enclosing
		// catch/3 still sees it.
		if sub.UncaughtBall != nil {
			panic(prologBall{Ball: sub.UncaughtBall})
		}
		return false
	}
	// Adopt the sub-VM's bindings. Var IDs are shared (Clone carries
	// NextVarId forward), so this is what makes the goal's bindings
	// visible to the caller.
	vm.Bindings = sub.Bindings
	vm.NextVarId = sub.NextVarId
	return true
}

// builtinCatch implements catch(Goal, Catcher, Recovery).
//
// A ball thrown while Goal runs is recovered here. If it unifies with
// Catcher, Recovery runs in its place and catch/3 reports Recovery's
// result; otherwise the ball is re-thrown for an outer catch/3 (or for
// Run's top-level recovery). Bindings made before the throw are undone
// before Recovery runs, matching ISO.
func (vm *WamState) builtinCatch(goal Value, catcher Value, recovery Value) (result bool) {
	mark := vm.TrailLen
	cpLen := len(vm.ChoicePoints)
	savedRegs := vm.snapshotAllRegs()

	caught := false
	var ball Value
	func() {
		defer func() {
			if r := recover(); r != nil {
				thrown, ok := r.(prologBall)
				if !ok {
					panic(r)
				}
				caught = true
				ball = thrown.Ball
			}
		}()
		result = vm.callGoalTerm(goal)
	}()

	if !caught {
		return result
	}

	// Undo the failed goal's bindings before trying the catcher.
	vm.unwindTrailTo(mark)
	if cpLen <= len(vm.ChoicePoints) {
		vm.ChoicePoints = vm.ChoicePoints[:cpLen]
	}
	vm.restoreSavedRegs(savedRegs)

	if !vm.Unify(catcher, ball) {
		// Not ours: let an outer catch/3 (or Run) see it.
		vm.unwindTrailTo(mark)
		vm.restoreSavedRegs(savedRegs)
		panic(prologBall{Ball: ball})
	}
	return vm.callGoalTerm(recovery)
}

func (vm *WamState) executeBuiltin(op string, arity int) bool {
	arg1 := vm.getReg(0) // A1
	var arg2, arg3, arg4, arg5 Value
	if arity >= 2 { arg2 = vm.getReg(1) } // A2
	if arity >= 3 { arg3 = vm.getReg(2) } // A3
	if arity >= 4 { arg4 = vm.getReg(3) } // A4
	if arity >= 5 { arg5 = vm.getReg(4) } // A5

	// ISO three-form keys. The _lax forms delegate to the plain key
	// (silent failure); the _iso forms throw. Handled ahead of the main
	// switch so the comparison family can share one branch.
	switch op {
	case "throw/1":
		if vm.hasUnboundDeep(arg1) {
			vm.throwTerm(makeInstantiationError("throw/1"))
		}
		vm.throwTerm(arg1)
	case "catch/3":
		return vm.builtinCatch(arg1, arg2, arg3)
	case "is_lax/2":
		return vm.executeBuiltin("is/2", 2)
	case "succ_lax/2":
		return vm.executeBuiltin("succ/2", 2)
	case "is_iso/2":
		val := vm.evalArithIso(arg2, "is/2")
		return vm.Unify(arg1, arithResultValue(val))
	case "succ_iso/2":
		a := vm.deref(arg1)
		b := vm.deref(arg2)
		if isUnbound(a) && isUnbound(b) {
			vm.throwTerm(makeInstantiationError("succ/2"))
		}
		if !isUnbound(a) {
			if n, ok := a.(*Integer); !ok {
				vm.throwTerm(makeTypeError("integer", a, "succ/2"))
			} else if n.Val < 0 {
				vm.throwTerm(makeTypeError("not_less_than_zero", a, "succ/2"))
			}
		}
		if !isUnbound(b) {
			if n, ok := b.(*Integer); !ok {
				vm.throwTerm(makeTypeError("integer", b, "succ/2"))
			} else if n.Val < 0 {
				vm.throwTerm(makeTypeError("not_less_than_zero", b, "succ/2"))
			}
		}
		return vm.executeBuiltin("succ/2", 2)
	}
	if base := isoComparisonOp(op, "_lax/2"); base != "" {
		return vm.executeBuiltin(base+"/2", 2)
	}
	if base := isoComparisonOp(op, "_iso/2"); base != "" {
		v1 := vm.evalArithIso(arg1, base+"/2")
		v2 := vm.evalArithIso(arg2, base+"/2")
		return isoCompare(base, v1, v2)
	}

	switch op {
	case "display/1":
		fmt.Print(vm.writeTermString(arg1))
		return true
	case "write/1":
		fmt.Print(vm.writeTermString(arg1))
		return true
	case "call/1":
		return vm.executeCall1(vm.deref(arg1))
	case "write_canonical/1":
		fmt.Print(vm.canonicalTermString(arg1))
		return true
	case "put_char/1":
		text, ok := outputAtomCharText(vm.deref(arg1))
		if !ok {
			return false
		}
		fmt.Print(text)
		return true
	case "put_code/1":
		text, ok := outputCodeText(vm.deref(arg1))
		if !ok {
			return false
		}
		fmt.Print(text)
		return true
	case "put_char/2":
		text, ok := outputAtomCharText(vm.deref(arg2))
		if !ok {
			return false
		}
		return writeStreamText(vm.deref(arg1), text)
	case "put_code/2":
		text, ok := outputCodeText(vm.deref(arg2))
		if !ok {
			return false
		}
		return writeStreamText(vm.deref(arg1), text)
	case "get_char/1":
		r, eof, ok := vm.readDefaultInputRune()
		if !ok {
			return false
		}
		return vm.Unify(arg1, inputCharValue(r, eof))
	case "get_char/2":
		r, eof, ok := readStreamRune(vm.deref(arg1))
		if !ok {
			return false
		}
		return vm.Unify(arg2, inputCharValue(r, eof))
	case "peek_char/1":
		r, eof, ok := vm.peekDefaultInputRune()
		if !ok {
			return false
		}
		return vm.Unify(arg1, inputCharValue(r, eof))
	case "peek_char/2":
		r, eof, ok := peekStreamRune(vm.deref(arg1))
		if !ok {
			return false
		}
		return vm.Unify(arg2, inputCharValue(r, eof))
	case "get_code/1":
		r, eof, ok := vm.readDefaultInputRune()
		if !ok {
			return false
		}
		return vm.Unify(arg1, inputCodeValue(r, eof))
	case "get_code/2":
		r, eof, ok := readStreamRune(vm.deref(arg1))
		if !ok {
			return false
		}
		return vm.Unify(arg2, inputCodeValue(r, eof))
	case "open/3":
		path, okPath := streamPathText(vm.deref(arg1))
		mode, okMode := streamOpenMode(vm.deref(arg2))
		if !okPath || !okMode {
			return false
		}
		handle, ok := vm.openStreamHandle(path, mode)
		if !ok {
			return false
		}
		if !vm.Unify(arg3, handle) {
			_ = handle.File.Close()
			handle.Closed = true
			return false
		}
		return true
	case "close/1":
		handle, ok := streamHandle(vm.deref(arg1))
		if !ok {
			return false
		}
		if err := handle.File.Close(); err != nil {
			return false
		}
		handle.Closed = true
		return true
	case "read_line_to_string/2":
		line, ok := readLineStreamValue(vm.deref(arg1))
		if !ok {
			return false
		}
		return vm.Unify(arg2, line)
	case "read_string/5":
		length, ok := vm.deref(arg2).(*Integer)
		if !ok {
			return false
		}
		text, ok := readStringStreamText(vm.deref(arg1), length.Val)
		if !ok {
			return false
		}
		if !vm.Unify(arg3, &Integer{Val: int64(len([]rune(text)))}) {
			return false
		}
		return vm.Unify(arg5, internAtom(text))
	case "at_end_of_stream/1":
		_, eof, ok := peekStreamRune(vm.deref(arg1))
		return ok && eof
	case "write_to_stream/2":
		return writeStreamText(vm.deref(arg1), vm.writeTermString(arg2))
	case "nl_to_stream/1":
		return writeStreamText(vm.deref(arg1), "\n")
	case "writeln/1":
		fmt.Println(vm.writeTermString(arg1))
		return true
	case "print/1":
		fmt.Print(vm.writeTermString(arg1))
		return true
	case "format/1":
		template, ok := formatTemplateText(vm.deref(arg1))
		if !ok {
			return false
		}
		out, ok := vm.formatTemplate(template, nil)
		if !ok {
			return false
		}
		fmt.Print(out)
		return true
	case "format/2":
		template, ok := formatTemplateText(vm.deref(arg1))
		if !ok {
			return false
		}
		args, ok := vm.listToSlice(arg2)
		if !ok {
			return false
		}
		out, ok := vm.formatTemplate(template, args)
		if !ok {
			return false
		}
		fmt.Print(out)
		return true
	case "nl/0":
		fmt.Println()
		return true
	case "tab/1":
		count, ok := vm.deref(arg1).(*Integer)
		if !ok || count.Val < 0 {
			return false
		}
		fmt.Print(strings.Repeat(" ", int(count.Val)))
		return true
	case "getenv/2":
		name, ok := vm.deref(arg1).(*Atom)
		if !ok {
			return false
		}
		value, ok := os.LookupEnv(name.Name)
		if !ok {
			return false
		}
		return vm.Unify(arg2, internAtom(value))
	case "setenv/2":
		name, okName := vm.deref(arg1).(*Atom)
		value, okValue := vm.deref(arg2).(*Atom)
		if !okName || !okValue {
			return false
		}
		return os.Setenv(name.Name, value.Name) == nil
	case "true/0":
		return true
	case "fail/0":
		return false
	case "!/0":
		// Cut is a barrier at PendingB0 (WAM B0), never a stack wipe.
		// PendingB0 is set by Call AND Execute (docs/WAM_BACKEND_CONVENTIONS.md §9).
		// A no-Allocate callee has no EnvFrame; using peekEnvFrame().CutB0
		// would cut to the caller's Allocate-time height and steal its
		// alternatives (cut-semantics p01).
		limit := vm.PendingB0
		if limit < 0 {
			limit = 0
		}
		if limit < len(vm.ChoicePoints) {
			vm.ChoicePoints = vm.ChoicePoints[:limit]
		}
		return true
	case "is/2":
		val, ok := vm.evalArithmetic(arg2)
		if !ok { return false }
		// Produce an Integer for integral results: Unify is type-strict
		// (Integer never unifies with Float), so always wrapping in Float
		// made `R is N + 1` fail whenever R was already bound to a ground
		// Integer (e.g. cack(0,5,6) / cfib(10,55) / cbi_arith). Mirrors the
		// Python target's int-vs-float result heuristic.
		var result Value
		if !math.IsInf(val, 0) && val == math.Trunc(val) {
			result = &Integer{Val: int64(val)}
		} else {
			result = &Float{Val: val}
		}
		return vm.Unify(arg1, result)

	case "=:=/2" :
		v1, ok1 := vm.evalArithmetic(arg1)
		v2, ok2 := vm.evalArithmetic(arg2)
		return ok1 && ok2 && v1 == v2
	case "=\\=/2" :
		v1, ok1 := vm.evalArithmetic(arg1)
		v2, ok2 := vm.evalArithmetic(arg2)
		return ok1 && ok2 && v1 != v2
	case "</2":
		v1, ok1 := vm.evalArithmetic(arg1)
		v2, ok2 := vm.evalArithmetic(arg2)
		return ok1 && ok2 && v1 < v2
	case ">/2":
		v1, ok1 := vm.evalArithmetic(arg1)
		v2, ok2 := vm.evalArithmetic(arg2)
		return ok1 && ok2 && v1 > v2
	case "=</2":
		v1, ok1 := vm.evalArithmetic(arg1)
		v2, ok2 := vm.evalArithmetic(arg2)
		return ok1 && ok2 && v1 <= v2
	case ">=/2":
		v1, ok1 := vm.evalArithmetic(arg1)
		v2, ok2 := vm.evalArithmetic(arg2)
		return ok1 && ok2 && v1 >= v2
	case "succ/2":
		a := vm.deref(arg1)
		b := vm.deref(arg2)
		aInt, aIsInt := a.(*Integer)
		bInt, bIsInt := b.(*Integer)
		aUnbound := isUnbound(a)
		bUnbound := isUnbound(b)
		if aUnbound && bUnbound {
			return false
		}
		if !aUnbound && !aIsInt {
			return false
		}
		if !bUnbound && !bIsInt {
			return false
		}
		if aIsInt {
			if aInt.Val < 0 {
				return false
			}
			return vm.Unify(arg2, &Integer{Val: aInt.Val + 1})
		}
		if bIsInt && bInt.Val > 0 {
			return vm.Unify(arg1, &Integer{Val: bInt.Val - 1})
		}
		return false
	case "between/3":
		lo, ok1 := vm.deref(arg1).(*Integer)
		hi, ok2 := vm.deref(arg2).(*Integer)
		if !ok1 || !ok2 || lo.Val > hi.Val {
			return false
		}
		v := vm.deref(arg3)
		if n, ok := v.(*Integer); ok {
			return n.Val >= lo.Val && n.Val <= hi.Val
		}
		if !isUnbound(v) {
			return false
		}
		if lo.Val < hi.Val {
			cp := ChoicePoint{
				ResumePC:       vm.PC + 1,
				CP:             vm.CP,
				E:              vm.E,
				StackLen:       len(vm.Stack),
				SavedRegs:      vm.snapshotAllRegs(),
				HeapTop:        vm.HeapLen,
				TrailMark:      vm.TrailLen,
				BetweenActive:  true,
				BetweenCurrent: lo.Val + 1,
				BetweenHigh:    hi.Val,
				BetweenReg:     2,
			}
			vm.fillBarrier(&cp)
			vm.ChoicePoints = append(vm.ChoicePoints, cp)
		}
		return vm.Unify(arg3, &Integer{Val: lo.Val})
	case "atom_number/2":
		atomVal := vm.deref(arg1)
		numberVal := vm.deref(arg2)
		if text, ok := atomNumberText(atomVal); ok {
			parsed, ok := parseAtomNumberText(text)
			if !ok {
				return false
			}
			return vm.Unify(arg2, parsed)
		}
		if parsed, ok := atomNumberValue(numberVal); ok {
			text, ok := atomNumberText(parsed)
			if !ok {
				return false
			}
			return vm.Unify(arg1, internAtom(text))
		}
		return false
	case "upcase_atom/2":
		atom, ok := vm.deref(arg1).(*Atom)
		if !ok {
			return false
		}
		return vm.Unify(arg2, internAtom(strings.ToUpper(atom.Name)))
	case "downcase_atom/2":
		atom, ok := vm.deref(arg1).(*Atom)
		if !ok {
			return false
		}
		return vm.Unify(arg2, internAtom(strings.ToLower(atom.Name)))
	case "atom_concat/3":
		left, okLeft := vm.deref(arg1).(*Atom)
		right, okRight := vm.deref(arg2).(*Atom)
		if !okLeft || !okRight {
			return false
		}
		return vm.Unify(arg3, internAtom(left.Name+right.Name))
	case "atom_length/2", "string_length/2":
		atom, ok := vm.deref(arg1).(*Atom)
		if !ok {
			return false
		}
		return vm.Unify(arg2, &Integer{Val: int64(len([]rune(atom.Name)))})
	case "char_code/2":
		charVal := vm.deref(arg1)
		codeVal := vm.deref(arg2)
		if atom, ok := charVal.(*Atom); ok {
			runes := []rune(atom.Name)
			if len(runes) != 1 {
				return false
			}
			return vm.Unify(arg2, &Integer{Val: int64(runes[0])})
		}
		if code, ok := codeVal.(*Integer); ok {
			if code.Val < 0 || code.Val > 65535 {
				return false
			}
			return vm.Unify(arg1, internAtom(string(rune(code.Val))))
		}
		return false
	case "atom_codes/2", "atom_chars/2", "string_codes/2", "string_chars/2":
		atomVal := vm.deref(arg1)
		listVal := vm.deref(arg2)
		if u, ok := listVal.(*Unbound); ok {
			if next, ok := vm.heapConsAfterUnbound(u); ok {
				listVal = vm.deref(next)
			}
		}
		textFromList := vm.codeListText
		listFromText := func(text string) *List { return atomCodeList(text) }
		if op == "atom_chars/2" || op == "string_chars/2" {
			textFromList = vm.charListText
			listFromText = func(text string) *List { return atomCharList(text) }
		}
		if atom, ok := atomVal.(*Atom); ok {
			if _, unbound := listVal.(*Unbound); unbound {
				lst := listFromText(atom.Name)
				return vm.Unify(arg2, listFromItems(lst.Elements))
			}
			text, ok := textFromList(listVal)
			return ok && atom.Name == text
		}
		if text, ok := textFromList(listVal); ok {
			return vm.Unify(arg1, internAtom(text))
		}
		return false
	case "number_codes/2", "number_chars/2":
		numberVal := vm.deref(arg1)
		listVal := vm.deref(arg2)
		if u, ok := listVal.(*Unbound); ok {
			if next, ok := vm.heapConsAfterUnbound(u); ok {
				listVal = vm.deref(next)
			}
		}
		textFromList := vm.codeListText
		listFromText := func(text string) *List { return atomCodeList(text) }
		if op == "number_chars/2" {
			textFromList = vm.charListText
			listFromText = func(text string) *List { return atomCharList(text) }
		}
		if number, ok := atomNumberValue(numberVal); ok {
			text, ok := atomNumberText(number)
			if !ok {
				return false
			}
			if _, unbound := listVal.(*Unbound); unbound {
				lst := listFromText(text)
				return vm.Unify(arg2, listFromItems(lst.Elements))
			}
			listText, ok := textFromList(listVal)
			return ok && text == listText
		}
		if text, ok := textFromList(listVal); ok {
			parsed, ok := parseAtomNumberText(text)
			return ok && vm.Unify(arg1, parsed)
		}
		return false
	case "atom_string/2", "string_to_atom/2":
		atomArg := arg1
		stringArg := arg2
		if op == "string_to_atom/2" {
			atomArg = arg2
			stringArg = arg1
		}
		atomVal := vm.deref(atomArg)
		stringVal := vm.deref(stringArg)
		if atom, ok := atomVal.(*Atom); ok {
			return vm.Unify(stringArg, internAtom(atom.Name))
		}
		if text, ok := stringVal.(*Atom); ok {
			return vm.Unify(atomArg, internAtom(text.Name))
		}
		return false
	case "sub_atom/5":
		source, ok := atomNumberText(vm.deref(arg1))
		if !ok {
			return false
		}
		before, okBefore := vm.deref(arg2).(*Integer)
		length, okLength := vm.deref(arg3).(*Integer)
		if !okBefore || !okLength || before.Val < 0 || length.Val < 0 {
			return false
		}
		runes := []rune(source)
		start := before.Val
		end := start + length.Val
		if end > int64(len(runes)) {
			return false
		}
		after := int64(len(runes)) - end
		sub := string(runes[start:end])
		mark := vm.TrailLen
		if !vm.Unify(arg4, &Integer{Val: after}) {
			return false
		}
		if !vm.Unify(arg5, internAtom(sub)) {
			vm.unwindTrailTo(mark)
			return false
		}
		return true
	case "char_type/2":
		charAtom, okChar := vm.deref(arg1).(*Atom)
		typeAtom, okType := vm.deref(arg2).(*Atom)
		if !okChar || !okType {
			return false
		}
		runes := []rune(charAtom.Name)
		if len(runes) != 1 {
			return false
		}
		r := runes[0]
		switch typeAtom.Name {
		case "alpha":
			return unicode.IsLetter(r)
		case "alnum":
			return unicode.IsLetter(r) || unicode.IsDigit(r)
		case "digit":
			return unicode.IsDigit(r)
		case "space":
			return unicode.IsSpace(r)
		case "white":
			return r == ' ' || r == '\t'
		case "upper":
			return unicode.IsUpper(r)
		case "lower":
			return unicode.IsLower(r)
		case "punct":
			return unicode.IsPunct(r)
		case "ascii":
			return r >= 1 && r <= 127
		case "csym":
			return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'
		case "csymf":
			return unicode.IsLetter(r) || r == '_'
		case "newline":
			return r == '\n'
		default:
			return false
		}
	case "string_code/3":
		index, okIndex := vm.deref(arg1).(*Integer)
		text, okText := vm.deref(arg2).(*Atom)
		if !okIndex || !okText || index.Val < 1 {
			return false
		}
		runes := []rune(text.Name)
		if index.Val > int64(len(runes)) {
			return false
		}
		return vm.Unify(arg3, &Integer{Val: int64(runes[index.Val-1])})
	case "split_string/4":
		source, okSource := atomNumberText(vm.deref(arg1))
		seps, okSeps := atomNumberText(vm.deref(arg2))
		pads, okPads := atomNumberText(vm.deref(arg3))
		if !okSource || !okSeps || !okPads {
			return false
		}
		sepSet := make(map[rune]bool)
		for _, r := range seps {
			sepSet[r] = true
		}
		padSet := make(map[rune]bool)
		for _, r := range pads {
			padSet[r] = true
		}
		parts := []string{}
		current := []rune{}
		for _, r := range source {
			if sepSet[r] {
				parts = append(parts, string(current))
				current = current[:0]
				continue
			}
			current = append(current, r)
		}
		parts = append(parts, string(current))
		items := make([]Value, 0, len(parts))
		for _, part := range parts {
			runes := []rune(part)
			start := 0
			for start < len(runes) && padSet[runes[start]] {
				start++
			}
			end := len(runes)
			for end > start && padSet[runes[end-1]] {
				end--
			}
			items = append(items, internAtom(string(runes[start:end])))
		}
		return vm.Unify(arg4, &List{Elements: items})

	case "var/1": return isUnbound(vm.deref(arg1))
	case "nonvar/1": return !isUnbound(vm.deref(arg1))
	case "atom/1": return isAtom(vm.deref(arg1))
	case "integer/1": return isInteger(vm.deref(arg1))
	case "float/1": return isFloat(vm.deref(arg1))
	case "number/1": return isNumber(vm.deref(arg1))
	case "compound/1": return isCompound(vm.deref(arg1))
	case "atomic/1": return isAtomic(vm.deref(arg1))
	case "ground/1": return vm.groundTerm(arg1, make(map[Value]bool))
	case "is_list/1": return isList(vm.deref(arg1))

	case "==/2": return vm.termIdentical(arg1, arg2)
	case "\\==/2": return !vm.termIdentical(arg1, arg2)
	case "@</2":
		return vm.compareTerms(arg1, arg2) < 0
	case "@=</2":
		return vm.compareTerms(arg1, arg2) <= 0
	case "@>/2":
		return vm.compareTerms(arg1, arg2) > 0
	case "@>=/2":
		return vm.compareTerms(arg1, arg2) >= 0
	case "compare/3":
		cmp := vm.compareTerms(arg2, arg3)
		switch {
		case cmp < 0:
			return vm.Unify(arg1, internAtom("<"))
		case cmp > 0:
			return vm.Unify(arg1, internAtom(">"))
		default:
			return vm.Unify(arg1, internAtom("="))
		}
	case "=/2": return vm.Unify(arg1, arg2)
	case "\\=/2":
		clone := vm.Clone()
		return !clone.Unify(arg1, arg2)
	case "\\+/1":
		goal := vm.deref(arg1)
		f, args := decompose(goal)
		if f != "" {
			// Save A-regs temporarily
			var oldRegs [3]Value
			for idx := 0; idx < 3; idx++ {
				oldRegs[idx] = vm.Regs[idx]
			}
			if len(args) > 0 { vm.Regs[0] = args[0] }
			if len(args) > 1 { vm.Regs[1] = args[1] }
			if len(args) > 2 { vm.Regs[2] = args[2] }
			opName := fmt.Sprintf("%s/%d", f, len(args))
			trailMark := vm.TrailLen
			cpLen := len(vm.ChoicePoints)
			res := vm.executeBuiltin(opName, len(args))
			vm.unwindTrailTo(trailMark)
			if cpLen <= len(vm.ChoicePoints) {
				vm.ChoicePoints = vm.ChoicePoints[:cpLen]
			}
			for idx := 0; idx < 3; idx++ {
				vm.Regs[idx] = oldRegs[idx]
			}
			if res {
				return false
			}
			if pc, ok := vm.Ctx.Labels[opName]; ok {
				if vm.hasParallelNegationChoices(pc) {
					return !vm.runNegationParallel(pc, args)
				}
				return !vm.runIsolatedGoal(pc, args)
			}
			return true
		}
		return false

	case "length/2":
		// Walk the list counting cells without materialising a
		// Go slice. listToSlice is recursive-ish + allocates;
		// length/2 just needs the count, so the bench's
		// `length(Visited, Depth)` (called per category_ancestor
		// recursion step) doesn't pay the slice alloc.
		v := vm.deref(arg1)
		count := int64(0)
		for {
			if isEmptyListValue(v) {
				break
			}
			head, tail, ok := vm.valueListHeadTail(v)
			if !ok {
				return false
			}
			_ = head
			count++
			v = vm.deref(tail)
		}
		return vm.Unify(arg2, &Integer{Val: count})
	case "member/2":
		v := vm.deref(arg2)
		for {
			if isEmptyListValue(v) {
				return false
			}
			head, tail, ok := vm.valueListHeadTail(v)
			if !ok {
				return false
			}
			mark := vm.TrailLen
			savedRegs := vm.snapshotAllRegs()
			if vm.Unify(arg1, head) {
				tail = vm.deref(tail)
				if !isEmptyListValue(tail) {
					cp := ChoicePoint{
						ResumePC: vm.PC + 1,
						CP:       vm.CP,
						E:        vm.E,
						StackLen: len(vm.Stack),
						SavedRegs: savedRegs,
						HeapTop:   vm.HeapLen,
						TrailMark: mark,
						MemberTail: tail,
					}
					vm.fillBarrier(&cp)
					vm.ChoicePoints = append(vm.ChoicePoints, cp)
				}
				return true
			}
			vm.unwindTrailTo(mark)
			v = vm.deref(tail)
		}
	case "memberchk/2":
		v := vm.deref(arg2)
		for {
			if isEmptyListValue(v) {
				return false
			}
			head, tail, ok := vm.valueListHeadTail(v)
			if !ok {
				return false
			}
			mark := vm.TrailLen
			if vm.Unify(arg1, head) {
				return true
			}
			vm.unwindTrailTo(mark)
			v = vm.deref(tail)
		}
	case "select/3":
		items, ok := vm.listToSlice(arg2)
		if !ok {
			return false
		}
		solutions := vm.selectSolutions(items)
		if len(solutions) == 0 {
			return false
		}
		resumePC := vm.PC + 1
		trailMark := vm.TrailLen
		savedRegs := vm.snapshotAllRegs()
		stackLen := len(vm.Stack)
		heapTop := vm.HeapLen
		for idx, solution := range solutions {
			vm.unwindTrailTo(trailMark)
			vm.restoreSavedRegs(savedRegs)
			if stackLen <= len(vm.Stack) {
				vm.Stack = vm.Stack[:stackLen]
			}
			if heapTop <= vm.HeapLen {
				vm.heapTrimTo(heapTop)
			}
			if !vm.applySelectSolution(solution) {
				continue
			}
			if idx+1 < len(solutions) {
				cp := ChoicePoint{
					ResumePC:      resumePC,
					CP:            vm.CP,
					E:             vm.E,
					StackLen:      stackLen,
					SavedRegs:     savedRegs,
					HeapTop:       heapTop,
					TrailMark:     trailMark,
					SelectResults: append([]SelectSolution(nil), solutions[idx+1:]...),
				}
				vm.fillBarrier(&cp)
				vm.ChoicePoints = append(vm.ChoicePoints, cp)
			}
			return true
		}
		return false
	case "delete/3":
		items, ok := vm.listToSlice(arg1)
		if !ok {
			return false
		}
		target := vm.deref(arg2)
		kept := make([]Value, 0, len(items))
		for _, item := range items {
			if !valueEquals(vm.deref(item), target) {
				kept = append(kept, item)
			}
		}
		return vm.Unify(arg3, &List{Elements: kept})
	case "subtract/3":
		left, okLeft := vm.listToSlice(arg1)
		right, okRight := vm.listToSlice(arg2)
		if !okLeft || !okRight {
			return false
		}
		result := make([]Value, 0, len(left))
		for _, item := range left {
			candidate := vm.deref(item)
			remove := false
			for _, other := range right {
				if valueEquals(candidate, vm.deref(other)) {
					remove = true
					break
				}
			}
			if !remove {
				result = append(result, item)
			}
		}
		return vm.Unify(arg3, &List{Elements: result})
	case "intersection/3":
		left, okLeft := vm.listToSlice(arg1)
		right, okRight := vm.listToSlice(arg2)
		if !okLeft || !okRight {
			return false
		}
		result := make([]Value, 0, len(left))
		for _, item := range left {
			candidate := vm.deref(item)
			for _, other := range right {
				if valueEquals(candidate, vm.deref(other)) {
					result = append(result, item)
					break
				}
			}
		}
		return vm.Unify(arg3, &List{Elements: result})
	case "union/3":
		left, okLeft := vm.listToSlice(arg1)
		right, okRight := vm.listToSlice(arg2)
		if !okLeft || !okRight {
			return false
		}
		result := append([]Value(nil), left...)
		for _, item := range right {
			candidate := vm.deref(item)
			found := false
			for _, existing := range left {
				if valueEquals(candidate, vm.deref(existing)) {
					found = true
					break
				}
			}
			if !found {
				result = append(result, item)
			}
		}
		return vm.Unify(arg3, &List{Elements: result})
	case "list_to_set/2":
		items, ok := vm.listToSlice(arg1)
		if !ok {
			return false
		}
		result := make([]Value, 0, len(items))
		for _, item := range items {
			candidate := vm.deref(item)
			found := false
			for _, existing := range result {
				if valueEquals(vm.deref(existing), candidate) {
					found = true
					break
				}
			}
			if !found {
				result = append(result, candidate)
			}
		}
		return vm.Unify(arg2, &List{Elements: result})
	case "permutation/2":
		items, ok := vm.listToSlice(arg1)
		if !ok {
			return false
		}
		d2 := vm.deref(arg2)
		if isUnbound(d2) {
			return vm.Unify(arg2, &List{Elements: append([]Value(nil), items...)})
		}
		other, ok := vm.listToSlice(arg2)
		if !ok || len(items) != len(other) {
			return false
		}
		left := append([]Value(nil), items...)
		right := append([]Value(nil), other...)
		sort.SliceStable(left, func(i, j int) bool {
			return compareValues(vm.deref(left[i]), vm.deref(left[j])) < 0
		})
		sort.SliceStable(right, func(i, j int) bool {
			return compareValues(vm.deref(right[i]), vm.deref(right[j])) < 0
		})
		for i := range left {
			if compareValues(vm.deref(left[i]), vm.deref(right[i])) != 0 {
				return false
			}
		}
		return true
	case "append/3":
		l1, ok1 := vm.listToSlice(arg1)
		l2, ok2 := vm.listToSlice(arg2)
		if !ok1 || !ok2 { return false }
		return vm.Unify(arg3, listFromItems(append(append([]Value{}, l1...), l2...)))
	case "reverse/2":
		d1 := vm.deref(arg1)
		d2 := vm.deref(arg2)
		if !isUnbound(d1) {
			items, ok := vm.listToSlice(d1)
			if !ok {
				return false
			}
			reversed := append([]Value{}, items...)
			for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
				reversed[i], reversed[j] = reversed[j], reversed[i]
			}
			return vm.Unify(arg2, listFromItems(reversed))
		}
		if !isUnbound(d2) {
			items, ok := vm.listToSlice(d2)
			if !ok {
				return false
			}
			reversed := append([]Value{}, items...)
			for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
				reversed[i], reversed[j] = reversed[j], reversed[i]
			}
			return vm.Unify(arg1, listFromItems(reversed))
		}
		return false
	case "last/2":
		items, ok := vm.listToSlice(arg1)
		if !ok || len(items) == 0 {
			return false
		}
		return vm.Unify(arg2, items[len(items)-1])
	case "nth0/3", "nth1/3":
		idxVal := vm.deref(arg1)
		idxInt, ok := idxVal.(*Integer)
		if !ok {
			return false
		}
		items, ok := vm.listToSlice(arg2)
		if !ok {
			return false
		}
		base := int64(0)
		if op == "nth1/3" {
			base = 1
		}
		idx := idxInt.Val - base
		if idx < 0 || idx >= int64(len(items)) {
			return false
		}
		return vm.Unify(arg3, items[int(idx)])
	case "numlist/3":
		lo, ok1 := vm.deref(arg1).(*Integer)
		hi, ok2 := vm.deref(arg2).(*Integer)
		if !ok1 || !ok2 || lo.Val > hi.Val {
			return false
		}
		items := make([]Value, 0, hi.Val-lo.Val+1)
		for n := lo.Val; n <= hi.Val; n++ {
			items = append(items, &Integer{Val: n})
		}
		return vm.Unify(arg3, &List{Elements: items})
	case "sum_list/2", "min_list/2", "max_list/2":
		items, ok := vm.listToSlice(arg1)
		if !ok {
			return false
		}
		if op != "sum_list/2" && len(items) == 0 {
			return false
		}
		allInts := true
		var intAcc int64
		var floatAcc float64
		if op == "sum_list/2" {
			for _, item := range items {
				switch n := vm.deref(item).(type) {
				case *Integer:
					intAcc += n.Val
					floatAcc += float64(n.Val)
				case *Float:
					allInts = false
					floatAcc += n.Val
				default:
					return false
				}
			}
		} else {
			for idx, item := range items {
				switch n := vm.deref(item).(type) {
				case *Integer:
					if idx == 0 || (op == "min_list/2" && n.Val < intAcc) || (op == "max_list/2" && n.Val > intAcc) {
						intAcc = n.Val
					}
					if idx == 0 || (op == "min_list/2" && float64(n.Val) < floatAcc) || (op == "max_list/2" && float64(n.Val) > floatAcc) {
						floatAcc = float64(n.Val)
					}
				case *Float:
					allInts = false
					if idx == 0 || (op == "min_list/2" && n.Val < floatAcc) || (op == "max_list/2" && n.Val > floatAcc) {
						floatAcc = n.Val
					}
				default:
					return false
				}
			}
		}
		if allInts {
			return vm.Unify(arg2, &Integer{Val: intAcc})
		}
		return vm.Unify(arg2, &Float{Val: floatAcc})
	case "sort/2", "msort/2":
		items, ok := vm.listToSlice(arg1)
		if !ok {
			return false
		}
		sorted := append([]Value(nil), items...)
		sort.SliceStable(sorted, func(i, j int) bool {
			return vm.compareTerms(sorted[i], sorted[j]) < 0
		})
		if op == "sort/2" {
			deduped := make([]Value, 0, len(sorted))
			for _, item := range sorted {
				if len(deduped) == 0 || vm.compareTerms(deduped[len(deduped)-1], item) != 0 {
					deduped = append(deduped, item)
				}
			}
			sorted = deduped
		}
		return vm.Unify(arg2, listFromItems(sorted))
	case "keysort/2":
		items, ok := vm.listToSlice(arg1)
		if !ok {
			return false
		}
		type keyPair struct {
			key  Value
			pair Value
		}
		pairAfterUnbound := func(u *Unbound) (Value, bool) {
			found := false
			for _, cell := range vm.Heap[:vm.HeapLen] {
				if cell == u {
					found = true
					continue
				}
				if !found {
					continue
				}
				pair := vm.deref(cell)
				switch p := pair.(type) {
				case *Compound:
					if parseFunctorName(p.Functor) == "-" && len(p.Args) == 2 {
						return pair, true
					}
				case *Structure:
					if parseFunctorName(p.Functor) == "-" && len(p.Args) == 2 {
						return pair, true
					}
				}
			}
			return nil, false
		}
		pairs := make([]keyPair, 0, len(items))
		for _, item := range items {
			pair := vm.deref(item)
			if u, ok := pair.(*Unbound); ok {
				if resolved, ok := pairAfterUnbound(u); ok {
					pair = resolved
				}
			}
			var key Value
			switch p := pair.(type) {
			case *Compound:
				if parseFunctorName(p.Functor) != "-" || len(p.Args) != 2 {
					return false
				}
				key = vm.deref(p.Args[0])
			case *Structure:
				if parseFunctorName(p.Functor) != "-" || len(p.Args) != 2 {
					return false
				}
				key = vm.deref(p.Args[0])
			default:
				return false
			}
			pairs = append(pairs, keyPair{key: key, pair: pair})
		}
		compareKeys := func(a, b Value) int {
			switch ka := a.(type) {
			case *Integer:
				switch kb := b.(type) {
				case *Integer:
					if ka.Val < kb.Val { return -1 }
					if ka.Val > kb.Val { return 1 }
					return 0
				case *Float:
					av := float64(ka.Val)
					if av < kb.Val { return -1 }
					if av > kb.Val { return 1 }
					return 0
				}
			case *Float:
				switch kb := b.(type) {
				case *Integer:
					bv := float64(kb.Val)
					if ka.Val < bv { return -1 }
					if ka.Val > bv { return 1 }
					return 0
				case *Float:
					if ka.Val < kb.Val { return -1 }
					if ka.Val > kb.Val { return 1 }
					return 0
				}
			}
			return compareValues(a, b)
		}
		sort.SliceStable(pairs, func(i, j int) bool {
			return compareKeys(pairs[i].key, pairs[j].key) < 0
		})
		sorted := make([]Value, len(pairs))
		for i, pair := range pairs {
			sorted[i] = pair.pair
		}
		return vm.Unify(arg2, listFromItems(sorted))
	case "functor/3":
		t := vm.deref(arg1)
		if isUnbound(t) {
			name := vm.deref(arg2)
			arityVal := vm.deref(arg3)
			arityInt, ok := arityVal.(*Integer)
			if !ok || arityInt.Val < 0 {
				return false
			}
			if arityInt.Val == 0 {
				return vm.Unify(arg1, name)
			}
			nameAtom, ok := name.(*Atom)
			if !ok {
				return false
			}
			args := make([]Value, int(arityInt.Val))
			for i := range args {
				args[i] = &Unbound{Name: fmt.Sprintf("_F%d", i), Idx: vm.allocVarId()}
			}
			return vm.Unify(arg1, makeStructureValue(nameAtom.Name, args))
		}
		name, arityOut, ok := vm.termFunctorArity(t)
		if !ok {
			return false
		}
		return vm.Unify(arg2, name) && vm.Unify(arg3, &Integer{Val: arityOut})
	case "arg/3":
		idxVal := vm.deref(arg1)
		idx, ok := idxVal.(*Integer)
		if !ok || idx.Val < 1 {
			return false
		}
		term := vm.deref(arg2)
		var selected Value
		switch t := term.(type) {
		case *Compound:
			if int(idx.Val) > len(t.Args) {
				return false
			}
			selected = t.Args[int(idx.Val)-1]
		case *Structure:
			if int(idx.Val) > len(t.Args) {
				return false
			}
			selected = t.Args[int(idx.Val)-1]
		case *List:
			head, tail, ok := vm.listHeadTail(t)
			if !ok {
				return false
			}
			if idx.Val == 1 {
				selected = head
			} else if idx.Val == 2 {
				selected = tail
			} else {
				return false
			}
		default:
			return false
		}
		return vm.Unify(arg3, selected)
	case "=../2":
		t := vm.deref(arg1)
		if isUnbound(t) {
			items, ok := vm.listToSlice(arg2)
			if !ok || len(items) == 0 {
				return false
			}
			if len(items) == 1 {
				return vm.Unify(arg1, items[0])
			}
			nameAtom, ok := vm.deref(items[0]).(*Atom)
			if !ok {
				return false
			}
			return vm.Unify(arg1, makeStructureValue(nameAtom.Name, items[1:]))
		}
		list, ok := vm.termToUnivList(t)
		if !ok {
			return false
		}
		return vm.Unify(arg2, list)
	case "copy_term/2":
		copy := vm.copyTermValue(arg1, make(map[int]*Unbound))
		return vm.Unify(arg2, copy)
	case "maplist/2":
		return vm.builtinMaplist(1)
	case "maplist/3":
		return vm.builtinMaplist(2)
	case "maplist/4":
		return vm.builtinMaplist(3)
	case "predsort/3":
		return vm.builtinPredsort()
	}
	return false
}

// predArityFromName reads the trailing /N of a "name/N" key. Used by
// Execute (the instruction has no Arity field) when falling back to
// executeBuiltin.
func predArityFromName(name string) int {
	if slash := strings.LastIndex(name, "/"); slash >= 0 {
		n, err := strconv.Atoi(name[slash+1:])
		if err == nil {
			return n
		}
	}
	return 0
}

// warnUnresolved is the A3 loud-unknown-builtin knob. Off by default;
// set UW_WAM_WARN_UNKNOWN=1 to print on Call/Execute of a goal that
// missed Labels, Foreign, indexed facts, and the builtin table.
func (vm *WamState) warnUnresolved(form, pred string) {
	if os.Getenv("UW_WAM_WARN_UNKNOWN") != "" {
		fmt.Fprintf(os.Stderr, "[wam_go] %s of unresolved goal %s failed\n", form, pred)
	}
}

// predExtraGoal builds Goal+extra-args for maplist/predsort, matching
// the JS runtime: an atom becomes name(Extra...), a structure appends.
func (vm *WamState) predExtraGoal(pred Value, extra []Value) Value {
	pred = vm.deref(pred)
	switch t := pred.(type) {
	case *Atom:
		if len(extra) == 0 {
			return t
		}
		return makeStructureValue(parseFunctorName(t.Name), extra)
	case *Structure:
		args := make([]Value, 0, len(t.Args)+len(extra))
		args = append(args, t.Args...)
		args = append(args, extra...)
		return makeStructureValue(parseFunctorName(t.Functor), args)
	case *Compound:
		args := make([]Value, 0, len(t.Args)+len(extra))
		args = append(args, t.Args...)
		args = append(args, extra...)
		return makeStructureValue(parseFunctorName(t.Functor), args)
	default:
		return nil
	}
}

// builtinMaplist implements maplist/2,3,4. Meta-calls USER predicates
// (resolver is_v3/1, store unpack helpers) via invokeGoal. nLists is
// the number of list arguments (maplist/2 → 1).
func (vm *WamState) builtinMaplist(nLists int) bool {
	pred := vm.deref(vm.getReg(0))
	lists := make([][]Value, nLists)
	unboundAt := -1
	for i := 0; i < nLists; i++ {
		items, ok := vm.listToSlice(vm.getReg(1 + i))
		if !ok {
			if unboundAt >= 0 {
				return false
			}
			unboundAt = i
			continue
		}
		lists[i] = items
	}
	if unboundAt < 0 {
		len0 := len(lists[0])
		for i := 1; i < nLists; i++ {
			if len(lists[i]) != len0 {
				return false
			}
		}
		for i := 0; i < len0; i++ {
			extra := make([]Value, nLists)
			for j := 0; j < nLists; j++ {
				extra[j] = lists[j][i]
			}
			goal := vm.predExtraGoal(pred, extra)
			if goal == nil {
				return false
			}
			if !vm.invokeGoal(goal, len(vm.ChoicePoints)) {
				return false
			}
		}
		return true
	}
	boundIdx := 0
	if unboundAt == 0 {
		boundIdx = 1
	}
	bound := lists[boundIdx]
	if bound == nil || nLists != 2 {
		return false
	}
	out := make([]Value, 0, len(bound))
	for i := 0; i < len(bound); i++ {
		y := &Unbound{Name: fmt.Sprintf("_ML%d", i), Idx: vm.allocVarId()}
		var extra []Value
		if unboundAt == 0 {
			extra = []Value{y, bound[i]}
		} else {
			extra = []Value{bound[i], y}
		}
		goal := vm.predExtraGoal(pred, extra)
		if goal == nil {
			return false
		}
		if !vm.invokeGoal(goal, len(vm.ChoicePoints)) {
			return false
		}
		out = append(out, y)
	}
	return vm.Unify(vm.getReg(1+unboundAt), listFromItems(out))
}

// builtinPredsort implements predsort/3. Comparator is Pred(Order,X,Y)
// binding Order to < / > / =; compare/3 short-circuits to term order
// (resolver uses cmp_ver/3 for mixed v/3 + deb/3).
func (vm *WamState) builtinPredsort() bool {
	pred := vm.deref(vm.getReg(0))
	items, ok := vm.listToSlice(vm.getReg(1))
	if !ok {
		return false
	}
	predName := ""
	if a, isAtom := pred.(*Atom); isAtom {
		predName = parseFunctorName(a.Name)
	}
	sorted := append([]Value(nil), items...)
	sort.SliceStable(sorted, func(i, j int) bool {
		x, y := sorted[i], sorted[j]
		if predName == "compare" {
			return vm.compareTerms(x, y) < 0
		}
		order := &Unbound{Name: "_Ord", Idx: vm.allocVarId()}
		goal := vm.predExtraGoal(pred, []Value{order, x, y})
		if goal == nil {
			return false
		}
		if !vm.invokeGoal(goal, len(vm.ChoicePoints)) {
			return false
		}
		if a, ok := vm.deref(order).(*Atom); ok {
			return a.Name == "<"
		}
		return false
	})
	return vm.Unify(vm.getReg(2), listFromItems(sorted))
}

// executeCall1 implements ISO call/1 as an opaque cut scope: a `!`
// inside the metacall prunes only choice points created during the
// call. Nested user goals run on this VM (bindings are shared) with
// Proceed treated as a local halt so the outer BuiltinCall can
// continue. Leftover CPs from a user goal inside call/1 are not
// resumed as extra solutions of the metacall (first-solution nested
// run); the cut-semantics probes that use call/1 (p09, p10) do not
// require that extra-solution path.
func (vm *WamState) executeCall1(goal Value) bool {
	return vm.invokeGoal(goal, len(vm.ChoicePoints))
}

func goalNameArgs(v Value) (name string, args []Value, ok bool) {
	if v == nil {
		return "", nil, false
	}
	switch t := v.(type) {
	case *Atom:
		return t.Name, nil, true
	case *Structure:
		return parseFunctorName(t.Functor), t.Args, true
	case *Compound:
		return parseFunctorName(t.Functor), t.Args, true
	default:
		return "", nil, false
	}
}

func (vm *WamState) invokeGoal(goal Value, cutFloor int) bool {
	goal = vm.deref(goal)
	name, args, ok := goalNameArgs(goal)
	if !ok {
		return false
	}
	if name == ":" && len(args) == 2 {
		return vm.invokeGoal(args[1], cutFloor)
	}
	switch name {
	case "true":
		return true
	case "fail", "false":
		return false
	case "!":
		if cutFloor < 0 {
			cutFloor = 0
		}
		if cutFloor < len(vm.ChoicePoints) {
			vm.ChoicePoints = vm.ChoicePoints[:cutFloor]
		}
		return true
	case ",":
		if len(args) != 2 {
			return false
		}
		if !vm.invokeGoal(args[0], cutFloor) {
			return false
		}
		return vm.invokeGoal(args[1], cutFloor)
	case ";":
		if len(args) != 2 {
			return false
		}
		left := vm.deref(args[0])
		lname, largs, lok := goalNameArgs(left)
		if lok && (lname == "->" || lname == "*->") && len(largs) == 2 {
			if vm.invokeGoal(largs[0], cutFloor) {
				return vm.invokeGoal(largs[1], cutFloor)
			}
			return vm.invokeGoal(args[1], cutFloor)
		}
		if vm.invokeGoal(args[0], cutFloor) {
			return true
		}
		return vm.invokeGoal(args[1], cutFloor)
	case "->", "*->":
		if len(args) != 2 {
			return false
		}
		if !vm.invokeGoal(args[0], cutFloor) {
			return false
		}
		return vm.invokeGoal(args[1], cutFloor)
	case "\\+", "not":
		if len(args) != 1 {
			return false
		}
		mark := len(vm.ChoicePoints)
		ok := vm.invokeGoal(args[0], mark)
		if mark < len(vm.ChoicePoints) {
			vm.ChoicePoints = vm.ChoicePoints[:mark]
		}
		return !ok
	case "once":
		if len(args) != 1 {
			return false
		}
		mark := len(vm.ChoicePoints)
		ok := vm.invokeGoal(args[0], mark)
		if mark < len(vm.ChoicePoints) {
			vm.ChoicePoints = vm.ChoicePoints[:mark]
		}
		return ok
	case "call":
		if len(args) == 0 {
			return false
		}
		return vm.invokeGoal(args[0], cutFloor)
	default:
		return vm.invokeCallable(name, args, cutFloor)
	}
}

func (vm *WamState) invokeCallable(name string, args []Value, cutFloor int) bool {
	key := fmt.Sprintf("%s/%d", name, len(args))
	if pc, found := vm.Ctx.Labels[key]; found {
		return vm.invokeAtPC(pc, args, cutFloor)
	}
	saved := vm.snapshotAllRegs()
	for i, a := range args {
		vm.putReg(i, a)
	}
	ok := vm.executeBuiltin(key, len(args))
	if !ok {
		vm.restoreSavedRegs(saved)
	}
	return ok
}

func (vm *WamState) invokeAtPC(pc int, args []Value, cutFloor int) bool {
	savedPC := vm.PC
	savedCP := vm.CP
	savedHalted := vm.Halted
	savedRegs := vm.snapshotAllRegs()
	for i, a := range args {
		vm.putReg(i, a)
	}
	savedPending := vm.PendingB0
	savedCutLen := len(vm.CutB0Stack)
	yStart := len(vm.YSaves)
	vm.pushCallFrame()
	vm.CP = 0
	vm.PC = pc
	vm.Halted = false
	ok := vm.runUntilHalt(cutFloor)
	if len(vm.YSaves) > yStart {
		copy(vm.Regs[200:300], vm.YSaves[yStart][:])
		vm.YSaves = vm.YSaves[:yStart]
	}
	if len(vm.CutB0Stack) > savedCutLen {
		vm.CutB0Stack = vm.CutB0Stack[:savedCutLen]
	}
	vm.PendingB0 = savedPending
	vm.Halted = savedHalted
	vm.PC = savedPC
	vm.CP = savedCP
	if !ok {
		vm.restoreSavedRegs(savedRegs)
	}
	return ok
}

func (vm *WamState) runUntilHalt(baseChoicePoints int) bool {
	for {
		if vm.Halted {
			return true
		}
		instr := vm.fetch()
		if instr == nil {
			if !vm.backtrackAbove(baseChoicePoints) {
				return false
			}
			continue
		}
		if !vm.Step(instr) {
			if !vm.backtrackAbove(baseChoicePoints) {
				return false
			}
		}
	}
}

func (vm *WamState) hasParallelNegationChoices(targetPC int) bool {
	return len(vm.negationChoiceTargets(targetPC)) > 1
}

func (vm *WamState) negationChoiceTargets(targetPC int) []int {
	if targetPC < 0 || targetPC >= len(vm.Ctx.Code) {
		return nil
	}
	targets := make([]int, 0, 2)
	seen := make(map[int]bool)
	pc := targetPC
	for pc >= 0 && pc < len(vm.Ctx.Code) {
		if seen[pc] {
			break
		}
		seen[pc] = true
		targets = append(targets, vm.indexedClauseBodyStart(pc))
		switch instr := vm.Ctx.Code[pc].(type) {
		case *TryMeElsePc:
			pc = instr.NextPC
		case *TryMeElse:
			next, ok := vm.Ctx.Labels[instr.Label]
			if !ok {
				return targets
			}
			pc = next
		case *RetryMeElsePc:
			pc = instr.NextPC
		case *RetryMeElse:
			next, ok := vm.Ctx.Labels[instr.Label]
			if !ok {
				return targets
			}
			pc = next
		case *TrustMe:
			return targets
		default:
			return targets
		}
	}
	return targets
}

func (vm *WamState) runNegationParallel(targetPC int, args []Value) bool {
	targets := vm.negationChoiceTargets(targetPC)
	if len(targets) == 0 {
		return false
	}
	if len(targets) == 1 {
		return vm.runIsolatedGoal(targets[0], args)
	}
	tasks := make([]func() bool, 0, len(targets))
	for _, target := range targets {
		targetPC := target
		tasks = append(tasks, func() bool {
			return vm.runIsolatedGoal(targetPC, args)
		})
	}
	return raceToTrue(tasks)
}

func raceToTrue(tasks []func() bool) bool {
	if len(tasks) == 0 {
		return false
	}
	done := make(chan bool, len(tasks))
	var wg sync.WaitGroup
	wg.Add(len(tasks))
	for _, task := range tasks {
		task := task
		go func() {
			defer wg.Done()
			done <- task()
		}()
	}
	go func() {
		wg.Wait()
		close(done)
	}()
	for ok := range done {
		if ok {
			return true
		}
	}
	return false
}

// Unify implements standard WAM unification without occurs check,
// matching ISO Prolog semantics (unify_with_occurs_check/2 is separate).
func (vm *WamState) Unify(v1, v2 Value) bool {
	v1 = vm.deref(v1)
	v2 = vm.deref(v2)

	if u1, ok := v1.(*Unbound); ok {
		vm.bindUnbound(u1, v2)
		return true
	}
	if u2, ok := v2.(*Unbound); ok {
		vm.bindUnbound(u2, v1)
		return true
	}

	if valueEquals(v1, v2) {
		return true
	}

	if isEmptyListValue(v1) || isEmptyListValue(v2) {
		return isEmptyListValue(v1) && isEmptyListValue(v2)
	}
	h1, t1, ok1 := vm.valueListHeadTail(v1)
	h2, t2, ok2 := vm.valueListHeadTail(v2)
	if ok1 || ok2 {
		return ok1 && ok2 && vm.Unify(h1, h2) && vm.Unify(t1, t2)
	}

	f1, args1 := decompose(v1)
	f2, args2 := decompose(v2)
	if f1 != "" && f1 == f2 && len(args1) == len(args2) {
		for i := range args1 {
			if !vm.Unify(args1[i], args2[i]) {
				return false
			}
		}
		return true
	}

	return false
}

func (vm *WamState) backtrack() bool {
    if len(vm.ChoicePoints) == 0 {
        return false
    }
    topIdx := len(vm.ChoicePoints) - 1
    // Pointer access avoids copying the ~150-byte ChoicePoint struct
    // value on every backtrack — the post-Phase-E profile flagged
    // `cp := vm.ChoicePoints[topIdx]` at ~210ms cum / 8.7%. The
    // pointer stays valid because backtrack only truncates the
    // ChoicePoints slice (no append), so the underlying array slot
    // doesn't get reused or moved while we're reading. Mutations to
    // `cp.IndexedClausePCs` / `cp.ForeignResults` now apply
    // in-place, removing the explicit `vm.ChoicePoints[topIdx] = cp`
    // write-back the value-copy version needed.
    cp := &vm.ChoicePoints[topIdx]
    vm.restoreBarrier(cp)
    if len(cp.IndexedClausePCs) > 0 {
        vm.unwindTrailTo(cp.TrailMark)
        vm.restoreSavedRegs(cp.SavedRegs)
        if cp.StackLen <= len(vm.Stack) {
            vm.Stack = vm.Stack[:cp.StackLen]
        }
        vm.E = cp.E
        if cp.HeapTop >= 0 && cp.HeapTop <= vm.HeapLen {
            vm.heapTrimTo(cp.HeapTop)
        }
        vm.CP = cp.CP
        vm.Halted = false
        vm.CurrentStruct = nil
        vm.CurrentList = nil
        nextPC := cp.IndexedClausePCs[0]
        cp.IndexedClausePCs = cp.IndexedClausePCs[1:]
        if len(cp.IndexedClausePCs) == 0 {
            vm.ChoicePoints = vm.ChoicePoints[:topIdx]
        }
        vm.PC = nextPC
        return true
    }
    if cp.MemberTail != nil {
        for cp.MemberTail != nil {
            vm.unwindTrailTo(cp.TrailMark)
            vm.restoreSavedRegs(cp.SavedRegs)
            if cp.StackLen <= len(vm.Stack) {
                vm.Stack = vm.Stack[:cp.StackLen]
            }
            vm.E = cp.E
            if cp.HeapTop >= 0 && cp.HeapTop <= vm.HeapLen {
                vm.heapTrimTo(cp.HeapTop)
            }
            vm.CP = cp.CP
            vm.Halted = false
            vm.CurrentStruct = nil
            vm.CurrentList = nil
            v := vm.deref(cp.MemberTail)
            if isEmptyListValue(v) {
                vm.ChoicePoints = vm.ChoicePoints[:topIdx]
                return vm.backtrack()
            }
            item, tail, ok := vm.valueListHeadTail(v)
            if !ok {
                vm.ChoicePoints = vm.ChoicePoints[:topIdx]
                return vm.backtrack()
            }
            cp.MemberTail = vm.deref(tail)
            resumePC := cp.ResumePC
            if isEmptyListValue(cp.MemberTail) {
                vm.ChoicePoints = vm.ChoicePoints[:topIdx]
            }
            if !vm.Unify(vm.getReg(0), item) {
                vm.unwindTrailTo(cp.TrailMark)
                continue
            }
            vm.PC = resumePC
            return true
        }
        return vm.backtrack()
    }
    if len(cp.SelectResults) > 0 {
        for len(cp.SelectResults) > 0 {
            vm.unwindTrailTo(cp.TrailMark)
            vm.restoreSavedRegs(cp.SavedRegs)
            if cp.StackLen <= len(vm.Stack) {
                vm.Stack = vm.Stack[:cp.StackLen]
            }
            if cp.HeapTop <= vm.HeapLen {
                vm.heapTrimTo(cp.HeapTop)
            }
            vm.PC = cp.ResumePC
            vm.CP = cp.CP
            vm.E = cp.E
            vm.Halted = false
            vm.CurrentStruct = nil
            vm.CurrentList = nil
            solution := cp.SelectResults[0]
            cp.SelectResults = cp.SelectResults[1:]
            resumePC := cp.ResumePC
            if len(cp.SelectResults) == 0 {
                vm.ChoicePoints = vm.ChoicePoints[:topIdx]
            }
            if !vm.applySelectSolution(solution) {
                continue
            }
            vm.PC = resumePC
            return true
        }
        return vm.backtrack()
    }
    if cp.BetweenActive {
        for cp.BetweenCurrent <= cp.BetweenHigh {
            vm.unwindTrailTo(cp.TrailMark)
            vm.restoreSavedRegs(cp.SavedRegs)
            if cp.StackLen <= len(vm.Stack) {
                vm.Stack = vm.Stack[:cp.StackLen]
            }
            if cp.HeapTop <= vm.HeapLen {
                vm.heapTrimTo(cp.HeapTop)
            }
            vm.PC = cp.ResumePC
            vm.CP = cp.CP
            vm.E = cp.E
            vm.Halted = false
            vm.CurrentStruct = nil
            vm.CurrentList = nil
            n := cp.BetweenCurrent
            cp.BetweenCurrent++
            resumePC := cp.ResumePC
            if cp.BetweenCurrent > cp.BetweenHigh {
                vm.ChoicePoints = vm.ChoicePoints[:topIdx]
            }
            if !vm.Unify(vm.getReg(cp.BetweenReg), &Integer{Val: n}) {
                continue
            }
            vm.PC = resumePC
            return true
        }
        return vm.backtrack()
    }
    if len(cp.ForeignResults) > 0 {
        for len(cp.ForeignResults) > 0 {
            vm.unwindTrailTo(cp.TrailMark)
            vm.restoreSavedRegs(cp.SavedRegs)
            if cp.StackLen <= len(vm.Stack) {
                vm.Stack = vm.Stack[:cp.StackLen]
            }
            vm.E = cp.E
            if cp.HeapTop >= 0 && cp.HeapTop <= vm.HeapLen {
                vm.heapTrimTo(cp.HeapTop)
            }
            vm.CP = cp.CP
            vm.Halted = false
            vm.CurrentStruct = nil
            vm.CurrentList = nil
            nextResult := cp.ForeignResults[0]
            cp.ForeignResults = cp.ForeignResults[1:]
            // Snapshot the fields we still need after truncation —
            // once `vm.ChoicePoints = vm.ChoicePoints[:topIdx]`
            // shrinks the slice, `cp` still points at the underlying
            // array slot but it''s tidier to read the values out
            // before truncating.
            predKey := cp.ForeignPredKey
            resultRegs := cp.ForeignResultRegs
            resumePC := cp.ResumePC
            if len(cp.ForeignResults) == 0 {
                vm.ChoicePoints = vm.ChoicePoints[:topIdx]
            }
            if !vm.applyForeignResult(predKey, resultRegs, nextResult) {
                continue
            }
            vm.PC = resumePC
            return true
        }
        // The final tuple may have bound an earlier correlated output before
        // a later aliased output conflicted.  Its ForeignCP was popped before
        // applyForeignResult so restore the pre-stream snapshot explicitly,
        // then continue with any older WAM choice point.
        vm.unwindTrailTo(cp.TrailMark)
        vm.restoreSavedRegs(cp.SavedRegs)
        if cp.StackLen <= len(vm.Stack) {
            vm.Stack = vm.Stack[:cp.StackLen]
        }
        vm.E = cp.E
        if cp.HeapTop >= 0 && cp.HeapTop <= vm.HeapLen {
            vm.heapTrimTo(cp.HeapTop)
        }
        vm.CP = cp.CP
        vm.Halted = false
        vm.CurrentStruct = nil
        vm.CurrentList = nil
        return vm.backtrack()
    }
    vm.unwindTrailTo(cp.TrailMark)
    vm.restoreSavedRegs(cp.SavedRegs)
    if cp.StackLen <= len(vm.Stack) {
        vm.Stack = vm.Stack[:cp.StackLen]
    }
    vm.E = cp.E
    if cp.HeapTop >= 0 && cp.HeapTop <= vm.HeapLen {
        vm.heapTrimTo(cp.HeapTop)
    }
    vm.PC = cp.NextPC
    vm.CP = cp.CP
    vm.Halted = false
    vm.CurrentStruct = nil
    vm.CurrentList = nil
    return true
}

func (vm *WamState) unwindTrailTo(mark int) {
    for i := vm.TrailLen - 1; i >= mark; i-- {
        entry := vm.Trail[i]
        if entry.RegIdx >= 0 {
            // Register-alias rewrite: restore the register to the variable
            // cell it shared before bindUnbound rewrote it to a value.
            vm.Regs[entry.RegIdx] = entry.RegOld
            continue
        }
        // setBinding handles the bookkeeping: writing nil reverts an
        // address to its unbound state, exactly matching the previous
        // map-based delete-on-!HadOld for the slice rep.
        vm.setBinding(entry.Addr, entry.Old)
    }
    vm.Trail = vm.Trail[:mark]
    vm.TrailLen = mark
}

// buildAtomInternTable interns all atom strings from fact pairs and weighted
// edge triples into integer IDs for fast comparison in kernel loops.
func (ctx *WamContext) buildAtomInternTable() {
    ctx.AtomIntern = make(map[string]int)
    ctx.InternedFacts = make(map[string][][]int)
    ctx.InternedWeightedFacts = make(map[string][]InternedWeightedEdge)
    nextId := 0

    internAtom := func(s string) int {
        if id, ok := ctx.AtomIntern[s]; ok {
            return id
        }
        ctx.AtomIntern[s] = nextId
        nextId++
        return nextId - 1
    }

    for predKey, pairs := range ctx.IndexedAtomFactPairs {
        rows := make([][]int, len(pairs))
        for i, p := range pairs {
            rows[i] = []int{internAtom(p.Left), internAtom(p.Right)}
        }
        ctx.InternedFacts[predKey] = rows
    }

    for predKey, triples := range ctx.IndexedWeightedEdgeTriples {
        rows := make([]InternedWeightedEdge, len(triples))
        for i, t := range triples {
            rows[i] = InternedWeightedEdge{
                Left:   internAtom(t.Left),
                Right:  internAtom(t.Right),
                Weight: t.Weight,
            }
        }
        ctx.InternedWeightedFacts[predKey] = rows
    }
}
