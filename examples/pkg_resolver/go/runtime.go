package wam

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"sync"
)

var (
	_ = fmt.Sprintf
	_ = math.NaN
	_ = os.ReadFile
	_ = exec.Command
	_ = runtime.GOMAXPROCS
	_ = strings.Split
	_ sync.Locker = (*sync.Mutex)(nil)
)

// Step executes a single WAM instruction.
func (vm *WamState) Step(instr Instruction) bool {
    switch i := instr.(type) {
    case *GetConstant:
        val := vm.Regs[i.Ai]
        if val == nil { return false }
        // Dereference before inspecting: the register may hold a *Ref or a
        // bound *Unbound (bound via the Bindings table), e.g. the tail of a
        // partial list carried through a recursive call. Without the deref,
        // isUnbound() sees the raw *Unbound as unbound and rebinds it to the
        // constant -- silently corrupting an already-bound value (get_constant
        // [] would overwrite a bound [H|T] tail with []). Mirrors the deref
        // GetStructure / GetList already do.
        val = vm.deref(val)
        if isUnbound(val) {
            u := val.(*Unbound)
            vm.bindUnbound(u, i.C)
            vm.PC++
            return true
        }
        if valueEquals(val, i.C) {
            vm.PC++
            return true
        }
        return false
    case *GetVariable:
        val := vm.Regs[i.Ai]
        if val == nil { return false }
        vm.trailBinding(i.Xn)
        vm.putReg(i.Xn, val)
        vm.PC++
        return true
    case *GetValue:
        valA := vm.Regs[i.Ai]
        valX := vm.getReg(i.Xn)
        if valA == nil { return false }
        if vm.Unify(valA, valX) {
            vm.PC++
            return true
        }
        return false
    case *GetStructure:
        val := vm.Regs[i.Ai]
        if val == nil { return false }
        // Dereference before deciding read vs write mode. The register
        // can hold a *Ref, or an *Unbound that is already bound through
        // the Bindings table -- which is exactly what unify_variable
        // leaves behind when it reads a nested term out of an enclosing
        // one. Testing isUnbound on the raw value made get_structure
        // take the *write* branch for an already-bound argument and
        // build a fresh structure over the top of it, so a head like
        // `p([tk(N)|R], N, R)` bound N to a new empty cell instead of
        // the incoming value. GetList and GetConstant already deref
        // first; this mirrors them.
        val = vm.deref(val)
        if isUnbound(val) {
            addr := vm.heapPush(nil)
            arity := parseFunctorArity(i.Functor)
            s := &Structure{Functor: i.Functor, Arity: arity, Args: make([]Value, arity)}
            vm.Heap[addr] = s
            vm.CurrentStruct = s
            vm.CurrentList = nil
            vm.bindUnbound(val.(*Unbound), &Ref{Addr: addr})
            vm.Stack = append(vm.Stack, &WriteCtx{N: arity, Struct: s})
            vm.PC++
            return true
        }
        if s, ok := val.(*Structure); ok {
            if s.Functor == i.Functor {
                vm.Stack = append(vm.Stack, &UnifyCtx{Args: s.Args})
                vm.PC++
                return true
            }
        }
        return false
    case *GetList:
        val := vm.Regs[i.Ai]
        if val == nil { return false }
        val = vm.deref(val)
        if isUnbound(val) {
            addr := vm.heapPush(nil)
            l := &List{Elements: make([]Value, 2)}
            vm.Heap[addr] = l
            vm.CurrentList = l
            vm.CurrentStruct = nil
            vm.bindUnbound(val.(*Unbound), &Ref{Addr: addr})
            vm.Stack = append(vm.Stack, &WriteCtx{N: 2, List: l})
            vm.PC++
            return true
        }
        // get_list reads a term as a *cons cell*: exactly two slots,
        // head and tail. Push those, not the raw Elements slice.
        //
        // *List is used for two different things in this runtime: a
        // cons pair built by put_list (Elements = [head, tail]) and a
        // flat list returned by a Go builtin such as reverse/2,
        // findall/3, sort/2 or append/3 (Elements = the items). Pushing
        // Elements directly conflated them — a flat one-element list
        // offered a single slot, so the following unify_* for the cons
        // tail found an empty context and the clause failed; a flat
        // three-element list offered three slots and bound the tail to
        // the second *item*. That is why heads like
        // `p([tk_atom(N)|R], ...)` matched a literal list but not one
        // that had been through reverse/2.
        //
        // valueListHeadTail normalises both representations (and the
        // heap-linked Compound/Structure cons form), so it is the one
        // place that knows how to split a list value.
        if h, t, ok := vm.valueListHeadTail(val); ok {
            vm.Stack = append(vm.Stack, &UnifyCtx{Args: []Value{h, t}})
            vm.PC++
            return true
        }
        if h, t, ok := consHeadTail(val); ok {
            vm.Stack = append(vm.Stack, &UnifyCtx{Args: []Value{h, t}})
            vm.PC++
            return true
        }
        return false
    case *UnifyVariable:
        if ctx := vm.peekUnifyCtx(); ctx != nil && len(ctx.Args) > 0 {
            arg := ctx.Args[0]
            ctx.Args = ctx.Args[1:]
            if len(ctx.Args) == 0 { vm.popStack() }
            vm.trailBinding(i.Xn)
            vm.putReg(i.Xn, arg)
            vm.PC++
            return true
        }
        if wctx := vm.peekWriteCtx(); wctx != nil && wctx.N > 0 {
            addr := vm.HeapLen
            v := &Unbound{Name: fmt.Sprintf("_H%d", addr), Idx: vm.allocVarId()}
            vm.heapPush(v)
            if vm.CurrentStruct != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = v
            } else if vm.CurrentList != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = v
            }
            wctx.N--
            if wctx.N == 0 {
                vm.popStackRestoringWriteTarget()
            }
            vm.putReg(i.Xn, v)
            vm.PC++
            return true
        }
        return false
    case *UnifyValue:
        if ctx := vm.peekUnifyCtx(); ctx != nil && len(ctx.Args) > 0 {
            expected := ctx.Args[0]
            ctx.Args = ctx.Args[1:]
            if len(ctx.Args) == 0 { vm.popStack() }
            actual := vm.getReg(i.Xn)
            if vm.Unify(expected, actual) {
                vm.PC++
                return true
            }
            return false
        }
        if wctx := vm.peekWriteCtx(); wctx != nil && wctx.N > 0 {
            val := vm.getReg(i.Xn)
            vm.heapPush(val)
            if vm.CurrentStruct != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = val
            } else if vm.CurrentList != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = val
            }
            wctx.N--
            if wctx.N == 0 {
                vm.popStackRestoringWriteTarget()
            }
            vm.PC++
            return true
        }
        return false
    case *UnifyConstant:
        if ctx := vm.peekUnifyCtx(); ctx != nil && len(ctx.Args) > 0 {
            expected := ctx.Args[0]
            ctx.Args = ctx.Args[1:]
            if len(ctx.Args) == 0 { vm.popStack() }
            if valueEquals(vm.deref(expected), i.C) {
                vm.PC++
                return true
            }
            return false
        }
        if wctx := vm.peekWriteCtx(); wctx != nil && wctx.N > 0 {
            vm.heapPush(i.C)
            if vm.CurrentStruct != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = i.C
            } else if vm.CurrentList != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = i.C
            }
            wctx.N--
            if wctx.N == 0 {
                vm.popStackRestoringWriteTarget()
            }
            vm.PC++
            return true
        }
        return false
    case *PutConstant:
        vm.putReg(i.Ai, i.C)
        vm.PC++
        return true
    case *PutVariable:
        // Allocate a globally-unique Idx for the new logical variable.
        // The earlier strategy reused i.Xn (the X-register slot index)
        // as the Idx, which collided across activations: an outer
        // activation's X207 (Idx=207) and an inner activation's X207
        // (also Idx=207) would share Bindings[207] and clobber each
        // other. The first attempt at a fix `delete(Bindings, i.Xn)`
        // worked for 2-hop recursion but broke at 3+ hops because the
        // delete could erase an outer activation's still-live binding.
        // Allocating a fresh Idx via allocVarId/0 sidesteps the
        // collision entirely.
        v := &Unbound{Name: fmt.Sprintf("_R%d", i.Xn), Idx: vm.allocVarId()}
        vm.putReg(i.Xn, v)
        vm.putReg(i.Ai, v)
        vm.PC++
        return true
    case *PutValue:
        val := vm.getReg(i.Xn)
        vm.putReg(i.Ai, val)
        vm.PC++
        return true
    case *PutStructure:
        addr := vm.heapPush(nil)
        arity := parseFunctorArity(i.Functor)
        s := &Structure{Functor: i.Functor, Arity: arity, Args: make([]Value, arity)}
        vm.Heap[addr] = s
        vm.CurrentStruct = s
        vm.CurrentList = nil
        ref := &Ref{Addr: addr}
        // If this register holds an unbound placeholder (one a prior
        // set_variable embedded into an enclosing structure/list arg), bind
        // it to the new structure so the embedded copy resolves here. The
        // compiler builds nested terms outer-first — e.g. +(+(A,B),C) or a
        // list tail cell [|]/2 — leaving a placeholder in the arg slot that
        // a later put_structure into the same register must fill. Trailing
        // via bindUnbound keeps it backtrack-safe.
        //
        // A-REGISTER EXCEPTION (M139/M140 bind-through class): A registers
        // (index < 100) are argument STAGING — their old occupant is an
        // unrelated variable (often a clause-head argument), and binding it
        // to the new cell creates a cyclic term (X = f(X)), making a later
        // X = 1 wrong-fail. Top-down chaining placeholders only ever live
        // in X/Y registers (set_variable Xn), so the bind-through is
        // conditioned on the register class — the same fix the Rust and
        // LLVM targets carry.
        if i.Ai >= 100 {
            if cur := vm.Regs[i.Ai]; cur != nil {
                if u, ok := vm.deref(cur).(*Unbound); ok {
                    vm.bindUnbound(u, ref)
                }
            }
        }
        vm.putReg(i.Ai, ref)
        vm.Stack = append(vm.Stack, &WriteCtx{N: arity, Struct: s})
        vm.PC++
        return true
    case *PutList:
        addr := vm.heapPush(nil)
        l := &List{Elements: make([]Value, 2)}
        vm.Heap[addr] = l
        vm.CurrentList = l
        vm.CurrentStruct = nil
        vm.putReg(i.Ai, &Ref{Addr: addr})
        vm.Stack = append(vm.Stack, &WriteCtx{N: 2, List: l})
        vm.PC++
        return true
    case *SetVariable:
        addr := vm.HeapLen
        v := &Unbound{Name: fmt.Sprintf("_H%d", addr), Idx: vm.allocVarId()}
        vm.heapPush(v)
        if vm.CurrentStruct != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = v
                wctx.N--
                if wctx.N == 0 {
                    vm.popStackRestoringWriteTarget()
                }
            }
        } else if vm.CurrentList != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = v
                wctx.N--
                if wctx.N == 0 {
                    vm.popStackRestoringWriteTarget()
                }
            }
        }
        vm.putReg(i.Xn, v)
        vm.PC++
        return true
    case *SetValue:
        val := vm.getReg(i.Xn)
        vm.heapPush(val)
        if vm.CurrentStruct != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = val
                wctx.N--
                if wctx.N == 0 {
                    vm.popStackRestoringWriteTarget()
                }
            }
        } else if vm.CurrentList != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = val
                wctx.N--
                if wctx.N == 0 {
                    vm.popStackRestoringWriteTarget()
                }
            }
        }
        vm.PC++
        return true
    case *SetConstant:
        vm.heapPush(i.C)
        if vm.CurrentStruct != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = i.C
                wctx.N--
                if wctx.N == 0 {
                    vm.popStackRestoringWriteTarget()
                }
            }
        } else if vm.CurrentList != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = i.C
                wctx.N--
                if wctx.N == 0 {
                    vm.popStackRestoringWriteTarget()
                }
            }
        }
        vm.PC++
        return true
    case *Allocate:
        // Env trimming: PrevE links the new frame back to the previously
        // active env frame. vm.E moves to the index of the just-pushed
        // frame so peekEnvFrame is O(1) and Deallocate can walk the
        // PrevE chain without scanning the stack.
        env := &EnvFrame{CP: vm.CP, B0: len(vm.ChoicePoints), CutB0: vm.PendingB0, PrevE: vm.E}
        // Snapshot the Y-reg range (200..299) into the env frame so a
        // nested predicate that uses the same slot numbers via
        // PutVariable doesn't silently clobber the caller's Y-regs.
        // Restored at Deallocate. Bindings on caller-passed Unbounds
        // still propagate via the global Bindings[Idx] table — deref
        // of the restored Y-reg follows that binding, so we don't
        // lose genuine results, only spurious leftover state.
        copy(env.SavedYRegs[:], vm.Regs[200:300])
        vm.Stack = append(vm.Stack, env)
        vm.E = len(vm.Stack) - 1
        vm.PC++
        return true
    case *Deallocate:
        if vm.E >= 0 && vm.E < len(vm.Stack) {
            if env, ok := vm.Stack[vm.E].(*EnvFrame); ok {
                vm.CP = env.CP
                copy(vm.Regs[200:300], env.SavedYRegs[:])
                prevE := env.PrevE
                // Physical-pop only if it's safe: the frame must be at
                // the top of the stack AND no younger choicepoint
                // references it (env.B0 == current CP count means the
                // frame was Allocated AFTER the youngest live CP, so
                // nothing depends on it staying around). Otherwise the
                // frame stays on the stack — backtrack truncation will
                // sweep it away when the referencing CPs are
                // exhausted, and a future Allocate can push above it
                // because vm.E now points at prevE, not at the dead
                // frame's slot.
                if env.B0 >= len(vm.ChoicePoints) && vm.E == len(vm.Stack)-1 {
                    vm.Stack = vm.Stack[:vm.E]
                }
                vm.E = prevE
            }
        }
        vm.PC++
        return true
    case *Call:
        vm.CP = vm.PC + 1
        if pc, ok := vm.Ctx.Labels[i.Pred]; ok {
            vm.pushCallFrame()
            vm.PC = pc
            return true
        }
        if _, ok := vm.Ctx.ForeignNativeKinds[i.Pred]; ok {
            if vm.executeForeignPredicate(i.Pred, i.Arity) {
                vm.PC = vm.CP
                return true
            }
            return false
        }
        if vm.executeIndexedAtomFact2(i.Pred) {
            vm.PC = vm.CP
            return true
        }
        return false
    case *GetArgInto:
        term := vm.deref(vm.getReg(i.Src))
        var selected Value
        switch t := term.(type) {
        case *Compound:
            if i.Index < 1 || i.Index > len(t.Args) {
                return false
            }
            selected = t.Args[i.Index-1]
        case *Structure:
            if i.Index < 1 || i.Index > len(t.Args) {
                return false
            }
            selected = t.Args[i.Index-1]
        case *List:
            head, tail, ok := vm.listHeadTail(t)
            if !ok {
                return false
            }
            if i.Index == 1 {
                selected = head
            } else if i.Index == 2 {
                selected = tail
            } else {
                return false
            }
        default:
            return false
        }
        vm.trailBinding(i.Dest)
        vm.putReg(i.Dest, selected)
        vm.PC++
        return true
    case *CallForeign:
        return vm.executeForeignPredicate(i.Pred, i.Arity)
    case *CallIndexedAtomFact2:
        return vm.executeIndexedAtomFact2(i.Pred)
    case *CallPc:
        vm.CP = vm.PC + 1
        vm.pushCallFrame()
        vm.PC = i.TargetPC
        return true
    case *Execute:
        if pc, ok := vm.Ctx.Labels[i.Pred]; ok {
            vm.enterExecute()
            vm.PC = pc
            return true
        }
        if _, ok := vm.Ctx.ForeignNativeKinds[i.Pred]; ok {
            return vm.executeForeignPredicate(i.Pred, 0)
        }
        return vm.executeIndexedAtomFact2(i.Pred)
    case *ExecutePc:
        vm.enterExecute()
        vm.PC = i.TargetPC
        return true
    case *Jump:
        if pc, ok := vm.Ctx.Labels[i.Label]; ok {
            vm.PC = pc
            return true
        }
        return false
    case *JumpPc:
        vm.PC = i.TargetPC
        return true
    case *CutIte:
        if len(vm.ChoicePoints) > 0 {
            vm.ChoicePoints = vm.ChoicePoints[:len(vm.ChoicePoints)-1]
        }
        vm.PC++
        return true
    case *GetLevel:
        vm.recordIteLevel(i.Reg)
        vm.PC++
        return true
    case *Cut:
        if target, ok := vm.lookupIteLevel(i.Reg); ok {
            if target >= 0 && target < len(vm.ChoicePoints) {
                vm.ChoicePoints = vm.ChoicePoints[:target]
            }
        }
        vm.PC++
        return true
    case *BeginAggregate:
        return vm.executeAggregate(i.AggType, i.ValueReg, i.ResultReg)
    case *EndAggregate:
        vm.PC++
        return true
    case *Proceed:
        vm.popCallFrame()
        if vm.CP > 0 {
            vm.PC = vm.CP
        } else {
            vm.Halted = true
        }
        return true
    case *BuiltinCall:
        result := vm.executeBuiltin(i.Op, i.Arity)
        if result {
            vm.PC++
        }
        return result
    case *BuiltinExecute:
        result := vm.executeBuiltin(i.Op, i.Arity)
        if !result {
            return false
        }
        vm.popCallFrame()
        if vm.CP > 0 {
            vm.PC = vm.CP
        } else {
            vm.Halted = true
        }
        return true
    case *TryMeElse:
        nextPC := 0
        if pc, ok := vm.Ctx.Labels[i.Label]; ok {
            nextPC = pc
        }
        vm.pushChoicePoint(nextPC, i.Arity)
        vm.PC++
        return true
    case *TryMeElsePc:
        vm.pushChoicePoint(i.NextPC, i.Arity)
        vm.PC++
        return true
    case *RetryMeElse:
        if pc, ok := vm.Ctx.Labels[i.Label]; ok {
            if len(vm.ChoicePoints) > 0 {
                vm.ChoicePoints[len(vm.ChoicePoints)-1].NextPC = pc
            }
        }
        vm.PC++
        return true
    case *RetryMeElsePc:
        if len(vm.ChoicePoints) > 0 {
            vm.ChoicePoints[len(vm.ChoicePoints)-1].NextPC = i.NextPC
        }
        vm.PC++
        return true
    case *TrustMe:
        if len(vm.ChoicePoints) > 0 {
            vm.ChoicePoints = vm.ChoicePoints[:len(vm.ChoicePoints)-1]
        }
        vm.PC++
        return true
    case *SwitchOnConstant:
        if val := vm.Regs[0]; val != nil && !isUnbound(val) {
            targets := make([]int, 0)
            for _, c := range i.Cases {
                if !valueEquals(c.Val, val) {
                    continue
                }
                if c.Label == "default" {
                    targets = append(targets, vm.indexedClauseBodyStart(vm.PC+1))
                    continue
                }
                if pc, ok := vm.Ctx.Labels[c.Label]; ok {
                    targets = append(targets, vm.indexedClauseBodyStart(pc))
                }
            }
            if len(targets) > 0 {
                return vm.enterIndexedAlternatives(targets)
            }
        }
        vm.PC++
        return true
    case *SwitchOnConstantPc:
        if val := vm.Regs[0]; val != nil && !isUnbound(val) {
            n := len(i.Cases)
            idx := sort.Search(n, func(j int) bool {
                return compareValues(i.Cases[j].Val, val) >= 0
            })
            targets := make([]int, 0)
            for idx < n && valueEquals(i.Cases[idx].Val, val) {
                targets = append(targets, vm.indexedClauseBodyStart(i.Cases[idx].TargetPC))
                idx++
            }
            if len(targets) > 0 {
                return vm.enterIndexedAlternatives(targets)
            }
        }
        vm.PC++
        return true
    case *SwitchOnStructure:
        if val := vm.Regs[0]; val != nil {
            if f, args := decompose(val); f != "" {
                key := fmt.Sprintf("%s/%d", f, len(args))
                targets := make([]int, 0)
                for _, c := range i.Cases {
                    if c.Functor != key {
                        continue
                    }
                    if c.Label == "default" {
                        targets = append(targets, vm.indexedClauseBodyStart(vm.PC+1))
                        continue
                    }
                    if pc, ok := vm.Ctx.Labels[c.Label]; ok {
                        targets = append(targets, vm.indexedClauseBodyStart(pc))
                    }
                }
                if len(targets) > 0 {
                    return vm.enterIndexedAlternatives(targets)
                }
            }
        }
        vm.PC++
        return true
    case *SwitchOnStructurePc:
        if val := vm.Regs[0]; val != nil {
            if f, args := decompose(val); f != "" {
                key := fmt.Sprintf("%s/%d", f, len(args))
                targets := make([]int, 0)
                for _, c := range i.Cases {
                    if c.Functor == key {
                        targets = append(targets, vm.indexedClauseBodyStart(c.TargetPC))
                    }
                }
                if len(targets) > 0 {
                    return vm.enterIndexedAlternatives(targets)
                }
            }
        }
        vm.PC++
        return true
    case *SwitchOnConstantA2:
        if val := vm.Regs[1]; val != nil && !isUnbound(val) {
            targets := make([]int, 0)
            for _, c := range i.Cases {
                if !valueEquals(c.Val, val) {
                    continue
                }
                if c.Label == "default" {
                    targets = append(targets, vm.indexedClauseBodyStart(vm.PC+1))
                    continue
                }
                if pc, ok := vm.Ctx.Labels[c.Label]; ok {
                    targets = append(targets, vm.indexedClauseBodyStart(pc))
                }
            }
            if len(targets) > 0 {
                return vm.enterIndexedAlternatives(targets)
            }
        }
        vm.PC++
        return true
    case *SwitchOnConstantA2Pc:
        if val := vm.Regs[1]; val != nil && !isUnbound(val) {
            n := len(i.Cases)
            idx := sort.Search(n, func(j int) bool {
                return compareValues(i.Cases[j].Val, val) >= 0
            })
            targets := make([]int, 0)
            for idx < n && valueEquals(i.Cases[idx].Val, val) {
                targets = append(targets, vm.indexedClauseBodyStart(i.Cases[idx].TargetPC))
                idx++
            }
            if len(targets) > 0 {
                return vm.enterIndexedAlternatives(targets)
            }
        }
        vm.PC++
        return true
    default:
        return false
    }
}

// Run executes the WAM instruction loop until halt or failure.
func (vm *WamState) Run() (result bool) {
    // An ISO error that escapes every catch/3 surfaces as failure with
    // the ball recorded in vm.UncaughtBall, rather than as a process
    // crash. Non-prologBall panics are genuine bugs and re-raise.
    defer func() {
        if r := recover(); r != nil {
            thrown, ok := r.(prologBall)
            if !ok {
                panic(r)
            }
            vm.UncaughtBall = thrown.Ball
            result = false
        }
    }()
    for {
        if vm.Halted {
            return true
        }
        instr := vm.fetch()
        if instr == nil {
            return false
        }
        if !vm.Step(instr) {
            if !vm.backtrack() {
                return false
            }
        }
    }
}

func (vm *WamState) indexedClauseBodyStart(targetPC int) int {
    if targetPC < 0 || targetPC >= len(vm.Ctx.Code) {
        return targetPC
    }
    switch vm.Ctx.Code[targetPC].(type) {
    case *TryMeElse, *TryMeElsePc, *RetryMeElse, *RetryMeElsePc, *TrustMe:
        return targetPC + 1
    default:
        return targetPC
    }
}

func (vm *WamState) enterIndexedAlternatives(targets []int) bool {
    if len(targets) == 0 {
        return false
    }
    if len(targets) > 1 {
        // count non-nil A-regs as arity approximation when not statically known
        arity := 0
        for i := 0; i < 32; i++ {
            if vm.Regs[i] != nil { arity = i + 1 }
        }
        if arity < 1 { arity = 1 }
        vm.pushIndexedChoicePoint(targets[1:], arity)
    }
    vm.PC = targets[0]
    return true
}

func (vm *WamState) enterIndexedClause(targetPC int) bool {
    return vm.enterIndexedAlternatives([]int{vm.indexedClauseBodyStart(targetPC)})
}

func (vm *WamState) runIsolatedGoal(targetPC int, args []Value) bool {
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
    return sub.Run()
}

// CollectResults gathers values from A registers (indices 0..N-1).
func (vm *WamState) CollectResults() []Value {
	results := make([]Value, 0)
	for i := 0; i < 100; i++ {
		val := vm.Regs[i]
		if val == nil {
			break
		}
		results = append(results, vm.deref(val))
	}
	return results
}

// fetch retrieves the instruction at the current PC.
func (vm *WamState) fetch() Instruction {
    if vm.PC >= 0 && vm.PC < len(vm.Ctx.Code) {
        return vm.Ctx.Code[vm.PC]
    }
    return nil
}

func resolveInstructions(code []Instruction, labels map[string]int) []Instruction {
    // resolveLabel handles the "default" sentinel that the constant-index
    // emitter (build_constant_index/5 in wam_target.pl) puts on the
    // first clause's entry — at runtime that label means "fall through
    // to the next instruction" (vm.PC+1). resolveInstructions runs over
    // the code linearly, so the "next" PC is always idx+1 of the
    // current SwitchOnConstant. Without this resolution, every
    // SwitchOnConstant with a default entry stays unresolved (because
    // labels["default"] doesn't exist), keeping the linear-scan
    // SwitchOnConstant runtime case alive — at scale-300 with
    // category_parent's 6000-clause table the O(N) scan dominates
    // the per-call cost.
    resolveLabel := func(label string, idx int) (int, bool) {
        if label == "default" {
            return idx + 1, true
        }
        pc, ok := labels[label]
        return pc, ok
    }
    resolved := make([]Instruction, 0, len(code))
    for idx, instr := range code {
        switch i := instr.(type) {
        case *Call:
            if pc, ok := labels[i.Pred]; ok {
                resolved = append(resolved, &CallPc{TargetPC: pc, Arity: i.Arity})
            } else {
                resolved = append(resolved, instr)
            }
        case *CallForeign:
            resolved = append(resolved, instr)
        case *Execute:
            if pc, ok := labels[i.Pred]; ok {
                resolved = append(resolved, &ExecutePc{TargetPC: pc})
            } else {
                resolved = append(resolved, instr)
            }
        case *Jump:
            if pc, ok := labels[i.Label]; ok {
                resolved = append(resolved, &JumpPc{TargetPC: pc})
            } else {
                resolved = append(resolved, instr)
            }
        case *TryMeElse:
            if pc, ok := labels[i.Label]; ok {
                resolved = append(resolved, &TryMeElsePc{NextPC: pc, Arity: i.Arity})
            } else {
                resolved = append(resolved, instr)
            }
        case *RetryMeElse:
            if pc, ok := labels[i.Label]; ok {
                resolved = append(resolved, &RetryMeElsePc{NextPC: pc, Arity: i.Arity})
            } else {
                resolved = append(resolved, instr)
            }
        case *SwitchOnConstant:
            cases := make([]ConstPcCase, 0, len(i.Cases))
            complete := true
            for _, c := range i.Cases {
                pc, ok := resolveLabel(c.Label, idx)
                if !ok {
                    complete = false
                    break
                }
                cases = append(cases, ConstPcCase{Val: c.Val, TargetPC: pc})
            }
            if complete {
                sort.Slice(cases, func(a, b int) bool {
                    return compareValues(cases[a].Val, cases[b].Val) < 0
                })
                resolved = append(resolved, &SwitchOnConstantPc{Cases: cases})
            } else {
                resolved = append(resolved, instr)
            }
        case *SwitchOnStructure:
            cases := make([]StructPcCase, 0, len(i.Cases))
            complete := true
            for _, c := range i.Cases {
                pc, ok := resolveLabel(c.Label, idx)
                if !ok {
                    complete = false
                    break
                }
                cases = append(cases, StructPcCase{Functor: c.Functor, TargetPC: pc})
            }
            if complete {
                resolved = append(resolved, &SwitchOnStructurePc{Cases: cases})
            } else {
                resolved = append(resolved, instr)
            }
        case *SwitchOnConstantA2:
            cases := make([]ConstPcCase, 0, len(i.Cases))
            complete := true
            for _, c := range i.Cases {
                pc, ok := resolveLabel(c.Label, idx)
                if !ok {
                    complete = false
                    break
                }
                cases = append(cases, ConstPcCase{Val: c.Val, TargetPC: pc})
            }
            if complete {
                sort.Slice(cases, func(a, b int) bool {
                    return compareValues(cases[a].Val, cases[b].Val) < 0
                })
                resolved = append(resolved, &SwitchOnConstantA2Pc{Cases: cases})
            } else {
                resolved = append(resolved, instr)
            }
        default:
            resolved = append(resolved, instr)
        }
    }
    return resolved
}

func (vm *WamState) executeAggregate(aggType string, valueReg int, resultReg int) bool {
    endPC, ok := vm.findMatchingAggregateEnd(vm.PC)
    if !ok {
        return false
    }
    sub := vm.Clone()
    baseChoicePoints := len(sub.ChoicePoints)
    sub.PC = vm.PC + 1
    sub.Halted = false
    sub.CurrentStruct = nil
    sub.CurrentList = nil

    values := make([]Value, 0)
    count := 0
    for {
        if !sub.runUntilPC(endPC, baseChoicePoints) {
            break
        }
        count++
        if aggType != "count" {
            val := sub.Regs[valueReg]
            if val == nil {
                return false
            }
            values = append(values, sub.freezeTerm(val))
        }
        if !sub.backtrackAbove(baseChoicePoints) {
            break
        }
    }

    if aggType == "set" || aggType == "setof" {
        sort.SliceStable(values, func(i, j int) bool {
            return sub.compareTerms(values[i], values[j]) < 0
        })
    }

    result, ok := aggregateResultValue(aggType, values, count)
    if !ok {
        return false
    }
    if !vm.bindOrUnifyReg(resultReg, result) {
        return false
    }
    vm.PC = endPC + 1
    return true
}

func (vm *WamState) findMatchingAggregateEnd(startPC int) (int, bool) {
    depth := 1
    for pc := startPC + 1; pc < len(vm.Ctx.Code); pc++ {
        switch vm.Ctx.Code[pc].(type) {
        case *BeginAggregate:
            depth++
        case *EndAggregate:
            depth--
            if depth == 0 {
                return pc, true
            }
        }
    }
    return 0, false
}

func (vm *WamState) runUntilPC(targetPC int, baseChoicePoints int) bool {
    for {
        if vm.Halted {
            return false
        }
        if vm.PC == targetPC {
            return true
        }
        instr := vm.fetch()
        if instr == nil {
            return false
        }
        if !vm.Step(instr) {
            if !vm.backtrackAbove(baseChoicePoints) {
                return false
            }
        }
    }
}

func (vm *WamState) backtrackAbove(limit int) bool {
    if len(vm.ChoicePoints) <= limit {
        return false
    }
    return vm.backtrack()
}

func (vm *WamState) bindOrUnifyReg(reg int, val Value) bool {
    existing := vm.Regs[reg]
    if existing == nil {
        vm.trailBinding(reg)
        vm.putReg(reg, val)
        return true
    }
    return vm.Unify(existing, val)
}

func aggregateResultValue(aggType string, values []Value, count int) (Value, bool) {
    switch aggType {
    case "count":
        return &Integer{Val: int64(count)}, true
    case "collect":
        return listFromItems(append([]Value(nil), values...)), true
    case "bag":
        return listFromItems(append([]Value(nil), values...)), true
    case "bagof":
        if len(values) == 0 {
            return nil, false
        }
        return listFromItems(append([]Value(nil), values...)), true
    case "set":
        return &List{Elements: uniqueAggregateValues(values)}, true
    case "setof":
        if len(values) == 0 {
            return nil, false
        }
        return &List{Elements: uniqueAggregateValues(values)}, true
    case "sum":
        total := 0.0
        for _, value := range values {
            number, ok := aggregateNumericValue(value)
            if !ok {
                return nil, false
            }
            total += number
        }
        return &Float{Val: total}, true
    case "max":
        if len(values) == 0 {
            return nil, false
        }
        best, ok := aggregateNumericValue(values[0])
        if !ok {
            return nil, false
        }
        for _, value := range values[1:] {
            number, ok := aggregateNumericValue(value)
            if !ok {
                return nil, false
            }
            if number > best {
                best = number
            }
        }
        return &Float{Val: best}, true
    case "min":
        if len(values) == 0 {
            return nil, false
        }
        best, ok := aggregateNumericValue(values[0])
        if !ok {
            return nil, false
        }
        for _, value := range values[1:] {
            number, ok := aggregateNumericValue(value)
            if !ok {
                return nil, false
            }
            if number < best {
                best = number
            }
        }
        return &Float{Val: best}, true
    default:
        return nil, false
    }
}

func uniqueAggregateValues(values []Value) []Value {
    out := make([]Value, 0, len(values))
    for _, value := range values {
        found := false
        for _, existing := range out {
            if valueEquals(existing, value) {
                found = true
                break
            }
        }
        if !found {
            out = append(out, value)
        }
    }
    return out
}

func aggregateNumericValue(value Value) (float64, bool) {
    switch t := value.(type) {
    case *Integer:
        return float64(t.Val), true
    case *Float:
        return t.Val, true
    default:
        return 0, false
    }
}

func (vm *WamState) registerForeignNativeKind(predKey string, kind string) {
    vm.Ctx.ForeignNativeKinds[predKey] = kind
}

func (vm *WamState) registerForeignResultLayout(predKey string, layout string) {
    vm.Ctx.ForeignResultLayouts[predKey] = layout
}

func (vm *WamState) registerForeignResultMode(predKey string, mode string) {
    vm.Ctx.ForeignResultModes[predKey] = mode
}

func (vm *WamState) registerForeignStringConfig(predKey string, key string, value string) {
    cfg, ok := vm.Ctx.ForeignStringConfigs[predKey]
    if !ok {
        cfg = make(map[string]string)
        vm.Ctx.ForeignStringConfigs[predKey] = cfg
    }
    cfg[key] = value
}

func (vm *WamState) registerForeignUsizeConfig(predKey string, key string, value int) {
    cfg, ok := vm.Ctx.ForeignUsizeConfigs[predKey]
    if !ok {
        cfg = make(map[string]int)
        vm.Ctx.ForeignUsizeConfigs[predKey] = cfg
    }
    cfg[key] = value
}

func (vm *WamState) registerIndexedAtomFact2Pairs(predKey string, pairs []AtomPair) {
    vm.Ctx.IndexedAtomFactPairs[predKey] = pairs
    vm.registerAtomFact2Source(predKey, newStaticAtomFact2Source(pairs))
}

func (vm *WamState) registerAtomFact2Source(predKey string, source AtomFact2Source) {
    if source == nil {
        return
    }
    vm.Ctx.AtomFact2Sources[predKey] = source
    vm.Ctx.IndexedAtomFactPairs[predKey] = source.Scan()
}

func (vm *WamState) registerTsvAtomFact2(predKey string, path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return err
    }
    lines := strings.Split(string(data), "\n")
    pairs := make([]AtomPair, 0, len(lines))
    for idx, line := range lines {
        line = strings.TrimSpace(line)
        if line == "" {
            continue
        }
        if idx == 0 {
            continue
        }
        cols := strings.Split(line, "\t")
        if len(cols) < 2 {
            continue
        }
        left := strings.TrimSpace(cols[0])
        right := strings.TrimSpace(cols[1])
        if left == "" || right == "" {
            continue
        }
        pairs = append(pairs, AtomPair{Left: left, Right: right})
    }
    vm.registerAtomFact2Source(predKey, newStaticAtomFact2Source(pairs))
    return nil
}

func (vm *WamState) registerLmdbAtomFact2(predKey string, artifactDir string, mode string, l2Capacity int) error {
    source := newLmdbAtomFact2Source(predKey, artifactDir, mode, l2Capacity)
    vm.Ctx.AtomFact2Sources[predKey] = source
    return nil
}

func (vm *WamState) registerIndexedWeightedEdgeTriples(predKey string, triples []WeightedEdgeTriple) {
    vm.Ctx.IndexedWeightedEdgeTriples[predKey] = triples
}

func (vm *WamState) foreignResultLayout(predKey string) string {
    return vm.Ctx.ForeignResultLayouts[predKey]
}

func (vm *WamState) foreignResultMode(predKey string) string {
    return vm.Ctx.ForeignResultModes[predKey]
}

func (vm *WamState) foreignStringConfig(predKey string, key string) string {
    cfg, ok := vm.Ctx.ForeignStringConfigs[predKey]
    if !ok {
        return ""
    }
    return cfg[key]
}

func (vm *WamState) foreignUsizeConfig(predKey string, key string) int {
    cfg, ok := vm.Ctx.ForeignUsizeConfigs[predKey]
    if !ok {
        return 0
    }
    return cfg[key]
}

func parseForeignTupleLayout(layout string) int {
    var arity int
    if _, err := fmt.Sscanf(layout, "tuple:%d", &arity); err == nil {
        return arity
    }
    return 0
}

type staticAtomFact2Source struct {
    pairs []AtomPair
    byLeft map[string][]AtomPair
}

func newStaticAtomFact2Source(pairs []AtomPair) *staticAtomFact2Source {
    copied := append([]AtomPair(nil), pairs...)
    byLeft := make(map[string][]AtomPair)
    for _, pair := range copied {
        byLeft[pair.Left] = append(byLeft[pair.Left], pair)
    }
    return &staticAtomFact2Source{pairs: copied, byLeft: byLeft}
}

func (source *staticAtomFact2Source) Scan() []AtomPair {
    if source == nil {
        return nil
    }
    return append([]AtomPair(nil), source.pairs...)
}

func (source *staticAtomFact2Source) LookupArg1(left string) []AtomPair {
    if source == nil {
        return nil
    }
    return append([]AtomPair(nil), source.byLeft[left]...)
}

// lmdbAtomFact2Source serves arity-2 atom facts out of an LMDB
// relation artifact through the `lmdb_relation_artifact` helper.
//
// Three materialisation tiers, selected at construction and mirroring
// the F# LmdbFactSource (see templates/targets/fsharp_wam/
// lmdb_fact_source.fs.mustache):
//
//   eager  — one Scan() at construction into an in-memory arg1 index.
//            No helper process afterwards. Lowest per-lookup cost;
//            pays a full materialisation up front, so it needs the
//            demand set to fit in memory.
//   lazy   — the original behaviour: one helper invocation per lookup,
//            nothing retained. Right for segregated workloads with no
//            cross-query key reuse.
//   cached — on-demand like lazy, but memoised through a two-level
//            cache. L1 is a small hot map; L2 is a larger bounded map.
//            Once L2 is full new keys are served but not retained
//            (bounded memory, no eviction churn), matching F#.
//
// Misses are cached too: a key with no rows records an empty result so
// a repeated probe does not re-spawn the helper. The artifact is
// read-only for the lifetime of the process, so this is safe.
type lmdbAtomFact2Source struct {
    predKey string
    artifactDir string
    helperBin string
    mode string
    l1Capacity int
    l2Capacity int

    mu sync.RWMutex
    eagerAll []AtomPair
    eagerByLeft map[string][]AtomPair
    l1 map[string][]AtomPair
    l2 map[string][]AtomPair
}

func newLmdbAtomFact2Source(predKey string, artifactDir string, mode string, l2Capacity int) *lmdbAtomFact2Source {
    helperBin := os.Getenv("UW_LMDB_RELATION_ARTIFACT_BIN")
    if helperBin == "" {
        helperBin = "lmdb_relation_artifact"
    }
    switch mode {
    case "eager", "lazy", "cached":
    default:
        mode = "cached"
    }
    if l2Capacity <= 0 {
        l2Capacity = 4096
    }
    l1Capacity := l2Capacity / 8
    if l1Capacity < 64 {
        l1Capacity = 64
    }
    source := &lmdbAtomFact2Source{
        predKey: predKey,
        artifactDir: artifactDir,
        helperBin: helperBin,
        mode: mode,
        l1Capacity: l1Capacity,
        l2Capacity: l2Capacity,
    }
    if mode == "eager" {
        source.materialise()
    }
    if mode == "cached" {
        source.l1 = make(map[string][]AtomPair)
        source.l2 = make(map[string][]AtomPair)
    }
    return source
}

// materialise runs the single eager-mode Scan and builds the arg1 index.
func (source *lmdbAtomFact2Source) materialise() {
    all := source.run("scan", source.artifactDir, source.predKey)
    byLeft := make(map[string][]AtomPair, len(all))
    for _, pair := range all {
        byLeft[pair.Left] = append(byLeft[pair.Left], pair)
    }
    source.eagerAll = all
    source.eagerByLeft = byLeft
}

func (source *lmdbAtomFact2Source) Scan() []AtomPair {
    if source == nil {
        return nil
    }
    if source.mode == "eager" {
        source.mu.RLock()
        defer source.mu.RUnlock()
        return append([]AtomPair(nil), source.eagerAll...)
    }
    // lazy and cached both stream the full relation on demand; a full
    // scan is not key-addressed, so there is nothing for the key caches
    // to serve it from.
    return source.run("scan", source.artifactDir, source.predKey)
}

func (source *lmdbAtomFact2Source) LookupArg1(left string) []AtomPair {
    if source == nil {
        return nil
    }
    switch source.mode {
    case "eager":
        source.mu.RLock()
        defer source.mu.RUnlock()
        return append([]AtomPair(nil), source.eagerByLeft[left]...)
    case "cached":
        return source.lookupCached(left)
    default:
        return source.run("get", source.artifactDir, source.predKey, left)
    }
}

// lookupCached implements the L1 -> L2 -> helper dispatch. A hit in L2
// promotes the entry into L1; a helper result populates both.
func (source *lmdbAtomFact2Source) lookupCached(left string) []AtomPair {
    source.mu.RLock()
    if rows, ok := source.l1[left]; ok {
        source.mu.RUnlock()
        return append([]AtomPair(nil), rows...)
    }
    rows, ok := source.l2[left]
    source.mu.RUnlock()
    if ok {
        source.mu.Lock()
        if len(source.l1) < source.l1Capacity {
            source.l1[left] = rows
        }
        source.mu.Unlock()
        return append([]AtomPair(nil), rows...)
    }

    fetched := source.run("get", source.artifactDir, source.predKey, left)
    if fetched == nil {
        // Distinguish "no rows" from nil so a repeated miss stays cached.
        fetched = []AtomPair{}
    }
    source.mu.Lock()
    if len(source.l2) < source.l2Capacity {
        source.l2[left] = fetched
    }
    if len(source.l1) < source.l1Capacity {
        source.l1[left] = fetched
    }
    source.mu.Unlock()
    return append([]AtomPair(nil), fetched...)
}

func (source *lmdbAtomFact2Source) run(args ...string) []AtomPair {
    if source.helperBin == "" {
        return nil
    }
    output, err := exec.Command(source.helperBin, args...).Output()
    if err != nil {
        return nil
    }
    return parseAtomFact2Rows(string(output))
}

func parseAtomFact2Rows(output string) []AtomPair {
    lines := strings.Split(output, "\n")
    pairs := make([]AtomPair, 0, len(lines))
    for _, line := range lines {
        line = strings.TrimSpace(line)
        if line == "" {
            continue
        }
        cols := strings.Split(line, "\t")
        if len(cols) < 2 {
            continue
        }
        left := strings.TrimSpace(cols[0])
        right := strings.TrimSpace(cols[1])
        if left == "" || right == "" {
            continue
        }
        pairs = append(pairs, AtomPair{Left: left, Right: right})
    }
    return pairs
}

func (vm *WamState) applyForeignResult(predKey string, resultRegs []int, result Value) bool {
    tupleArity := parseForeignTupleLayout(vm.foreignResultLayout(predKey))
    if tupleArity <= 1 {
        if len(resultRegs) < 1 {
            return false
        }
        return vm.Unify(vm.getReg(resultRegs[0]), result)
    }
    tuple, ok := result.(*Compound)
    if !ok || tuple.Functor != "__tuple__" || len(tuple.Args) != tupleArity || len(resultRegs) < tupleArity {
        return false
    }
    for idx := 0; idx < tupleArity; idx++ {
        if !vm.Unify(vm.getReg(resultRegs[idx]), tuple.Args[idx]) {
            return false
        }
    }
    return true
}

func (vm *WamState) finishForeignResults(predKey string, resultRegs []int, results []Value) bool {
    if len(results) == 0 {
        return false
    }
    resumePC := vm.PC + 1
    mode := vm.foreignResultMode(predKey)
    switch mode {
    case "stream":
        return vm.finishStreamResults(predKey, resultRegs, results)
    default:
        baseRegs := vm.Regs
        trailMark := vm.TrailLen
        heapTop := vm.HeapLen
        if !vm.applyForeignResult(predKey, resultRegs, results[0]) {
            vm.unwindTrailTo(trailMark)
            vm.Regs = baseRegs
            if heapTop >= 0 && heapTop <= vm.HeapLen {
                vm.heapTrimTo(heapTop)
            }
            return false
        }
        vm.PC = resumePC
        return true
    }
}

func (vm *WamState) finishStreamResults(predKey string, resultRegs []int, results []Value) bool {
    if len(results) == 0 {
        return false
    }
    resumePC := vm.PC + 1
    var baseRegs [320]Value
    baseRegs = vm.Regs
    // Capture stack length and the env-pointer instead of cloning
    // the stack — backtrack truncates to baseStackLen and restores
    // baseE the same way the regular CP path does.
    baseStackLen := len(vm.Stack)
    baseE := vm.E
    trailMark := vm.TrailLen
    heapTop := vm.HeapLen
    for idx, result := range results {
        vm.unwindTrailTo(trailMark)
        vm.Regs = baseRegs
        if baseStackLen <= len(vm.Stack) {
            vm.Stack = vm.Stack[:baseStackLen]
        }
        vm.E = baseE
        if heapTop >= 0 && heapTop <= vm.HeapLen {
            vm.heapTrimTo(heapTop)
        }
        vm.Halted = false
        vm.CurrentStruct = nil
        vm.CurrentList = nil
        if !vm.applyForeignResult(predKey, resultRegs, result) {
            continue
        }
        if idx+1 < len(results) {
            remaining := append([]Value(nil), results[idx+1:]...)
            ycount := vm.MaxYReg - 200
            if ycount < 0 {
                ycount = 0
            }
            savedRegs := make([]Value, 8+ycount)
            copy(savedRegs[:8], baseRegs[:8])
            if ycount > 0 {
                copy(savedRegs[8:], baseRegs[200:200+ycount])
            }
            vm.ChoicePoints = append(vm.ChoicePoints, ChoicePoint{
                NextPC: resumePC,
                ResumePC: resumePC,
                CP: vm.CP,
                E: baseE,
                StackLen: baseStackLen,
                SavedRegs: savedRegs,
                HeapTop: heapTop,
                TrailMark: trailMark,
                ForeignPredKey: predKey,
                ForeignResultRegs: append([]int(nil), resultRegs...),
                ForeignResults: remaining,
                PendingB0: vm.PendingB0,
                CutB0Stack: vm.copyCutB0Stack(),
                YSaves: vm.copyYSaveStack(),
                YSaveLen: len(vm.YSaves),
            })
        }
        vm.PC = resumePC
        return true
    }
    // The last candidate can fail only after binding an earlier tuple
    // component (notably when output registers alias).  There is no next
    // loop iteration to perform the usual reset, so restore the complete
    // pre-stream snapshot before reporting exhaustion.
    vm.unwindTrailTo(trailMark)
    vm.Regs = baseRegs
    if baseStackLen <= len(vm.Stack) {
        vm.Stack = vm.Stack[:baseStackLen]
    }
    vm.E = baseE
    if heapTop >= 0 && heapTop <= vm.HeapLen {
        vm.heapTrimTo(heapTop)
    }
    vm.Halted = false
    vm.CurrentStruct = nil
    vm.CurrentList = nil
    return false
}

func (vm *WamState) executeIndexedAtomFact2(predKey string) bool {
    key, ok := valueAsAtomString(vm, vm.getReg(0))
    if !ok {
        return false
    }
    pairs := vm.Ctx.IndexedAtomFactPairs[predKey]
    if source, ok := vm.Ctx.AtomFact2Sources[predKey]; ok {
        pairs = source.LookupArg1(key)
    }
    results := make([]Value, 0)
    for _, pair := range pairs {
        if pair.Left == key {
            results = append(results, internAtom(pair.Right))
        }
    }
    return vm.finishStreamResults(predKey, []int{1}, results)
}

func valueAsAtomString(vm *WamState, v Value) (string, bool) {
    val := vm.deref(v)
    atom, ok := val.(*Atom)
    if !ok {
        return "", false
    }
    return atom.Name, true
}

func valueAsInteger(vm *WamState, v Value) (int64, bool) {
    val := vm.deref(v)
    integer, ok := val.(*Integer)
    if !ok {
        return 0, false
    }
    return integer.Val, true
}

func valueAsFloat(vm *WamState, v Value) (float64, bool) {
    val := vm.deref(v)
    switch n := val.(type) {
    case *Integer:
        return float64(n.Val), true
    case *Float:
        return n.Val, true
    default:
        return 0, false
    }
}

func listAsSlice(vm *WamState, v Value) ([]Value, bool) {
    val := vm.deref(v)
    list, ok := val.(*List)
    if !ok {
        return nil, false
    }
    return list.Elements, true
}

func (vm *WamState) collectNativeListSuffixes(items []Value, out *[]Value) {
    for idx := 0; idx <= len(items); idx++ {
        suffix := append([]Value(nil), items[idx:]...)
        *out = append(*out, &List{Elements: suffix})
    }
}

func tupleValue(items ...Value) Value {
    return &Compound{Functor: "__tuple__", Args: items}
}

func atomAdjacency(pairs []AtomPair) map[string][]string {
    adjacency := make(map[string][]string)
    for _, pair := range pairs {
        adjacency[pair.Left] = append(adjacency[pair.Left], pair.Right)
    }
    return adjacency
}

func weightedAdjacency(triples []WeightedEdgeTriple) map[string][]WeightedEdgeTriple {
    adjacency := make(map[string][]WeightedEdgeTriple)
    for _, triple := range triples {
        adjacency[triple.Left] = append(adjacency[triple.Left], triple)
    }
    return adjacency
}

func (vm *WamState) collectNativeTransitiveClosureResults(source string, pairs []AtomPair) []Value {
    adjacency := atomAdjacency(pairs)
    visited := make(map[string]bool)
    queue := append([]string(nil), adjacency[source]...)
    results := make([]Value, 0)
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        if visited[node] {
            continue
        }
        visited[node] = true
        results = append(results, internAtom(node))
        queue = append(queue, adjacency[node]...)
    }
    return results
}

func (vm *WamState) collectNativeTransitiveDistanceResults(source string, pairs []AtomPair) []Value {
    // dist+ (docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md): BFS shortest
    // positive distance. Visited tracks edge-discovered nodes — do not seed
    // with source (Source appears only for self-loop / nonempty cycle).
    adjacency := atomAdjacency(pairs)
    visited := make(map[string]bool)
    type qd struct {
        node string
        dist int
    }
    queue := []qd{{source, 0}}
    results := make([]Value, 0)
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        for _, next := range adjacency[current.node] {
            if visited[next] {
                continue
            }
            visited[next] = true
            nextDist := current.dist + 1
            queue = append(queue, qd{next, nextDist})
            results = append(results, tupleValue(
                internAtom(next),
                &Integer{Val: int64(nextDist)},
            ))
        }
    }
    return results
}

func (vm *WamState) collectNativeTransitiveParentDistanceResults(source string, pairs []AtomPair) []Value {
    // Shortest-positive parents
    // (docs/design/WAM_TRANSITIVE_PARENT_DISTANCE4_CONTRACT.md): BFS with
    // parent sets. dist tracks edge-discovered nodes — do not seed with
    // source. Equal-shortest parents are all emitted.
    adjacency := atomAdjacency(pairs)
    dist := make(map[string]int)
    parents := make(map[string]map[string]bool)
    type qd struct {
        node string
        dist int
    }
    queue := []qd{{source, 0}}
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        nextDist := current.dist + 1
        for _, next := range adjacency[current.node] {
            if d0, ok := dist[next]; ok {
                if d0 == nextDist {
                    parents[next][current.node] = true
                }
                continue
            }
            dist[next] = nextDist
            parents[next] = map[string]bool{current.node: true}
            queue = append(queue, qd{next, nextDist})
        }
    }
    keys := make([]string, 0, len(dist))
    for k := range dist {
        keys = append(keys, k)
    }
    sort.Strings(keys)
    results := make([]Value, 0)
    for _, target := range keys {
        d := dist[target]
        pars := make([]string, 0, len(parents[target]))
        for p := range parents[target] {
            pars = append(pars, p)
        }
        sort.Strings(pars)
        for _, parent := range pars {
            results = append(results, tupleValue(
                internAtom(target),
                internAtom(parent),
                &Integer{Val: int64(d)},
            ))
        }
    }
    return results
}

func (vm *WamState) collectNativeTransitiveStepParentDistanceResults(source string, pairs []AtomPair) []Value {
    // Shortest-positive correlated step/parent
    // (docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md).
    // Level-synchronous BFS stores correlated (Step, Parent) pairs per
    // Target — never an independent Step×Parent cross-product. Dist is
    // not seeded with Source.
    adjacency := atomAdjacency(pairs)
    dist := make(map[string]int)
    pairSets := make(map[string]map[[2]string]bool)
    type qd struct {
        node string
        d    int
    }
    queue := []qd{{source, 0}}
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        nd := current.d + 1
        for _, next := range adjacency[current.node] {
            var cands [][2]string
            if current.node == source {
                cands = [][2]string{{next, source}}
            } else {
                seenStep := make(map[string]bool)
                for pair := range pairSets[current.node] {
                    if !seenStep[pair[0]] {
                        seenStep[pair[0]] = true
                        cands = append(cands, [2]string{pair[0], current.node})
                    }
                }
            }
            if d0, ok := dist[next]; !ok {
                dist[next] = nd
                set := make(map[[2]string]bool)
                for _, c := range cands {
                    set[c] = true
                }
                pairSets[next] = set
                queue = append(queue, qd{next, nd})
            } else if d0 == nd {
                set := pairSets[next]
                if set == nil {
                    set = make(map[[2]string]bool)
                    pairSets[next] = set
                }
                for _, c := range cands {
                    set[c] = true
                }
            }
        }
    }
    results := make([]Value, 0)
    targets := make([]string, 0, len(dist))
    for t := range dist {
        targets = append(targets, t)
    }
    sort.Strings(targets)
    for _, t := range targets {
        d := dist[t]
        type sp struct{ step, parent string }
        list := make([]sp, 0, len(pairSets[t]))
        for pair := range pairSets[t] {
            list = append(list, sp{pair[0], pair[1]})
        }
        sort.Slice(list, func(i, j int) bool {
            if list[i].step != list[j].step {
                return list[i].step < list[j].step
            }
            return list[i].parent < list[j].parent
        })
        for _, p := range list {
            results = append(results, tupleValue(
                internAtom(t),
                internAtom(p.step),
                internAtom(p.parent),
                &Integer{Val: int64(d)},
            ))
        }
    }
    return results
}

func pickShortestCandidate(dist map[string]float64, settled map[string]bool) (string, bool) {
    bestNode := ""
    bestDist := 0.0
    found := false
    for node, d := range dist {
        if settled[node] {
            continue
        }
        if !found || d < bestDist || (d == bestDist && node < bestNode) {
            bestNode = node
            bestDist = d
            found = true
        }
    }
    return bestNode, found
}

// heuristicLookup returns the min valid (from,target) weight.
// Missing → 0. ok=false on malformed relevant heuristic (NaN/Inf/neg).
func heuristicLookup(triples []WeightedEdgeTriple, from string, target string) (float64, bool) {
    best := 0.0
    saw := false
    for _, triple := range triples {
        if triple.Left != from || triple.Right != target {
            continue
        }
        w := triple.Weight
        if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
            return 0, false
        }
        if !saw || w < best {
            best = w
            saw = true
        }
    }
    if saw {
        return best, true
    }
    return 0, true
}

// collectNativeWeightedShortestPathResults — finite nonnegative Dijkstra
// for weighted_shortest_path3
// (docs/design/WAM_WEIGHTED_SHORTEST_PATH3_CONTRACT.md).
// Emit one (Target, float64) per reachable non-Source target. Source is
// never emitted (even via self-loop / cycle). Indexed triples are
// atom-keyed; a reachable invalid weight (NaN / Inf / negative) fails
// the complete call (nil result → foreign fail).
func (vm *WamState) collectNativeWeightedShortestPathResults(source string, triples []WeightedEdgeTriple) []Value {
    adjacency := weightedAdjacency(triples)
    dist := map[string]float64{source: 0}
    settled := make(map[string]bool)
    results := make([]Value, 0)
    for {
        current, ok := pickShortestCandidate(dist, settled)
        if !ok {
            break
        }
        settled[current] = true
        if current != source {
            results = append(results, tupleValue(
                internAtom(current),
                &Float{Val: dist[current]},
            ))
        }
        for _, edge := range adjacency[current] {
            w := edge.Weight
            if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
                return nil
            }
            candidate := dist[current] + w
            prev, exists := dist[edge.Right]
            if !exists || candidate < prev {
                dist[edge.Right] = candidate
            }
        }
    }
    return results
}

// collectNativeAstarShortestPathResult — correctness-safe A* for
// astar_shortest_path4 (docs/design/WAM_ASTAR_SHORTEST_PATH4_CONTRACT.md).
// Final cost is always the finite Dijkstra minimum (settled by g-cost).
// Missing h = 0.0. Source=Target → 0.0. Malformed reachable edge or
// relevant heuristic fails the complete call (nil).
func (vm *WamState) collectNativeAstarShortestPathResult(source string, target string, dim int64, weighted []WeightedEdgeTriple, direct []WeightedEdgeTriple) []Value {
    if dim <= 0 {
        return nil
    }
    if source == target {
        return []Value{&Float{Val: 0}}
    }
    adjacency := weightedAdjacency(weighted)
    gScore := map[string]float64{source: 0}
    open := map[string]bool{source: true}
    for len(open) > 0 {
        current := ""
        bestG := 0.0
        bestF := 0.0
        found := false
        for node := range open {
            h, ok := heuristicLookup(direct, node, target)
            if !ok {
                return nil
            }
            g := gScore[node]
            // Secondary scheduling key only (Dim/h); settle by g.
            f := math.Pow(g, float64(dim)) + math.Pow(h, float64(dim))
            if !found || g < bestG || (g == bestG && (f < bestF || (f == bestF && node < current))) {
                current = node
                bestG = g
                bestF = f
                found = true
            }
        }
        if !found {
            break
        }
        delete(open, current)
        // Settled by g-cost (Dijkstra). Safe even if h overestimates.
        if current == target {
            return []Value{&Float{Val: gScore[current]}}
        }
        for _, edge := range adjacency[current] {
            w := edge.Weight
            if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
                return nil
            }
            candidate := gScore[current] + w
            prev, exists := gScore[edge.Right]
            if !exists || candidate < prev {
                gScore[edge.Right] = candidate
                open[edge.Right] = true
            }
        }
    }
    return nil
}

// listAsAtomStrings walks a Prolog cons-list and pulls each element
// out as an atom string. Uses vm.listToSlice rather than listAsSlice
// because the latter returns the 2-element [head, tail] cons cell,
// not the full flattened list — `category_ancestor`'s visited list
// can be many cells deep, and reading just the first cons would
// silently truncate it (causing the kernel to walk paths through
// already-visited nodes and produce wrong hop counts).
func listAsAtomStrings(vm *WamState, v Value) ([]string, bool) {
    items, ok := vm.listToSlice(v)
    if !ok {
        return nil, false
    }
    out := make([]string, 0, len(items))
    for _, item := range items {
        s, ok := valueAsAtomString(vm, item)
        if !ok {
            return nil, false
        }
        out = append(out, s)
    }
    return out, true
}

// collectNativeCategoryAncestorHops walks the parent edges of `cat`
// looking for `root`, emitting one hop count per matching path. Mirrors
// the Rust implementation at wam_rust_target.pl:1797 — same DFS, same
// max-depth semantics, same visited-set skip. The adjacency map is
// built once at the top level and threaded into the recursive helper
// to avoid rebuilding it per call.
func (vm *WamState) collectNativeCategoryAncestorHops(cat string, root string, visited []string, maxDepth int, pairs []AtomPair) []int64 {
    adjacency := atomAdjacency(pairs)
    var out []int64
    vm.collectNativeCategoryAncestorHopsRec(cat, root, visited, maxDepth, adjacency, &out)
    return out
}

func (vm *WamState) collectNativeCategoryAncestorHopsRec(cat string, root string, visited []string, maxDepth int, adjacency map[string][]string, out *[]int64) {
    rootSeen := false
    for _, v := range visited {
        if v == root {
            rootSeen = true
            break
        }
    }
    parents := adjacency[cat]
    if !rootSeen {
        for _, p := range parents {
            if p == root {
                *out = append(*out, 1)
                break
            }
        }
    }
    if len(visited) >= maxDepth {
        return
    }
    for _, parent := range parents {
        skip := false
        for _, v := range visited {
            if v == parent {
                skip = true
                break
            }
        }
        if skip {
            continue
        }
        nextVisited := make([]string, 0, len(visited)+1)
        nextVisited = append(nextVisited, parent)
        nextVisited = append(nextVisited, visited...)
        before := len(*out)
        vm.collectNativeCategoryAncestorHopsRec(parent, root, nextVisited, maxDepth, adjacency, out)
        for i := before; i < len(*out); i++ {
            (*out)[i] += 1
        }
    }
}

func (vm *WamState) executeForeignPredicate(pred string, arity int) bool {
    predKey := fmt.Sprintf("%s/%d", pred, arity)
    nativeKind, ok := vm.Ctx.ForeignNativeKinds[predKey]
    if !ok {
        return false
    }
    switch nativeKind {
    case "countdown_sum2":
        n, ok := valueAsInteger(vm, vm.getReg(0))
        if !ok {
            return false
        }
        sum := n * (n + 1) / 2
        return vm.finishForeignResults(predKey, []int{1}, []Value{&Integer{Val: sum}})
    case "list_suffix2":
        items, ok := listAsSlice(vm, vm.getReg(0))
        if !ok {
            return false
        }
        suffixes := make([]Value, 0, len(items)+1)
        vm.collectNativeListSuffixes(items, &suffixes)
        packed := make([]Value, 0, len(suffixes))
        for _, suffix := range suffixes {
            packed = append(packed, suffix)
        }
        return vm.finishForeignResults(predKey, []int{1}, packed)
    case "list_suffixes2":
        items, ok := listAsSlice(vm, vm.getReg(0))
        if !ok {
            return false
        }
        suffixes := make([]Value, 0, len(items)+1)
        vm.collectNativeListSuffixes(items, &suffixes)
        return vm.finishForeignResults(predKey, []int{1}, []Value{&List{Elements: suffixes}})
    case "transitive_closure2":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveClosureResults(source, pairs)
        return vm.finishForeignResults(predKey, []int{1}, results)
    case "transitive_distance3":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []int{1, 2}, results)
    case "transitive_parent_distance4":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveParentDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []int{1, 2, 3}, results)
    case "transitive_step_parent_distance5":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveStepParentDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []int{1, 2, 3, 4}, results)
    case "category_ancestor":
        // category_ancestor(Cat, Root, Hops, Visited) — output is Hops
        // (A3, reg index 2). Cat=A1, Root=A2, Visited=A4. The WAM
        // semantics: walk parents of Cat up to max_depth hops; emit one
        // integer hop count per path that reaches Root, skipping any
        // node already in Visited. See
        // src/unifyweaver/core/recursive_kernel_detection.pl:135 for the
        // canonical register layout and call spec.
        cat, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        root, ok := valueAsAtomString(vm, vm.getReg(1))
        if !ok {
            return false
        }
        visited, ok := listAsAtomStrings(vm, vm.getReg(3))
        if !ok {
            return false
        }
        maxDepth := vm.foreignUsizeConfig(predKey, "max_depth")
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        hops := vm.collectNativeCategoryAncestorHops(cat, root, visited, maxDepth, pairs)
        if len(hops) == 0 {
            return false
        }
        results := make([]Value, len(hops))
        for i, h := range hops {
            results[i] = &Integer{Val: h}
        }
        return vm.finishForeignResults(predKey, []int{2}, results)
    case "weighted_shortest_path3":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        weightPred := vm.foreignStringConfig(predKey, "weight_pred")
        triples := vm.Ctx.IndexedWeightedEdgeTriples[weightPred]
        results := vm.collectNativeWeightedShortestPathResults(source, triples)
        return vm.finishForeignResults(predKey, []int{1, 2}, results)
    case "astar_shortest_path4":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        target, ok := valueAsAtomString(vm, vm.getReg(1))
        if !ok {
            return false
        }
        dim, ok := valueAsInteger(vm, vm.getReg(2))
        if !ok || dim <= 0 {
            return false
        }
        weightPred := vm.foreignStringConfig(predKey, "weight_pred")
        directPred := vm.foreignStringConfig(predKey, "direct_dist_pred")
        weighted := vm.Ctx.IndexedWeightedEdgeTriples[weightPred]
        direct := vm.Ctx.IndexedWeightedEdgeTriples[directPred]
        results := vm.collectNativeAstarShortestPathResult(source, target, dim, weighted, direct)
        return vm.finishForeignResults(predKey, []int{3}, results)
    default:
        return false
    }
}

