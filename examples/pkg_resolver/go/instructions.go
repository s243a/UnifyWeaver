package wam

type Instruction interface {
	instrTag()
}

// Data structures for Switch instructions
type ConstCase struct {
	Val   Value
	Label string
}

type StructCase struct {
	Functor string
	Label   string
}

type GetConstant struct { C Value; Ai int }
func (i *GetConstant) instrTag() {}

type GetVariable struct { Xn int; Ai int }
func (i *GetVariable) instrTag() {}

type GetValue struct { Xn int; Ai int }
func (i *GetValue) instrTag() {}

type GetStructure struct { Functor string; Ai int }
func (i *GetStructure) instrTag() {}

type GetList struct { Ai int }
func (i *GetList) instrTag() {}

type UnifyVariable struct { Xn int }
func (i *UnifyVariable) instrTag() {}

type UnifyValue struct { Xn int }
func (i *UnifyValue) instrTag() {}

type UnifyConstant struct { C Value }
func (i *UnifyConstant) instrTag() {}

type PutConstant struct { C Value; Ai int }
func (i *PutConstant) instrTag() {}

type PutVariable struct { Xn int; Ai int }
func (i *PutVariable) instrTag() {}

type PutValue struct { Xn int; Ai int }
func (i *PutValue) instrTag() {}

type PutStructure struct { Functor string; Ai int }
func (i *PutStructure) instrTag() {}

type PutList struct { Ai int }
func (i *PutList) instrTag() {}

type SetVariable struct { Xn int }
func (i *SetVariable) instrTag() {}

type SetValue struct { Xn int }
func (i *SetValue) instrTag() {}

type SetConstant struct { C Value }
func (i *SetConstant) instrTag() {}

type Allocate struct{}
func (i *Allocate) instrTag() {}

type Deallocate struct{}
func (i *Deallocate) instrTag() {}

type Call struct { Pred string; Arity int }
func (i *Call) instrTag() {}

type CallForeign struct { Pred string; Arity int }
func (i *CallForeign) instrTag() {}

type CallIndexedAtomFact2 struct { Pred string }
func (i *CallIndexedAtomFact2) instrTag() {}

type CallPc struct { TargetPC int; Arity int }
func (i *CallPc) instrTag() {}

type Execute struct { Pred string }
func (i *Execute) instrTag() {}

type ExecutePc struct { TargetPC int }
func (i *ExecutePc) instrTag() {}

type Jump struct { Label string }
func (i *Jump) instrTag() {}

type JumpPc struct { TargetPC int }
func (i *JumpPc) instrTag() {}

type CutIte struct{}
func (i *CutIte) instrTag() {}

// M17 soft cut. GetLevel snapshots the choicepoint-stack height (before an
// if-then-else / negation try_me_else); Cut truncates the choicepoint stack
// back to that snapshot at the commit site, removing the ITE/negation
// choicepoint AND any CPs the condition pushed above it. Replaces the legacy
// CutIte (which dropped only the single topmost CP).
//
// Reg names the Y register the shared emitter reserved for the barrier, but
// it is only a KEY: the level is kept on the if-then-else's own choice point
// (ChoicePoint.Levels), never written into Regs[Reg]. The emitter plants
// `get_level Yn` in clauses that get no `allocate`, and Y slots are global
// here, so writing the register clobbered the caller's Y (WAM_FLEET_GAPS A2,
// frameless-Y-write form). See recordIteLevel / lookupIteLevel in state.go.
type GetLevel struct { Reg int }
func (i *GetLevel) instrTag() {}

type Cut struct { Reg int }
func (i *Cut) instrTag() {}

type BeginAggregate struct { AggType string; ValueReg int; ResultReg int }
func (i *BeginAggregate) instrTag() {}

type EndAggregate struct { ValueReg int }
func (i *EndAggregate) instrTag() {}

// GetArgInto implements the `arg N, Src, Dest` WAM instruction: extract
// argument N (1-based) of the compound/structure/list term held in
// register Src into register Dest. Emitted by arg/3 when the source term
// is a runtime value already in a register (e.g. arg(1, C, Y) after
// copy_term), as opposed to the arg/3 builtin_call form.
type GetArgInto struct { Index int; Src int; Dest int }
func (i *GetArgInto) instrTag() {}

type Proceed struct{}
func (i *Proceed) instrTag() {}

type BuiltinCall struct { Op string; Arity int }
func (i *BuiltinCall) instrTag() {}

// BuiltinExecute is the last-call form of BuiltinCall, emitted for
// `execute <builtin>/N` (a clause whose final goal is a builtin, e.g.
// `s(X) :- succ(1, X).`). BuiltinCall advances to the following
// instruction; `execute` has no following instruction, so running the
// builtin and then falling off the end of the predicate would bind the
// output arguments correctly and still report failure. BuiltinExecute
// runs the builtin and then performs Proceed's return-to-caller step.
//
// Kept as a distinct instruction rather than emitting BuiltinCall +
// Proceed so the WAM-text line/instruction mapping stays 1:1 — inserting
// an extra instruction would shift every later PC and label index in the
// shared code table.
type BuiltinExecute struct { Op string; Arity int }
func (i *BuiltinExecute) instrTag() {}

type TryMeElse struct { Label string; Arity int }
func (i *TryMeElse) instrTag() {}

type TryMeElsePc struct { NextPC int; Arity int }
func (i *TryMeElsePc) instrTag() {}

type RetryMeElse struct { Label string; Arity int }
func (i *RetryMeElse) instrTag() {}

type RetryMeElsePc struct { NextPC int; Arity int }
func (i *RetryMeElsePc) instrTag() {}

type TrustMe struct{}
func (i *TrustMe) instrTag() {}

type SwitchOnConstant struct { Cases []ConstCase }
func (i *SwitchOnConstant) instrTag() {}

type ConstPcCase struct {
	Val      Value
	TargetPC int
}

type SwitchOnConstantPc struct { Cases []ConstPcCase }
func (i *SwitchOnConstantPc) instrTag() {}

type SwitchOnStructure struct { Cases []StructCase }
func (i *SwitchOnStructure) instrTag() {}

type StructPcCase struct {
	Functor  string
	TargetPC int
}

type SwitchOnStructurePc struct { Cases []StructPcCase }
func (i *SwitchOnStructurePc) instrTag() {}

type SwitchOnConstantA2 struct { Cases []ConstCase }
func (i *SwitchOnConstantA2) instrTag() {}

type SwitchOnConstantA2Pc struct { Cases []ConstPcCase }
func (i *SwitchOnConstantA2Pc) instrTag() {}
