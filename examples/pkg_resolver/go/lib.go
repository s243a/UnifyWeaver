package wam

func setupSharedForeignPredicates(vm *WamState) {

}

var sharedWamCodeRaw = []Instruction{
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 200, Ai: 2},
        &GetVariable{Xn: 105, Ai: 3},
        &PutValue{Xn: 105, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &PutVariable{Xn: 203, Ai: 2},
        &Call{Pred: "member_selected/3", Arity: 3},
        &TryMeElse{Label: "L_ite_else_1", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 200, Ai: 2},
        &PutValue{Xn: 202, Ai: 3},
        &Call{Pred: "conflicts_in/4", Arity: 4},
        &Jump{Label: "L_ite_cont_1"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &Call{Pred: "conflicts_in/4", Arity: 4},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 107},
        &GetStructure{Functor: "a/3", Ai: 107},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetVariable{Xn: 206, Ai: 3},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_2", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_2"},
        &TrustMe{},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_3", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_3"},
        &TrustMe{},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Call{Pred: "active_member/4", Arity: 4},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_alias_list_2_2_body"}, {Functor: "catalog/10", Label: "L_alias_list_2_3_body"}, {Functor: "icat/3", Label: "L_alias_list_2_4_body"}}},
        &TryMeElse{Label: "L_alias_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_alias_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 108, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_alias_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 108, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "alias_list/2"},
        &TryMeElse{Label: "L_alias_lookup_3_2", Arity: 3},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetValue{Xn: 100, Ai: 2},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 105},
        &GetStructure{Functor: "alias/2", Ai: 105},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_4", Arity: 3},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_4"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Call{Pred: "alias_lookup/3", Arity: 3},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 105, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 204, Ai: 3},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetVariable{Xn: 201},
        &SetVariable{Xn: 202},
        &PutValue{Xn: 105, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &PutValue{Xn: 204, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "provides_sat/5"},
        &TryMeElse{Label: "L_already_satisfied_4_2", Arity: 4},
        &Allocate{},
        &GetVariable{Xn: 102, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 104, Ai: 2},
        &GetVariable{Xn: 201, Ai: 3},
        &PutValue{Xn: 103, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "satisfies/2"},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetVariable{Xn: 103, Ai: 3},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 103, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "already_provided/4"},
        &TryMeElse{Label: "L_alt_reasons_4_2", Arity: 4},
        &GetVariable{Xn: 100, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetConstant{C: wamAtom____0, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 203, Ai: 0},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 107},
        &GetStructure{Functor: "dep/2", Ai: 107},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 204},
        &GetVariable{Xn: 205, Ai: 2},
        &GetList{Ai: 3},
        &UnifyVariable{Xn: 108},
        &GetStructure{Functor: "alt/2", Ai: 108},
        &UnifyValue{Xn: 200},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 206},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_5", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &PutValue{Xn: 202, Ai: 4},
        &Call{Pred: "explain_alt/5", Arity: 5},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_5"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutConstant{C: wamAtom_unsatisfiable_1, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "alt_reasons/4"},
        &TryMeElse{Label: "L_audit_holds_4_2", Arity: 4},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetValue{Xn: 101, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 107},
        &GetStructure{Functor: "hold/3", Ai: 107},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 205, Ai: 2},
        &GetVariable{Xn: 206, Ai: 3},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_6", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom_blanket_2, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 207},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_7", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &Call{Pred: "tight_base_revdep/2", Arity: 2},
        &Cut{Reg: 208},
        &PutVariable{Xn: 204, Ai: 0},
        &PutStructure{Functor: "audit/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetVariable{Xn: 110},
        &PutStructure{Functor: "suggest/1", Ai: 110},
        &SetConstant{C: wamAtom_abi_anchor_3},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_7"},
        &TrustMe{},
        &PutVariable{Xn: 204, Ai: 0},
        &PutStructure{Functor: "audit/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetConstant{C: wamAtom_over_frozen_4},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_6"},
        &TrustMe{},
        &PutVariable{Xn: 204, Ai: 0},
        &PutStructure{Functor: "audit/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetVariable{Xn: 110},
        &PutStructure{Functor: "held/1", Ai: 110},
        &SetValue{Xn: 201},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutStructure{Functor: "[|]/2", Ai: 2},
        &SetValue{Xn: 204},
        &SetValue{Xn: 205},
        &PutValue{Xn: 206, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "audit_holds/4"},
        &Allocate{},
        &GetVariable{Xn: 103, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &PutValue{Xn: 103, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "base_list/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "scan_base_holds/3", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_base_list_2_2_body"}, {Functor: "catalog/10", Label: "L_base_list_2_3_body"}, {Functor: "icat/3", Label: "L_base_list_2_4_body"}}},
        &TryMeElse{Label: "L_base_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetValue{Xn: 103, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_base_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 103, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_base_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 103, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "base_list/2"},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutVariable{Xn: 102, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "base_ver/3"},
        &Allocate{},
        &GetVariable{Xn: 103, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &PutValue{Xn: 103, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "base_holds/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "hold_reason/3"},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 205, Ai: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "base_list/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &Call{Pred: "layers_list/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 203, Ai: 2},
        &BuiltinCall{Op: "append/3", Arity: 3},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "lookup_held/3"},
        &TryMeElse{Label: "L_blocked_acc_5_2", Arity: 5},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 106},
        &GetStructure{Functor: "alternatives/1", Ai: 106},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 107},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 205, Ai: 3},
        &GetVariable{Xn: 203, Ai: 4},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutVariable{Xn: 204, Ai: 3},
        &Call{Pred: "alt_reasons/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 109},
        &SetValue{Xn: 205},
        &PutStructure{Functor: "blocked/1", Ai: 109},
        &SetVariable{Xn: 110},
        &PutStructure{Functor: "alternatives/1", Ai: 110},
        &SetValue{Xn: 204},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 203, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 204},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 209, Ai: 2},
        &GetVariable{Xn: 202, Ai: 3},
        &GetVariable{Xn: 211, Ai: 4},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_8", Arity: 5},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_9", Arity: 5},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 213},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_9"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 212},
        &PutVariable{Xn: 210, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 113},
        &SetValue{Xn: 202},
        &PutStructure{Functor: "blocked/3", Ai: 113},
        &SetValue{Xn: 204},
        &SetVariable{Xn: 114},
        &SetVariable{Xn: 115},
        &PutStructure{Functor: "needs/1", Ai: 114},
        &SetValue{Xn: 205},
        &PutStructure{Functor: "base_has/1", Ai: 115},
        &SetValue{Xn: 200},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_8"},
        &TrustMe{},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_10", Arity: 5},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutVariable{Xn: 201, Ai: 3},
        &Call{Pred: "virtual_provider_ceilings/4", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &Cut{Reg: 213},
        &PutVariable{Xn: 210, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 113},
        &SetValue{Xn: 202},
        &PutStructure{Functor: "blocked/3", Ai: 113},
        &SetValue{Xn: 204},
        &SetVariable{Xn: 114},
        &SetVariable{Xn: 115},
        &PutStructure{Functor: "needs/1", Ai: 114},
        &SetValue{Xn: 205},
        &PutStructure{Functor: "providers/1", Ai: 115},
        &SetValue{Xn: 201},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_10"},
        &TrustMe{},
        &PutVariable{Xn: 210, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 214},
        &TryMeElse{Label: "L_ite_else_11", Arity: 5},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Cut{Reg: 214},
        &PutValue{Xn: 211, Ai: 0},
        &PutValue{Xn: 210, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_11"},
        &TrustMe{},
        &GetLevel{Reg: 215},
        &TryMeElse{Label: "L_ite_else_12", Arity: 5},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutVariable{Xn: 206, Ai: 3},
        &PutVariable{Xn: 207, Ai: 4},
        &Call{Pred: "walk_pkg_for_blocked/5", Arity: 5},
        &Cut{Reg: 215},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutVariable{Xn: 208, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &PutStructure{Functor: "[|]/2", Ai: 2},
        &SetValue{Xn: 204},
        &SetValue{Xn: 209},
        &PutValue{Xn: 210, Ai: 3},
        &PutValue{Xn: 211, Ai: 4},
        &Call{Pred: "blocked_acc_list/5", Arity: 5},
        &Jump{Label: "L_ite_cont_12"},
        &TrustMe{},
        &PutValue{Xn: 211, Ai: 0},
        &PutValue{Xn: 210, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_blocked_acc_list_5_2", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetValue{Xn: 102, Ai: 4},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 106, Ai: 3},
        &GetVariable{Xn: 204, Ai: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 105, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 106, Ai: 3},
        &PutVariable{Xn: 203, Ai: 4},
        &Call{Pred: "blocked_acc/5", Arity: 5},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &PutValue{Xn: 204, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "blocked_acc_list/5"},
        &TryMeElse{Label: "L_blocked_from_4_2", Arity: 4},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 105},
        &GetStructure{Functor: "alternatives/1", Ai: 105},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 106},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutVariable{Xn: 204, Ai: 3},
        &Call{Pred: "alt_reasons/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutStructure{Functor: "blocked/1", Ai: 1},
        &SetVariable{Xn: 108},
        &PutStructure{Functor: "alternatives/1", Ai: 108},
        &SetValue{Xn: 204},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_blocked_from_4_3", Arity: 4},
        &Allocate{},
        &GetVariable{Xn: 104, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 105, Ai: 2},
        &GetVariable{Xn: 200, Ai: 3},
        &PutValue{Xn: 104, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutVariable{Xn: 203, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &GetLevel{Reg: 204},
        &TryMeElse{Label: "L_ite_else_13", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 204},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_13"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutStructure{Functor: "blocked/3", Ai: 1},
        &SetValue{Xn: 201},
        &SetVariable{Xn: 107},
        &SetVariable{Xn: 108},
        &PutStructure{Functor: "needs/1", Ai: 107},
        &SetValue{Xn: 202},
        &PutStructure{Functor: "base_has/1", Ai: 108},
        &SetValue{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_blocked_from_4_4", Arity: 4},
        &Allocate{},
        &GetVariable{Xn: 104, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 105, Ai: 2},
        &GetVariable{Xn: 200, Ai: 3},
        &PutValue{Xn: 104, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutVariable{Xn: 203, Ai: 3},
        &Call{Pred: "virtual_provider_ceilings/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutStructure{Functor: "blocked/3", Ai: 1},
        &SetValue{Xn: 201},
        &SetVariable{Xn: 107},
        &SetVariable{Xn: 108},
        &PutStructure{Functor: "needs/1", Ai: 107},
        &SetValue{Xn: 202},
        &PutStructure{Functor: "providers/1", Ai: 108},
        &SetValue{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 204, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 206},
        &UnifyVariable{Xn: 200},
        &GetVariable{Xn: 207, Ai: 2},
        &GetVariable{Xn: 208, Ai: 3},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_14", Arity: 4},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Cut{Reg: 209},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_14"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 200, Ai: 2},
        &PutVariable{Xn: 201, Ai: 3},
        &PutVariable{Xn: 202, Ai: 4},
        &Call{Pred: "walk_pkg_for_blocked/5", Arity: 5},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutVariable{Xn: 203, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutVariable{Xn: 205, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutStructure{Functor: "[|]/2", Ai: 2},
        &SetValue{Xn: 206},
        &SetValue{Xn: 207},
        &PutValue{Xn: 208, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "blocked_from/4"},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 206, Ai: 2},
        &GetVariable{Xn: 205, Ai: 3},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_15", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 0}, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &Cut{Reg: 211},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom_t_5, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_15"},
        &TrustMe{},
        &PutVariable{Xn: 201, Ai: 0},
        &PutStructure{Functor: "///2", Ai: 1},
        &SetVariable{Xn: 112},
        &SetConstant{C: &Integer{Val: 2}},
        &PutStructure{Functor: "-/2", Ai: 112},
        &SetValue{Xn: 200},
        &SetConstant{C: &Integer{Val: 1}},
        &BuiltinCall{Op: "is/2", Arity: 2},
        &PutVariable{Xn: 203, Ai: 0},
        &PutStructure{Functor: "-/2", Ai: 1},
        &SetVariable{Xn: 114},
        &SetValue{Xn: 201},
        &PutStructure{Functor: "-/2", Ai: 114},
        &SetValue{Xn: 200},
        &SetConstant{C: &Integer{Val: 1}},
        &BuiltinCall{Op: "is/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 207, Ai: 2},
        &PutStructure{Functor: "[|]/2", Ai: 3},
        &SetVariable{Xn: 116},
        &SetVariable{Xn: 204},
        &PutStructure{Functor: "-/2", Ai: 116},
        &SetVariable{Xn: 208},
        &SetVariable{Xn: 209},
        &Call{Pred: "build_tree/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutVariable{Xn: 210, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &Call{Pred: "build_tree/4", Arity: 4},
        &PutValue{Xn: 206, Ai: 0},
        &PutStructure{Functor: "t/4", Ai: 1},
        &SetValue{Xn: 207},
        &SetValue{Xn: 208},
        &SetValue{Xn: 209},
        &SetValue{Xn: 210},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 204, Ai: 3},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_16", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "excluded_name/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_16"},
        &TrustMe{},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutVariable{Xn: 203, Ai: 3},
        &Call{Pred: "matching_versions_in/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Call{Pred: "sort_versions_desc/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 102, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 104, Ai: 2},
        &GetVariable{Xn: 200, Ai: 3},
        &PutValue{Xn: 102, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &PutValue{Xn: 104, Ai: 2},
        &PutVariable{Xn: 201, Ai: 3},
        &Call{Pred: "candidate_versions/4", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 103, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &PutValue{Xn: 103, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "alias_list/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "alias_lookup/3"},
        &Allocate{},
        &GetVariable{Xn: 203, Ai: 0},
        &GetVariable{Xn: 205, Ai: 1},
        &GetVariable{Xn: 207, Ai: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "base_holds/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutVariable{Xn: 202, Ai: 3},
        &Call{Pred: "first_broken/4", Arity: 4},
        &GetLevel{Reg: 210},
        &TryMeElse{Label: "L_ite_else_17", Arity: 3},
        &PutValue{Xn: 202, Ai: 0},
        &PutConstant{C: wamAtom_none_6, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 210},
        &PutValue{Xn: 205, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 207, Ai: 0},
        &PutStructure{Functor: "ok/1", Ai: 1},
        &SetValue{Xn: 201},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_17"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "broken/3", Ai: 1},
        &SetVariable{Xn: 204},
        &SetVariable{Xn: 209},
        &SetVariable{Xn: 208},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_18", Arity: 3},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutVariable{Xn: 206, Ai: 3},
        &Call{Pred: "pick_repair/4", Arity: 4},
        &Cut{Reg: 211},
        &PutValue{Xn: 203, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 112},
        &SetValue{Xn: 205},
        &PutStructure{Functor: "-/2", Ai: 112},
        &SetValue{Xn: 204},
        &SetValue{Xn: 206},
        &PutValue{Xn: 207, Ai: 2},
        &Call{Pred: "close_moving/3", Arity: 3},
        &Jump{Label: "L_ite_cont_18"},
        &TrustMe{},
        &PutValue{Xn: 207, Ai: 0},
        &PutStructure{Functor: "blocked/3", Ai: 1},
        &SetValue{Xn: 204},
        &SetVariable{Xn: 112},
        &SetVariable{Xn: 113},
        &PutStructure{Functor: "needs/1", Ai: 112},
        &SetValue{Xn: 208},
        &PutStructure{Functor: "base_has/1", Ai: 113},
        &SetValue{Xn: 209},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnConstant{Cases: []ConstCase{{Val: wamAtom___7, Label: "default"}, {Val: wamAtom___8, Label: "L_cmp_ver_3_2_body"}, {Val: wamAtom___9, Label: "L_cmp_ver_3_3_body"}}},
        &TryMeElse{Label: "L_cmp_ver_3_2", Arity: 3},
        &Allocate{},
        &GetConstant{C: wamAtom___7, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_cmp_ver_3_3", Arity: 3},
        &Allocate{},
        &GetConstant{C: wamAtom___8, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &PutValue{Xn: 101, Ai: 0},
        &PutValue{Xn: 100, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &GetConstant{C: wamAtom___9, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 202, Ai: 0},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 205, Ai: 2},
        &GetVariable{Xn: 206, Ai: 3},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_19", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "dep_index/2", Arity: 2},
        &Cut{Reg: 207},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_20", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutStructure{Functor: "-/2", Ai: 1},
        &SetValue{Xn: 204},
        &SetValue{Xn: 205},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "tree_lookup/3", Arity: 3},
        &Cut{Reg: 208},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_20"},
        &TrustMe{},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_19"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 203, Ai: 1},
        &Call{Pred: "depends_list/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Call{Pred: "matching_deps/4", Arity: 4},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 104, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 201, Ai: 2},
        &GetVariable{Xn: 202, Ai: 3},
        &PutValue{Xn: 104, Ai: 0},
        &PutVariable{Xn: 203, Ai: 1},
        &Call{Pred: "conflicts_list/2", Arity: 2},
        &PutStructure{Functor: "conflicts/3", Ai: 0},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &SetValue{Xn: 202},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_conflicts_list_2_2_body"}, {Functor: "catalog/10", Label: "L_conflicts_list_2_3_body"}, {Functor: "icat/3", Label: "L_conflicts_list_2_4_body"}}},
        &TryMeElse{Label: "L_conflicts_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetValue{Xn: 102, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_conflicts_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 102, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_conflicts_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 102, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "conflicts_list/2"},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 110},
        &GetStructure{Functor: "depends/4", Ai: 110},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 206, Ai: 1},
        &GetVariable{Xn: 207, Ai: 2},
        &GetVariable{Xn: 208, Ai: 3},
        &GetVariable{Xn: 209, Ai: 4},
        &GetLevel{Reg: 210},
        &TryMeElse{Label: "L_ite_else_21", Arity: 5},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 208, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 204, Ai: 3},
        &Call{Pred: "dep_breaks_need/4", Arity: 4},
        &Cut{Reg: 210},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_21"},
        &TrustMe{},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 208, Ai: 3},
        &PutValue{Xn: 209, Ai: 4},
        &Call{Pred: "dep_breaks/5", Arity: 5},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 105, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &GetVariable{Xn: 204, Ai: 4},
        &PutValue{Xn: 105, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "depends_list/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &PutValue{Xn: 204, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "dep_breaks/5"},
        &TryMeElse{Label: "L_dep_breaks_need_4_2", Arity: 4},
        &Allocate{},
        &GetVariable{Xn: 206, Ai: 0},
        &GetStructure{Functor: "alternatives/1", Ai: 1},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 108, Ai: 2},
        &GetVariable{Xn: 202, Ai: 3},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetVariable{Xn: 200},
        &SetValue{Xn: 202},
        &PutValue{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_22", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_22"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_23", Arity: 4},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetVariable{Xn: 203},
        &SetVariable{Xn: 204},
        &PutValue{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutVariable{Xn: 207, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 209},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_23"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 102, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 201, Ai: 2},
        &GetValue{Xn: 201, Ai: 3},
        &PutValue{Xn: 102, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &GetLevel{Reg: 202},
        &TryMeElse{Label: "L_ite_else_24", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 202},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_24"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetValue{Xn: 101, Ai: 1},
        &Proceed{},
        &TryMeElse{Label: "L_dep_mentions_2_2", Arity: 2},
        &Allocate{},
        &GetStructure{Functor: "alternatives/1", Ai: 0},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetVariable{Xn: 201},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 105, Ai: 0},
        &GetVariable{Xn: 106, Ai: 1},
        &GetVariable{Xn: 107, Ai: 2},
        &GetVariable{Xn: 201, Ai: 3},
        &GetVariable{Xn: 203, Ai: 4},
        &PutValue{Xn: 105, Ai: 0},
        &PutValue{Xn: 106, Ai: 1},
        &PutValue{Xn: 107, Ai: 2},
        &PutVariable{Xn: 202, Ai: 3},
        &PutVariable{Xn: 204, Ai: 4},
        &Call{Pred: "depends_in/5", Arity: 5},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_25", Arity: 5},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "alternatives/1", Ai: 1},
        &SetVariable{Xn: 200},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 205},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetValue{Xn: 201},
        &SetValue{Xn: 203},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Jump{Label: "L_ite_cont_25"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_dep_to_req_3_2", Arity: 3},
        &Allocate{},
        &GetStructure{Functor: "alternatives/1", Ai: 0},
        &UnifyVariable{Xn: 100},
        &GetVariable{Xn: 101, Ai: 1},
        &GetStructure{Functor: "req/2", Ai: 2},
        &UnifyVariable{Xn: 102},
        &GetStructure{Functor: "alternatives/1", Ai: 102},
        &UnifyValue{Xn: 100},
        &UnifyConstant{C: wamAtom_any_10},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetStructure{Functor: "req/2", Ai: 2},
        &UnifyValue{Xn: 100},
        &UnifyValue{Xn: 101},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 105, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 105, Ai: 1},
        &PutVariable{Xn: 202, Ai: 2},
        &Call{Pred: "canonicalize_name/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "depends_list/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutConstant{C: wamAtom____0, Ai: 2},
        &PutVariable{Xn: 203, Ai: 3},
        &Call{Pred: "direct_on/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 104, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "dependents/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutConstant{C: wamAtom____0, Ai: 2},
        &PutVariable{Xn: 202, Ai: 3},
        &Call{Pred: "keep_installed_or_base/4", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 105, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 201, Ai: 2},
        &GetVariable{Xn: 202, Ai: 3},
        &GetVariable{Xn: 203, Ai: 4},
        &PutValue{Xn: 105, Ai: 0},
        &PutVariable{Xn: 204, Ai: 1},
        &Call{Pred: "depends_list/2", Arity: 2},
        &PutStructure{Functor: "depends/4", Ai: 0},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &SetValue{Xn: 202},
        &SetValue{Xn: 203},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_depends_list_2_2_body"}, {Functor: "catalog/10", Label: "L_depends_list_2_3_body"}, {Functor: "icat/3", Label: "L_depends_list_2_4_body"}}},
        &TryMeElse{Label: "L_depends_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetValue{Xn: 101, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_depends_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 101, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_depends_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 101, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "depends_list/2"},
        &TryMeElse{Label: "L_direct_on_4_2", Arity: 4},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetValue{Xn: 101, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 108},
        &GetStructure{Functor: "depends/4", Ai: 108},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 109},
        &UnifyVariable{Xn: 204},
        &GetVariable{Xn: 205, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 207, Ai: 3},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_26", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &Call{Pred: "dep_mentions/2", Arity: 2},
        &Cut{Reg: 208},
        &PutVariable{Xn: 206, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 111},
        &SetValue{Xn: 203},
        &PutStructure{Functor: "-/2", Ai: 111},
        &SetValue{Xn: 201},
        &SetValue{Xn: 202},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_26"},
        &TrustMe{},
        &PutVariable{Xn: 206, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "direct_on/4"},
        &TryMeElse{Label: "L_exclude_name_3_2", Arity: 3},
        &GetVariable{Xn: 100, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &Proceed{},
        &RetryMeElse{Label: "L_exclude_name_3_3", Arity: 3},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetList{Ai: 1},
        &UnifyValue{Xn: 200},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 202, Ai: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "exclude_name/3"},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetList{Ai: 2},
        &UnifyValue{Xn: 101},
        &UnifyVariable{Xn: 103},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 102, Ai: 1},
        &PutValue{Xn: 103, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "exclude_name/3"},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_excluded_list_2_2_body"}, {Functor: "catalog/10", Label: "L_excluded_list_2_3_body"}, {Functor: "icat/3", Label: "L_excluded_list_2_4_body"}}},
        &TryMeElse{Label: "L_excluded_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_excluded_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 107, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_excluded_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 107, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "excluded_list/2"},
        &Allocate{},
        &GetVariable{Xn: 102, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &PutValue{Xn: 102, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "excluded_list/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 200, Ai: 3},
        &GetVariable{Xn: 207, Ai: 4},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_27", Arity: 5},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "req/2", Ai: 1},
        &SetValue{Xn: 202},
        &SetValue{Xn: 203},
        &PutValue{Xn: 200, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &Call{Pred: "blocked_from/4", Arity: 4},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_27"},
        &TrustMe{},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_28", Arity: 5},
        &GetLevel{Reg: 210},
        &TryMeElse{Label: "L_ite_else_29", Arity: 5},
        &PutConstant{C: wamAtom_layered_11, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &PutConstant{C: wamAtom____0, Ai: 4},
        &PutVariable{Xn: 204, Ai: 5},
        &PutVariable{Xn: 205, Ai: 6},
        &PutVariable{Xn: 206, Ai: 7},
        &Call{Pred: "pick_need/8", Arity: 8},
        &Cut{Reg: 210},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_29"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 209},
        &PutValue{Xn: 207, Ai: 0},
        &PutConstant{C: wamAtom_unsatisfiable_1, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_28"},
        &TrustMe{},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "request_to_req/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutConstant{C: wamAtom____0, Ai: 2},
        &PutValue{Xn: 202, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "blocked_from/4"},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 104, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "request_to_req/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutConstant{C: wamAtom____0, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &PutVariable{Xn: 202, Ai: 4},
        &Call{Pred: "blocked_acc/5", Arity: 5},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_filter_satisfies_3_2", Arity: 3},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 201, Ai: 2},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_30", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetVariable{Xn: 204},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_30"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutVariable{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "filter_satisfies/3"},
        &TryMeElse{Label: "L_first_alt_already_4_2", Arity: 4},
        &Allocate{},
        &GetVariable{Xn: 104, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 201, Ai: 2},
        &GetVariable{Xn: 105, Ai: 3},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetVariable{Xn: 202},
        &SetVariable{Xn: 203},
        &PutValue{Xn: 105, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Call{Pred: "already_satisfied/4", Arity: 4},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetConstant{C: wamAtom_layered_11, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 103, Ai: 2},
        &GetVariable{Xn: 104, Ai: 3},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetVariable{Xn: 201},
        &SetVariable{Xn: 202},
        &PutValue{Xn: 104, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Call{Pred: "layer_satisfies/3", Arity: 3},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_first_broken_4_2", Arity: 4},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetConstant{C: wamAtom_none_6, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 108},
        &GetStructure{Functor: "hold/3", Ai: 108},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 109},
        &UnifyVariable{Xn: 207},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetVariable{Xn: 206, Ai: 3},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_31", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Cut{Reg: 208},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Call{Pred: "first_broken/4", Arity: 4},
        &Jump{Label: "L_ite_cont_31"},
        &TrustMe{},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_32", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutVariable{Xn: 205, Ai: 4},
        &Call{Pred: "dep_breaks_moving/5", Arity: 5},
        &Cut{Reg: 209},
        &PutValue{Xn: 206, Ai: 0},
        &PutStructure{Functor: "broken/3", Ai: 1},
        &SetValue{Xn: 202},
        &SetValue{Xn: 203},
        &SetValue{Xn: 205},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_32"},
        &TrustMe{},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Call{Pred: "first_broken/4", Arity: 4},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 104, Ai: 0},
        &GetVariable{Xn: 105, Ai: 1},
        &GetVariable{Xn: 106, Ai: 2},
        &GetVariable{Xn: 202, Ai: 3},
        &GetVariable{Xn: 203, Ai: 4},
        &PutValue{Xn: 104, Ai: 0},
        &PutValue{Xn: 105, Ai: 1},
        &PutValue{Xn: 106, Ai: 2},
        &PutVariable{Xn: 200, Ai: 3},
        &PutVariable{Xn: 201, Ai: 4},
        &Call{Pred: "depends_in/5", Arity: 5},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "follow_raw_dep/4"},
        &TryMeElse{Label: "L_follow_raw_dep_4_2", Arity: 4},
        &Allocate{},
        &GetStructure{Functor: "alternatives/1", Ai: 0},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 106, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 204, Ai: 3},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetVariable{Xn: 205},
        &SetVariable{Xn: 200},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 205},
        &SetVariable{Xn: 202},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetValue{Xn: 200, Ai: 3},
        &PutValue{Xn: 200, Ai: 0},
        &BuiltinCall{Op: "atom/1", Arity: 1},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetVariable{Xn: 201},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 203, Ai: 1},
        &PutValue{Xn: 201, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "base_holds/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutConstant{C: wamAtom____0, Ai: 2},
        &PutVariable{Xn: 202, Ai: 3},
        &Call{Pred: "audit_holds/4", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_group_keyed_2_2", Arity: 2},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 102},
        &GetStructure{Functor: "-/2", Ai: 102},
        &UnifyVariable{Xn: 103},
        &GetStructure{Functor: "-/2", Ai: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 108},
        &GetStructure{Functor: "-/2", Ai: 108},
        &UnifyValue{Xn: 104},
        &UnifyVariable{Xn: 109},
        &GetStructure{Functor: "[|]/2", Ai: 109},
        &UnifyValue{Xn: 106},
        &UnifyVariable{Xn: 110},
        &UnifyVariable{Xn: 201},
        &PutValue{Xn: 107, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &PutValue{Xn: 110, Ai: 2},
        &PutVariable{Xn: 200, Ai: 3},
        &Call{Pred: "same_key/4", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "group_keyed/2"},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 105},
        &GetStructure{Functor: "hold/3", Ai: 105},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_33", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_33"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Call{Pred: "hold_reason/3", Arity: 3},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 211, Ai: 0},
        &GetVariable{Xn: 210, Ai: 1},
        &PutValue{Xn: 211, Ai: 0},
        &Call{Pred: "is_public_catalog/1", Arity: 1},
        &PutValue{Xn: 211, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "depends_list/2", Arity: 2},
        &PutValue{Xn: 211, Ai: 0},
        &PutVariable{Xn: 204, Ai: 1},
        &Call{Pred: "packages/2", Arity: 2},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_34", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Call{Pred: "worth_indexing/2", Arity: 2},
        &Cut{Reg: 212},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 0}, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "key_dep_rows/3", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 203, Ai: 1},
        &Call{Pred: "group_keyed/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutVariable{Xn: 208, Ai: 1},
        &Call{Pred: "list_to_tree/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutConstant{C: &Integer{Val: 0}, Ai: 1},
        &PutVariable{Xn: 205, Ai: 2},
        &Call{Pred: "key_pkg_rows/3", Arity: 3},
        &PutValue{Xn: 205, Ai: 0},
        &PutVariable{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutVariable{Xn: 207, Ai: 1},
        &Call{Pred: "group_keyed/2", Arity: 2},
        &PutValue{Xn: 207, Ai: 0},
        &PutVariable{Xn: 209, Ai: 1},
        &Call{Pred: "list_to_tree/2", Arity: 2},
        &PutValue{Xn: 210, Ai: 0},
        &PutStructure{Functor: "icat/3", Ai: 1},
        &SetValue{Xn: 211},
        &SetValue{Xn: 208},
        &SetValue{Xn: 209},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_34"},
        &TrustMe{},
        &PutValue{Xn: 210, Ai: 0},
        &PutValue{Xn: 211, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &GetConstant{C: &Integer{Val: 64}, Ai: 0},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetVariable{Xn: 103, Ai: 3},
        &GetVariable{Xn: 104, Ai: 4},
        &PutList{Ai: 0},
        &SetVariable{Xn: 106},
        &SetConstant{C: wamAtom____0},
        &PutStructure{Functor: "-/2", Ai: 106},
        &SetValue{Xn: 102},
        &SetValue{Xn: 103},
        &PutValue{Xn: 100, Ai: 1},
        &PutValue{Xn: 101, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &PutConstant{C: wamAtom____0, Ai: 4},
        &PutValue{Xn: 104, Ai: 5},
        &Deallocate{},
        &Execute{Pred: "inst_walk/6"},
        &TryMeElse{Label: "L_inst_walk_6_2", Arity: 6},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetVariable{Xn: 103, Ai: 4},
        &GetValue{Xn: 103, Ai: 5},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 112},
        &GetStructure{Functor: "-/2", Ai: 112},
        &UnifyVariable{Xn: 208},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 204},
        &GetVariable{Xn: 206, Ai: 1},
        &GetVariable{Xn: 207, Ai: 2},
        &GetVariable{Xn: 209, Ai: 3},
        &GetVariable{Xn: 210, Ai: 4},
        &GetVariable{Xn: 211, Ai: 5},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_35", Arity: 6},
        &PutValue{Xn: 208, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Cut{Reg: 212},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 209, Ai: 3},
        &PutValue{Xn: 210, Ai: 4},
        &PutValue{Xn: 211, Ai: 5},
        &Call{Pred: "inst_walk/6", Arity: 6},
        &Jump{Label: "L_ite_cont_35"},
        &TrustMe{},
        &PutVariable{Xn: 203, Ai: 203},
        &PutVariable{Xn: 200, Ai: 200},
        &PutVariable{Xn: 201, Ai: 201},
        &BeginAggregate{AggType: "collect", ValueReg: 0, ResultReg: 203},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &PutValue{Xn: 200, Ai: 4},
        &Call{Pred: "follow_dep_name/5", Arity: 5},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &EndAggregate{ValueReg: 0},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutVariable{Xn: 205, Ai: 2},
        &BuiltinCall{Op: "append/3", Arity: 3},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutStructure{Functor: "[|]/2", Ai: 3},
        &SetValue{Xn: 208},
        &SetValue{Xn: 209},
        &PutStructure{Functor: "[|]/2", Ai: 4},
        &SetValue{Xn: 208},
        &SetValue{Xn: 210},
        &PutValue{Xn: 211, Ai: 5},
        &Call{Pred: "inst_walk/6", Arity: 6},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_installed_list_2_2_body"}, {Functor: "catalog/10", Label: "L_installed_list_2_3_body"}, {Functor: "icat/3", Label: "L_installed_list_2_4_body"}}},
        &TryMeElse{Label: "L_installed_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetValue{Xn: 104, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_installed_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 104, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_installed_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 104, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "installed_list/2"},
        &TryMeElse{Label: "L_installed_or_base_3_2", Arity: 3},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "installed_ver/3"},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 102, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 200, Ai: 2},
        &PutValue{Xn: 102, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 103, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 201, Ai: 2},
        &PutValue{Xn: 103, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &Call{Pred: "installed_list/2", Arity: 2},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_is_public_catalog_1_2_body"}, {Functor: "catalog/10", Label: "L_is_public_catalog_1_3_body"}}},
        &TryMeElse{Label: "L_is_public_catalog_1_2", Arity: 1},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &Proceed{},
        &RetryMeElse{Label: "L_is_public_catalog_1_3", Arity: 1},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &Proceed{},
        &TrustMe{},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &Proceed{},
        &GetStructure{Functor: "v/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "-/2", Label: "default"}, {Functor: "base/2", Label: "L_item_ver_3_2_body"}, {Functor: "layer/2", Label: "L_item_ver_3_3_body"}}},
        &TryMeElse{Label: "L_item_ver_3_2", Arity: 3},
        &Allocate{},
        &GetStructure{Functor: "-/2", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &GetVariable{Xn: 102, Ai: 1},
        &GetValue{Xn: 101, Ai: 2},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 102, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_item_ver_3_3", Arity: 3},
        &Allocate{},
        &GetStructure{Functor: "base/2", Ai: 0},
        &UnifyVariable{Xn: 100},
        &GetStructure{Functor: "-/2", Ai: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &GetVariable{Xn: 104, Ai: 1},
        &GetValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 101, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "layer/2", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &GetVariable{Xn: 102, Ai: 1},
        &GetVariable{Xn: 103, Ai: 2},
        &PutValue{Xn: 101, Ai: 0},
        &PutValue{Xn: 102, Ai: 1},
        &PutValue{Xn: 103, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "lookup_held/3"},
        &TryMeElse{Label: "L_keep_installed_or_base_4_2", Arity: 4},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetValue{Xn: 101, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 107},
        &GetStructure{Functor: "-/2", Ai: 107},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 203},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 206, Ai: 3},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_36", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &Call{Pred: "installed_or_base/3", Arity: 3},
        &Cut{Reg: 207},
        &PutVariable{Xn: 205, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 109},
        &SetValue{Xn: 202},
        &PutStructure{Functor: "-/2", Ai: 109},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_36"},
        &TrustMe{},
        &PutVariable{Xn: 205, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "keep_installed_or_base/4"},
        &TryMeElse{Label: "L_key_dep_rows_3_2", Arity: 3},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 104},
        &GetStructure{Functor: "depends/4", Ai: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 200, Ai: 1},
        &GetList{Ai: 2},
        &UnifyVariable{Xn: 109},
        &GetStructure{Functor: "-/2", Ai: 109},
        &UnifyVariable{Xn: 110},
        &GetStructure{Functor: "-/2", Ai: 110},
        &UnifyVariable{Xn: 111},
        &GetStructure{Functor: "-/2", Ai: 111},
        &UnifyValue{Xn: 105},
        &UnifyValue{Xn: 106},
        &UnifyValue{Xn: 200},
        &UnifyVariable{Xn: 112},
        &UnifyVariable{Xn: 203},
        &PutValue{Xn: 107, Ai: 0},
        &PutValue{Xn: 108, Ai: 1},
        &PutValue{Xn: 112, Ai: 2},
        &Call{Pred: "dep_to_req/3", Arity: 3},
        &PutVariable{Xn: 202, Ai: 0},
        &PutStructure{Functor: "+/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetConstant{C: &Integer{Val: 1}},
        &BuiltinCall{Op: "is/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "key_dep_rows/3"},
        &TryMeElse{Label: "L_key_pkg_rows_3_2", Arity: 3},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 103},
        &GetStructure{Functor: "package/2", Ai: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 200},
        &GetVariable{Xn: 106, Ai: 1},
        &GetList{Ai: 2},
        &UnifyVariable{Xn: 107},
        &GetStructure{Functor: "-/2", Ai: 107},
        &UnifyVariable{Xn: 108},
        &GetStructure{Functor: "-/2", Ai: 108},
        &UnifyValue{Xn: 104},
        &UnifyValue{Xn: 106},
        &UnifyValue{Xn: 105},
        &UnifyVariable{Xn: 202},
        &PutVariable{Xn: 201, Ai: 0},
        &PutStructure{Functor: "+/2", Ai: 1},
        &SetValue{Xn: 106},
        &SetConstant{C: &Integer{Val: 1}},
        &BuiltinCall{Op: "is/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "key_pkg_rows/3"},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutList{Ai: 1},
        &SetValue{Xn: 103},
        &SetConstant{C: wamAtom____0},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "resolve_layered/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Call{Pred: "topo_sort_sel/3", Arity: 3},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_layer_provider_5_2", Arity: 5},
        &Allocate{},
        &GetVariable{Xn: 202, Ai: 0},
        &GetVariable{Xn: 205, Ai: 1},
        &GetVariable{Xn: 206, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &GetVariable{Xn: 204, Ai: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "base_holds/2", Arity: 2},
        &PutStructure{Functor: "hold/3", Ai: 0},
        &SetValue{Xn: 203},
        &SetValue{Xn: 204},
        &SetVariable{Xn: 200},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &PutValue{Xn: 206, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "provides_sat/5"},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 203, Ai: 0},
        &GetVariable{Xn: 206, Ai: 1},
        &GetVariable{Xn: 207, Ai: 2},
        &GetVariable{Xn: 204, Ai: 3},
        &GetVariable{Xn: 205, Ai: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "layers_list/2", Arity: 2},
        &PutStructure{Functor: "layer/2", Ai: 0},
        &SetVariable{Xn: 200},
        &SetVariable{Xn: 202},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &Call{Pred: "lookup_held/3", Arity: 3},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &PutValue{Xn: 207, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "provides_sat/5"},
        &TryMeElse{Label: "L_layer_satisfies_3_2", Arity: 3},
        &Allocate{},
        &GetVariable{Xn: 102, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 201, Ai: 2},
        &PutValue{Xn: 102, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "satisfies/2"},
        &RetryMeElse{Label: "L_layer_satisfies_3_3", Arity: 3},
        &Allocate{},
        &GetVariable{Xn: 202, Ai: 0},
        &GetVariable{Xn: 205, Ai: 1},
        &GetVariable{Xn: 206, Ai: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "base_holds/2", Arity: 2},
        &PutStructure{Functor: "hold/3", Ai: 0},
        &SetVariable{Xn: 203},
        &SetVariable{Xn: 204},
        &SetVariable{Xn: 200},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &PutValue{Xn: 206, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "provides_sat/5"},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 203, Ai: 0},
        &GetVariable{Xn: 206, Ai: 1},
        &GetVariable{Xn: 207, Ai: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "layers_list/2", Arity: 2},
        &PutStructure{Functor: "layer/2", Ai: 0},
        &SetVariable{Xn: 200},
        &SetVariable{Xn: 202},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 204, Ai: 1},
        &PutVariable{Xn: 205, Ai: 2},
        &Call{Pred: "lookup_held/3", Arity: 3},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_37", Arity: 3},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_37"},
        &TrustMe{},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &PutValue{Xn: 207, Ai: 4},
        &Call{Pred: "provides_sat/5", Arity: 5},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 204, Ai: 3},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_38", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &Cut{Reg: 205},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_38"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &Call{Pred: "candidates_high_first/4", Arity: 4},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_layers_list_2_2_body"}, {Functor: "catalog/10", Label: "L_layers_list_2_3_body"}, {Functor: "icat/3", Label: "L_layers_list_2_4_body"}}},
        &TryMeElse{Label: "L_layers_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_layers_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 106, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_layers_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 106, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "layers_list/2"},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &PutValue{Xn: 201, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "length/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "build_tree/4"},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 200, Ai: 1},
        &GetLevel{Reg: 203},
        &TryMeElse{Label: "L_ite_else_39", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 1}, Ai: 1},
        &BuiltinCall{Op: "=</2", Arity: 2},
        &Cut{Reg: 203},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_39"},
        &TrustMe{},
        &PutVariable{Xn: 202, Ai: 0},
        &PutStructure{Functor: "+/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetConstant{C: &Integer{Val: -1}},
        &BuiltinCall{Op: "is/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "long_enough/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_40", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "item_ver/3", Arity: 3},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_40"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Call{Pred: "lookup_held/3", Arity: 3},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_map_requests_3_2", Arity: 3},
        &GetVariable{Xn: 100, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 201},
        &GetList{Ai: 2},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 202},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &PutValue{Xn: 104, Ai: 2},
        &Call{Pred: "request_to_req/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "map_requests/3"},
        &TryMeElse{Label: "L_matching_deps_4_2", Arity: 4},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetConstant{C: wamAtom____0, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 110},
        &GetStructure{Functor: "depends/4", Ai: 110},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 206},
        &GetVariable{Xn: 207, Ai: 1},
        &GetVariable{Xn: 208, Ai: 2},
        &GetVariable{Xn: 205, Ai: 3},
        &GetLevel{Reg: 210},
        &TryMeElse{Label: "L_ite_else_41", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 210},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutVariable{Xn: 204, Ai: 2},
        &Call{Pred: "dep_to_req/3", Arity: 3},
        &PutValue{Xn: 205, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetValue{Xn: 204},
        &SetVariable{Xn: 209},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_41"},
        &TrustMe{},
        &PutValue{Xn: 205, Ai: 0},
        &PutVariable{Xn: 209, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutValue{Xn: 208, Ai: 2},
        &PutValue{Xn: 209, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "matching_deps/4"},
        &TryMeElse{Label: "L_matching_versions_4_2", Arity: 4},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetConstant{C: wamAtom____0, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 107},
        &GetStructure{Functor: "package/2", Ai: 107},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 203},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 205, Ai: 2},
        &GetVariable{Xn: 202, Ai: 3},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_42", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 207},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetValue{Xn: 201},
        &SetVariable{Xn: 206},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_42"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "matching_versions/4"},
        &Allocate{},
        &GetVariable{Xn: 202, Ai: 0},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 205, Ai: 2},
        &GetVariable{Xn: 206, Ai: 3},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_43", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "pkg_index/2", Arity: 2},
        &Cut{Reg: 207},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_44", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "tree_lookup/3", Arity: 3},
        &Cut{Reg: 208},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &Call{Pred: "filter_satisfies/3", Arity: 3},
        &Jump{Label: "L_ite_cont_44"},
        &TrustMe{},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_43"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 203, Ai: 1},
        &Call{Pred: "packages/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Call{Pred: "matching_versions/4", Arity: 4},
        &Deallocate{},
        &Proceed{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 101},
        &SetValue{Xn: 102},
        &PutValue{Xn: 100, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Proceed{},
        &TryMeElse{Label: "L_names_of_2_2", Arity: 2},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 100},
        &GetStructure{Functor: "-/2", Ai: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &GetList{Ai: 1},
        &UnifyValue{Xn: 101},
        &UnifyVariable{Xn: 104},
        &PutValue{Xn: 103, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "names_of/2"},
        &TryMeElse{Label: "L_needed_names_4_2", Arity: 4},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &GetConstant{C: wamAtom____0, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 104, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &PutValue{Xn: 104, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "roots_to_pairs/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &PutConstant{C: wamAtom____0, Ai: 4},
        &PutValue{Xn: 203, Ai: 5},
        &Deallocate{},
        &Execute{Pred: "inst_walk/6"},
        &TryMeElse{Label: "L_no_acc_conflicts_4_2", Arity: 4},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom____0, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 202, Ai: 0},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetList{Ai: 3},
        &UnifyVariable{Xn: 106},
        &GetStructure{Functor: "-/2", Ai: 106},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 205},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_45", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &Call{Pred: "conflicts_in/4", Arity: 4},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_45"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_46", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Call{Pred: "conflicts_in/4", Arity: 4},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_46"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "no_acc_conflicts/4"},
        &TryMeElse{Label: "L_order_lt_2_2", Arity: 2},
        &Allocate{},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_order_lt_2_3", Arity: 2},
        &Allocate{},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &PutValue{Xn: 101, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "order_val/2", Arity: 2},
        &PutConstant{C: &Integer{Val: 0}, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_order_lt_2_4", Arity: 2},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &PutValue{Xn: 101, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "order_val/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 0}, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 203},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 204},
        &PutValue{Xn: 105, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "order_val/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &Call{Pred: "order_val/2", Arity: 2},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_47", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 205},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_47"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Call{Pred: "order_lt/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_order_val_2_2", Arity: 2},
        &Allocate{},
        &GetConstant{C: &Integer{Val: 126}, Ai: 0},
        &GetConstant{C: &Integer{Val: -1}, Ai: 1},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_order_val_2_3", Arity: 2},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 65}, Ai: 1},
        &BuiltinCall{Op: ">=/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 90}, Ai: 1},
        &BuiltinCall{Op: "=</2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_order_val_2_4", Arity: 2},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 97}, Ai: 1},
        &BuiltinCall{Op: ">=/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 122}, Ai: 1},
        &BuiltinCall{Op: "=</2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &PutValue{Xn: 101, Ai: 0},
        &PutStructure{Functor: "+/2", Ai: 1},
        &SetValue{Xn: 100},
        &SetConstant{C: &Integer{Val: 256}},
        &BuiltinCall{Op: "is/2", Arity: 2},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 103, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 201, Ai: 2},
        &PutValue{Xn: 103, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &Call{Pred: "packages/2", Arity: 2},
        &PutStructure{Functor: "package/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 103, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &PutValue{Xn: 103, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &Call{Pred: "packages/2", Arity: 2},
        &PutStructure{Functor: "package/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetVariable{Xn: 201},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_packages_2_2_body"}, {Functor: "catalog/10", Label: "L_packages_2_3_body"}, {Functor: "icat/3", Label: "L_packages_2_4_body"}}},
        &TryMeElse{Label: "L_packages_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetValue{Xn: 100, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_packages_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 100, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_packages_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 100, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "packages/2"},
        &TryMeElse{Label: "L_pad_head_2_2", Arity: 2},
        &Allocate{},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 100},
        &GetStructure{Functor: "s/2", Ai: 100},
        &UnifyConstant{C: wamAtom____0},
        &UnifyConstant{C: &Integer{Val: 0}},
        &UnifyConstant{C: wamAtom____0},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetValue{Xn: 100, Ai: 1},
        &Proceed{},
        &SwitchOnConstant{Cases: []ConstCase{{Val: wamAtom_classic_12, Label: "default"}, {Val: wamAtom_layered_11, Label: "L_pick_7_2_body"}}},
        &TryMeElse{Label: "L_pick_7_2", Arity: 7},
        &Allocate{},
        &GetConstant{C: wamAtom_classic_12, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetVariable{Xn: 103, Ai: 4},
        &GetVariable{Xn: 104, Ai: 5},
        &GetConstant{C: wamAtom_from_catalog_13, Ai: 6},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 104, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "candidates_high_first/4"},
        &TrustMe{},
        &Allocate{},
        &GetConstant{C: wamAtom_layered_11, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &GetVariable{Xn: 106, Ai: 4},
        &GetVariable{Xn: 204, Ai: 5},
        &GetVariable{Xn: 205, Ai: 6},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_48", Arity: 7},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &Cut{Reg: 206},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutConstant{C: wamAtom_from_base_14, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_48"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &Call{Pred: "candidates_high_first/4", Arity: 4},
        &PutValue{Xn: 205, Ai: 0},
        &PutConstant{C: wamAtom_from_catalog_13, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnConstant{Cases: []ConstCase{{Val: wamAtom_classic_12, Label: "default"}, {Val: wamAtom_layered_11, Label: "L_pick_need_8_3_body"}}},
        &TryMeElse{Label: "L_pick_need_8_2", Arity: 8},
        &Allocate{},
        &GetConstant{C: wamAtom_classic_12, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetVariable{Xn: 103, Ai: 4},
        &GetValue{Xn: 101, Ai: 5},
        &GetVariable{Xn: 104, Ai: 6},
        &GetConstant{C: wamAtom_from_catalog_13, Ai: 7},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 104, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "candidates_high_first/4"},
        &RetryMeElse{Label: "L_pick_need_8_3", Arity: 8},
        &Allocate{},
        &GetConstant{C: wamAtom_classic_12, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetVariable{Xn: 103, Ai: 4},
        &GetVariable{Xn: 104, Ai: 5},
        &GetVariable{Xn: 105, Ai: 6},
        &GetConstant{C: wamAtom_from_catalog_13, Ai: 7},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 104, Ai: 3},
        &PutValue{Xn: 105, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "provider_candidate/5"},
        &TrustMe{},
        &Allocate{},
        &GetConstant{C: wamAtom_layered_11, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &GetVariable{Xn: 110, Ai: 4},
        &GetVariable{Xn: 204, Ai: 5},
        &GetVariable{Xn: 205, Ai: 6},
        &GetVariable{Xn: 206, Ai: 7},
        &GetLevel{Reg: 210},
        &TryMeElse{Label: "L_ite_else_49", Arity: 8},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &Cut{Reg: 210},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom_from_base_14, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_49"},
        &TrustMe{},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_50", Arity: 8},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutValue{Xn: 205, Ai: 4},
        &Call{Pred: "layer_provider/5", Arity: 5},
        &Cut{Reg: 211},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom_from_base_14, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_50"},
        &TrustMe{},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_51", Arity: 8},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 207, Ai: 3},
        &Call{Pred: "candidate_versions/4", Arity: 4},
        &PutValue{Xn: 207, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 208},
        &SetVariable{Xn: 209},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 212},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom_from_catalog_13, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_51"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutValue{Xn: 205, Ai: 4},
        &Call{Pred: "provider_candidate/5", Arity: 5},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom_from_catalog_13, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 202, Ai: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutConstant{C: wamAtom_any_10, Ai: 2},
        &PutValue{Xn: 202, Ai: 3},
        &Call{Pred: "candidates_high_first/4", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "repairs_moving/4"},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetValue{Xn: 102, Ai: 1},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "provides/3", Label: "default"}, {Functor: "provides/4", Label: "L_provide_row_5_2_body"}}},
        &TryMeElse{Label: "L_provide_row_5_2", Arity: 5},
        &GetStructure{Functor: "provides/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetValue{Xn: 100, Ai: 1},
        &GetValue{Xn: 101, Ai: 2},
        &GetValue{Xn: 102, Ai: 3},
        &GetConstant{C: wamAtom_unversioned_15, Ai: 4},
        &Proceed{},
        &TrustMe{},
        &GetStructure{Functor: "provides/4", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &GetValue{Xn: 100, Ai: 1},
        &GetValue{Xn: 101, Ai: 2},
        &GetValue{Xn: 102, Ai: 3},
        &GetValue{Xn: 103, Ai: 4},
        &Proceed{},
        &TryMeElse{Label: "L_provide_satisfies_2_2", Arity: 2},
        &GetConstant{C: wamAtom_unversioned_15, Ai: 0},
        &GetConstant{C: wamAtom_any_10, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: wamAtom_unversioned_15, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "satisfies/2"},
        &Allocate{},
        &GetVariable{Xn: 205, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetVariable{Xn: 206, Ai: 3},
        &GetVariable{Xn: 207, Ai: 4},
        &PutValue{Xn: 205, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "provides_list/2", Arity: 2},
        &PutVariable{Xn: 201, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 202, Ai: 3},
        &PutVariable{Xn: 203, Ai: 4},
        &Call{Pred: "provide_row/5", Arity: 5},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_52", Arity: 5},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &Call{Pred: "excluded_name/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_52"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Call{Pred: "provide_satisfies/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "package_in/3"},
        &Allocate{},
        &GetVariable{Xn: 106, Ai: 0},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &GetVariable{Xn: 205, Ai: 4},
        &PutValue{Xn: 106, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "provides_list/2", Arity: 2},
        &PutVariable{Xn: 201, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutValue{Xn: 205, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "provide_row/5"},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_provides_list_2_2_body"}, {Functor: "catalog/10", Label: "L_provides_list_2_3_body"}, {Functor: "icat/3", Label: "L_provides_list_2_4_body"}}},
        &TryMeElse{Label: "L_provides_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_provides_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_provides_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 109, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "provides_list/2"},
        &Allocate{},
        &GetVariable{Xn: 107, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 204, Ai: 3},
        &GetVariable{Xn: 206, Ai: 4},
        &PutValue{Xn: 107, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "provides_list/2", Arity: 2},
        &PutVariable{Xn: 201, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutVariable{Xn: 205, Ai: 4},
        &Call{Pred: "provide_row/5", Arity: 5},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "provide_satisfies/2"},
        &Allocate{},
        &GetVariable{Xn: 209, Ai: 0},
        &GetVariable{Xn: 112, Ai: 1},
        &GetVariable{Xn: 211, Ai: 2},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 112, Ai: 1},
        &PutVariable{Xn: 206, Ai: 2},
        &Call{Pred: "canonicalize_name/3", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutVariable{Xn: 205, Ai: 1},
        &Call{Pred: "installed_list/2", Arity: 2},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_53", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "installed_ver/3", Arity: 3},
        &Cut{Reg: 212},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_53"},
        &TrustMe{},
        &PutVariable{Xn: 200, Ai: 0},
        &PutConstant{C: wamAtom_none_6, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_54", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: wamAtom_none_6, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 213},
        &PutValue{Xn: 211, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_54"},
        &TrustMe{},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &PutVariable{Xn: 207, Ai: 4},
        &Call{Pred: "inst_closure_names/5", Arity: 5},
        &PutValue{Xn: 209, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "requested_list/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutVariable{Xn: 202, Ai: 2},
        &Call{Pred: "exclude_name/3", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutVariable{Xn: 208, Ai: 3},
        &Call{Pred: "needed_names/4", Arity: 4},
        &PutVariable{Xn: 210, Ai: 210},
        &PutVariable{Xn: 203, Ai: 203},
        &PutVariable{Xn: 204, Ai: 204},
        &BeginAggregate{AggType: "collect", ValueReg: 0, ResultReg: 210},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 203},
        &SetValue{Xn: 204},
        &PutValue{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &GetLevel{Reg: 214},
        &TryMeElse{Label: "L_ite_else_55", Arity: 3},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Cut{Reg: 214},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_55"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 215},
        &TryMeElse{Label: "L_ite_else_56", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "base_name/2", Arity: 2},
        &Cut{Reg: 215},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_56"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 203},
        &SetValue{Xn: 204},
        &EndAggregate{ValueReg: 0},
        &PutValue{Xn: 210, Ai: 0},
        &PutValue{Xn: 211, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 102, Ai: 0},
        &GetVariable{Xn: 103, Ai: 1},
        &GetVariable{Xn: 104, Ai: 2},
        &GetVariable{Xn: 201, Ai: 3},
        &PutValue{Xn: 102, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &PutValue{Xn: 104, Ai: 2},
        &PutVariable{Xn: 200, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "reqs_ok_moving/2"},
        &TryMeElse{Label: "L_reqs_ok_moving_2_2", Arity: 2},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_reqs_ok_moving_2_3", Arity: 2},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 106},
        &GetStructure{Functor: "req/2", Ai: 106},
        &UnifyVariable{Xn: 107},
        &GetStructure{Functor: "alternatives/1", Ai: 107},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 204},
        &GetVariable{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_57", Arity: 2},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetVariable{Xn: 201},
        &SetVariable{Xn: 203},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutVariable{Xn: 202, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Cut{Reg: 206},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Jump{Label: "L_ite_cont_57"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "reqs_ok_moving/2"},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 105},
        &GetStructure{Functor: "req/2", Ai: 105},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 203},
        &GetVariable{Xn: 204, Ai: 1},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_58", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Cut{Reg: 205},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Jump{Label: "L_ite_cont_58"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "reqs_ok_moving/2"},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetStructure{Functor: "req/2", Ai: 2},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 204},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_59", Arity: 3},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "req/2", Ai: 1},
        &SetVariable{Xn: 200},
        &SetValue{Xn: 204},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &Call{Pred: "canonicalize_name/3", Arity: 3},
        &Jump{Label: "L_ite_cont_59"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &Call{Pred: "canonicalize_name/3", Arity: 3},
        &PutValue{Xn: 204, Ai: 0},
        &PutConstant{C: wamAtom_any_10, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_requested_list_2_2_body"}, {Functor: "catalog/10", Label: "L_requested_list_2_3_body"}, {Functor: "icat/3", Label: "L_requested_list_2_4_body"}}},
        &TryMeElse{Label: "L_requested_list_2_2", Arity: 2},
        &GetStructure{Functor: "catalog/6", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &GetValue{Xn: 105, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_requested_list_2_3", Arity: 2},
        &GetStructure{Functor: "catalog/9", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &GetValue{Xn: 105, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_requested_list_2_4", Arity: 2},
        &GetStructure{Functor: "catalog/10", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 104},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 108},
        &UnifyVariable{Xn: 109},
        &GetValue{Xn: 105, Ai: 1},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "icat/3", Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 1},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "requested_list/2"},
        &Allocate{},
        &GetVariable{Xn: 105, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &PutValue{Xn: 105, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "index_catalog/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutVariable{Xn: 202, Ai: 2},
        &Call{Pred: "map_requests/3", Arity: 3},
        &PutConstant{C: wamAtom_classic_12, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &PutVariable{Xn: 203, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 200, Ai: 2},
        &GetVariable{Xn: 205, Ai: 3},
        &GetVariable{Xn: 206, Ai: 4},
        &GetVariable{Xn: 207, Ai: 5},
        &GetVariable{Xn: 208, Ai: 6},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_60", Arity: 7},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &Call{Pred: "first_alt_already/4", Arity: 4},
        &Cut{Reg: 209},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &PutValue{Xn: 207, Ai: 4},
        &PutValue{Xn: 208, Ai: 5},
        &Call{Pred: "resolve_pending/6", Arity: 6},
        &Jump{Label: "L_ite_cont_60"},
        &TrustMe{},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetVariable{Xn: 203},
        &SetVariable{Xn: 204},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutStructure{Functor: "[|]/2", Ai: 2},
        &SetVariable{Xn: 111},
        &SetValue{Xn: 205},
        &PutStructure{Functor: "req/2", Ai: 111},
        &SetValue{Xn: 203},
        &SetValue{Xn: 204},
        &PutValue{Xn: 206, Ai: 3},
        &PutValue{Xn: 207, Ai: 4},
        &PutValue{Xn: 208, Ai: 5},
        &Call{Pred: "resolve_pending/6", Arity: 6},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 105, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &PutValue{Xn: 105, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "index_catalog/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutVariable{Xn: 202, Ai: 2},
        &Call{Pred: "map_requests/3", Arity: 3},
        &PutConstant{C: wamAtom_layered_11, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &PutVariable{Xn: 203, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetVariable{Xn: 103, Ai: 3},
        &GetVariable{Xn: 104, Ai: 4},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 103, Ai: 3},
        &PutStructure{Functor: "st/2", Ai: 4},
        &SetConstant{C: &Integer{Val: 0}},
        &SetConstant{C: wamAtom____0},
        &PutValue{Xn: 104, Ai: 5},
        &Deallocate{},
        &Execute{Pred: "resolve_pending/6"},
        &TryMeElse{Label: "L_resolve_pending_6_2", Arity: 6},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetVariable{Xn: 103, Ai: 4},
        &GetValue{Xn: 102, Ai: 5},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 205, Ai: 0},
        &GetVariable{Xn: 206, Ai: 1},
        &GetList{Ai: 2},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 207},
        &GetVariable{Xn: 208, Ai: 3},
        &GetVariable{Xn: 209, Ai: 4},
        &GetVariable{Xn: 210, Ai: 5},
        &GetLevel{Reg: 220},
        &TryMeElse{Label: "L_ite_else_61", Arity: 6},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "done/3", Ai: 1},
        &SetVariable{Xn: 213},
        &SetVariable{Xn: 211},
        &SetVariable{Xn: 215},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 220},
        &PutValue{Xn: 209, Ai: 0},
        &PutStructure{Functor: "st/2", Ai: 1},
        &SetVariable{Xn: 200},
        &SetVariable{Xn: 122},
        &PutStructure{Functor: "[|]/2", Ai: 122},
        &SetVariable{Xn: 123},
        &SetVariable{Xn: 201},
        &PutStructure{Functor: "a/3", Ai: 123},
        &SetValue{Xn: 213},
        &SetValue{Xn: 211},
        &SetValue{Xn: 215},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 208, Ai: 3},
        &PutStructure{Functor: "st/2", Ai: 4},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &PutValue{Xn: 210, Ai: 5},
        &Call{Pred: "resolve_pending/6", Arity: 6},
        &Jump{Label: "L_ite_cont_61"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "req/2", Ai: 1},
        &SetVariable{Xn: 203},
        &SetVariable{Xn: 212},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 221},
        &TryMeElse{Label: "L_ite_else_62", Arity: 6},
        &PutValue{Xn: 203, Ai: 0},
        &PutStructure{Functor: "alternatives/1", Ai: 1},
        &SetVariable{Xn: 204},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 221},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &PutValue{Xn: 208, Ai: 4},
        &PutValue{Xn: 209, Ai: 5},
        &PutValue{Xn: 210, Ai: 6},
        &Call{Pred: "resolve_alternatives/7", Arity: 7},
        &Jump{Label: "L_ite_cont_62"},
        &TrustMe{},
        &GetLevel{Reg: 222},
        &TryMeElse{Label: "L_ite_else_63", Arity: 6},
        &PutValue{Xn: 208, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutVariable{Xn: 211, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Cut{Reg: 222},
        &PutValue{Xn: 211, Ai: 0},
        &PutValue{Xn: 212, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 208, Ai: 3},
        &PutValue{Xn: 209, Ai: 4},
        &PutValue{Xn: 210, Ai: 5},
        &Call{Pred: "resolve_pending/6", Arity: 6},
        &Jump{Label: "L_ite_cont_63"},
        &TrustMe{},
        &GetLevel{Reg: 223},
        &TryMeElse{Label: "L_ite_else_64", Arity: 6},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 212, Ai: 3},
        &Call{Pred: "already_provided/4", Arity: 4},
        &Cut{Reg: 223},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 208, Ai: 3},
        &PutValue{Xn: 209, Ai: 4},
        &PutValue{Xn: 210, Ai: 5},
        &Call{Pred: "resolve_pending/6", Arity: 6},
        &Jump{Label: "L_ite_cont_64"},
        &TrustMe{},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 212, Ai: 3},
        &PutValue{Xn: 208, Ai: 4},
        &PutVariable{Xn: 213, Ai: 5},
        &PutVariable{Xn: 211, Ai: 6},
        &PutVariable{Xn: 214, Ai: 7},
        &Call{Pred: "pick_need/8", Arity: 8},
        &GetLevel{Reg: 224},
        &TryMeElse{Label: "L_ite_else_65", Arity: 6},
        &PutValue{Xn: 214, Ai: 0},
        &PutConstant{C: wamAtom_from_base_14, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 224},
        &PutValue{Xn: 209, Ai: 0},
        &PutStructure{Functor: "st/2", Ai: 1},
        &SetVariable{Xn: 215},
        &SetVariable{Xn: 216},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 225},
        &TryMeElse{Label: "L_ite_else_66", Arity: 6},
        &PutValue{Xn: 216, Ai: 0},
        &PutValue{Xn: 213, Ai: 1},
        &PutValue{Xn: 211, Ai: 2},
        &PutValue{Xn: 215, Ai: 3},
        &Call{Pred: "active_member/4", Arity: 4},
        &Cut{Reg: 225},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 208, Ai: 3},
        &PutValue{Xn: 209, Ai: 4},
        &PutValue{Xn: 210, Ai: 5},
        &Call{Pred: "resolve_pending/6", Arity: 6},
        &Jump{Label: "L_ite_cont_66"},
        &TrustMe{},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 213, Ai: 1},
        &PutValue{Xn: 211, Ai: 2},
        &PutVariable{Xn: 217, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 217, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 123},
        &SetValue{Xn: 207},
        &PutStructure{Functor: "done/3", Ai: 123},
        &SetValue{Xn: 213},
        &SetValue{Xn: 211},
        &SetValue{Xn: 215},
        &PutVariable{Xn: 218, Ai: 2},
        &BuiltinCall{Op: "append/3", Arity: 3},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 218, Ai: 2},
        &PutValue{Xn: 208, Ai: 3},
        &PutStructure{Functor: "st/2", Ai: 4},
        &SetValue{Xn: 215},
        &SetVariable{Xn: 125},
        &PutStructure{Functor: "[|]/2", Ai: 125},
        &SetVariable{Xn: 126},
        &SetValue{Xn: 216},
        &PutStructure{Functor: "a/3", Ai: 126},
        &SetValue{Xn: 213},
        &SetValue{Xn: 211},
        &SetValue{Xn: 215},
        &PutValue{Xn: 210, Ai: 5},
        &Call{Pred: "resolve_pending/6", Arity: 6},
        &Jump{Label: "L_ite_cont_65"},
        &TrustMe{},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 213, Ai: 1},
        &PutValue{Xn: 211, Ai: 2},
        &PutValue{Xn: 208, Ai: 3},
        &Call{Pred: "no_acc_conflicts/4", Arity: 4},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 213, Ai: 1},
        &PutValue{Xn: 211, Ai: 2},
        &PutVariable{Xn: 217, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 217, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutVariable{Xn: 218, Ai: 2},
        &BuiltinCall{Op: "append/3", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutStructure{Functor: "st/2", Ai: 1},
        &SetVariable{Xn: 215},
        &SetVariable{Xn: 216},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutVariable{Xn: 219, Ai: 0},
        &PutStructure{Functor: "+/2", Ai: 1},
        &SetValue{Xn: 215},
        &SetConstant{C: &Integer{Val: 1}},
        &BuiltinCall{Op: "is/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 218, Ai: 2},
        &PutStructure{Functor: "[|]/2", Ai: 3},
        &SetVariable{Xn: 124},
        &SetValue{Xn: 208},
        &PutStructure{Functor: "-/2", Ai: 124},
        &SetValue{Xn: 213},
        &SetValue{Xn: 211},
        &PutStructure{Functor: "st/2", Ai: 4},
        &SetValue{Xn: 219},
        &SetValue{Xn: 216},
        &PutValue{Xn: 210, Ai: 5},
        &Call{Pred: "resolve_pending/6", Arity: 6},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_roots_to_pairs_3_2", Arity: 3},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &Proceed{},
        &RetryMeElse{Label: "L_roots_to_pairs_3_3", Arity: 3},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 103},
        &UnifyVariable{Xn: 200},
        &GetVariable{Xn: 201, Ai: 1},
        &GetList{Ai: 2},
        &UnifyVariable{Xn: 104},
        &GetStructure{Functor: "-/2", Ai: 104},
        &UnifyValue{Xn: 103},
        &UnifyVariable{Xn: 105},
        &UnifyVariable{Xn: 202},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 103},
        &SetValue{Xn: 105},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "roots_to_pairs/3"},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 100},
        &UnifyVariable{Xn: 101},
        &GetVariable{Xn: 102, Ai: 1},
        &GetVariable{Xn: 103, Ai: 2},
        &PutValue{Xn: 101, Ai: 0},
        &PutValue{Xn: 102, Ai: 1},
        &PutValue{Xn: 103, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "roots_to_pairs/3"},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 106, Ai: 1},
        &GetVariable{Xn: 205, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 106, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "canonicalize_name/3", Arity: 3},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_67", Arity: 4},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_68", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &Call{Pred: "package_in/3", Arity: 3},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_68"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 206},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_no_candidate_16, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_67"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_69", Arity: 4},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_70", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutVariable{Xn: 202, Ai: 2},
        &Call{Pred: "base_reason/3", Arity: 3},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_70"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 207},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_no_candidate_16, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_69"},
        &TrustMe{},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutVariable{Xn: 204, Ai: 2},
        &Call{Pred: "base_reason/3", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutValue{Xn: 203, Ai: 4},
        &Call{Pred: "safe_upgrade_reason/5", Arity: 5},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_safe_upgrade_reason_5_2", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_modified_17, Ai: 3},
        &GetStructure{Functor: "unsafe/1", Ai: 4},
        &UnifyConstant{C: wamAtom_modified_17},
        &Proceed{},
        &RetryMeElse{Label: "L_safe_upgrade_reason_5_3", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_footprint_18, Ai: 3},
        &GetStructure{Functor: "safe/1", Ai: 4},
        &UnifyVariable{Xn: 103},
        &GetStructure{Functor: "cost/1", Ai: 103},
        &UnifyConstant{C: wamAtom_footprint_18},
        &Proceed{},
        &RetryMeElse{Label: "L_safe_upgrade_reason_5_4", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_blanket_2, Ai: 3},
        &GetStructure{Functor: "safe/1", Ai: 4},
        &UnifyVariable{Xn: 103},
        &GetStructure{Functor: "cost/1", Ai: 103},
        &UnifyConstant{C: wamAtom_blanket_2},
        &Proceed{},
        &RetryMeElse{Label: "L_safe_upgrade_reason_5_5", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_layer_shadow_19, Ai: 3},
        &GetStructure{Functor: "safe/1", Ai: 4},
        &UnifyVariable{Xn: 103},
        &GetStructure{Functor: "cost/1", Ai: 103},
        &UnifyConstant{C: wamAtom_layer_shadow_19},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_abi_anchor_3, Ai: 3},
        &GetStructure{Functor: "coordinated/1", Ai: 4},
        &UnifyVariable{Xn: 103},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutStructure{Functor: "ok/1", Ai: 3},
        &SetValue{Xn: 103},
        &Deallocate{},
        &Execute{Pred: "upgrade_set_result/4"},
        &TryMeElse{Label: "L_same_key_4_2", Arity: 4},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &GetConstant{C: wamAtom____0, Ai: 3},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 108},
        &GetStructure{Functor: "-/2", Ai: 108},
        &UnifyVariable{Xn: 109},
        &GetStructure{Functor: "-/2", Ai: 109},
        &UnifyVariable{Xn: 204},
        &UnifyVariable{Xn: 205},
        &UnifyVariable{Xn: 206},
        &UnifyVariable{Xn: 207},
        &GetVariable{Xn: 200, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_71", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 208},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetValue{Xn: 206},
        &SetVariable{Xn: 201},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Call{Pred: "same_key/4", Arity: 4},
        &Jump{Label: "L_ite_cont_71"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 111},
        &SetValue{Xn: 207},
        &PutStructure{Functor: "-/2", Ai: 111},
        &SetVariable{Xn: 112},
        &SetValue{Xn: 206},
        &PutStructure{Functor: "-/2", Ai: 112},
        &SetValue{Xn: 204},
        &SetValue{Xn: 205},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_satisfies_2_2", Arity: 2},
        &GetVariable{Xn: 100, Ai: 0},
        &GetConstant{C: wamAtom_any_10, Ai: 1},
        &Proceed{},
        &RetryMeElse{Label: "L_satisfies_2_3", Arity: 2},
        &GetVariable{Xn: 100, Ai: 0},
        &GetStructure{Functor: "eq/1", Ai: 1},
        &UnifyVariable{Xn: 101},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Proceed{},
        &RetryMeElse{Label: "L_satisfies_2_4", Arity: 2},
        &GetVariable{Xn: 100, Ai: 0},
        &GetStructure{Functor: "gte/1", Ai: 1},
        &UnifyVariable{Xn: 101},
        &GetLevel{Reg: 200},
        &TryMeElse{Label: "L_ite_else_72", Arity: 2},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &Cut{Reg: 200},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_72"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Proceed{},
        &RetryMeElse{Label: "L_satisfies_2_5", Arity: 2},
        &GetVariable{Xn: 100, Ai: 0},
        &GetStructure{Functor: "lte/1", Ai: 1},
        &UnifyVariable{Xn: 101},
        &GetLevel{Reg: 200},
        &TryMeElse{Label: "L_ite_else_73", Arity: 2},
        &PutValue{Xn: 101, Ai: 0},
        &PutValue{Xn: 100, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &Cut{Reg: 200},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_73"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Proceed{},
        &RetryMeElse{Label: "L_satisfies_2_6", Arity: 2},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetStructure{Functor: "lt/1", Ai: 1},
        &UnifyVariable{Xn: 101},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "version_lt/2"},
        &RetryMeElse{Label: "L_satisfies_2_7", Arity: 2},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetStructure{Functor: "gt/1", Ai: 1},
        &UnifyVariable{Xn: 101},
        &PutValue{Xn: 101, Ai: 0},
        &PutValue{Xn: 100, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "version_lt/2"},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetStructure{Functor: "range/2", Ai: 1},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 201},
        &GetLevel{Reg: 202},
        &TryMeElse{Label: "L_ite_else_74", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 102, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &Cut{Reg: 202},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_74"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "version_lt/2"},
        &TryMeElse{Label: "L_scan_base_holds_3_2", Arity: 3},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetValue{Xn: 100, Ai: 2},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 208},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 210, Ai: 2},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_75", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "layer/2", Ai: 1},
        &SetConstant{C: wamAtom_base_20},
        &SetVariable{Xn: 200},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 211},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutVariable{Xn: 209, Ai: 2},
        &Call{Pred: "scan_base_holds/3", Arity: 3},
        &Jump{Label: "L_ite_cont_75"},
        &TrustMe{},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_76", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "layer/2", Ai: 1},
        &SetVariable{Xn: 202},
        &SetVariable{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 212},
        &PutVariable{Xn: 209, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_76"},
        &TrustMe{},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_77", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "base/2", Ai: 1},
        &SetVariable{Xn: 112},
        &SetVariable{Xn: 207},
        &PutStructure{Functor: "-/2", Ai: 112},
        &SetVariable{Xn: 205},
        &SetVariable{Xn: 206},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 213},
        &PutVariable{Xn: 209, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 114},
        &SetValue{Xn: 204},
        &PutStructure{Functor: "hold/3", Ai: 114},
        &SetValue{Xn: 205},
        &SetValue{Xn: 206},
        &SetValue{Xn: 207},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_77"},
        &TrustMe{},
        &GetLevel{Reg: 214},
        &TryMeElse{Label: "L_ite_else_78", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "-/2", Ai: 1},
        &SetVariable{Xn: 205},
        &SetVariable{Xn: 206},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 214},
        &PutVariable{Xn: 209, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 113},
        &SetValue{Xn: 204},
        &PutStructure{Functor: "hold/3", Ai: 113},
        &SetValue{Xn: 205},
        &SetValue{Xn: 206},
        &SetConstant{C: wamAtom_blanket_2},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_78"},
        &TrustMe{},
        &PutVariable{Xn: 209, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 208, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutValue{Xn: 210, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "scan_base_holds/3"},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 202, Ai: 1},
        &GetLevel{Reg: 203},
        &TryMeElse{Label: "L_ite_else_79", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 203},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_79"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_segs_lt_2_2", Arity: 2},
        &Allocate{},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 103, Ai: 0},
        &GetVariable{Xn: 200, Ai: 1},
        &PutValue{Xn: 103, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "pad_head/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &Call{Pred: "pad_head/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "segs_lt_1/2"},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 106},
        &GetStructure{Functor: "s/2", Ai: 106},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 204},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 107},
        &GetStructure{Functor: "s/2", Ai: 107},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 205},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_80", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "order_lt/2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_80"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_81", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_81"},
        &TrustMe{},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &Call{Pred: "segs_lt/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetLevel{Reg: 204},
        &TryMeElse{Label: "L_ite_else_82", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutStructure{Functor: "-/2", Ai: 1},
        &SetValue{Xn: 202},
        &SetValue{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 204},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_82"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetLevel{Reg: 203},
        &TryMeElse{Label: "L_ite_else_83", Arity: 2},
        &PutConstant{C: wamAtom_is_v3_21, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "maplist/2", Arity: 2},
        &Cut{Reg: 203},
        &PutValue{Xn: 200, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "reverse/2", Arity: 2},
        &Jump{Label: "L_ite_cont_83"},
        &TrustMe{},
        &PutConstant{C: wamAtom_cmp_ver_22, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &BuiltinCall{Op: "predsort/3", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "reverse/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &PutValue{Xn: 201, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "base_holds/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Deallocate{},
        &Execute{Pred: "tight_rev_in/3"},
        &Allocate{},
        &PutConstant{C: wamAtom_any_10, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 108},
        &GetStructure{Functor: "hold/3", Ai: 108},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 109},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 206, Ai: 1},
        &GetVariable{Xn: 207, Ai: 2},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_84", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutVariable{Xn: 202, Ai: 3},
        &PutVariable{Xn: 203, Ai: 4},
        &Call{Pred: "dep_targets/5", Arity: 5},
        &TryMeElse{Label: "L_ite_else_85", Arity: 3},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &Call{Pred: "tight_constraint/1", Arity: 1},
        &Jump{Label: "L_ite_cont_85"},
        &TrustMe{},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutVariable{Xn: 204, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 202, Ai: 3},
        &PutValue{Xn: 203, Ai: 4},
        &Call{Pred: "provides_sat/5", Arity: 5},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_84"},
        &TrustMe{},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &Call{Pred: "tight_rev_in/3", Arity: 3},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_topo_all_7_2", Arity: 7},
        &GetVariable{Xn: 100, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetValue{Xn: 102, Ai: 4},
        &GetVariable{Xn: 103, Ai: 5},
        &GetValue{Xn: 103, Ai: 6},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetList{Ai: 1},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 108, Ai: 3},
        &GetVariable{Xn: 204, Ai: 4},
        &GetVariable{Xn: 109, Ai: 5},
        &GetVariable{Xn: 206, Ai: 6},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 107, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 108, Ai: 3},
        &PutVariable{Xn: 203, Ai: 4},
        &PutValue{Xn: 109, Ai: 5},
        &PutVariable{Xn: 205, Ai: 6},
        &Call{Pred: "topo_one/7", Arity: 7},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &PutValue{Xn: 204, Ai: 4},
        &PutValue{Xn: 205, Ai: 5},
        &PutValue{Xn: 206, Ai: 6},
        &Deallocate{},
        &Execute{Pred: "topo_all/7"},
        &TryMeElse{Label: "L_topo_one_7_2", Arity: 7},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetVariable{Xn: 103, Ai: 3},
        &GetValue{Xn: 103, Ai: 4},
        &GetVariable{Xn: 104, Ai: 5},
        &GetValue{Xn: 104, Ai: 6},
        &PutValue{Xn: 101, Ai: 0},
        &PutValue{Xn: 103, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 202, Ai: 0},
        &GetVariable{Xn: 209, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetVariable{Xn: 210, Ai: 3},
        &GetVariable{Xn: 208, Ai: 4},
        &GetVariable{Xn: 212, Ai: 5},
        &GetVariable{Xn: 211, Ai: 6},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_86", Arity: 7},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 209},
        &SetVariable{Xn: 205},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Cut{Reg: 213},
        &PutVariable{Xn: 201, Ai: 201},
        &PutVariable{Xn: 200, Ai: 200},
        &BeginAggregate{AggType: "collect", ValueReg: 200, ResultReg: 201},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutValue{Xn: 200, Ai: 4},
        &Call{Pred: "follow_dep_name/5", Arity: 5},
        &EndAggregate{ValueReg: 200},
        &PutValue{Xn: 201, Ai: 0},
        &PutVariable{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutStructure{Functor: "[|]/2", Ai: 3},
        &SetValue{Xn: 209},
        &SetValue{Xn: 210},
        &PutVariable{Xn: 207, Ai: 4},
        &PutValue{Xn: 212, Ai: 5},
        &PutVariable{Xn: 206, Ai: 6},
        &Call{Pred: "topo_all/7", Arity: 7},
        &PutValue{Xn: 211, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 116},
        &SetValue{Xn: 206},
        &PutStructure{Functor: "-/2", Ai: 116},
        &SetValue{Xn: 209},
        &SetValue{Xn: 205},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 208, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_86"},
        &TrustMe{},
        &PutValue{Xn: 208, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetValue{Xn: 209},
        &SetValue{Xn: 210},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 211, Ai: 0},
        &PutValue{Xn: 212, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_topo_sort_sel_3_2", Arity: 3},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetConstant{C: wamAtom____0, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 206, Ai: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutVariable{Xn: 202, Ai: 1},
        &Call{Pred: "names_of/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &PutVariable{Xn: 204, Ai: 4},
        &PutConstant{C: wamAtom____0, Ai: 5},
        &PutVariable{Xn: 205, Ai: 6},
        &Call{Pred: "topo_all/7", Arity: 7},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "reverse/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetStructure{Functor: "t/4", Ai: 0},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 106},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &PutVariable{Xn: 201, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 106, Ai: 2},
        &BuiltinCall{Op: "compare/3", Arity: 3},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_87", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom___9, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 206},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_87"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_88", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom___7, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 207},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Call{Pred: "tree_lookup/3", Arity: 3},
        &Jump{Label: "L_ite_cont_88"},
        &TrustMe{},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Call{Pred: "tree_lookup/3", Arity: 3},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetVariable{Xn: 103, Ai: 3},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutStructure{Functor: "ok/1", Ai: 3},
        &SetValue{Xn: 103},
        &Call{Pred: "upgrade_set_result/4", Arity: 4},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 104, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "canonicalize_name/3", Arity: 3},
        &GetLevel{Reg: 204},
        &TryMeElse{Label: "L_ite_else_89", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &Call{Pred: "package_in/3", Arity: 3},
        &Cut{Reg: 204},
        &PutValue{Xn: 200, Ai: 0},
        &PutList{Ai: 1},
        &SetVariable{Xn: 106},
        &SetConstant{C: wamAtom____0},
        &PutStructure{Functor: "-/2", Ai: 106},
        &SetValue{Xn: 201},
        &SetValue{Xn: 202},
        &PutValue{Xn: 203, Ai: 2},
        &Call{Pred: "close_moving/3", Arity: 3},
        &Jump{Label: "L_ite_cont_89"},
        &TrustMe{},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_no_candidate_16, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "v/3", Label: "default"}, {Functor: "deb/3", Label: "L_version_lt_2_2_body"}}},
        &TryMeElse{Label: "L_version_lt_2_2", Arity: 2},
        &Allocate{},
        &GetStructure{Functor: "v/3", Ai: 0},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 204},
        &GetStructure{Functor: "v/3", Ai: 1},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 205},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_90", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_90"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_91", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_91"},
        &TrustMe{},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetStructure{Functor: "deb/3", Ai: 0},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 204},
        &GetStructure{Functor: "deb/3", Ai: 1},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 205},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_92", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_92"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_93", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "segs_lt/2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_93"},
        &TrustMe{},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_94", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "segs_lt/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_94"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_95", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "segs_lt/2", Arity: 2},
        &Cut{Reg: 209},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_95"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &Call{Pred: "segs_lt/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 202, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 206, Ai: 2},
        &GetVariable{Xn: 200, Ai: 3},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_96", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "package_in_name/2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_96"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutVariable{Xn: 203, Ai: 203},
        &PutVariable{Xn: 206, Ai: 206},
        &PutVariable{Xn: 204, Ai: 204},
        &BeginAggregate{AggType: "collect", ValueReg: 0, ResultReg: 200},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutVariable{Xn: 205, Ai: 4},
        &Call{Pred: "provides_for/5", Arity: 5},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_97", Arity: 4},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &Call{Pred: "provide_satisfies/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_97"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutStructure{Functor: "blocked/3", Ai: 0},
        &SetValue{Xn: 203},
        &SetVariable{Xn: 0},
        &SetVariable{Xn: 0},
        &PutStructure{Functor: "needs/1", Ai: 0},
        &SetValue{Xn: 206},
        &PutStructure{Functor: "base_has/1", Ai: 0},
        &SetValue{Xn: 204},
        &EndAggregate{ValueReg: 0},
        &Deallocate{},
        &Proceed{},
        &TryMeElse{Label: "L_walk_pkg_for_blocked_5_2", Arity: 5},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetValue{Xn: 101, Ai: 3},
        &GetVariable{Xn: 103, Ai: 4},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 103, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "layered_walk_ver/4"},
        &RetryMeElse{Label: "L_walk_pkg_for_blocked_5_3", Arity: 5},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetVariable{Xn: 103, Ai: 3},
        &GetVariable{Xn: 104, Ai: 4},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 103, Ai: 3},
        &PutValue{Xn: 104, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "layer_provider/5"},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetVariable{Xn: 103, Ai: 3},
        &GetVariable{Xn: 104, Ai: 4},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 103, Ai: 3},
        &PutValue{Xn: 104, Ai: 4},
        &Deallocate{},
        &Execute{Pred: "provider_candidate/5"},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &PutVariable{Xn: 202, Ai: 0},
        &Call{Pred: "index_threshold/1", Arity: 1},
        &GetLevel{Reg: 203},
        &TryMeElse{Label: "L_ite_else_98", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "long_enough/2", Arity: 2},
        &Cut{Reg: 203},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_98"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "long_enough/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
}

var sharedWamLabels = map[string]int{
        "acc_conflicts/4": 0,
        "L_ite_else_1": 16,
        "L_ite_cont_1": 22,
        "active_member/4": 24,
        "L_ite_else_2": 43,
        "L_ite_else_3": 55,
        "L_ite_cont_3": 61,
        "L_ite_cont_2": 61,
        "alias_list/2": 63,
        "L_alias_list_2_2": 74,
        "L_alias_list_2_2_body": 75,
        "L_alias_list_2_3": 87,
        "L_alias_list_2_3_body": 88,
        "L_alias_list_2_4": 101,
        "L_alias_list_2_4_body": 102,
        "alias_lookup/3": 112,
        "L_alias_lookup_3_2": 117,
        "L_alias_lookup_3_2_body": 118,
        "L_ite_else_4": 137,
        "L_ite_cont_4": 142,
        "already_provided/4": 144,
        "already_satisfied/4": 161,
        "L_already_satisfied_4_2": 175,
        "L_already_satisfied_4_2_body": 176,
        "alt_reasons/4": 187,
        "L_alt_reasons_4_2": 193,
        "L_alt_reasons_4_2_body": 194,
        "L_ite_else_5": 220,
        "L_ite_cont_5": 224,
        "audit_holds/4": 230,
        "L_audit_holds_4_2": 236,
        "L_audit_holds_4_2_body": 237,
        "L_ite_else_7": 268,
        "L_ite_cont_7": 274,
        "L_ite_else_6": 275,
        "L_ite_cont_6": 283,
        "base_holds/2": 291,
        "base_list/2": 306,
        "L_base_list_2_2": 317,
        "L_base_list_2_2_body": 318,
        "L_base_list_2_3": 330,
        "L_base_list_2_3_body": 331,
        "L_base_list_2_4": 344,
        "L_base_list_2_4_body": 345,
        "base_name/2": 355,
        "base_reason/3": 363,
        "base_ver/3": 375,
        "blocked_acc/5": 394,
        "L_blocked_acc_5_2": 422,
        "L_blocked_acc_5_2_body": 423,
        "L_ite_else_9": 445,
        "L_ite_cont_9": 447,
        "L_ite_else_8": 462,
        "L_ite_else_10": 488,
        "L_ite_cont_10": 492,
        "L_ite_cont_8": 492,
        "L_ite_else_11": 502,
        "L_ite_else_12": 526,
        "L_ite_cont_12": 530,
        "L_ite_cont_11": 530,
        "blocked_acc_list/5": 532,
        "L_blocked_acc_list_5_2": 539,
        "L_blocked_acc_list_5_2_body": 540,
        "blocked_from/4": 561,
        "L_blocked_from_4_2": 585,
        "L_blocked_from_4_2_body": 586,
        "L_ite_else_13": 605,
        "L_ite_cont_13": 607,
        "L_blocked_from_4_3": 619,
        "L_blocked_from_4_3_body": 620,
        "L_blocked_from_4_4": 647,
        "L_blocked_from_4_4_body": 648,
        "L_ite_else_14": 663,
        "L_ite_cont_14": 665,
        "build_tree/4": 687,
        "L_ite_else_15": 705,
        "L_ite_cont_15": 744,
        "candidate_versions/4": 746,
        "L_ite_else_16": 761,
        "L_ite_cont_16": 770,
        "candidates_high_first/4": 772,
        "canonicalize_name/3": 787,
        "close_moving/3": 799,
        "L_ite_else_17": 825,
        "L_ite_else_18": 850,
        "L_ite_cont_18": 861,
        "L_ite_cont_17": 861,
        "cmp_ver/3": 863,
        "L_cmp_ver_3_2": 875,
        "L_cmp_ver_3_2_body": 876,
        "L_cmp_ver_3_3": 886,
        "L_cmp_ver_3_3_body": 887,
        "collect_deps/4": 891,
        "L_ite_else_20": 915,
        "L_ite_cont_20": 919,
        "L_ite_else_19": 920,
        "L_ite_cont_19": 929,
        "conflicts_in/4": 931,
        "conflicts_list/2": 947,
        "L_conflicts_list_2_2": 958,
        "L_conflicts_list_2_2_body": 959,
        "L_conflicts_list_2_3": 971,
        "L_conflicts_list_2_3_body": 972,
        "L_conflicts_list_2_4": 985,
        "L_conflicts_list_2_4_body": 986,
        "dep_breaks/5": 996,
        "L_ite_else_21": 1027,
        "L_ite_cont_21": 1034,
        "dep_breaks_moving/5": 1036,
        "dep_breaks_need/4": 1052,
        "L_ite_else_22": 1077,
        "L_ite_cont_22": 1079,
        "L_ite_else_23": 1096,
        "L_ite_cont_23": 1098,
        "L_dep_breaks_need_4_2": 1100,
        "L_dep_breaks_need_4_2_body": 1101,
        "L_ite_else_24": 1118,
        "L_ite_cont_24": 1120,
        "dep_index/2": 1122,
        "dep_mentions/2": 1128,
        "L_dep_mentions_2_2": 1141,
        "L_dep_mentions_2_2_body": 1142,
        "dep_targets/5": 1150,
        "L_ite_else_25": 1175,
        "L_ite_cont_25": 1182,
        "dep_to_req/3": 1184,
        "L_dep_to_req_3_2": 1197,
        "L_dep_to_req_3_2_body": 1198,
        "dependents/3": 1204,
        "dependents_installed/3": 1226,
        "depends_in/5": 1245,
        "depends_list/2": 1263,
        "L_depends_list_2_2": 1274,
        "L_depends_list_2_2_body": 1275,
        "L_depends_list_2_3": 1287,
        "L_depends_list_2_3_body": 1288,
        "L_depends_list_2_4": 1301,
        "L_depends_list_2_4_body": 1302,
        "direct_on/4": 1312,
        "L_direct_on_4_2": 1318,
        "L_direct_on_4_2_body": 1319,
        "L_ite_else_26": 1346,
        "L_ite_cont_26": 1350,
        "exclude_name/3": 1356,
        "L_exclude_name_3_2": 1361,
        "L_exclude_name_3_2_body": 1362,
        "L_exclude_name_3_3": 1374,
        "L_exclude_name_3_3_body": 1375,
        "excluded_list/2": 1388,
        "L_excluded_list_2_2": 1399,
        "L_excluded_list_2_2_body": 1400,
        "L_excluded_list_2_3": 1412,
        "L_excluded_list_2_3_body": 1413,
        "L_excluded_list_2_4": 1426,
        "L_excluded_list_2_4_body": 1427,
        "excluded_name/2": 1437,
        "explain_alt/5": 1448,
        "L_ite_else_27": 1466,
        "L_ite_else_29": 1483,
        "L_ite_cont_29": 1485,
        "L_ite_else_28": 1490,
        "L_ite_cont_28": 1492,
        "L_ite_cont_27": 1492,
        "explain_blocked/3": 1494,
        "explain_blocked_list/3": 1508,
        "filter_satisfies/3": 1528,
        "L_filter_satisfies_3_2": 1533,
        "L_filter_satisfies_3_2_body": 1534,
        "L_ite_else_30": 1552,
        "L_ite_cont_30": 1556,
        "first_alt_already/4": 1561,
        "L_first_alt_already_4_2": 1580,
        "L_first_alt_already_4_2_body": 1581,
        "first_broken/4": 1598,
        "L_first_broken_4_2": 1604,
        "L_first_broken_4_2_body": 1605,
        "L_ite_else_31": 1629,
        "L_ite_else_32": 1646,
        "L_ite_cont_32": 1652,
        "L_ite_cont_31": 1652,
        "follow_dep_name/5": 1654,
        "follow_raw_dep/4": 1672,
        "L_follow_raw_dep_4_2": 1695,
        "L_follow_raw_dep_4_2_body": 1696,
        "freeze_audit/2": 1710,
        "group_keyed/2": 1727,
        "L_group_keyed_2_2": 1731,
        "L_group_keyed_2_2_body": 1732,
        "hold_reason/3": 1760,
        "L_ite_else_33": 1780,
        "L_ite_cont_33": 1785,
        "index_catalog/2": 1787,
        "L_ite_else_34": 1837,
        "L_ite_cont_34": 1841,
        "index_threshold/1": 1843,
        "inst_closure_names/5": 1845,
        "inst_walk/6": 1864,
        "L_inst_walk_6_2": 1872,
        "L_inst_walk_6_2_body": 1873,
        "L_ite_else_35": 1899,
        "L_ite_cont_35": 1934,
        "installed_list/2": 1936,
        "L_installed_list_2_2": 1947,
        "L_installed_list_2_2_body": 1948,
        "L_installed_list_2_3": 1960,
        "L_installed_list_2_3_body": 1961,
        "L_installed_list_2_4": 1974,
        "L_installed_list_2_4_body": 1975,
        "installed_or_base/3": 1985,
        "L_installed_or_base_3_2": 1995,
        "L_installed_or_base_3_2_body": 1996,
        "installed_ver/3": 2009,
        "is_public_catalog/1": 2023,
        "L_is_public_catalog_1_2": 2033,
        "L_is_public_catalog_1_2_body": 2034,
        "L_is_public_catalog_1_3": 2045,
        "L_is_public_catalog_1_3_body": 2046,
        "is_v3/1": 2058,
        "item_ver/3": 2063,
        "L_item_ver_3_2": 2076,
        "L_item_ver_3_2_body": 2077,
        "L_item_ver_3_3": 2091,
        "L_item_ver_3_3_body": 2092,
        "keep_installed_or_base/4": 2103,
        "L_keep_installed_or_base_4_2": 2109,
        "L_keep_installed_or_base_4_2_body": 2110,
        "L_ite_else_36": 2136,
        "L_ite_cont_36": 2140,
        "key_dep_rows/3": 2146,
        "L_key_dep_rows_3_2": 2151,
        "L_key_dep_rows_3_2_body": 2152,
        "key_pkg_rows/3": 2188,
        "L_key_pkg_rows_3_2": 2193,
        "L_key_pkg_rows_3_2_body": 2194,
        "layer_closure/3": 2221,
        "layer_provider/5": 2238,
        "L_layer_provider_5_2": 2261,
        "L_layer_provider_5_2_body": 2262,
        "layer_satisfies/3": 2287,
        "L_layer_satisfies_3_2": 2300,
        "L_layer_satisfies_3_2_body": 2301,
        "L_layer_satisfies_3_3": 2321,
        "L_layer_satisfies_3_3_body": 2322,
        "L_ite_else_37": 2349,
        "L_ite_cont_37": 2356,
        "layered_walk_ver/4": 2358,
        "L_ite_else_38": 2377,
        "L_ite_cont_38": 2383,
        "layers_list/2": 2386,
        "L_layers_list_2_2": 2397,
        "L_layers_list_2_2_body": 2398,
        "L_layers_list_2_3": 2410,
        "L_layers_list_2_3_body": 2411,
        "L_layers_list_2_4": 2424,
        "L_layers_list_2_4_body": 2425,
        "list_to_tree/2": 2435,
        "long_enough/2": 2447,
        "L_ite_else_39": 2460,
        "L_ite_cont_39": 2469,
        "lookup_held/3": 2471,
        "L_ite_else_40": 2488,
        "L_ite_cont_40": 2493,
        "map_requests/3": 2495,
        "L_map_requests_3_2": 2500,
        "L_map_requests_3_2_body": 2501,
        "matching_deps/4": 2518,
        "L_matching_deps_4_2": 2524,
        "L_matching_deps_4_2_body": 2525,
        "L_ite_else_41": 2556,
        "L_ite_cont_41": 2560,
        "matching_versions/4": 2566,
        "L_matching_versions_4_2": 2572,
        "L_matching_versions_4_2_body": 2573,
        "L_ite_else_42": 2598,
        "L_ite_cont_42": 2602,
        "matching_versions_in/4": 2608,
        "L_ite_else_44": 2631,
        "L_ite_cont_44": 2635,
        "L_ite_else_43": 2636,
        "L_ite_cont_43": 2645,
        "member_selected/3": 2647,
        "names_of/2": 2656,
        "L_names_of_2_2": 2660,
        "L_names_of_2_2_body": 2661,
        "needed_names/4": 2675,
        "L_needed_names_4_2": 2681,
        "L_needed_names_4_2_body": 2682,
        "no_acc_conflicts/4": 2699,
        "L_no_acc_conflicts_4_2": 2705,
        "L_no_acc_conflicts_4_2_body": 2706,
        "L_ite_else_45": 2726,
        "L_ite_cont_45": 2728,
        "L_ite_else_46": 2738,
        "L_ite_cont_46": 2740,
        "order_lt/2": 2746,
        "L_order_lt_2_2": 2754,
        "L_order_lt_2_2_body": 2755,
        "L_order_lt_2_3": 2768,
        "L_order_lt_2_3_body": 2769,
        "L_order_lt_2_4": 2782,
        "L_order_lt_2_4_body": 2783,
        "L_ite_else_47": 2804,
        "L_ite_cont_47": 2811,
        "L_order_lt_2_list_dispatch": 2813,
        "order_val/2": 2813,
        "L_order_val_2_2": 2820,
        "L_order_val_2_2_body": 2821,
        "L_order_val_2_3": 2833,
        "L_order_val_2_3_body": 2834,
        "L_order_val_2_4": 2846,
        "L_order_val_2_4_body": 2847,
        "package_in/3": 2855,
        "package_in_name/2": 2869,
        "packages/2": 2882,
        "L_packages_2_2": 2893,
        "L_packages_2_2_body": 2894,
        "L_packages_2_3": 2906,
        "L_packages_2_3_body": 2907,
        "L_packages_2_4": 2920,
        "L_packages_2_4_body": 2921,
        "pad_head/2": 2931,
        "L_pad_head_2_2": 2943,
        "L_pad_head_2_2_body": 2944,
        "pick/7": 2947,
        "L_pick_7_2": 2963,
        "L_pick_7_2_body": 2964,
        "L_ite_else_48": 2989,
        "L_ite_cont_48": 2998,
        "pick_need/8": 3000,
        "L_pick_need_8_2": 3017,
        "L_pick_need_8_2_body": 3018,
        "L_pick_need_8_3": 3034,
        "L_pick_need_8_3_body": 3035,
        "L_ite_else_49": 3064,
        "L_ite_else_50": 3078,
        "L_ite_else_51": 3102,
        "L_ite_cont_51": 3112,
        "L_ite_cont_50": 3112,
        "L_ite_cont_49": 3112,
        "pick_repair/4": 3114,
        "pkg_index/2": 3130,
        "provide_row/5": 3136,
        "L_provide_row_5_2": 3147,
        "L_provide_row_5_2_body": 3148,
        "provide_satisfies/2": 3158,
        "L_provide_satisfies_2_2": 3162,
        "L_provide_satisfies_2_2_body": 3163,
        "provider_candidate/5": 3173,
        "L_ite_else_52": 3199,
        "L_ite_cont_52": 3201,
        "provides_for/5": 3209,
        "provides_list/2": 3228,
        "L_provides_list_2_2": 3239,
        "L_provides_list_2_2_body": 3240,
        "L_provides_list_2_3": 3252,
        "L_provides_list_2_3_body": 3253,
        "L_provides_list_2_4": 3266,
        "L_provides_list_2_4_body": 3267,
        "provides_sat/5": 3277,
        "removal_orphans/3": 3299,
        "L_ite_else_53": 3319,
        "L_ite_cont_53": 3323,
        "L_ite_else_54": 3333,
        "L_ite_else_55": 3375,
        "L_ite_cont_55": 3377,
        "L_ite_else_56": 3385,
        "L_ite_cont_56": 3387,
        "L_ite_cont_54": 3394,
        "repairs_moving/4": 3397,
        "reqs_ok_moving/2": 3411,
        "L_reqs_ok_moving_2_2": 3415,
        "L_reqs_ok_moving_2_2_body": 3416,
        "L_ite_else_57": 3443,
        "L_ite_cont_57": 3445,
        "L_reqs_ok_moving_2_3": 3449,
        "L_reqs_ok_moving_2_3_body": 3450,
        "L_ite_else_58": 3469,
        "L_ite_cont_58": 3471,
        "L_reqs_ok_moving_2_list_dispatch": 3475,
        "request_to_req/3": 3475,
        "L_ite_else_59": 3494,
        "L_ite_cont_59": 3502,
        "requested_list/2": 3504,
        "L_requested_list_2_2": 3515,
        "L_requested_list_2_2_body": 3516,
        "L_requested_list_2_3": 3528,
        "L_requested_list_2_3_body": 3529,
        "L_requested_list_2_4": 3542,
        "L_requested_list_2_4_body": 3543,
        "resolve/3": 3553,
        "resolve_alternatives/7": 3576,
        "L_ite_else_60": 3600,
        "L_ite_cont_60": 3618,
        "resolve_layered/3": 3620,
        "resolve_pending/5": 3643,
        "resolve_pending/6": 3659,
        "L_resolve_pending_6_2": 3667,
        "L_resolve_pending_6_2_body": 3668,
        "L_ite_else_61": 3708,
        "L_ite_else_62": 3730,
        "L_ite_else_63": 3749,
        "L_ite_else_64": 3766,
        "L_ite_else_66": 3803,
        "L_ite_cont_66": 3835,
        "L_ite_else_65": 3836,
        "L_ite_cont_65": 3875,
        "L_ite_cont_64": 3875,
        "L_ite_cont_63": 3875,
        "L_ite_cont_62": 3875,
        "L_ite_cont_61": 3875,
        "roots_to_pairs/3": 3877,
        "L_roots_to_pairs_3_2": 3882,
        "L_roots_to_pairs_3_2_body": 3883,
        "L_roots_to_pairs_3_3": 3905,
        "L_roots_to_pairs_3_3_body": 3906,
        "L_roots_to_pairs_3_list_dispatch": 3917,
        "safe_upgrade/4": 3917,
        "L_ite_else_68": 3937,
        "L_ite_cont_68": 3939,
        "L_ite_else_67": 3944,
        "L_ite_else_70": 3956,
        "L_ite_cont_70": 3958,
        "L_ite_else_69": 3963,
        "L_ite_cont_69": 3974,
        "L_ite_cont_67": 3974,
        "safe_upgrade_reason/5": 3977,
        "L_safe_upgrade_reason_5_2": 3985,
        "L_safe_upgrade_reason_5_2_body": 3986,
        "L_safe_upgrade_reason_5_3": 3995,
        "L_safe_upgrade_reason_5_3_body": 3996,
        "L_safe_upgrade_reason_5_4": 4005,
        "L_safe_upgrade_reason_5_4_body": 4006,
        "L_safe_upgrade_reason_5_5": 4015,
        "L_safe_upgrade_reason_5_5_body": 4016,
        "same_key/4": 4030,
        "L_same_key_4_2": 4036,
        "L_same_key_4_2_body": 4037,
        "L_ite_else_71": 4067,
        "L_ite_cont_71": 4082,
        "satisfies/2": 4084,
        "L_satisfies_2_2": 4088,
        "L_satisfies_2_2_body": 4089,
        "L_satisfies_2_3": 4096,
        "L_satisfies_2_3_body": 4097,
        "L_ite_else_72": 4108,
        "L_ite_cont_72": 4110,
        "L_satisfies_2_4": 4111,
        "L_satisfies_2_4_body": 4112,
        "L_ite_else_73": 4123,
        "L_ite_cont_73": 4125,
        "L_satisfies_2_5": 4126,
        "L_satisfies_2_5_body": 4127,
        "L_satisfies_2_6": 4135,
        "L_satisfies_2_6_body": 4136,
        "L_satisfies_2_7": 4144,
        "L_satisfies_2_7_body": 4145,
        "L_ite_else_74": 4158,
        "L_ite_cont_74": 4160,
        "scan_base_holds/3": 4164,
        "L_scan_base_holds_3_2": 4169,
        "L_scan_base_holds_3_2_body": 4170,
        "L_ite_else_75": 4189,
        "L_ite_else_76": 4202,
        "L_ite_else_77": 4224,
        "L_ite_else_78": 4243,
        "L_ite_cont_78": 4247,
        "L_ite_cont_77": 4247,
        "L_ite_cont_76": 4247,
        "L_ite_cont_75": 4247,
        "seen_name/2": 4252,
        "L_ite_else_79": 4265,
        "L_ite_cont_79": 4269,
        "segs_lt/2": 4271,
        "L_segs_lt_2_2": 4279,
        "L_segs_lt_2_2_body": 4280,
        "segs_lt_1/2": 4293,
        "L_ite_else_80": 4314,
        "L_ite_else_81": 4326,
        "L_ite_cont_81": 4336,
        "L_ite_cont_80": 4336,
        "selected_ver/3": 4338,
        "L_ite_else_82": 4354,
        "L_ite_cont_82": 4359,
        "sort_versions_desc/2": 4361,
        "L_ite_else_83": 4377,
        "L_ite_cont_83": 4385,
        "tight_base_revdep/2": 4387,
        "tight_constraint/1": 4398,
        "tight_rev_in/3": 4403,
        "L_ite_else_85": 4431,
        "L_ite_cont_85": 4442,
        "L_ite_else_84": 4445,
        "L_ite_cont_84": 4450,
        "topo_all/7": 4452,
        "L_topo_all_7_2": 4461,
        "L_topo_all_7_2_body": 4462,
        "topo_one/7": 4489,
        "L_topo_one_7_2": 4504,
        "L_topo_one_7_2_body": 4505,
        "L_ite_else_86": 4556,
        "L_ite_cont_86": 4565,
        "topo_sort_sel/3": 4567,
        "L_topo_sort_sel_3_2": 4575,
        "L_topo_sort_sel_3_2_body": 4576,
        "tree_lookup/3": 4599,
        "L_ite_else_87": 4621,
        "L_ite_else_88": 4633,
        "L_ite_cont_88": 4638,
        "L_ite_cont_87": 4638,
        "upgrade_set/4": 4640,
        "upgrade_set_result/4": 4654,
        "L_ite_else_89": 4680,
        "L_ite_cont_89": 4684,
        "version_lt/2": 4687,
        "L_ite_else_90": 4706,
        "L_ite_else_91": 4718,
        "L_ite_cont_91": 4728,
        "L_ite_cont_90": 4728,
        "L_version_lt_2_2": 4730,
        "L_version_lt_2_2_body": 4731,
        "L_ite_else_92": 4748,
        "L_ite_else_93": 4760,
        "L_ite_else_94": 4772,
        "L_ite_cont_94": 4774,
        "L_ite_else_95": 4782,
        "L_ite_cont_95": 4784,
        "L_ite_cont_93": 4787,
        "L_ite_cont_92": 4787,
        "virtual_provider_ceilings/4": 4789,
        "L_ite_else_96": 4802,
        "L_ite_cont_96": 4804,
        "L_ite_else_97": 4826,
        "L_ite_cont_97": 4828,
        "walk_pkg_for_blocked/5": 4839,
        "L_walk_pkg_for_blocked_5_2": 4852,
        "L_walk_pkg_for_blocked_5_2_body": 4853,
        "L_walk_pkg_for_blocked_5_3": 4866,
        "L_walk_pkg_for_blocked_5_3_body": 4867,
        "worth_indexing/2": 4880,
        "L_ite_else_98": 4893,
        "L_ite_cont_98": 4897,
}

var sharedWamCode = resolveInstructions(sharedWamCodeRaw, sharedWamLabels)

// Exported aliases for main.go / parallel runner
var SharedWamCode = sharedWamCode
var SharedWamLabels = sharedWamLabels

// Strategy: wam
// WAM-compiled predicate: acc_conflicts/4 (shared table, pc=0)
var Acc_conflictsCode = sharedWamCode
var Acc_conflictsLabels = sharedWamLabels
const Acc_conflictsStartPC = 0

func Acc_conflicts(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 0
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: active_member/4 (shared table, pc=24)
var Active_memberCode = sharedWamCode
var Active_memberLabels = sharedWamLabels
const Active_memberStartPC = 24

func Active_member(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 24
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: alias_list/2 (shared table, pc=63)
var Alias_listCode = sharedWamCode
var Alias_listLabels = sharedWamLabels
const Alias_listStartPC = 63

func Alias_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 63
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: alias_lookup/3 (shared table, pc=112)
var Alias_lookupCode = sharedWamCode
var Alias_lookupLabels = sharedWamLabels
const Alias_lookupStartPC = 112

func Alias_lookup(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 112
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: already_provided/4 (shared table, pc=144)
var Already_providedCode = sharedWamCode
var Already_providedLabels = sharedWamLabels
const Already_providedStartPC = 144

func Already_provided(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 144
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: already_satisfied/4 (shared table, pc=161)
var Already_satisfiedCode = sharedWamCode
var Already_satisfiedLabels = sharedWamLabels
const Already_satisfiedStartPC = 161

func Already_satisfied(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 161
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: alt_reasons/4 (shared table, pc=187)
var Alt_reasonsCode = sharedWamCode
var Alt_reasonsLabels = sharedWamLabels
const Alt_reasonsStartPC = 187

func Alt_reasons(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 187
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: audit_holds/4 (shared table, pc=230)
var Audit_holdsCode = sharedWamCode
var Audit_holdsLabels = sharedWamLabels
const Audit_holdsStartPC = 230

func Audit_holds(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 230
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_holds/2 (shared table, pc=291)
var Base_holdsCode = sharedWamCode
var Base_holdsLabels = sharedWamLabels
const Base_holdsStartPC = 291

func Base_holds(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 291
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_list/2 (shared table, pc=306)
var Base_listCode = sharedWamCode
var Base_listLabels = sharedWamLabels
const Base_listStartPC = 306

func Base_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 306
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_name/2 (shared table, pc=355)
var Base_nameCode = sharedWamCode
var Base_nameLabels = sharedWamLabels
const Base_nameStartPC = 355

func Base_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 355
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_reason/3 (shared table, pc=363)
var Base_reasonCode = sharedWamCode
var Base_reasonLabels = sharedWamLabels
const Base_reasonStartPC = 363

func Base_reason(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 363
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_ver/3 (shared table, pc=375)
var Base_verCode = sharedWamCode
var Base_verLabels = sharedWamLabels
const Base_verStartPC = 375

func Base_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 375
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_acc/5 (shared table, pc=394)
var Blocked_accCode = sharedWamCode
var Blocked_accLabels = sharedWamLabels
const Blocked_accStartPC = 394

func Blocked_acc(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 394
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_acc_list/5 (shared table, pc=532)
var Blocked_acc_listCode = sharedWamCode
var Blocked_acc_listLabels = sharedWamLabels
const Blocked_acc_listStartPC = 532

func Blocked_acc_list(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 532
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_from/4 (shared table, pc=561)
var Blocked_fromCode = sharedWamCode
var Blocked_fromLabels = sharedWamLabels
const Blocked_fromStartPC = 561

func Blocked_from(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 561
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: build_tree/4 (shared table, pc=687)
var Build_treeCode = sharedWamCode
var Build_treeLabels = sharedWamLabels
const Build_treeStartPC = 687

func Build_tree(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 687
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: candidate_versions/4 (shared table, pc=746)
var Candidate_versionsCode = sharedWamCode
var Candidate_versionsLabels = sharedWamLabels
const Candidate_versionsStartPC = 746

func Candidate_versions(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 746
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: candidates_high_first/4 (shared table, pc=772)
var Candidates_high_firstCode = sharedWamCode
var Candidates_high_firstLabels = sharedWamLabels
const Candidates_high_firstStartPC = 772

func Candidates_high_first(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 772
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: canonicalize_name/3 (shared table, pc=787)
var Canonicalize_nameCode = sharedWamCode
var Canonicalize_nameLabels = sharedWamLabels
const Canonicalize_nameStartPC = 787

func Canonicalize_name(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 787
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: close_moving/3 (shared table, pc=799)
var Close_movingCode = sharedWamCode
var Close_movingLabels = sharedWamLabels
const Close_movingStartPC = 799

func Close_moving(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 799
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: cmp_ver/3 (shared table, pc=863)
var Cmp_verCode = sharedWamCode
var Cmp_verLabels = sharedWamLabels
const Cmp_verStartPC = 863

func Cmp_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 863
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: collect_deps/4 (shared table, pc=891)
var Collect_depsCode = sharedWamCode
var Collect_depsLabels = sharedWamLabels
const Collect_depsStartPC = 891

func Collect_deps(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 891
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: conflicts_in/4 (shared table, pc=931)
var Conflicts_inCode = sharedWamCode
var Conflicts_inLabels = sharedWamLabels
const Conflicts_inStartPC = 931

func Conflicts_in(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 931
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: conflicts_list/2 (shared table, pc=947)
var Conflicts_listCode = sharedWamCode
var Conflicts_listLabels = sharedWamLabels
const Conflicts_listStartPC = 947

func Conflicts_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 947
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_breaks/5 (shared table, pc=996)
var Dep_breaksCode = sharedWamCode
var Dep_breaksLabels = sharedWamLabels
const Dep_breaksStartPC = 996

func Dep_breaks(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 996
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_breaks_moving/5 (shared table, pc=1036)
var Dep_breaks_movingCode = sharedWamCode
var Dep_breaks_movingLabels = sharedWamLabels
const Dep_breaks_movingStartPC = 1036

func Dep_breaks_moving(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1036
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_breaks_need/4 (shared table, pc=1052)
var Dep_breaks_needCode = sharedWamCode
var Dep_breaks_needLabels = sharedWamLabels
const Dep_breaks_needStartPC = 1052

func Dep_breaks_need(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1052
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_index/2 (shared table, pc=1122)
var Dep_indexCode = sharedWamCode
var Dep_indexLabels = sharedWamLabels
const Dep_indexStartPC = 1122

func Dep_index(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1122
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_mentions/2 (shared table, pc=1128)
var Dep_mentionsCode = sharedWamCode
var Dep_mentionsLabels = sharedWamLabels
const Dep_mentionsStartPC = 1128

func Dep_mentions(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1128
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_targets/5 (shared table, pc=1150)
var Dep_targetsCode = sharedWamCode
var Dep_targetsLabels = sharedWamLabels
const Dep_targetsStartPC = 1150

func Dep_targets(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1150
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_to_req/3 (shared table, pc=1184)
var Dep_to_reqCode = sharedWamCode
var Dep_to_reqLabels = sharedWamLabels
const Dep_to_reqStartPC = 1184

func Dep_to_req(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1184
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dependents/3 (shared table, pc=1204)
var DependentsCode = sharedWamCode
var DependentsLabels = sharedWamLabels
const DependentsStartPC = 1204

func Dependents(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1204
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dependents_installed/3 (shared table, pc=1226)
var Dependents_installedCode = sharedWamCode
var Dependents_installedLabels = sharedWamLabels
const Dependents_installedStartPC = 1226

func Dependents_installed(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1226
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: depends_in/5 (shared table, pc=1245)
var Depends_inCode = sharedWamCode
var Depends_inLabels = sharedWamLabels
const Depends_inStartPC = 1245

func Depends_in(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1245
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: depends_list/2 (shared table, pc=1263)
var Depends_listCode = sharedWamCode
var Depends_listLabels = sharedWamLabels
const Depends_listStartPC = 1263

func Depends_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1263
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: direct_on/4 (shared table, pc=1312)
var Direct_onCode = sharedWamCode
var Direct_onLabels = sharedWamLabels
const Direct_onStartPC = 1312

func Direct_on(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1312
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: exclude_name/3 (shared table, pc=1356)
var Exclude_nameCode = sharedWamCode
var Exclude_nameLabels = sharedWamLabels
const Exclude_nameStartPC = 1356

func Exclude_name(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1356
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: excluded_list/2 (shared table, pc=1388)
var Excluded_listCode = sharedWamCode
var Excluded_listLabels = sharedWamLabels
const Excluded_listStartPC = 1388

func Excluded_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1388
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: excluded_name/2 (shared table, pc=1437)
var Excluded_nameCode = sharedWamCode
var Excluded_nameLabels = sharedWamLabels
const Excluded_nameStartPC = 1437

func Excluded_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1437
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: explain_alt/5 (shared table, pc=1448)
var Explain_altCode = sharedWamCode
var Explain_altLabels = sharedWamLabels
const Explain_altStartPC = 1448

func Explain_alt(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1448
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: explain_blocked/3 (shared table, pc=1494)
var Explain_blockedCode = sharedWamCode
var Explain_blockedLabels = sharedWamLabels
const Explain_blockedStartPC = 1494

func Explain_blocked(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1494
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: explain_blocked_list/3 (shared table, pc=1508)
var Explain_blocked_listCode = sharedWamCode
var Explain_blocked_listLabels = sharedWamLabels
const Explain_blocked_listStartPC = 1508

func Explain_blocked_list(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1508
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: filter_satisfies/3 (shared table, pc=1528)
var Filter_satisfiesCode = sharedWamCode
var Filter_satisfiesLabels = sharedWamLabels
const Filter_satisfiesStartPC = 1528

func Filter_satisfies(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1528
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: first_alt_already/4 (shared table, pc=1561)
var First_alt_alreadyCode = sharedWamCode
var First_alt_alreadyLabels = sharedWamLabels
const First_alt_alreadyStartPC = 1561

func First_alt_already(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1561
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: first_broken/4 (shared table, pc=1598)
var First_brokenCode = sharedWamCode
var First_brokenLabels = sharedWamLabels
const First_brokenStartPC = 1598

func First_broken(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1598
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: follow_dep_name/5 (shared table, pc=1654)
var Follow_dep_nameCode = sharedWamCode
var Follow_dep_nameLabels = sharedWamLabels
const Follow_dep_nameStartPC = 1654

func Follow_dep_name(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1654
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: follow_raw_dep/4 (shared table, pc=1672)
var Follow_raw_depCode = sharedWamCode
var Follow_raw_depLabels = sharedWamLabels
const Follow_raw_depStartPC = 1672

func Follow_raw_dep(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1672
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: freeze_audit/2 (shared table, pc=1710)
var Freeze_auditCode = sharedWamCode
var Freeze_auditLabels = sharedWamLabels
const Freeze_auditStartPC = 1710

func Freeze_audit(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1710
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: group_keyed/2 (shared table, pc=1727)
var Group_keyedCode = sharedWamCode
var Group_keyedLabels = sharedWamLabels
const Group_keyedStartPC = 1727

func Group_keyed(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1727
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: hold_reason/3 (shared table, pc=1760)
var Hold_reasonCode = sharedWamCode
var Hold_reasonLabels = sharedWamLabels
const Hold_reasonStartPC = 1760

func Hold_reason(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1760
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: index_catalog/2 (shared table, pc=1787)
var Index_catalogCode = sharedWamCode
var Index_catalogLabels = sharedWamLabels
const Index_catalogStartPC = 1787

func Index_catalog(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1787
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: index_threshold/1 (shared table, pc=1843)
var Index_thresholdCode = sharedWamCode
var Index_thresholdLabels = sharedWamLabels
const Index_thresholdStartPC = 1843

func Index_threshold(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1843
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: inst_closure_names/5 (shared table, pc=1845)
var Inst_closure_namesCode = sharedWamCode
var Inst_closure_namesLabels = sharedWamLabels
const Inst_closure_namesStartPC = 1845

func Inst_closure_names(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1845
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: inst_walk/6 (shared table, pc=1864)
var Inst_walkCode = sharedWamCode
var Inst_walkLabels = sharedWamLabels
const Inst_walkStartPC = 1864

func Inst_walk(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1864
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_list/2 (shared table, pc=1936)
var Installed_listCode = sharedWamCode
var Installed_listLabels = sharedWamLabels
const Installed_listStartPC = 1936

func Installed_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1936
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_or_base/3 (shared table, pc=1985)
var Installed_or_baseCode = sharedWamCode
var Installed_or_baseLabels = sharedWamLabels
const Installed_or_baseStartPC = 1985

func Installed_or_base(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1985
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_ver/3 (shared table, pc=2009)
var Installed_verCode = sharedWamCode
var Installed_verLabels = sharedWamLabels
const Installed_verStartPC = 2009

func Installed_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2009
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: is_public_catalog/1 (shared table, pc=2023)
var Is_public_catalogCode = sharedWamCode
var Is_public_catalogLabels = sharedWamLabels
const Is_public_catalogStartPC = 2023

func Is_public_catalog(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2023
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: is_v3/1 (shared table, pc=2058)
var Is_v3Code = sharedWamCode
var Is_v3Labels = sharedWamLabels
const Is_v3StartPC = 2058

func Is_v3(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2058
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: item_ver/3 (shared table, pc=2063)
var Item_verCode = sharedWamCode
var Item_verLabels = sharedWamLabels
const Item_verStartPC = 2063

func Item_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2063
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: keep_installed_or_base/4 (shared table, pc=2103)
var Keep_installed_or_baseCode = sharedWamCode
var Keep_installed_or_baseLabels = sharedWamLabels
const Keep_installed_or_baseStartPC = 2103

func Keep_installed_or_base(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2103
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: key_dep_rows/3 (shared table, pc=2146)
var Key_dep_rowsCode = sharedWamCode
var Key_dep_rowsLabels = sharedWamLabels
const Key_dep_rowsStartPC = 2146

func Key_dep_rows(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2146
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: key_pkg_rows/3 (shared table, pc=2188)
var Key_pkg_rowsCode = sharedWamCode
var Key_pkg_rowsLabels = sharedWamLabels
const Key_pkg_rowsStartPC = 2188

func Key_pkg_rows(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2188
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layer_closure/3 (shared table, pc=2221)
var Layer_closureCode = sharedWamCode
var Layer_closureLabels = sharedWamLabels
const Layer_closureStartPC = 2221

func Layer_closure(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2221
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layer_provider/5 (shared table, pc=2238)
var Layer_providerCode = sharedWamCode
var Layer_providerLabels = sharedWamLabels
const Layer_providerStartPC = 2238

func Layer_provider(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2238
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layer_satisfies/3 (shared table, pc=2287)
var Layer_satisfiesCode = sharedWamCode
var Layer_satisfiesLabels = sharedWamLabels
const Layer_satisfiesStartPC = 2287

func Layer_satisfies(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2287
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layered_walk_ver/4 (shared table, pc=2358)
var Layered_walk_verCode = sharedWamCode
var Layered_walk_verLabels = sharedWamLabels
const Layered_walk_verStartPC = 2358

func Layered_walk_ver(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2358
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layers_list/2 (shared table, pc=2386)
var Layers_listCode = sharedWamCode
var Layers_listLabels = sharedWamLabels
const Layers_listStartPC = 2386

func Layers_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2386
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: list_to_tree/2 (shared table, pc=2435)
var List_to_treeCode = sharedWamCode
var List_to_treeLabels = sharedWamLabels
const List_to_treeStartPC = 2435

func List_to_tree(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2435
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: long_enough/2 (shared table, pc=2447)
var Long_enoughCode = sharedWamCode
var Long_enoughLabels = sharedWamLabels
const Long_enoughStartPC = 2447

func Long_enough(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2447
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: lookup_held/3 (shared table, pc=2471)
var Lookup_heldCode = sharedWamCode
var Lookup_heldLabels = sharedWamLabels
const Lookup_heldStartPC = 2471

func Lookup_held(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2471
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: map_requests/3 (shared table, pc=2495)
var Map_requestsCode = sharedWamCode
var Map_requestsLabels = sharedWamLabels
const Map_requestsStartPC = 2495

func Map_requests(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2495
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: matching_deps/4 (shared table, pc=2518)
var Matching_depsCode = sharedWamCode
var Matching_depsLabels = sharedWamLabels
const Matching_depsStartPC = 2518

func Matching_deps(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2518
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: matching_versions/4 (shared table, pc=2566)
var Matching_versionsCode = sharedWamCode
var Matching_versionsLabels = sharedWamLabels
const Matching_versionsStartPC = 2566

func Matching_versions(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2566
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: matching_versions_in/4 (shared table, pc=2608)
var Matching_versions_inCode = sharedWamCode
var Matching_versions_inLabels = sharedWamLabels
const Matching_versions_inStartPC = 2608

func Matching_versions_in(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2608
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: member_selected/3 (shared table, pc=2647)
var Member_selectedCode = sharedWamCode
var Member_selectedLabels = sharedWamLabels
const Member_selectedStartPC = 2647

func Member_selected(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2647
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: names_of/2 (shared table, pc=2656)
var Names_ofCode = sharedWamCode
var Names_ofLabels = sharedWamLabels
const Names_ofStartPC = 2656

func Names_of(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2656
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: needed_names/4 (shared table, pc=2675)
var Needed_namesCode = sharedWamCode
var Needed_namesLabels = sharedWamLabels
const Needed_namesStartPC = 2675

func Needed_names(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2675
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: no_acc_conflicts/4 (shared table, pc=2699)
var No_acc_conflictsCode = sharedWamCode
var No_acc_conflictsLabels = sharedWamLabels
const No_acc_conflictsStartPC = 2699

func No_acc_conflicts(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2699
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: order_lt/2 (shared table, pc=2746)
var Order_ltCode = sharedWamCode
var Order_ltLabels = sharedWamLabels
const Order_ltStartPC = 2746

func Order_lt(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2746
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: order_val/2 (shared table, pc=2813)
var Order_valCode = sharedWamCode
var Order_valLabels = sharedWamLabels
const Order_valStartPC = 2813

func Order_val(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2813
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: package_in/3 (shared table, pc=2855)
var Package_inCode = sharedWamCode
var Package_inLabels = sharedWamLabels
const Package_inStartPC = 2855

func Package_in(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2855
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: package_in_name/2 (shared table, pc=2869)
var Package_in_nameCode = sharedWamCode
var Package_in_nameLabels = sharedWamLabels
const Package_in_nameStartPC = 2869

func Package_in_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2869
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: packages/2 (shared table, pc=2882)
var PackagesCode = sharedWamCode
var PackagesLabels = sharedWamLabels
const PackagesStartPC = 2882

func Packages(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2882
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pad_head/2 (shared table, pc=2931)
var Pad_headCode = sharedWamCode
var Pad_headLabels = sharedWamLabels
const Pad_headStartPC = 2931

func Pad_head(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2931
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pick/7 (shared table, pc=2947)
var PickCode = sharedWamCode
var PickLabels = sharedWamLabels
const PickStartPC = 2947

func Pick(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2947
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    vm.Regs[6] = a7
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pick_need/8 (shared table, pc=3000)
var Pick_needCode = sharedWamCode
var Pick_needLabels = sharedWamLabels
const Pick_needStartPC = 3000

func Pick_need(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value, a8 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3000
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    vm.Regs[6] = a7
    vm.Regs[7] = a8
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pick_repair/4 (shared table, pc=3114)
var Pick_repairCode = sharedWamCode
var Pick_repairLabels = sharedWamLabels
const Pick_repairStartPC = 3114

func Pick_repair(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3114
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pkg_index/2 (shared table, pc=3130)
var Pkg_indexCode = sharedWamCode
var Pkg_indexLabels = sharedWamLabels
const Pkg_indexStartPC = 3130

func Pkg_index(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3130
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provide_row/5 (shared table, pc=3136)
var Provide_rowCode = sharedWamCode
var Provide_rowLabels = sharedWamLabels
const Provide_rowStartPC = 3136

func Provide_row(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3136
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provide_satisfies/2 (shared table, pc=3158)
var Provide_satisfiesCode = sharedWamCode
var Provide_satisfiesLabels = sharedWamLabels
const Provide_satisfiesStartPC = 3158

func Provide_satisfies(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3158
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provider_candidate/5 (shared table, pc=3173)
var Provider_candidateCode = sharedWamCode
var Provider_candidateLabels = sharedWamLabels
const Provider_candidateStartPC = 3173

func Provider_candidate(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3173
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provides_for/5 (shared table, pc=3209)
var Provides_forCode = sharedWamCode
var Provides_forLabels = sharedWamLabels
const Provides_forStartPC = 3209

func Provides_for(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3209
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provides_list/2 (shared table, pc=3228)
var Provides_listCode = sharedWamCode
var Provides_listLabels = sharedWamLabels
const Provides_listStartPC = 3228

func Provides_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3228
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provides_sat/5 (shared table, pc=3277)
var Provides_satCode = sharedWamCode
var Provides_satLabels = sharedWamLabels
const Provides_satStartPC = 3277

func Provides_sat(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3277
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: removal_orphans/3 (shared table, pc=3299)
var Removal_orphansCode = sharedWamCode
var Removal_orphansLabels = sharedWamLabels
const Removal_orphansStartPC = 3299

func Removal_orphans(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3299
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: repairs_moving/4 (shared table, pc=3397)
var Repairs_movingCode = sharedWamCode
var Repairs_movingLabels = sharedWamLabels
const Repairs_movingStartPC = 3397

func Repairs_moving(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3397
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: reqs_ok_moving/2 (shared table, pc=3411)
var Reqs_ok_movingCode = sharedWamCode
var Reqs_ok_movingLabels = sharedWamLabels
const Reqs_ok_movingStartPC = 3411

func Reqs_ok_moving(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3411
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: request_to_req/3 (shared table, pc=3475)
var Request_to_reqCode = sharedWamCode
var Request_to_reqLabels = sharedWamLabels
const Request_to_reqStartPC = 3475

func Request_to_req(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3475
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: requested_list/2 (shared table, pc=3504)
var Requested_listCode = sharedWamCode
var Requested_listLabels = sharedWamLabels
const Requested_listStartPC = 3504

func Requested_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3504
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve/3 (shared table, pc=3553)
var ResolveCode = sharedWamCode
var ResolveLabels = sharedWamLabels
const ResolveStartPC = 3553

func Resolve(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3553
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_alternatives/7 (shared table, pc=3576)
var Resolve_alternativesCode = sharedWamCode
var Resolve_alternativesLabels = sharedWamLabels
const Resolve_alternativesStartPC = 3576

func Resolve_alternatives(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3576
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    vm.Regs[6] = a7
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_layered/3 (shared table, pc=3620)
var Resolve_layeredCode = sharedWamCode
var Resolve_layeredLabels = sharedWamLabels
const Resolve_layeredStartPC = 3620

func Resolve_layered(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3620
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_pending/5 (shared table, pc=3643)
var Resolve_pending5Code = sharedWamCode
var Resolve_pending5Labels = sharedWamLabels
const Resolve_pending5StartPC = 3643

func Resolve_pending5(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3643
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_pending/6 (shared table, pc=3659)
var Resolve_pending6Code = sharedWamCode
var Resolve_pending6Labels = sharedWamLabels
const Resolve_pending6StartPC = 3659

func Resolve_pending6(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3659
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: roots_to_pairs/3 (shared table, pc=3877)
var Roots_to_pairsCode = sharedWamCode
var Roots_to_pairsLabels = sharedWamLabels
const Roots_to_pairsStartPC = 3877

func Roots_to_pairs(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3877
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: safe_upgrade/4 (shared table, pc=3917)
var Safe_upgradeCode = sharedWamCode
var Safe_upgradeLabels = sharedWamLabels
const Safe_upgradeStartPC = 3917

func Safe_upgrade(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3917
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: safe_upgrade_reason/5 (shared table, pc=3977)
var Safe_upgrade_reasonCode = sharedWamCode
var Safe_upgrade_reasonLabels = sharedWamLabels
const Safe_upgrade_reasonStartPC = 3977

func Safe_upgrade_reason(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3977
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: same_key/4 (shared table, pc=4030)
var Same_keyCode = sharedWamCode
var Same_keyLabels = sharedWamLabels
const Same_keyStartPC = 4030

func Same_key(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4030
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: satisfies/2 (shared table, pc=4084)
var SatisfiesCode = sharedWamCode
var SatisfiesLabels = sharedWamLabels
const SatisfiesStartPC = 4084

func Satisfies(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4084
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: scan_base_holds/3 (shared table, pc=4164)
var Scan_base_holdsCode = sharedWamCode
var Scan_base_holdsLabels = sharedWamLabels
const Scan_base_holdsStartPC = 4164

func Scan_base_holds(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4164
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: seen_name/2 (shared table, pc=4252)
var Seen_nameCode = sharedWamCode
var Seen_nameLabels = sharedWamLabels
const Seen_nameStartPC = 4252

func Seen_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4252
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: segs_lt/2 (shared table, pc=4271)
var Segs_ltCode = sharedWamCode
var Segs_ltLabels = sharedWamLabels
const Segs_ltStartPC = 4271

func Segs_lt(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4271
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: segs_lt_1/2 (shared table, pc=4293)
var Segs_lt_1Code = sharedWamCode
var Segs_lt_1Labels = sharedWamLabels
const Segs_lt_1StartPC = 4293

func Segs_lt_1(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4293
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: selected_ver/3 (shared table, pc=4338)
var Selected_verCode = sharedWamCode
var Selected_verLabels = sharedWamLabels
const Selected_verStartPC = 4338

func Selected_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4338
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: sort_versions_desc/2 (shared table, pc=4361)
var Sort_versions_descCode = sharedWamCode
var Sort_versions_descLabels = sharedWamLabels
const Sort_versions_descStartPC = 4361

func Sort_versions_desc(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4361
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_base_revdep/2 (shared table, pc=4387)
var Tight_base_revdepCode = sharedWamCode
var Tight_base_revdepLabels = sharedWamLabels
const Tight_base_revdepStartPC = 4387

func Tight_base_revdep(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4387
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_constraint/1 (shared table, pc=4398)
var Tight_constraintCode = sharedWamCode
var Tight_constraintLabels = sharedWamLabels
const Tight_constraintStartPC = 4398

func Tight_constraint(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4398
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_rev_in/3 (shared table, pc=4403)
var Tight_rev_inCode = sharedWamCode
var Tight_rev_inLabels = sharedWamLabels
const Tight_rev_inStartPC = 4403

func Tight_rev_in(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4403
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: topo_all/7 (shared table, pc=4452)
var Topo_allCode = sharedWamCode
var Topo_allLabels = sharedWamLabels
const Topo_allStartPC = 4452

func Topo_all(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4452
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    vm.Regs[6] = a7
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: topo_one/7 (shared table, pc=4489)
var Topo_oneCode = sharedWamCode
var Topo_oneLabels = sharedWamLabels
const Topo_oneStartPC = 4489

func Topo_one(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4489
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    vm.Regs[6] = a7
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: topo_sort_sel/3 (shared table, pc=4567)
var Topo_sort_selCode = sharedWamCode
var Topo_sort_selLabels = sharedWamLabels
const Topo_sort_selStartPC = 4567

func Topo_sort_sel(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4567
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tree_lookup/3 (shared table, pc=4599)
var Tree_lookupCode = sharedWamCode
var Tree_lookupLabels = sharedWamLabels
const Tree_lookupStartPC = 4599

func Tree_lookup(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4599
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: upgrade_set/4 (shared table, pc=4640)
var Upgrade_setCode = sharedWamCode
var Upgrade_setLabels = sharedWamLabels
const Upgrade_setStartPC = 4640

func Upgrade_set(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4640
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: upgrade_set_result/4 (shared table, pc=4654)
var Upgrade_set_resultCode = sharedWamCode
var Upgrade_set_resultLabels = sharedWamLabels
const Upgrade_set_resultStartPC = 4654

func Upgrade_set_result(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4654
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: version_lt/2 (shared table, pc=4687)
var Version_ltCode = sharedWamCode
var Version_ltLabels = sharedWamLabels
const Version_ltStartPC = 4687

func Version_lt(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4687
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: virtual_provider_ceilings/4 (shared table, pc=4789)
var Virtual_provider_ceilingsCode = sharedWamCode
var Virtual_provider_ceilingsLabels = sharedWamLabels
const Virtual_provider_ceilingsStartPC = 4789

func Virtual_provider_ceilings(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4789
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: walk_pkg_for_blocked/5 (shared table, pc=4839)
var Walk_pkg_for_blockedCode = sharedWamCode
var Walk_pkg_for_blockedLabels = sharedWamLabels
const Walk_pkg_for_blockedStartPC = 4839

func Walk_pkg_for_blocked(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4839
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: worth_indexing/2 (shared table, pc=4880)
var Worth_indexingCode = sharedWamCode
var Worth_indexingLabels = sharedWamLabels
const Worth_indexingStartPC = 4880

func Worth_indexing(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4880
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}

