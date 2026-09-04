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
        &TryMeElse{Label: "L_ite_else_2", Arity: 3},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_2"},
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
        &TryMeElse{Label: "L_ite_else_3", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &PutValue{Xn: 202, Ai: 4},
        &Call{Pred: "explain_alt/5", Arity: 5},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_3"},
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
        &TryMeElse{Label: "L_ite_else_4", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom_blanket_2, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 207},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_5", Arity: 4},
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
        &Jump{Label: "L_ite_cont_5"},
        &TrustMe{},
        &PutVariable{Xn: 204, Ai: 0},
        &PutStructure{Functor: "audit/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetConstant{C: wamAtom_over_frozen_4},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_4"},
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
        &GetVariable{Xn: 102, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 103},
        &GetVariable{Xn: 200, Ai: 2},
        &GetVariable{Xn: 104, Ai: 3},
        &GetValue{Xn: 104, Ai: 4},
        &PutValue{Xn: 201, Ai: 0},
        &BuiltinCall{Op: "atom/1", Arity: 1},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_blocked_acc_5_3", Arity: 5},
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
        &GetVariable{Xn: 206, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 208},
        &UnifyVariable{Xn: 203},
        &GetVariable{Xn: 209, Ai: 2},
        &GetVariable{Xn: 202, Ai: 3},
        &GetVariable{Xn: 210, Ai: 4},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_6", Arity: 5},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_7", Arity: 5},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 213},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_7"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 212},
        &PutVariable{Xn: 211, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 113},
        &SetValue{Xn: 202},
        &PutStructure{Functor: "blocked/3", Ai: 113},
        &SetValue{Xn: 208},
        &SetVariable{Xn: 114},
        &SetVariable{Xn: 115},
        &PutStructure{Functor: "needs/1", Ai: 114},
        &SetValue{Xn: 203},
        &PutStructure{Functor: "base_has/1", Ai: 115},
        &SetValue{Xn: 200},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_6"},
        &TrustMe{},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_8", Arity: 5},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 201, Ai: 3},
        &Call{Pred: "virtual_provider_ceilings/4", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &Cut{Reg: 213},
        &PutVariable{Xn: 211, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 113},
        &SetValue{Xn: 202},
        &PutStructure{Functor: "blocked/3", Ai: 113},
        &SetValue{Xn: 208},
        &SetVariable{Xn: 114},
        &SetVariable{Xn: 115},
        &PutStructure{Functor: "needs/1", Ai: 114},
        &SetValue{Xn: 203},
        &PutStructure{Functor: "providers/1", Ai: 115},
        &SetValue{Xn: 201},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_8"},
        &TrustMe{},
        &PutVariable{Xn: 211, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 214},
        &TryMeElse{Label: "L_ite_else_9", Arity: 5},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 204, Ai: 3},
        &PutVariable{Xn: 205, Ai: 4},
        &Call{Pred: "walk_pkg_for_blocked/5", Arity: 5},
        &Cut{Reg: 214},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutVariable{Xn: 207, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutStructure{Functor: "[|]/2", Ai: 2},
        &SetValue{Xn: 208},
        &SetValue{Xn: 209},
        &PutValue{Xn: 211, Ai: 3},
        &PutValue{Xn: 210, Ai: 4},
        &Call{Pred: "blocked_acc_list/5", Arity: 5},
        &Jump{Label: "L_ite_cont_9"},
        &TrustMe{},
        &PutValue{Xn: 210, Ai: 0},
        &PutValue{Xn: 211, Ai: 1},
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
        &GetVariable{Xn: 200, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 203},
        &GetVariable{Xn: 105, Ai: 2},
        &GetVariable{Xn: 201, Ai: 3},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_10", Arity: 4},
        &PutValue{Xn: 105, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Cut{Reg: 205},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_10"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 204, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_11", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_11"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "blocked/3", Ai: 1},
        &SetValue{Xn: 202},
        &SetVariable{Xn: 107},
        &SetVariable{Xn: 108},
        &PutStructure{Functor: "needs/1", Ai: 107},
        &SetValue{Xn: 203},
        &PutStructure{Functor: "base_has/1", Ai: 108},
        &SetValue{Xn: 204},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &RetryMeElse{Label: "L_blocked_from_4_4", Arity: 4},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 203},
        &GetVariable{Xn: 105, Ai: 2},
        &GetVariable{Xn: 201, Ai: 3},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_12", Arity: 4},
        &PutValue{Xn: 105, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Cut{Reg: 205},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_12"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 204, Ai: 3},
        &Call{Pred: "virtual_provider_ceilings/4", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "blocked/3", Ai: 1},
        &SetValue{Xn: 202},
        &SetVariable{Xn: 107},
        &SetVariable{Xn: 108},
        &PutStructure{Functor: "needs/1", Ai: 107},
        &SetValue{Xn: 203},
        &PutStructure{Functor: "providers/1", Ai: 108},
        &SetValue{Xn: 204},
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
        &TryMeElse{Label: "L_ite_else_13", Arity: 4},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Cut{Reg: 209},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_13"},
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
        &TryMeElse{Label: "L_ite_else_14", Arity: 4},
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
        &Jump{Label: "L_ite_cont_14"},
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
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_15", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "excluded_name/2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_15"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutVariable{Xn: 203, Ai: 3},
        &Call{Pred: "matching_versions_in/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutVariable{Xn: 205, Ai: 1},
        &Call{Pred: "sort_versions_desc/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
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
        &TryMeElse{Label: "L_ite_else_16", Arity: 3},
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
        &Jump{Label: "L_ite_cont_16"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "broken/3", Ai: 1},
        &SetVariable{Xn: 204},
        &SetVariable{Xn: 209},
        &SetVariable{Xn: 208},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_17", Arity: 3},
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
        &Jump{Label: "L_ite_cont_17"},
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
        &TryMeElse{Label: "L_ite_else_18", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "dep_index/2", Arity: 2},
        &Cut{Reg: 207},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_19", Arity: 4},
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
        &Jump{Label: "L_ite_cont_19"},
        &TrustMe{},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_18"},
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
        &TryMeElse{Label: "L_ite_else_20", Arity: 5},
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
        &Jump{Label: "L_ite_cont_20"},
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
        &TryMeElse{Label: "L_ite_else_21", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_21"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_22", Arity: 4},
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
        &Jump{Label: "L_ite_cont_22"},
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
        &TryMeElse{Label: "L_ite_else_23", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 202},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_23"},
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
        &TryMeElse{Label: "L_ite_else_24", Arity: 5},
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
        &Jump{Label: "L_ite_cont_24"},
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
        &TryMeElse{Label: "L_ite_else_25", Arity: 4},
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
        &Jump{Label: "L_ite_cont_25"},
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
        &TryMeElse{Label: "L_ite_else_26", Arity: 5},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "req/2", Ai: 1},
        &SetValue{Xn: 202},
        &SetValue{Xn: 203},
        &PutValue{Xn: 200, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &Call{Pred: "blocked_from/4", Arity: 4},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_26"},
        &TrustMe{},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_27", Arity: 5},
        &GetLevel{Reg: 210},
        &TryMeElse{Label: "L_ite_else_28", Arity: 5},
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
        &Jump{Label: "L_ite_cont_28"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 209},
        &PutValue{Xn: 207, Ai: 0},
        &PutConstant{C: wamAtom_unsatisfiable_1, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_27"},
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
        &TryMeElse{Label: "L_ite_else_29", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetVariable{Xn: 204},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_29"},
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
        &TryMeElse{Label: "L_ite_else_30", Arity: 4},
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
        &Jump{Label: "L_ite_cont_30"},
        &TrustMe{},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_31", Arity: 4},
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
        &Jump{Label: "L_ite_cont_31"},
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
        &TryMeElse{Label: "L_ite_else_32", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_32"},
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
        &TryMeElse{Label: "L_ite_else_33", Arity: 2},
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
        &Jump{Label: "L_ite_cont_33"},
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
        &TryMeElse{Label: "L_ite_else_34", Arity: 6},
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
        &Jump{Label: "L_ite_cont_34"},
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
        &TryMeElse{Label: "L_ite_else_35", Arity: 4},
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
        &Jump{Label: "L_ite_cont_35"},
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
        &TryMeElse{Label: "L_ite_else_36", Arity: 3},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_36"},
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
        &TryMeElse{Label: "L_ite_else_37", Arity: 4},
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
        &Jump{Label: "L_ite_cont_37"},
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
        &TryMeElse{Label: "L_ite_else_38", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: &Integer{Val: 1}, Ai: 1},
        &BuiltinCall{Op: "=</2", Arity: 2},
        &Cut{Reg: 203},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_38"},
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
        &TryMeElse{Label: "L_ite_else_39", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "item_ver/3", Arity: 3},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_39"},
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
        &TryMeElse{Label: "L_ite_else_40", Arity: 4},
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
        &Jump{Label: "L_ite_cont_40"},
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
        &TryMeElse{Label: "L_ite_else_41", Arity: 4},
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
        &Jump{Label: "L_ite_cont_41"},
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
        &TryMeElse{Label: "L_ite_else_42", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "pkg_index/2", Arity: 2},
        &Cut{Reg: 207},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_43", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "tree_lookup/3", Arity: 3},
        &Cut{Reg: 208},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &Call{Pred: "filter_satisfies/3", Arity: 3},
        &Jump{Label: "L_ite_cont_43"},
        &TrustMe{},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_42"},
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
        &TryMeElse{Label: "L_ite_else_44", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &Call{Pred: "conflicts_in/4", Arity: 4},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_44"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_45", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Call{Pred: "conflicts_in/4", Arity: 4},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_45"},
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
        &TryMeElse{Label: "L_ite_else_46", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 205},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_46"},
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
        &TryMeElse{Label: "L_ite_else_47", Arity: 7},
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
        &Jump{Label: "L_ite_cont_47"},
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
        &GetVariable{Xn: 107, Ai: 4},
        &GetVariable{Xn: 204, Ai: 5},
        &GetVariable{Xn: 205, Ai: 6},
        &GetVariable{Xn: 206, Ai: 7},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_48", Arity: 8},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &Cut{Reg: 207},
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
        &Jump{Label: "L_ite_cont_48"},
        &TrustMe{},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_49", Arity: 8},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutValue{Xn: 205, Ai: 4},
        &Call{Pred: "layer_provider/5", Arity: 5},
        &Cut{Reg: 208},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom_from_base_14, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_49"},
        &TrustMe{},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_50", Arity: 8},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &Call{Pred: "candidates_high_first/4", Arity: 4},
        &Cut{Reg: 209},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutConstant{C: wamAtom_from_catalog_13, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_50"},
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
        &TryMeElse{Label: "L_ite_else_51", Arity: 5},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &Call{Pred: "excluded_name/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_51"},
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
        &TryMeElse{Label: "L_ite_else_52", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "installed_ver/3", Arity: 3},
        &Cut{Reg: 212},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_52"},
        &TrustMe{},
        &PutVariable{Xn: 200, Ai: 0},
        &PutConstant{C: wamAtom_none_6, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_53", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: wamAtom_none_6, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 213},
        &PutValue{Xn: 211, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_53"},
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
        &TryMeElse{Label: "L_ite_else_54", Arity: 3},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Cut{Reg: 214},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_54"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 215},
        &TryMeElse{Label: "L_ite_else_55", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "base_name/2", Arity: 2},
        &Cut{Reg: 215},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_55"},
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
        &TryMeElse{Label: "L_ite_else_56", Arity: 2},
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
        &Jump{Label: "L_ite_cont_56"},
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
        &TryMeElse{Label: "L_ite_else_57", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Cut{Reg: 205},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Jump{Label: "L_ite_cont_57"},
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
        &TryMeElse{Label: "L_ite_else_58", Arity: 3},
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
        &Jump{Label: "L_ite_cont_58"},
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
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_59", Arity: 6},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &Call{Pred: "first_alt_already/4", Arity: 4},
        &Cut{Reg: 208},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &PutValue{Xn: 207, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &Jump{Label: "L_ite_cont_59"},
        &TrustMe{},
        &PutStructure{Functor: "dep/2", Ai: 0},
        &SetVariable{Xn: 203},
        &SetVariable{Xn: 204},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutStructure{Functor: "[|]/2", Ai: 2},
        &SetVariable{Xn: 110},
        &SetValue{Xn: 205},
        &PutStructure{Functor: "req/2", Ai: 110},
        &SetValue{Xn: 203},
        &SetValue{Xn: 204},
        &PutValue{Xn: 206, Ai: 3},
        &PutValue{Xn: 207, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
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
        &TryMeElse{Label: "L_resolve_pending_5_2", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetConstant{C: wamAtom____0, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetValue{Xn: 102, Ai: 4},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 205, Ai: 0},
        &GetVariable{Xn: 206, Ai: 1},
        &GetList{Ai: 2},
        &UnifyVariable{Xn: 113},
        &GetStructure{Functor: "req/2", Ai: 113},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 204},
        &UnifyVariable{Xn: 207},
        &GetVariable{Xn: 201, Ai: 3},
        &GetVariable{Xn: 208, Ai: 4},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_60", Arity: 5},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "alternatives/1", Ai: 1},
        &SetVariable{Xn: 200},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 213},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 200, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &PutValue{Xn: 201, Ai: 4},
        &PutValue{Xn: 208, Ai: 5},
        &Call{Pred: "resolve_alternatives/6", Arity: 6},
        &Jump{Label: "L_ite_cont_60"},
        &TrustMe{},
        &GetLevel{Reg: 214},
        &TryMeElse{Label: "L_ite_else_61", Arity: 5},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 203, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Cut{Reg: 214},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 201, Ai: 3},
        &PutValue{Xn: 208, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &Jump{Label: "L_ite_cont_61"},
        &TrustMe{},
        &GetLevel{Reg: 215},
        &TryMeElse{Label: "L_ite_else_62", Arity: 5},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &Call{Pred: "already_provided/4", Arity: 4},
        &Cut{Reg: 215},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 201, Ai: 3},
        &PutValue{Xn: 208, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &Jump{Label: "L_ite_cont_62"},
        &TrustMe{},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &PutValue{Xn: 201, Ai: 4},
        &PutVariable{Xn: 209, Ai: 5},
        &PutVariable{Xn: 203, Ai: 6},
        &PutVariable{Xn: 210, Ai: 7},
        &Call{Pred: "pick_need/8", Arity: 8},
        &GetLevel{Reg: 216},
        &TryMeElse{Label: "L_ite_else_63", Arity: 5},
        &PutValue{Xn: 210, Ai: 0},
        &PutConstant{C: wamAtom_from_base_14, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 216},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 211, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 211, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutVariable{Xn: 212, Ai: 2},
        &BuiltinCall{Op: "append/3", Arity: 3},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 212, Ai: 2},
        &PutValue{Xn: 201, Ai: 3},
        &PutValue{Xn: 208, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &Jump{Label: "L_ite_cont_63"},
        &TrustMe{},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 201, Ai: 3},
        &Call{Pred: "no_acc_conflicts/4", Arity: 4},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 211, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 211, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutVariable{Xn: 212, Ai: 2},
        &BuiltinCall{Op: "append/3", Arity: 3},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 212, Ai: 2},
        &PutStructure{Functor: "[|]/2", Ai: 3},
        &SetVariable{Xn: 115},
        &SetValue{Xn: 201},
        &PutStructure{Functor: "-/2", Ai: 115},
        &SetValue{Xn: 209},
        &SetValue{Xn: 203},
        &PutValue{Xn: 208, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
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
        &TryMeElse{Label: "L_ite_else_64", Arity: 4},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_65", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &Call{Pred: "package_in/3", Arity: 3},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_65"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 206},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_no_candidate_16, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_64"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_66", Arity: 4},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_67", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutVariable{Xn: 202, Ai: 2},
        &Call{Pred: "base_reason/3", Arity: 3},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_67"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 207},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_no_candidate_16, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_66"},
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
        &TryMeElse{Label: "L_ite_else_68", Arity: 4},
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
        &Jump{Label: "L_ite_cont_68"},
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
        &TryMeElse{Label: "L_ite_else_69", Arity: 2},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &Cut{Reg: 200},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_69"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Proceed{},
        &RetryMeElse{Label: "L_satisfies_2_5", Arity: 2},
        &GetVariable{Xn: 100, Ai: 0},
        &GetStructure{Functor: "lte/1", Ai: 1},
        &UnifyVariable{Xn: 101},
        &GetLevel{Reg: 200},
        &TryMeElse{Label: "L_ite_else_70", Arity: 2},
        &PutValue{Xn: 101, Ai: 0},
        &PutValue{Xn: 100, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &Cut{Reg: 200},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_70"},
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
        &TryMeElse{Label: "L_ite_else_71", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 102, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &Cut{Reg: 202},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_71"},
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
        &TryMeElse{Label: "L_ite_else_72", Arity: 3},
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
        &Jump{Label: "L_ite_cont_72"},
        &TrustMe{},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_73", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "layer/2", Ai: 1},
        &SetVariable{Xn: 202},
        &SetVariable{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 212},
        &PutVariable{Xn: 209, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_73"},
        &TrustMe{},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_74", Arity: 3},
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
        &Jump{Label: "L_ite_cont_74"},
        &TrustMe{},
        &GetLevel{Reg: 214},
        &TryMeElse{Label: "L_ite_else_75", Arity: 3},
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
        &Jump{Label: "L_ite_cont_75"},
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
        &TryMeElse{Label: "L_ite_else_76", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 203},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_76"},
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
        &TryMeElse{Label: "L_ite_else_77", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "order_lt/2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_77"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_78", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_78"},
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
        &TryMeElse{Label: "L_ite_else_79", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutStructure{Functor: "-/2", Ai: 1},
        &SetValue{Xn: 202},
        &SetValue{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 204},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_79"},
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
        &TryMeElse{Label: "L_ite_else_80", Arity: 2},
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
        &Jump{Label: "L_ite_cont_80"},
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
        &TryMeElse{Label: "L_ite_else_81", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutVariable{Xn: 202, Ai: 3},
        &PutVariable{Xn: 203, Ai: 4},
        &Call{Pred: "dep_targets/5", Arity: 5},
        &TryMeElse{Label: "L_ite_else_82", Arity: 3},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &Call{Pred: "tight_constraint/1", Arity: 1},
        &Jump{Label: "L_ite_cont_82"},
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
        &Jump{Label: "L_ite_cont_81"},
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
        &TryMeElse{Label: "L_ite_else_83", Arity: 7},
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
        &Jump{Label: "L_ite_cont_83"},
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
        &TryMeElse{Label: "L_ite_else_84", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom___9, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 206},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_84"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_85", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom___7, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 207},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Call{Pred: "tree_lookup/3", Arity: 3},
        &Jump{Label: "L_ite_cont_85"},
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
        &TryMeElse{Label: "L_ite_else_86", Arity: 4},
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
        &Jump{Label: "L_ite_cont_86"},
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
        &TryMeElse{Label: "L_ite_else_87", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_87"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_88", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_88"},
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
        &TryMeElse{Label: "L_ite_else_89", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_89"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_90", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "segs_lt/2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_90"},
        &TrustMe{},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_91", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "segs_lt/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_91"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_92", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "segs_lt/2", Arity: 2},
        &Cut{Reg: 209},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_92"},
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
        &TryMeElse{Label: "L_ite_else_93", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &Call{Pred: "package_in_name/2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_93"},
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
        &TryMeElse{Label: "L_ite_else_94", Arity: 4},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &Call{Pred: "provide_satisfies/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_94"},
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
        &TryMeElse{Label: "L_ite_else_95", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "long_enough/2", Arity: 2},
        &Cut{Reg: 203},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_95"},
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
        "alias_list/2": 24,
        "L_alias_list_2_2": 35,
        "L_alias_list_2_2_body": 36,
        "L_alias_list_2_3": 48,
        "L_alias_list_2_3_body": 49,
        "L_alias_list_2_4": 62,
        "L_alias_list_2_4_body": 63,
        "alias_lookup/3": 73,
        "L_alias_lookup_3_2": 78,
        "L_alias_lookup_3_2_body": 79,
        "L_ite_else_2": 98,
        "L_ite_cont_2": 103,
        "already_provided/4": 105,
        "already_satisfied/4": 122,
        "L_already_satisfied_4_2": 136,
        "L_already_satisfied_4_2_body": 137,
        "alt_reasons/4": 148,
        "L_alt_reasons_4_2": 154,
        "L_alt_reasons_4_2_body": 155,
        "L_ite_else_3": 181,
        "L_ite_cont_3": 185,
        "audit_holds/4": 191,
        "L_audit_holds_4_2": 197,
        "L_audit_holds_4_2_body": 198,
        "L_ite_else_5": 229,
        "L_ite_cont_5": 235,
        "L_ite_else_4": 236,
        "L_ite_cont_4": 244,
        "base_holds/2": 252,
        "base_list/2": 267,
        "L_base_list_2_2": 278,
        "L_base_list_2_2_body": 279,
        "L_base_list_2_3": 291,
        "L_base_list_2_3_body": 292,
        "L_base_list_2_4": 305,
        "L_base_list_2_4_body": 306,
        "base_name/2": 316,
        "base_reason/3": 324,
        "base_ver/3": 336,
        "blocked_acc/5": 355,
        "L_blocked_acc_5_2": 372,
        "L_blocked_acc_5_2_body": 373,
        "L_blocked_acc_5_3": 400,
        "L_blocked_acc_5_3_body": 401,
        "L_ite_else_7": 423,
        "L_ite_cont_7": 425,
        "L_ite_else_6": 440,
        "L_ite_else_8": 466,
        "L_ite_cont_8": 470,
        "L_ite_cont_6": 470,
        "L_ite_else_9": 493,
        "L_ite_cont_9": 497,
        "blocked_acc_list/5": 499,
        "L_blocked_acc_list_5_2": 506,
        "L_blocked_acc_list_5_2_body": 507,
        "blocked_from/4": 528,
        "L_blocked_from_4_2": 552,
        "L_blocked_from_4_2_body": 553,
        "L_ite_else_10": 568,
        "L_ite_cont_10": 570,
        "L_ite_else_11": 582,
        "L_ite_cont_11": 584,
        "L_blocked_from_4_3": 596,
        "L_blocked_from_4_3_body": 597,
        "L_ite_else_12": 612,
        "L_ite_cont_12": 614,
        "L_blocked_from_4_4": 634,
        "L_blocked_from_4_4_body": 635,
        "L_ite_else_13": 650,
        "L_ite_cont_13": 652,
        "build_tree/4": 674,
        "L_ite_else_14": 692,
        "L_ite_cont_14": 731,
        "candidates_high_first/4": 733,
        "L_ite_else_15": 746,
        "L_ite_cont_15": 748,
        "canonicalize_name/3": 761,
        "close_moving/3": 773,
        "L_ite_else_16": 799,
        "L_ite_else_17": 824,
        "L_ite_cont_17": 835,
        "L_ite_cont_16": 835,
        "cmp_ver/3": 837,
        "L_cmp_ver_3_2": 849,
        "L_cmp_ver_3_2_body": 850,
        "L_cmp_ver_3_3": 860,
        "L_cmp_ver_3_3_body": 861,
        "collect_deps/4": 865,
        "L_ite_else_19": 889,
        "L_ite_cont_19": 893,
        "L_ite_else_18": 894,
        "L_ite_cont_18": 903,
        "conflicts_in/4": 905,
        "conflicts_list/2": 921,
        "L_conflicts_list_2_2": 932,
        "L_conflicts_list_2_2_body": 933,
        "L_conflicts_list_2_3": 945,
        "L_conflicts_list_2_3_body": 946,
        "L_conflicts_list_2_4": 959,
        "L_conflicts_list_2_4_body": 960,
        "dep_breaks/5": 970,
        "L_ite_else_20": 1001,
        "L_ite_cont_20": 1008,
        "dep_breaks_moving/5": 1010,
        "dep_breaks_need/4": 1026,
        "L_ite_else_21": 1051,
        "L_ite_cont_21": 1053,
        "L_ite_else_22": 1070,
        "L_ite_cont_22": 1072,
        "L_dep_breaks_need_4_2": 1074,
        "L_dep_breaks_need_4_2_body": 1075,
        "L_ite_else_23": 1092,
        "L_ite_cont_23": 1094,
        "dep_index/2": 1096,
        "dep_mentions/2": 1102,
        "L_dep_mentions_2_2": 1115,
        "L_dep_mentions_2_2_body": 1116,
        "dep_targets/5": 1124,
        "L_ite_else_24": 1149,
        "L_ite_cont_24": 1156,
        "dep_to_req/3": 1158,
        "L_dep_to_req_3_2": 1171,
        "L_dep_to_req_3_2_body": 1172,
        "dependents/3": 1178,
        "dependents_installed/3": 1200,
        "depends_in/5": 1219,
        "depends_list/2": 1237,
        "L_depends_list_2_2": 1248,
        "L_depends_list_2_2_body": 1249,
        "L_depends_list_2_3": 1261,
        "L_depends_list_2_3_body": 1262,
        "L_depends_list_2_4": 1275,
        "L_depends_list_2_4_body": 1276,
        "direct_on/4": 1286,
        "L_direct_on_4_2": 1292,
        "L_direct_on_4_2_body": 1293,
        "L_ite_else_25": 1320,
        "L_ite_cont_25": 1324,
        "exclude_name/3": 1330,
        "L_exclude_name_3_2": 1335,
        "L_exclude_name_3_2_body": 1336,
        "L_exclude_name_3_3": 1348,
        "L_exclude_name_3_3_body": 1349,
        "excluded_list/2": 1362,
        "L_excluded_list_2_2": 1373,
        "L_excluded_list_2_2_body": 1374,
        "L_excluded_list_2_3": 1386,
        "L_excluded_list_2_3_body": 1387,
        "L_excluded_list_2_4": 1400,
        "L_excluded_list_2_4_body": 1401,
        "excluded_name/2": 1411,
        "explain_alt/5": 1422,
        "L_ite_else_26": 1440,
        "L_ite_else_28": 1457,
        "L_ite_cont_28": 1459,
        "L_ite_else_27": 1464,
        "L_ite_cont_27": 1466,
        "L_ite_cont_26": 1466,
        "explain_blocked/3": 1468,
        "explain_blocked_list/3": 1482,
        "filter_satisfies/3": 1502,
        "L_filter_satisfies_3_2": 1507,
        "L_filter_satisfies_3_2_body": 1508,
        "L_ite_else_29": 1526,
        "L_ite_cont_29": 1530,
        "first_alt_already/4": 1535,
        "L_first_alt_already_4_2": 1554,
        "L_first_alt_already_4_2_body": 1555,
        "first_broken/4": 1572,
        "L_first_broken_4_2": 1578,
        "L_first_broken_4_2_body": 1579,
        "L_ite_else_30": 1603,
        "L_ite_else_31": 1620,
        "L_ite_cont_31": 1626,
        "L_ite_cont_30": 1626,
        "follow_dep_name/5": 1628,
        "follow_raw_dep/4": 1646,
        "L_follow_raw_dep_4_2": 1669,
        "L_follow_raw_dep_4_2_body": 1670,
        "freeze_audit/2": 1684,
        "group_keyed/2": 1701,
        "L_group_keyed_2_2": 1705,
        "L_group_keyed_2_2_body": 1706,
        "hold_reason/3": 1734,
        "L_ite_else_32": 1754,
        "L_ite_cont_32": 1759,
        "index_catalog/2": 1761,
        "L_ite_else_33": 1811,
        "L_ite_cont_33": 1815,
        "index_threshold/1": 1817,
        "inst_closure_names/5": 1819,
        "inst_walk/6": 1838,
        "L_inst_walk_6_2": 1846,
        "L_inst_walk_6_2_body": 1847,
        "L_ite_else_34": 1873,
        "L_ite_cont_34": 1908,
        "installed_list/2": 1910,
        "L_installed_list_2_2": 1921,
        "L_installed_list_2_2_body": 1922,
        "L_installed_list_2_3": 1934,
        "L_installed_list_2_3_body": 1935,
        "L_installed_list_2_4": 1948,
        "L_installed_list_2_4_body": 1949,
        "installed_or_base/3": 1959,
        "L_installed_or_base_3_2": 1969,
        "L_installed_or_base_3_2_body": 1970,
        "installed_ver/3": 1983,
        "is_public_catalog/1": 1997,
        "L_is_public_catalog_1_2": 2007,
        "L_is_public_catalog_1_2_body": 2008,
        "L_is_public_catalog_1_3": 2019,
        "L_is_public_catalog_1_3_body": 2020,
        "is_v3/1": 2032,
        "item_ver/3": 2037,
        "L_item_ver_3_2": 2050,
        "L_item_ver_3_2_body": 2051,
        "L_item_ver_3_3": 2065,
        "L_item_ver_3_3_body": 2066,
        "keep_installed_or_base/4": 2077,
        "L_keep_installed_or_base_4_2": 2083,
        "L_keep_installed_or_base_4_2_body": 2084,
        "L_ite_else_35": 2110,
        "L_ite_cont_35": 2114,
        "key_dep_rows/3": 2120,
        "L_key_dep_rows_3_2": 2125,
        "L_key_dep_rows_3_2_body": 2126,
        "key_pkg_rows/3": 2162,
        "L_key_pkg_rows_3_2": 2167,
        "L_key_pkg_rows_3_2_body": 2168,
        "layer_closure/3": 2195,
        "layer_provider/5": 2212,
        "L_layer_provider_5_2": 2235,
        "L_layer_provider_5_2_body": 2236,
        "layer_satisfies/3": 2261,
        "L_layer_satisfies_3_2": 2274,
        "L_layer_satisfies_3_2_body": 2275,
        "L_layer_satisfies_3_3": 2295,
        "L_layer_satisfies_3_3_body": 2296,
        "L_ite_else_36": 2323,
        "L_ite_cont_36": 2330,
        "layered_walk_ver/4": 2332,
        "L_ite_else_37": 2351,
        "L_ite_cont_37": 2357,
        "layers_list/2": 2360,
        "L_layers_list_2_2": 2371,
        "L_layers_list_2_2_body": 2372,
        "L_layers_list_2_3": 2384,
        "L_layers_list_2_3_body": 2385,
        "L_layers_list_2_4": 2398,
        "L_layers_list_2_4_body": 2399,
        "list_to_tree/2": 2409,
        "long_enough/2": 2421,
        "L_ite_else_38": 2434,
        "L_ite_cont_38": 2443,
        "lookup_held/3": 2445,
        "L_ite_else_39": 2462,
        "L_ite_cont_39": 2467,
        "map_requests/3": 2469,
        "L_map_requests_3_2": 2474,
        "L_map_requests_3_2_body": 2475,
        "matching_deps/4": 2492,
        "L_matching_deps_4_2": 2498,
        "L_matching_deps_4_2_body": 2499,
        "L_ite_else_40": 2530,
        "L_ite_cont_40": 2534,
        "matching_versions/4": 2540,
        "L_matching_versions_4_2": 2546,
        "L_matching_versions_4_2_body": 2547,
        "L_ite_else_41": 2572,
        "L_ite_cont_41": 2576,
        "matching_versions_in/4": 2582,
        "L_ite_else_43": 2605,
        "L_ite_cont_43": 2609,
        "L_ite_else_42": 2610,
        "L_ite_cont_42": 2619,
        "member_selected/3": 2621,
        "names_of/2": 2630,
        "L_names_of_2_2": 2634,
        "L_names_of_2_2_body": 2635,
        "needed_names/4": 2649,
        "L_needed_names_4_2": 2655,
        "L_needed_names_4_2_body": 2656,
        "no_acc_conflicts/4": 2673,
        "L_no_acc_conflicts_4_2": 2679,
        "L_no_acc_conflicts_4_2_body": 2680,
        "L_ite_else_44": 2700,
        "L_ite_cont_44": 2702,
        "L_ite_else_45": 2712,
        "L_ite_cont_45": 2714,
        "order_lt/2": 2720,
        "L_order_lt_2_2": 2728,
        "L_order_lt_2_2_body": 2729,
        "L_order_lt_2_3": 2742,
        "L_order_lt_2_3_body": 2743,
        "L_order_lt_2_4": 2756,
        "L_order_lt_2_4_body": 2757,
        "L_ite_else_46": 2778,
        "L_ite_cont_46": 2785,
        "L_order_lt_2_list_dispatch": 2787,
        "order_val/2": 2787,
        "L_order_val_2_2": 2794,
        "L_order_val_2_2_body": 2795,
        "L_order_val_2_3": 2807,
        "L_order_val_2_3_body": 2808,
        "L_order_val_2_4": 2820,
        "L_order_val_2_4_body": 2821,
        "package_in/3": 2829,
        "package_in_name/2": 2843,
        "packages/2": 2856,
        "L_packages_2_2": 2867,
        "L_packages_2_2_body": 2868,
        "L_packages_2_3": 2880,
        "L_packages_2_3_body": 2881,
        "L_packages_2_4": 2894,
        "L_packages_2_4_body": 2895,
        "pad_head/2": 2905,
        "L_pad_head_2_2": 2917,
        "L_pad_head_2_2_body": 2918,
        "pick/7": 2921,
        "L_pick_7_2": 2937,
        "L_pick_7_2_body": 2938,
        "L_ite_else_47": 2963,
        "L_ite_cont_47": 2972,
        "pick_need/8": 2974,
        "L_pick_need_8_2": 2991,
        "L_pick_need_8_2_body": 2992,
        "L_pick_need_8_3": 3008,
        "L_pick_need_8_3_body": 3009,
        "L_ite_else_48": 3038,
        "L_ite_else_49": 3052,
        "L_ite_else_50": 3068,
        "L_ite_cont_50": 3078,
        "L_ite_cont_49": 3078,
        "L_ite_cont_48": 3078,
        "pick_repair/4": 3080,
        "pkg_index/2": 3096,
        "provide_row/5": 3102,
        "L_provide_row_5_2": 3113,
        "L_provide_row_5_2_body": 3114,
        "provide_satisfies/2": 3124,
        "L_provide_satisfies_2_2": 3128,
        "L_provide_satisfies_2_2_body": 3129,
        "provider_candidate/5": 3139,
        "L_ite_else_51": 3165,
        "L_ite_cont_51": 3167,
        "provides_for/5": 3175,
        "provides_list/2": 3194,
        "L_provides_list_2_2": 3205,
        "L_provides_list_2_2_body": 3206,
        "L_provides_list_2_3": 3218,
        "L_provides_list_2_3_body": 3219,
        "L_provides_list_2_4": 3232,
        "L_provides_list_2_4_body": 3233,
        "provides_sat/5": 3243,
        "removal_orphans/3": 3265,
        "L_ite_else_52": 3285,
        "L_ite_cont_52": 3289,
        "L_ite_else_53": 3299,
        "L_ite_else_54": 3341,
        "L_ite_cont_54": 3343,
        "L_ite_else_55": 3351,
        "L_ite_cont_55": 3353,
        "L_ite_cont_53": 3360,
        "repairs_moving/4": 3363,
        "reqs_ok_moving/2": 3377,
        "L_reqs_ok_moving_2_2": 3381,
        "L_reqs_ok_moving_2_2_body": 3382,
        "L_ite_else_56": 3409,
        "L_ite_cont_56": 3411,
        "L_reqs_ok_moving_2_3": 3415,
        "L_reqs_ok_moving_2_3_body": 3416,
        "L_ite_else_57": 3435,
        "L_ite_cont_57": 3437,
        "L_reqs_ok_moving_2_list_dispatch": 3441,
        "request_to_req/3": 3441,
        "L_ite_else_58": 3460,
        "L_ite_cont_58": 3468,
        "requested_list/2": 3470,
        "L_requested_list_2_2": 3481,
        "L_requested_list_2_2_body": 3482,
        "L_requested_list_2_3": 3494,
        "L_requested_list_2_3_body": 3495,
        "L_requested_list_2_4": 3508,
        "L_requested_list_2_4_body": 3509,
        "resolve/3": 3519,
        "resolve_alternatives/6": 3542,
        "L_ite_else_59": 3564,
        "L_ite_cont_59": 3581,
        "resolve_layered/3": 3583,
        "resolve_pending/5": 3606,
        "L_resolve_pending_5_2": 3613,
        "L_resolve_pending_5_2_body": 3614,
        "L_ite_else_60": 3640,
        "L_ite_else_61": 3658,
        "L_ite_else_62": 3674,
        "L_ite_else_63": 3706,
        "L_ite_cont_63": 3732,
        "L_ite_cont_62": 3732,
        "L_ite_cont_61": 3732,
        "L_ite_cont_60": 3732,
        "roots_to_pairs/3": 3734,
        "L_roots_to_pairs_3_2": 3739,
        "L_roots_to_pairs_3_2_body": 3740,
        "L_roots_to_pairs_3_3": 3762,
        "L_roots_to_pairs_3_3_body": 3763,
        "L_roots_to_pairs_3_list_dispatch": 3774,
        "safe_upgrade/4": 3774,
        "L_ite_else_65": 3794,
        "L_ite_cont_65": 3796,
        "L_ite_else_64": 3801,
        "L_ite_else_67": 3813,
        "L_ite_cont_67": 3815,
        "L_ite_else_66": 3820,
        "L_ite_cont_66": 3831,
        "L_ite_cont_64": 3831,
        "safe_upgrade_reason/5": 3834,
        "L_safe_upgrade_reason_5_2": 3842,
        "L_safe_upgrade_reason_5_2_body": 3843,
        "L_safe_upgrade_reason_5_3": 3852,
        "L_safe_upgrade_reason_5_3_body": 3853,
        "L_safe_upgrade_reason_5_4": 3862,
        "L_safe_upgrade_reason_5_4_body": 3863,
        "L_safe_upgrade_reason_5_5": 3872,
        "L_safe_upgrade_reason_5_5_body": 3873,
        "same_key/4": 3887,
        "L_same_key_4_2": 3893,
        "L_same_key_4_2_body": 3894,
        "L_ite_else_68": 3924,
        "L_ite_cont_68": 3939,
        "satisfies/2": 3941,
        "L_satisfies_2_2": 3945,
        "L_satisfies_2_2_body": 3946,
        "L_satisfies_2_3": 3953,
        "L_satisfies_2_3_body": 3954,
        "L_ite_else_69": 3965,
        "L_ite_cont_69": 3967,
        "L_satisfies_2_4": 3968,
        "L_satisfies_2_4_body": 3969,
        "L_ite_else_70": 3980,
        "L_ite_cont_70": 3982,
        "L_satisfies_2_5": 3983,
        "L_satisfies_2_5_body": 3984,
        "L_satisfies_2_6": 3992,
        "L_satisfies_2_6_body": 3993,
        "L_satisfies_2_7": 4001,
        "L_satisfies_2_7_body": 4002,
        "L_ite_else_71": 4015,
        "L_ite_cont_71": 4017,
        "scan_base_holds/3": 4021,
        "L_scan_base_holds_3_2": 4026,
        "L_scan_base_holds_3_2_body": 4027,
        "L_ite_else_72": 4046,
        "L_ite_else_73": 4059,
        "L_ite_else_74": 4081,
        "L_ite_else_75": 4100,
        "L_ite_cont_75": 4104,
        "L_ite_cont_74": 4104,
        "L_ite_cont_73": 4104,
        "L_ite_cont_72": 4104,
        "seen_name/2": 4109,
        "L_ite_else_76": 4122,
        "L_ite_cont_76": 4126,
        "segs_lt/2": 4128,
        "L_segs_lt_2_2": 4136,
        "L_segs_lt_2_2_body": 4137,
        "segs_lt_1/2": 4150,
        "L_ite_else_77": 4171,
        "L_ite_else_78": 4183,
        "L_ite_cont_78": 4193,
        "L_ite_cont_77": 4193,
        "selected_ver/3": 4195,
        "L_ite_else_79": 4211,
        "L_ite_cont_79": 4216,
        "sort_versions_desc/2": 4218,
        "L_ite_else_80": 4234,
        "L_ite_cont_80": 4242,
        "tight_base_revdep/2": 4244,
        "tight_constraint/1": 4255,
        "tight_rev_in/3": 4260,
        "L_ite_else_82": 4288,
        "L_ite_cont_82": 4299,
        "L_ite_else_81": 4302,
        "L_ite_cont_81": 4307,
        "topo_all/7": 4309,
        "L_topo_all_7_2": 4318,
        "L_topo_all_7_2_body": 4319,
        "topo_one/7": 4346,
        "L_topo_one_7_2": 4361,
        "L_topo_one_7_2_body": 4362,
        "L_ite_else_83": 4413,
        "L_ite_cont_83": 4422,
        "topo_sort_sel/3": 4424,
        "L_topo_sort_sel_3_2": 4432,
        "L_topo_sort_sel_3_2_body": 4433,
        "tree_lookup/3": 4456,
        "L_ite_else_84": 4478,
        "L_ite_else_85": 4490,
        "L_ite_cont_85": 4495,
        "L_ite_cont_84": 4495,
        "upgrade_set/4": 4497,
        "upgrade_set_result/4": 4511,
        "L_ite_else_86": 4537,
        "L_ite_cont_86": 4541,
        "version_lt/2": 4544,
        "L_ite_else_87": 4563,
        "L_ite_else_88": 4575,
        "L_ite_cont_88": 4585,
        "L_ite_cont_87": 4585,
        "L_version_lt_2_2": 4587,
        "L_version_lt_2_2_body": 4588,
        "L_ite_else_89": 4605,
        "L_ite_else_90": 4617,
        "L_ite_else_91": 4629,
        "L_ite_cont_91": 4631,
        "L_ite_else_92": 4639,
        "L_ite_cont_92": 4641,
        "L_ite_cont_90": 4644,
        "L_ite_cont_89": 4644,
        "virtual_provider_ceilings/4": 4646,
        "L_ite_else_93": 4659,
        "L_ite_cont_93": 4661,
        "L_ite_else_94": 4683,
        "L_ite_cont_94": 4685,
        "walk_pkg_for_blocked/5": 4696,
        "L_walk_pkg_for_blocked_5_2": 4709,
        "L_walk_pkg_for_blocked_5_2_body": 4710,
        "L_walk_pkg_for_blocked_5_3": 4723,
        "L_walk_pkg_for_blocked_5_3_body": 4724,
        "worth_indexing/2": 4737,
        "L_ite_else_95": 4750,
        "L_ite_cont_95": 4754,
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
// WAM-compiled predicate: alias_list/2 (shared table, pc=24)
var Alias_listCode = sharedWamCode
var Alias_listLabels = sharedWamLabels
const Alias_listStartPC = 24

func Alias_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 24
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: alias_lookup/3 (shared table, pc=73)
var Alias_lookupCode = sharedWamCode
var Alias_lookupLabels = sharedWamLabels
const Alias_lookupStartPC = 73

func Alias_lookup(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 73
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: already_provided/4 (shared table, pc=105)
var Already_providedCode = sharedWamCode
var Already_providedLabels = sharedWamLabels
const Already_providedStartPC = 105

func Already_provided(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 105
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: already_satisfied/4 (shared table, pc=122)
var Already_satisfiedCode = sharedWamCode
var Already_satisfiedLabels = sharedWamLabels
const Already_satisfiedStartPC = 122

func Already_satisfied(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 122
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: alt_reasons/4 (shared table, pc=148)
var Alt_reasonsCode = sharedWamCode
var Alt_reasonsLabels = sharedWamLabels
const Alt_reasonsStartPC = 148

func Alt_reasons(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 148
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: audit_holds/4 (shared table, pc=191)
var Audit_holdsCode = sharedWamCode
var Audit_holdsLabels = sharedWamLabels
const Audit_holdsStartPC = 191

func Audit_holds(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 191
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_holds/2 (shared table, pc=252)
var Base_holdsCode = sharedWamCode
var Base_holdsLabels = sharedWamLabels
const Base_holdsStartPC = 252

func Base_holds(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 252
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_list/2 (shared table, pc=267)
var Base_listCode = sharedWamCode
var Base_listLabels = sharedWamLabels
const Base_listStartPC = 267

func Base_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 267
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_name/2 (shared table, pc=316)
var Base_nameCode = sharedWamCode
var Base_nameLabels = sharedWamLabels
const Base_nameStartPC = 316

func Base_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 316
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_reason/3 (shared table, pc=324)
var Base_reasonCode = sharedWamCode
var Base_reasonLabels = sharedWamLabels
const Base_reasonStartPC = 324

func Base_reason(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 324
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_ver/3 (shared table, pc=336)
var Base_verCode = sharedWamCode
var Base_verLabels = sharedWamLabels
const Base_verStartPC = 336

func Base_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 336
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_acc/5 (shared table, pc=355)
var Blocked_accCode = sharedWamCode
var Blocked_accLabels = sharedWamLabels
const Blocked_accStartPC = 355

func Blocked_acc(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 355
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_acc_list/5 (shared table, pc=499)
var Blocked_acc_listCode = sharedWamCode
var Blocked_acc_listLabels = sharedWamLabels
const Blocked_acc_listStartPC = 499

func Blocked_acc_list(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 499
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_from/4 (shared table, pc=528)
var Blocked_fromCode = sharedWamCode
var Blocked_fromLabels = sharedWamLabels
const Blocked_fromStartPC = 528

func Blocked_from(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 528
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: build_tree/4 (shared table, pc=674)
var Build_treeCode = sharedWamCode
var Build_treeLabels = sharedWamLabels
const Build_treeStartPC = 674

func Build_tree(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 674
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: candidates_high_first/4 (shared table, pc=733)
var Candidates_high_firstCode = sharedWamCode
var Candidates_high_firstLabels = sharedWamLabels
const Candidates_high_firstStartPC = 733

func Candidates_high_first(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 733
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: canonicalize_name/3 (shared table, pc=761)
var Canonicalize_nameCode = sharedWamCode
var Canonicalize_nameLabels = sharedWamLabels
const Canonicalize_nameStartPC = 761

func Canonicalize_name(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 761
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: close_moving/3 (shared table, pc=773)
var Close_movingCode = sharedWamCode
var Close_movingLabels = sharedWamLabels
const Close_movingStartPC = 773

func Close_moving(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 773
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: cmp_ver/3 (shared table, pc=837)
var Cmp_verCode = sharedWamCode
var Cmp_verLabels = sharedWamLabels
const Cmp_verStartPC = 837

func Cmp_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 837
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: collect_deps/4 (shared table, pc=865)
var Collect_depsCode = sharedWamCode
var Collect_depsLabels = sharedWamLabels
const Collect_depsStartPC = 865

func Collect_deps(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 865
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: conflicts_in/4 (shared table, pc=905)
var Conflicts_inCode = sharedWamCode
var Conflicts_inLabels = sharedWamLabels
const Conflicts_inStartPC = 905

func Conflicts_in(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 905
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: conflicts_list/2 (shared table, pc=921)
var Conflicts_listCode = sharedWamCode
var Conflicts_listLabels = sharedWamLabels
const Conflicts_listStartPC = 921

func Conflicts_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 921
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_breaks/5 (shared table, pc=970)
var Dep_breaksCode = sharedWamCode
var Dep_breaksLabels = sharedWamLabels
const Dep_breaksStartPC = 970

func Dep_breaks(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 970
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_breaks_moving/5 (shared table, pc=1010)
var Dep_breaks_movingCode = sharedWamCode
var Dep_breaks_movingLabels = sharedWamLabels
const Dep_breaks_movingStartPC = 1010

func Dep_breaks_moving(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1010
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_breaks_need/4 (shared table, pc=1026)
var Dep_breaks_needCode = sharedWamCode
var Dep_breaks_needLabels = sharedWamLabels
const Dep_breaks_needStartPC = 1026

func Dep_breaks_need(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1026
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_index/2 (shared table, pc=1096)
var Dep_indexCode = sharedWamCode
var Dep_indexLabels = sharedWamLabels
const Dep_indexStartPC = 1096

func Dep_index(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1096
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_mentions/2 (shared table, pc=1102)
var Dep_mentionsCode = sharedWamCode
var Dep_mentionsLabels = sharedWamLabels
const Dep_mentionsStartPC = 1102

func Dep_mentions(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1102
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_targets/5 (shared table, pc=1124)
var Dep_targetsCode = sharedWamCode
var Dep_targetsLabels = sharedWamLabels
const Dep_targetsStartPC = 1124

func Dep_targets(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1124
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_to_req/3 (shared table, pc=1158)
var Dep_to_reqCode = sharedWamCode
var Dep_to_reqLabels = sharedWamLabels
const Dep_to_reqStartPC = 1158

func Dep_to_req(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1158
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dependents/3 (shared table, pc=1178)
var DependentsCode = sharedWamCode
var DependentsLabels = sharedWamLabels
const DependentsStartPC = 1178

func Dependents(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1178
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dependents_installed/3 (shared table, pc=1200)
var Dependents_installedCode = sharedWamCode
var Dependents_installedLabels = sharedWamLabels
const Dependents_installedStartPC = 1200

func Dependents_installed(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1200
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: depends_in/5 (shared table, pc=1219)
var Depends_inCode = sharedWamCode
var Depends_inLabels = sharedWamLabels
const Depends_inStartPC = 1219

func Depends_in(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1219
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: depends_list/2 (shared table, pc=1237)
var Depends_listCode = sharedWamCode
var Depends_listLabels = sharedWamLabels
const Depends_listStartPC = 1237

func Depends_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1237
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: direct_on/4 (shared table, pc=1286)
var Direct_onCode = sharedWamCode
var Direct_onLabels = sharedWamLabels
const Direct_onStartPC = 1286

func Direct_on(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1286
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: exclude_name/3 (shared table, pc=1330)
var Exclude_nameCode = sharedWamCode
var Exclude_nameLabels = sharedWamLabels
const Exclude_nameStartPC = 1330

func Exclude_name(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1330
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: excluded_list/2 (shared table, pc=1362)
var Excluded_listCode = sharedWamCode
var Excluded_listLabels = sharedWamLabels
const Excluded_listStartPC = 1362

func Excluded_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1362
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: excluded_name/2 (shared table, pc=1411)
var Excluded_nameCode = sharedWamCode
var Excluded_nameLabels = sharedWamLabels
const Excluded_nameStartPC = 1411

func Excluded_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1411
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: explain_alt/5 (shared table, pc=1422)
var Explain_altCode = sharedWamCode
var Explain_altLabels = sharedWamLabels
const Explain_altStartPC = 1422

func Explain_alt(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1422
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: explain_blocked/3 (shared table, pc=1468)
var Explain_blockedCode = sharedWamCode
var Explain_blockedLabels = sharedWamLabels
const Explain_blockedStartPC = 1468

func Explain_blocked(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1468
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: explain_blocked_list/3 (shared table, pc=1482)
var Explain_blocked_listCode = sharedWamCode
var Explain_blocked_listLabels = sharedWamLabels
const Explain_blocked_listStartPC = 1482

func Explain_blocked_list(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1482
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: filter_satisfies/3 (shared table, pc=1502)
var Filter_satisfiesCode = sharedWamCode
var Filter_satisfiesLabels = sharedWamLabels
const Filter_satisfiesStartPC = 1502

func Filter_satisfies(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1502
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: first_alt_already/4 (shared table, pc=1535)
var First_alt_alreadyCode = sharedWamCode
var First_alt_alreadyLabels = sharedWamLabels
const First_alt_alreadyStartPC = 1535

func First_alt_already(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1535
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: first_broken/4 (shared table, pc=1572)
var First_brokenCode = sharedWamCode
var First_brokenLabels = sharedWamLabels
const First_brokenStartPC = 1572

func First_broken(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1572
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: follow_dep_name/5 (shared table, pc=1628)
var Follow_dep_nameCode = sharedWamCode
var Follow_dep_nameLabels = sharedWamLabels
const Follow_dep_nameStartPC = 1628

func Follow_dep_name(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1628
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: follow_raw_dep/4 (shared table, pc=1646)
var Follow_raw_depCode = sharedWamCode
var Follow_raw_depLabels = sharedWamLabels
const Follow_raw_depStartPC = 1646

func Follow_raw_dep(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1646
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: freeze_audit/2 (shared table, pc=1684)
var Freeze_auditCode = sharedWamCode
var Freeze_auditLabels = sharedWamLabels
const Freeze_auditStartPC = 1684

func Freeze_audit(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1684
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: group_keyed/2 (shared table, pc=1701)
var Group_keyedCode = sharedWamCode
var Group_keyedLabels = sharedWamLabels
const Group_keyedStartPC = 1701

func Group_keyed(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1701
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: hold_reason/3 (shared table, pc=1734)
var Hold_reasonCode = sharedWamCode
var Hold_reasonLabels = sharedWamLabels
const Hold_reasonStartPC = 1734

func Hold_reason(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1734
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: index_catalog/2 (shared table, pc=1761)
var Index_catalogCode = sharedWamCode
var Index_catalogLabels = sharedWamLabels
const Index_catalogStartPC = 1761

func Index_catalog(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1761
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: index_threshold/1 (shared table, pc=1817)
var Index_thresholdCode = sharedWamCode
var Index_thresholdLabels = sharedWamLabels
const Index_thresholdStartPC = 1817

func Index_threshold(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1817
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: inst_closure_names/5 (shared table, pc=1819)
var Inst_closure_namesCode = sharedWamCode
var Inst_closure_namesLabels = sharedWamLabels
const Inst_closure_namesStartPC = 1819

func Inst_closure_names(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1819
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: inst_walk/6 (shared table, pc=1838)
var Inst_walkCode = sharedWamCode
var Inst_walkLabels = sharedWamLabels
const Inst_walkStartPC = 1838

func Inst_walk(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1838
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_list/2 (shared table, pc=1910)
var Installed_listCode = sharedWamCode
var Installed_listLabels = sharedWamLabels
const Installed_listStartPC = 1910

func Installed_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1910
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_or_base/3 (shared table, pc=1959)
var Installed_or_baseCode = sharedWamCode
var Installed_or_baseLabels = sharedWamLabels
const Installed_or_baseStartPC = 1959

func Installed_or_base(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1959
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_ver/3 (shared table, pc=1983)
var Installed_verCode = sharedWamCode
var Installed_verLabels = sharedWamLabels
const Installed_verStartPC = 1983

func Installed_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1983
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: is_public_catalog/1 (shared table, pc=1997)
var Is_public_catalogCode = sharedWamCode
var Is_public_catalogLabels = sharedWamLabels
const Is_public_catalogStartPC = 1997

func Is_public_catalog(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1997
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: is_v3/1 (shared table, pc=2032)
var Is_v3Code = sharedWamCode
var Is_v3Labels = sharedWamLabels
const Is_v3StartPC = 2032

func Is_v3(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2032
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: item_ver/3 (shared table, pc=2037)
var Item_verCode = sharedWamCode
var Item_verLabels = sharedWamLabels
const Item_verStartPC = 2037

func Item_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2037
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: keep_installed_or_base/4 (shared table, pc=2077)
var Keep_installed_or_baseCode = sharedWamCode
var Keep_installed_or_baseLabels = sharedWamLabels
const Keep_installed_or_baseStartPC = 2077

func Keep_installed_or_base(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2077
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: key_dep_rows/3 (shared table, pc=2120)
var Key_dep_rowsCode = sharedWamCode
var Key_dep_rowsLabels = sharedWamLabels
const Key_dep_rowsStartPC = 2120

func Key_dep_rows(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2120
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: key_pkg_rows/3 (shared table, pc=2162)
var Key_pkg_rowsCode = sharedWamCode
var Key_pkg_rowsLabels = sharedWamLabels
const Key_pkg_rowsStartPC = 2162

func Key_pkg_rows(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2162
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layer_closure/3 (shared table, pc=2195)
var Layer_closureCode = sharedWamCode
var Layer_closureLabels = sharedWamLabels
const Layer_closureStartPC = 2195

func Layer_closure(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2195
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layer_provider/5 (shared table, pc=2212)
var Layer_providerCode = sharedWamCode
var Layer_providerLabels = sharedWamLabels
const Layer_providerStartPC = 2212

func Layer_provider(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2212
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layer_satisfies/3 (shared table, pc=2261)
var Layer_satisfiesCode = sharedWamCode
var Layer_satisfiesLabels = sharedWamLabels
const Layer_satisfiesStartPC = 2261

func Layer_satisfies(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2261
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layered_walk_ver/4 (shared table, pc=2332)
var Layered_walk_verCode = sharedWamCode
var Layered_walk_verLabels = sharedWamLabels
const Layered_walk_verStartPC = 2332

func Layered_walk_ver(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2332
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layers_list/2 (shared table, pc=2360)
var Layers_listCode = sharedWamCode
var Layers_listLabels = sharedWamLabels
const Layers_listStartPC = 2360

func Layers_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2360
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: list_to_tree/2 (shared table, pc=2409)
var List_to_treeCode = sharedWamCode
var List_to_treeLabels = sharedWamLabels
const List_to_treeStartPC = 2409

func List_to_tree(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2409
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: long_enough/2 (shared table, pc=2421)
var Long_enoughCode = sharedWamCode
var Long_enoughLabels = sharedWamLabels
const Long_enoughStartPC = 2421

func Long_enough(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2421
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: lookup_held/3 (shared table, pc=2445)
var Lookup_heldCode = sharedWamCode
var Lookup_heldLabels = sharedWamLabels
const Lookup_heldStartPC = 2445

func Lookup_held(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2445
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: map_requests/3 (shared table, pc=2469)
var Map_requestsCode = sharedWamCode
var Map_requestsLabels = sharedWamLabels
const Map_requestsStartPC = 2469

func Map_requests(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2469
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: matching_deps/4 (shared table, pc=2492)
var Matching_depsCode = sharedWamCode
var Matching_depsLabels = sharedWamLabels
const Matching_depsStartPC = 2492

func Matching_deps(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2492
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: matching_versions/4 (shared table, pc=2540)
var Matching_versionsCode = sharedWamCode
var Matching_versionsLabels = sharedWamLabels
const Matching_versionsStartPC = 2540

func Matching_versions(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2540
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: matching_versions_in/4 (shared table, pc=2582)
var Matching_versions_inCode = sharedWamCode
var Matching_versions_inLabels = sharedWamLabels
const Matching_versions_inStartPC = 2582

func Matching_versions_in(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2582
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: member_selected/3 (shared table, pc=2621)
var Member_selectedCode = sharedWamCode
var Member_selectedLabels = sharedWamLabels
const Member_selectedStartPC = 2621

func Member_selected(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2621
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: names_of/2 (shared table, pc=2630)
var Names_ofCode = sharedWamCode
var Names_ofLabels = sharedWamLabels
const Names_ofStartPC = 2630

func Names_of(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2630
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: needed_names/4 (shared table, pc=2649)
var Needed_namesCode = sharedWamCode
var Needed_namesLabels = sharedWamLabels
const Needed_namesStartPC = 2649

func Needed_names(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2649
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: no_acc_conflicts/4 (shared table, pc=2673)
var No_acc_conflictsCode = sharedWamCode
var No_acc_conflictsLabels = sharedWamLabels
const No_acc_conflictsStartPC = 2673

func No_acc_conflicts(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2673
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: order_lt/2 (shared table, pc=2720)
var Order_ltCode = sharedWamCode
var Order_ltLabels = sharedWamLabels
const Order_ltStartPC = 2720

func Order_lt(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2720
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: order_val/2 (shared table, pc=2787)
var Order_valCode = sharedWamCode
var Order_valLabels = sharedWamLabels
const Order_valStartPC = 2787

func Order_val(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2787
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: package_in/3 (shared table, pc=2829)
var Package_inCode = sharedWamCode
var Package_inLabels = sharedWamLabels
const Package_inStartPC = 2829

func Package_in(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2829
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: package_in_name/2 (shared table, pc=2843)
var Package_in_nameCode = sharedWamCode
var Package_in_nameLabels = sharedWamLabels
const Package_in_nameStartPC = 2843

func Package_in_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2843
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: packages/2 (shared table, pc=2856)
var PackagesCode = sharedWamCode
var PackagesLabels = sharedWamLabels
const PackagesStartPC = 2856

func Packages(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2856
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pad_head/2 (shared table, pc=2905)
var Pad_headCode = sharedWamCode
var Pad_headLabels = sharedWamLabels
const Pad_headStartPC = 2905

func Pad_head(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2905
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pick/7 (shared table, pc=2921)
var PickCode = sharedWamCode
var PickLabels = sharedWamLabels
const PickStartPC = 2921

func Pick(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2921
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
// WAM-compiled predicate: pick_need/8 (shared table, pc=2974)
var Pick_needCode = sharedWamCode
var Pick_needLabels = sharedWamLabels
const Pick_needStartPC = 2974

func Pick_need(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value, a8 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2974
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
// WAM-compiled predicate: pick_repair/4 (shared table, pc=3080)
var Pick_repairCode = sharedWamCode
var Pick_repairLabels = sharedWamLabels
const Pick_repairStartPC = 3080

func Pick_repair(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3080
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pkg_index/2 (shared table, pc=3096)
var Pkg_indexCode = sharedWamCode
var Pkg_indexLabels = sharedWamLabels
const Pkg_indexStartPC = 3096

func Pkg_index(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3096
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provide_row/5 (shared table, pc=3102)
var Provide_rowCode = sharedWamCode
var Provide_rowLabels = sharedWamLabels
const Provide_rowStartPC = 3102

func Provide_row(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3102
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provide_satisfies/2 (shared table, pc=3124)
var Provide_satisfiesCode = sharedWamCode
var Provide_satisfiesLabels = sharedWamLabels
const Provide_satisfiesStartPC = 3124

func Provide_satisfies(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3124
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provider_candidate/5 (shared table, pc=3139)
var Provider_candidateCode = sharedWamCode
var Provider_candidateLabels = sharedWamLabels
const Provider_candidateStartPC = 3139

func Provider_candidate(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3139
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provides_for/5 (shared table, pc=3175)
var Provides_forCode = sharedWamCode
var Provides_forLabels = sharedWamLabels
const Provides_forStartPC = 3175

func Provides_for(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3175
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provides_list/2 (shared table, pc=3194)
var Provides_listCode = sharedWamCode
var Provides_listLabels = sharedWamLabels
const Provides_listStartPC = 3194

func Provides_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3194
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: provides_sat/5 (shared table, pc=3243)
var Provides_satCode = sharedWamCode
var Provides_satLabels = sharedWamLabels
const Provides_satStartPC = 3243

func Provides_sat(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3243
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: removal_orphans/3 (shared table, pc=3265)
var Removal_orphansCode = sharedWamCode
var Removal_orphansLabels = sharedWamLabels
const Removal_orphansStartPC = 3265

func Removal_orphans(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3265
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: repairs_moving/4 (shared table, pc=3363)
var Repairs_movingCode = sharedWamCode
var Repairs_movingLabels = sharedWamLabels
const Repairs_movingStartPC = 3363

func Repairs_moving(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3363
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: reqs_ok_moving/2 (shared table, pc=3377)
var Reqs_ok_movingCode = sharedWamCode
var Reqs_ok_movingLabels = sharedWamLabels
const Reqs_ok_movingStartPC = 3377

func Reqs_ok_moving(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3377
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: request_to_req/3 (shared table, pc=3441)
var Request_to_reqCode = sharedWamCode
var Request_to_reqLabels = sharedWamLabels
const Request_to_reqStartPC = 3441

func Request_to_req(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3441
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: requested_list/2 (shared table, pc=3470)
var Requested_listCode = sharedWamCode
var Requested_listLabels = sharedWamLabels
const Requested_listStartPC = 3470

func Requested_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3470
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve/3 (shared table, pc=3519)
var ResolveCode = sharedWamCode
var ResolveLabels = sharedWamLabels
const ResolveStartPC = 3519

func Resolve(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3519
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_alternatives/6 (shared table, pc=3542)
var Resolve_alternativesCode = sharedWamCode
var Resolve_alternativesLabels = sharedWamLabels
const Resolve_alternativesStartPC = 3542

func Resolve_alternatives(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3542
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_layered/3 (shared table, pc=3583)
var Resolve_layeredCode = sharedWamCode
var Resolve_layeredLabels = sharedWamLabels
const Resolve_layeredStartPC = 3583

func Resolve_layered(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3583
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_pending/5 (shared table, pc=3606)
var Resolve_pendingCode = sharedWamCode
var Resolve_pendingLabels = sharedWamLabels
const Resolve_pendingStartPC = 3606

func Resolve_pending(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3606
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: roots_to_pairs/3 (shared table, pc=3734)
var Roots_to_pairsCode = sharedWamCode
var Roots_to_pairsLabels = sharedWamLabels
const Roots_to_pairsStartPC = 3734

func Roots_to_pairs(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3734
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: safe_upgrade/4 (shared table, pc=3774)
var Safe_upgradeCode = sharedWamCode
var Safe_upgradeLabels = sharedWamLabels
const Safe_upgradeStartPC = 3774

func Safe_upgrade(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3774
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: safe_upgrade_reason/5 (shared table, pc=3834)
var Safe_upgrade_reasonCode = sharedWamCode
var Safe_upgrade_reasonLabels = sharedWamLabels
const Safe_upgrade_reasonStartPC = 3834

func Safe_upgrade_reason(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3834
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: same_key/4 (shared table, pc=3887)
var Same_keyCode = sharedWamCode
var Same_keyLabels = sharedWamLabels
const Same_keyStartPC = 3887

func Same_key(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3887
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: satisfies/2 (shared table, pc=3941)
var SatisfiesCode = sharedWamCode
var SatisfiesLabels = sharedWamLabels
const SatisfiesStartPC = 3941

func Satisfies(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 3941
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: scan_base_holds/3 (shared table, pc=4021)
var Scan_base_holdsCode = sharedWamCode
var Scan_base_holdsLabels = sharedWamLabels
const Scan_base_holdsStartPC = 4021

func Scan_base_holds(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4021
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: seen_name/2 (shared table, pc=4109)
var Seen_nameCode = sharedWamCode
var Seen_nameLabels = sharedWamLabels
const Seen_nameStartPC = 4109

func Seen_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4109
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: segs_lt/2 (shared table, pc=4128)
var Segs_ltCode = sharedWamCode
var Segs_ltLabels = sharedWamLabels
const Segs_ltStartPC = 4128

func Segs_lt(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4128
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: segs_lt_1/2 (shared table, pc=4150)
var Segs_lt_1Code = sharedWamCode
var Segs_lt_1Labels = sharedWamLabels
const Segs_lt_1StartPC = 4150

func Segs_lt_1(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4150
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: selected_ver/3 (shared table, pc=4195)
var Selected_verCode = sharedWamCode
var Selected_verLabels = sharedWamLabels
const Selected_verStartPC = 4195

func Selected_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4195
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: sort_versions_desc/2 (shared table, pc=4218)
var Sort_versions_descCode = sharedWamCode
var Sort_versions_descLabels = sharedWamLabels
const Sort_versions_descStartPC = 4218

func Sort_versions_desc(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4218
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_base_revdep/2 (shared table, pc=4244)
var Tight_base_revdepCode = sharedWamCode
var Tight_base_revdepLabels = sharedWamLabels
const Tight_base_revdepStartPC = 4244

func Tight_base_revdep(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4244
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_constraint/1 (shared table, pc=4255)
var Tight_constraintCode = sharedWamCode
var Tight_constraintLabels = sharedWamLabels
const Tight_constraintStartPC = 4255

func Tight_constraint(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4255
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_rev_in/3 (shared table, pc=4260)
var Tight_rev_inCode = sharedWamCode
var Tight_rev_inLabels = sharedWamLabels
const Tight_rev_inStartPC = 4260

func Tight_rev_in(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4260
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: topo_all/7 (shared table, pc=4309)
var Topo_allCode = sharedWamCode
var Topo_allLabels = sharedWamLabels
const Topo_allStartPC = 4309

func Topo_all(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4309
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
// WAM-compiled predicate: topo_one/7 (shared table, pc=4346)
var Topo_oneCode = sharedWamCode
var Topo_oneLabels = sharedWamLabels
const Topo_oneStartPC = 4346

func Topo_one(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4346
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
// WAM-compiled predicate: topo_sort_sel/3 (shared table, pc=4424)
var Topo_sort_selCode = sharedWamCode
var Topo_sort_selLabels = sharedWamLabels
const Topo_sort_selStartPC = 4424

func Topo_sort_sel(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4424
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tree_lookup/3 (shared table, pc=4456)
var Tree_lookupCode = sharedWamCode
var Tree_lookupLabels = sharedWamLabels
const Tree_lookupStartPC = 4456

func Tree_lookup(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4456
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: upgrade_set/4 (shared table, pc=4497)
var Upgrade_setCode = sharedWamCode
var Upgrade_setLabels = sharedWamLabels
const Upgrade_setStartPC = 4497

func Upgrade_set(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4497
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: upgrade_set_result/4 (shared table, pc=4511)
var Upgrade_set_resultCode = sharedWamCode
var Upgrade_set_resultLabels = sharedWamLabels
const Upgrade_set_resultStartPC = 4511

func Upgrade_set_result(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4511
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: version_lt/2 (shared table, pc=4544)
var Version_ltCode = sharedWamCode
var Version_ltLabels = sharedWamLabels
const Version_ltStartPC = 4544

func Version_lt(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4544
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: virtual_provider_ceilings/4 (shared table, pc=4646)
var Virtual_provider_ceilingsCode = sharedWamCode
var Virtual_provider_ceilingsLabels = sharedWamLabels
const Virtual_provider_ceilingsStartPC = 4646

func Virtual_provider_ceilings(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4646
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: walk_pkg_for_blocked/5 (shared table, pc=4696)
var Walk_pkg_for_blockedCode = sharedWamCode
var Walk_pkg_for_blockedLabels = sharedWamLabels
const Walk_pkg_for_blockedStartPC = 4696

func Walk_pkg_for_blocked(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4696
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: worth_indexing/2 (shared table, pc=4737)
var Worth_indexingCode = sharedWamCode
var Worth_indexingLabels = sharedWamLabels
const Worth_indexingStartPC = 4737

func Worth_indexing(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 4737
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}

