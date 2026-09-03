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
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_alias_list_2_2_body"}}},
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
        &TrustMe{},
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
        &TryMeElse{Label: "L_ite_else_3", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutConstant{C: wamAtom_blanket_1, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 207},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &Call{Pred: "tight_base_revdep/2", Arity: 2},
        &Cut{Reg: 208},
        &PutVariable{Xn: 204, Ai: 0},
        &PutStructure{Functor: "audit/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetVariable{Xn: 110},
        &PutStructure{Functor: "suggest/1", Ai: 110},
        &SetConstant{C: wamAtom_abi_anchor_2},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_4"},
        &TrustMe{},
        &PutVariable{Xn: 204, Ai: 0},
        &PutStructure{Functor: "audit/2", Ai: 1},
        &SetValue{Xn: 200},
        &SetConstant{C: wamAtom_over_frozen_3},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_3"},
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
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_base_list_2_2_body"}}},
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
        &TrustMe{},
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
        &GetVariable{Xn: 100, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 101},
        &UnifyVariable{Xn: 102},
        &GetVariable{Xn: 103, Ai: 2},
        &GetVariable{Xn: 104, Ai: 3},
        &GetValue{Xn: 104, Ai: 4},
        &PutValue{Xn: 103, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 204, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 206},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 207, Ai: 2},
        &GetVariable{Xn: 201, Ai: 3},
        &GetVariable{Xn: 208, Ai: 4},
        &GetLevel{Reg: 210},
        &TryMeElse{Label: "L_ite_else_5", Arity: 5},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_6", Arity: 5},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 211},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_6"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 210},
        &PutVariable{Xn: 209, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 111},
        &SetValue{Xn: 201},
        &PutStructure{Functor: "blocked/3", Ai: 111},
        &SetValue{Xn: 206},
        &SetVariable{Xn: 112},
        &SetVariable{Xn: 113},
        &PutStructure{Functor: "needs/1", Ai: 112},
        &SetValue{Xn: 202},
        &PutStructure{Functor: "base_has/1", Ai: 113},
        &SetValue{Xn: 200},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_5"},
        &TrustMe{},
        &PutVariable{Xn: 209, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_7", Arity: 5},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutVariable{Xn: 203, Ai: 3},
        &Call{Pred: "layered_walk_ver/4", Arity: 4},
        &Cut{Reg: 212},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 205, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutStructure{Functor: "[|]/2", Ai: 2},
        &SetValue{Xn: 206},
        &SetValue{Xn: 207},
        &PutValue{Xn: 209, Ai: 3},
        &PutValue{Xn: 208, Ai: 4},
        &Call{Pred: "blocked_acc_list/5", Arity: 5},
        &Jump{Label: "L_ite_cont_7"},
        &TrustMe{},
        &PutValue{Xn: 208, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
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
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 203},
        &GetVariable{Xn: 105, Ai: 2},
        &GetVariable{Xn: 201, Ai: 3},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_8", Arity: 4},
        &PutValue{Xn: 105, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Cut{Reg: 205},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_8"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 204, Ai: 2},
        &Call{Pred: "base_ver/3", Arity: 3},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_9", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_9"},
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
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 203, Ai: 0},
        &GetStructure{Functor: "req/2", Ai: 1},
        &UnifyVariable{Xn: 205},
        &UnifyVariable{Xn: 200},
        &GetVariable{Xn: 206, Ai: 2},
        &GetVariable{Xn: 207, Ai: 3},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_10", Arity: 4},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_10"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 200, Ai: 2},
        &PutVariable{Xn: 201, Ai: 3},
        &Call{Pred: "layered_walk_ver/4", Arity: 4},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutVariable{Xn: 202, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutVariable{Xn: 204, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutStructure{Functor: "[|]/2", Ai: 2},
        &SetValue{Xn: 205},
        &SetValue{Xn: 206},
        &PutValue{Xn: 207, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "blocked_from/4"},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 206, Ai: 3},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_11", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "excluded_name/2", Arity: 2},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_11"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 200, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &Call{Pred: "packages/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutVariable{Xn: 204, Ai: 3},
        &Call{Pred: "matching_versions/4", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutVariable{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutVariable{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "reverse/2", Arity: 2},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
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
        &TryMeElse{Label: "L_ite_else_12", Arity: 3},
        &PutValue{Xn: 202, Ai: 0},
        &PutConstant{C: wamAtom_none_4, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 210},
        &PutValue{Xn: 205, Ai: 0},
        &PutVariable{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 207, Ai: 0},
        &PutStructure{Functor: "ok/1", Ai: 1},
        &SetValue{Xn: 201},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_12"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutStructure{Functor: "broken/3", Ai: 1},
        &SetVariable{Xn: 204},
        &SetVariable{Xn: 209},
        &SetVariable{Xn: 208},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_13", Arity: 3},
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
        &Jump{Label: "L_ite_cont_13"},
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
        &Allocate{},
        &GetVariable{Xn: 104, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &PutValue{Xn: 104, Ai: 0},
        &PutVariable{Xn: 200, Ai: 1},
        &Call{Pred: "depends_list/2", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "matching_deps/4"},
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
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_conflicts_list_2_2_body"}}},
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
        &TrustMe{},
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
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 110},
        &GetStructure{Functor: "depends/4", Ai: 110},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 204},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 206, Ai: 1},
        &GetVariable{Xn: 207, Ai: 2},
        &GetVariable{Xn: 208, Ai: 3},
        &GetVariable{Xn: 209, Ai: 4},
        &GetLevel{Reg: 210},
        &TryMeElse{Label: "L_ite_else_14", Arity: 5},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 208, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 203, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_15", Arity: 5},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Cut{Reg: 211},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_15"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 210},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_14"},
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
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_depends_list_2_2_body"}}},
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
        &TrustMe{},
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
        &TryMeElse{Label: "L_ite_else_16", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 208},
        &PutVariable{Xn: 206, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 111},
        &SetValue{Xn: 203},
        &PutStructure{Functor: "-/2", Ai: 111},
        &SetValue{Xn: 201},
        &SetValue{Xn: 202},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_16"},
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
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_excluded_list_2_2_body"}}},
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
        &TrustMe{},
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
        &TryMeElse{Label: "L_first_broken_4_2", Arity: 4},
        &GetConstant{C: wamAtom____0, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetConstant{C: wamAtom_none_4, Ai: 3},
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
        &TryMeElse{Label: "L_ite_else_17", Arity: 4},
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
        &Jump{Label: "L_ite_cont_17"},
        &TrustMe{},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_18", Arity: 4},
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
        &Jump{Label: "L_ite_cont_18"},
        &TrustMe{},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 206, Ai: 3},
        &Call{Pred: "first_broken/4", Arity: 4},
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
        &TryMeElse{Label: "L_ite_else_19", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_19"},
        &TrustMe{},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &Call{Pred: "hold_reason/3", Arity: 3},
        &Deallocate{},
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
        &UnifyVariable{Xn: 113},
        &GetStructure{Functor: "-/2", Ai: 113},
        &UnifyVariable{Xn: 209},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 207, Ai: 1},
        &GetVariable{Xn: 208, Ai: 2},
        &GetVariable{Xn: 210, Ai: 3},
        &GetVariable{Xn: 211, Ai: 4},
        &GetVariable{Xn: 212, Ai: 5},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_20", Arity: 6},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 210, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Cut{Reg: 213},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutValue{Xn: 208, Ai: 2},
        &PutValue{Xn: 210, Ai: 3},
        &PutValue{Xn: 211, Ai: 4},
        &PutValue{Xn: 212, Ai: 5},
        &Call{Pred: "inst_walk/6", Arity: 6},
        &Jump{Label: "L_ite_cont_20"},
        &TrustMe{},
        &PutVariable{Xn: 204, Ai: 204},
        &PutVariable{Xn: 200, Ai: 200},
        &PutVariable{Xn: 201, Ai: 201},
        &BeginAggregate{AggType: "collect", ValueReg: 0, ResultReg: 204},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &PutVariable{Xn: 203, Ai: 4},
        &Call{Pred: "depends_in/5", Arity: 5},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &PutValue{Xn: 208, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 200},
        &SetValue{Xn: 201},
        &EndAggregate{ValueReg: 0},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutVariable{Xn: 206, Ai: 2},
        &BuiltinCall{Op: "append/3", Arity: 3},
        &PutValue{Xn: 206, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &PutValue{Xn: 208, Ai: 2},
        &PutStructure{Functor: "[|]/2", Ai: 3},
        &SetValue{Xn: 209},
        &SetValue{Xn: 210},
        &PutStructure{Functor: "[|]/2", Ai: 4},
        &SetValue{Xn: 209},
        &SetValue{Xn: 211},
        &PutValue{Xn: 212, Ai: 5},
        &Call{Pred: "inst_walk/6", Arity: 6},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_installed_list_2_2_body"}}},
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
        &TrustMe{},
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
        &TryMeElse{Label: "L_ite_else_21", Arity: 4},
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
        &Jump{Label: "L_ite_cont_21"},
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
        &Allocate{},
        &GetVariable{Xn: 201, Ai: 0},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetVariable{Xn: 204, Ai: 3},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_22", Arity: 4},
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
        &Jump{Label: "L_ite_cont_22"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &Call{Pred: "candidates_high_first/4", Arity: 4},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_layers_list_2_2_body"}}},
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
        &TrustMe{},
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
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 203, Ai: 1},
        &GetVariable{Xn: 204, Ai: 2},
        &GetLevel{Reg: 205},
        &TryMeElse{Label: "L_ite_else_23", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "item_ver/3", Arity: 3},
        &Cut{Reg: 205},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_23"},
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
        &UnifyVariable{Xn: 109},
        &GetStructure{Functor: "depends/4", Ai: 109},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 202},
        &UnifyVariable{Xn: 203},
        &UnifyVariable{Xn: 205},
        &GetVariable{Xn: 206, Ai: 1},
        &GetVariable{Xn: 207, Ai: 2},
        &GetVariable{Xn: 204, Ai: 3},
        &GetLevel{Reg: 209},
        &TryMeElse{Label: "L_ite_else_24", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 207, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 209},
        &PutValue{Xn: 204, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 111},
        &SetVariable{Xn: 208},
        &PutStructure{Functor: "req/2", Ai: 111},
        &SetValue{Xn: 202},
        &SetValue{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_24"},
        &TrustMe{},
        &PutValue{Xn: 204, Ai: 0},
        &PutVariable{Xn: 208, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutValue{Xn: 207, Ai: 2},
        &PutValue{Xn: 208, Ai: 3},
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
        &TryMeElse{Label: "L_ite_else_25", Arity: 4},
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
        &Jump{Label: "L_ite_cont_25"},
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
        &TryMeElse{Label: "L_ite_else_26", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &Call{Pred: "conflicts_in/4", Arity: 4},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_26"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_27", Arity: 4},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Call{Pred: "conflicts_in/4", Arity: 4},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_27"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &PutValue{Xn: 204, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "no_acc_conflicts/4"},
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
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_packages_2_2_body"}}},
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
        &TrustMe{},
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
        &SwitchOnConstant{Cases: []ConstCase{{Val: wamAtom_classic_5, Label: "default"}, {Val: wamAtom_layered_6, Label: "L_pick_7_2_body"}}},
        &TryMeElse{Label: "L_pick_7_2", Arity: 7},
        &Allocate{},
        &GetConstant{C: wamAtom_classic_5, Ai: 0},
        &GetVariable{Xn: 100, Ai: 1},
        &GetVariable{Xn: 101, Ai: 2},
        &GetVariable{Xn: 102, Ai: 3},
        &GetVariable{Xn: 103, Ai: 4},
        &GetVariable{Xn: 104, Ai: 5},
        &GetConstant{C: wamAtom_from_catalog_7, Ai: 6},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutValue{Xn: 104, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "candidates_high_first/4"},
        &TrustMe{},
        &Allocate{},
        &GetConstant{C: wamAtom_layered_6, Ai: 0},
        &GetVariable{Xn: 201, Ai: 1},
        &GetVariable{Xn: 202, Ai: 2},
        &GetVariable{Xn: 203, Ai: 3},
        &GetVariable{Xn: 106, Ai: 4},
        &GetVariable{Xn: 204, Ai: 5},
        &GetVariable{Xn: 205, Ai: 6},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_28", Arity: 7},
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
        &PutConstant{C: wamAtom_from_base_8, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_28"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &PutValue{Xn: 204, Ai: 3},
        &Call{Pred: "candidates_high_first/4", Arity: 4},
        &PutValue{Xn: 205, Ai: 0},
        &PutConstant{C: wamAtom_from_catalog_7, Ai: 1},
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
        &PutConstant{C: wamAtom_any_9, Ai: 2},
        &PutValue{Xn: 202, Ai: 3},
        &Call{Pred: "candidates_high_first/4", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 203, Ai: 3},
        &Deallocate{},
        &Execute{Pred: "repairs_moving/4"},
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
        &TryMeElse{Label: "L_ite_else_29", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 206, Ai: 1},
        &PutVariable{Xn: 200, Ai: 2},
        &Call{Pred: "installed_ver/3", Arity: 3},
        &Cut{Reg: 212},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_29"},
        &TrustMe{},
        &PutVariable{Xn: 200, Ai: 0},
        &PutConstant{C: wamAtom_none_4, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_30", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutConstant{C: wamAtom_none_4, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 213},
        &PutValue{Xn: 211, Ai: 0},
        &PutConstant{C: wamAtom____0, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_30"},
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
        &TryMeElse{Label: "L_ite_else_31", Arity: 3},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Cut{Reg: 214},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_31"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &GetLevel{Reg: 215},
        &TryMeElse{Label: "L_ite_else_32", Arity: 3},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &Call{Pred: "base_name/2", Arity: 2},
        &Cut{Reg: 215},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_32"},
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
        &TryMeElse{Label: "L_ite_else_33", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Cut{Reg: 205},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &Jump{Label: "L_ite_cont_33"},
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
        &TryMeElse{Label: "L_ite_else_34", Arity: 3},
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
        &Jump{Label: "L_ite_cont_34"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &Call{Pred: "canonicalize_name/3", Arity: 3},
        &PutValue{Xn: 204, Ai: 0},
        &PutConstant{C: wamAtom_any_9, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &SwitchOnStructure{Cases: []StructCase{{Functor: "catalog/6", Label: "default"}, {Functor: "catalog/9", Label: "L_requested_list_2_2_body"}}},
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
        &TrustMe{},
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
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 104, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "map_requests/3", Arity: 3},
        &PutConstant{C: wamAtom_classic_5, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &PutVariable{Xn: 202, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetVariable{Xn: 104, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 104, Ai: 1},
        &PutVariable{Xn: 201, Ai: 2},
        &Call{Pred: "map_requests/3", Arity: 3},
        &PutConstant{C: wamAtom_layered_6, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutConstant{C: wamAtom____0, Ai: 3},
        &PutVariable{Xn: 202, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
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
        &GetVariable{Xn: 204, Ai: 0},
        &GetVariable{Xn: 205, Ai: 1},
        &GetList{Ai: 2},
        &UnifyVariable{Xn: 111},
        &GetStructure{Functor: "req/2", Ai: 111},
        &UnifyVariable{Xn: 209},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 202},
        &GetVariable{Xn: 207, Ai: 3},
        &GetVariable{Xn: 208, Ai: 4},
        &GetLevel{Reg: 211},
        &TryMeElse{Label: "L_ite_else_35", Arity: 5},
        &PutValue{Xn: 207, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutVariable{Xn: 210, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
        &Cut{Reg: 211},
        &PutValue{Xn: 210, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &Call{Pred: "satisfies/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 202, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &PutValue{Xn: 208, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &Jump{Label: "L_ite_cont_35"},
        &TrustMe{},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 209, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &PutValue{Xn: 207, Ai: 4},
        &PutVariable{Xn: 210, Ai: 5},
        &PutVariable{Xn: 203, Ai: 6},
        &Call{Pred: "pick/7", Arity: 7},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutValue{Xn: 210, Ai: 2},
        &PutVariable{Xn: 201, Ai: 3},
        &Call{Pred: "collect_deps/4", Arity: 4},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutVariable{Xn: 206, Ai: 2},
        &BuiltinCall{Op: "append/3", Arity: 3},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_36", Arity: 5},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_from_base_8, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 212},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &PutValue{Xn: 208, Ai: 4},
        &Call{Pred: "resolve_pending/5", Arity: 5},
        &Jump{Label: "L_ite_cont_36"},
        &TrustMe{},
        &PutValue{Xn: 205, Ai: 0},
        &PutValue{Xn: 209, Ai: 1},
        &PutValue{Xn: 210, Ai: 2},
        &PutValue{Xn: 207, Ai: 3},
        &Call{Pred: "no_acc_conflicts/4", Arity: 4},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &PutStructure{Functor: "[|]/2", Ai: 3},
        &SetVariable{Xn: 113},
        &SetValue{Xn: 207},
        &PutStructure{Functor: "-/2", Ai: 113},
        &SetValue{Xn: 209},
        &SetValue{Xn: 210},
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
        &TryMeElse{Label: "L_ite_else_37", Arity: 4},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_38", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &Call{Pred: "package_in/3", Arity: 3},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_38"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 206},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_no_candidate_10, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_37"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_39", Arity: 4},
        &GetLevel{Reg: 208},
        &TryMeElse{Label: "L_ite_else_40", Arity: 4},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &PutVariable{Xn: 202, Ai: 2},
        &Call{Pred: "base_reason/3", Arity: 3},
        &Cut{Reg: 208},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_40"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Cut{Reg: 207},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_no_candidate_10, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_39"},
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
        &GetConstant{C: wamAtom_modified_11, Ai: 3},
        &GetStructure{Functor: "unsafe/1", Ai: 4},
        &UnifyConstant{C: wamAtom_modified_11},
        &Proceed{},
        &RetryMeElse{Label: "L_safe_upgrade_reason_5_3", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_footprint_12, Ai: 3},
        &GetStructure{Functor: "safe/1", Ai: 4},
        &UnifyVariable{Xn: 103},
        &GetStructure{Functor: "cost/1", Ai: 103},
        &UnifyConstant{C: wamAtom_footprint_12},
        &Proceed{},
        &RetryMeElse{Label: "L_safe_upgrade_reason_5_4", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_blanket_1, Ai: 3},
        &GetStructure{Functor: "safe/1", Ai: 4},
        &UnifyVariable{Xn: 103},
        &GetStructure{Functor: "cost/1", Ai: 103},
        &UnifyConstant{C: wamAtom_blanket_1},
        &Proceed{},
        &RetryMeElse{Label: "L_safe_upgrade_reason_5_5", Arity: 5},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_layer_shadow_13, Ai: 3},
        &GetStructure{Functor: "safe/1", Ai: 4},
        &UnifyVariable{Xn: 103},
        &GetStructure{Functor: "cost/1", Ai: 103},
        &UnifyConstant{C: wamAtom_layer_shadow_13},
        &Proceed{},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetVariable{Xn: 101, Ai: 1},
        &GetVariable{Xn: 102, Ai: 2},
        &GetConstant{C: wamAtom_abi_anchor_2, Ai: 3},
        &GetStructure{Functor: "coordinated/1", Ai: 4},
        &UnifyVariable{Xn: 103},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &PutValue{Xn: 102, Ai: 2},
        &PutStructure{Functor: "ok/1", Ai: 3},
        &SetValue{Xn: 103},
        &Deallocate{},
        &Execute{Pred: "upgrade_set_result/4"},
        &TryMeElse{Label: "L_satisfies_2_2", Arity: 2},
        &GetVariable{Xn: 100, Ai: 0},
        &GetConstant{C: wamAtom_any_9, Ai: 1},
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
        &TryMeElse{Label: "L_ite_else_41", Arity: 2},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &Cut{Reg: 200},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_41"},
        &TrustMe{},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Proceed{},
        &RetryMeElse{Label: "L_satisfies_2_5", Arity: 2},
        &Allocate{},
        &GetVariable{Xn: 100, Ai: 0},
        &GetStructure{Functor: "lt/1", Ai: 1},
        &UnifyVariable{Xn: 101},
        &PutValue{Xn: 100, Ai: 0},
        &PutValue{Xn: 101, Ai: 1},
        &Deallocate{},
        &Execute{Pred: "version_lt/2"},
        &TrustMe{},
        &Allocate{},
        &GetVariable{Xn: 200, Ai: 0},
        &GetStructure{Functor: "range/2", Ai: 1},
        &UnifyVariable{Xn: 102},
        &UnifyVariable{Xn: 201},
        &GetLevel{Reg: 202},
        &TryMeElse{Label: "L_ite_else_42", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 102, Ai: 1},
        &Call{Pred: "version_lt/2", Arity: 2},
        &Cut{Reg: 202},
        &BuiltinCall{Op: "fail/0", Arity: 0},
        &Jump{Label: "L_ite_cont_42"},
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
        &TryMeElse{Label: "L_ite_else_43", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "layer/2", Ai: 1},
        &SetConstant{C: wamAtom_base_14},
        &SetVariable{Xn: 200},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 211},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutVariable{Xn: 209, Ai: 2},
        &Call{Pred: "scan_base_holds/3", Arity: 3},
        &Jump{Label: "L_ite_cont_43"},
        &TrustMe{},
        &GetLevel{Reg: 212},
        &TryMeElse{Label: "L_ite_else_44", Arity: 3},
        &PutValue{Xn: 201, Ai: 0},
        &PutStructure{Functor: "layer/2", Ai: 1},
        &SetVariable{Xn: 202},
        &SetVariable{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 212},
        &PutVariable{Xn: 209, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_44"},
        &TrustMe{},
        &GetLevel{Reg: 213},
        &TryMeElse{Label: "L_ite_else_45", Arity: 3},
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
        &Jump{Label: "L_ite_cont_45"},
        &TrustMe{},
        &GetLevel{Reg: 214},
        &TryMeElse{Label: "L_ite_else_46", Arity: 3},
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
        &SetConstant{C: wamAtom_blanket_1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_46"},
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
        &TryMeElse{Label: "L_ite_else_47", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &BuiltinCall{Op: "==/2", Arity: 2},
        &Cut{Reg: 203},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_47"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &Call{Pred: "seen_name/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &GetVariable{Xn: 202, Ai: 1},
        &GetVariable{Xn: 203, Ai: 2},
        &GetLevel{Reg: 204},
        &TryMeElse{Label: "L_ite_else_48", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutStructure{Functor: "-/2", Ai: 1},
        &SetValue{Xn: 202},
        &SetValue{Xn: 203},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Cut{Reg: 204},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_48"},
        &TrustMe{},
        &PutValue{Xn: 201, Ai: 0},
        &PutValue{Xn: 202, Ai: 1},
        &PutValue{Xn: 203, Ai: 2},
        &Call{Pred: "selected_ver/3", Arity: 3},
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
        &PutConstant{C: wamAtom_any_9, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &Deallocate{},
        &Proceed{},
        &Allocate{},
        &GetList{Ai: 0},
        &UnifyVariable{Xn: 106},
        &GetStructure{Functor: "hold/3", Ai: 106},
        &UnifyVariable{Xn: 200},
        &UnifyVariable{Xn: 201},
        &UnifyVariable{Xn: 107},
        &UnifyVariable{Xn: 203},
        &GetVariable{Xn: 204, Ai: 1},
        &GetVariable{Xn: 205, Ai: 2},
        &GetLevel{Reg: 206},
        &TryMeElse{Label: "L_ite_else_49", Arity: 3},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "\\==/2", Arity: 2},
        &PutValue{Xn: 204, Ai: 0},
        &PutValue{Xn: 200, Ai: 1},
        &PutValue{Xn: 201, Ai: 2},
        &PutValue{Xn: 205, Ai: 3},
        &PutVariable{Xn: 202, Ai: 4},
        &Call{Pred: "depends_in/5", Arity: 5},
        &PutValue{Xn: 202, Ai: 0},
        &Call{Pred: "tight_constraint/1", Arity: 1},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_49"},
        &TrustMe{},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
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
        &GetVariable{Xn: 203, Ai: 0},
        &GetVariable{Xn: 210, Ai: 1},
        &GetVariable{Xn: 205, Ai: 2},
        &GetVariable{Xn: 211, Ai: 3},
        &GetVariable{Xn: 209, Ai: 4},
        &GetVariable{Xn: 213, Ai: 5},
        &GetVariable{Xn: 212, Ai: 6},
        &GetLevel{Reg: 214},
        &TryMeElse{Label: "L_ite_else_50", Arity: 7},
        &PutStructure{Functor: "-/2", Ai: 0},
        &SetValue{Xn: 210},
        &SetVariable{Xn: 206},
        &PutValue{Xn: 205, Ai: 1},
        &BuiltinCall{Op: "member/2", Arity: 2},
        &Cut{Reg: 214},
        &PutVariable{Xn: 202, Ai: 202},
        &PutVariable{Xn: 200, Ai: 200},
        &BeginAggregate{AggType: "collect", ValueReg: 200, ResultReg: 202},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 210, Ai: 1},
        &PutValue{Xn: 206, Ai: 2},
        &PutValue{Xn: 200, Ai: 3},
        &PutVariable{Xn: 201, Ai: 4},
        &Call{Pred: "depends_in/5", Arity: 5},
        &EndAggregate{ValueReg: 200},
        &PutValue{Xn: 202, Ai: 0},
        &PutVariable{Xn: 204, Ai: 1},
        &BuiltinCall{Op: "sort/2", Arity: 2},
        &PutValue{Xn: 203, Ai: 0},
        &PutValue{Xn: 204, Ai: 1},
        &PutValue{Xn: 205, Ai: 2},
        &PutStructure{Functor: "[|]/2", Ai: 3},
        &SetValue{Xn: 210},
        &SetValue{Xn: 211},
        &PutVariable{Xn: 208, Ai: 4},
        &PutValue{Xn: 213, Ai: 5},
        &PutVariable{Xn: 207, Ai: 6},
        &Call{Pred: "topo_all/7", Arity: 7},
        &PutValue{Xn: 212, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetVariable{Xn: 117},
        &SetValue{Xn: 207},
        &PutStructure{Functor: "-/2", Ai: 117},
        &SetValue{Xn: 210},
        &SetValue{Xn: 206},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 209, Ai: 0},
        &PutValue{Xn: 208, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &Jump{Label: "L_ite_cont_50"},
        &TrustMe{},
        &PutValue{Xn: 209, Ai: 0},
        &PutStructure{Functor: "[|]/2", Ai: 1},
        &SetValue{Xn: 210},
        &SetValue{Xn: 211},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &PutValue{Xn: 212, Ai: 0},
        &PutValue{Xn: 213, Ai: 1},
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
        &TryMeElse{Label: "L_ite_else_51", Arity: 4},
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
        &Jump{Label: "L_ite_cont_51"},
        &TrustMe{},
        &PutValue{Xn: 203, Ai: 0},
        &PutConstant{C: wamAtom_no_candidate_10, Ai: 1},
        &BuiltinCall{Op: "=/2", Arity: 2},
        &BuiltinCall{Op: "!/0", Arity: 0},
        &Deallocate{},
        &Proceed{},
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
        &TryMeElse{Label: "L_ite_else_52", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 206},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_52"},
        &TrustMe{},
        &GetLevel{Reg: 207},
        &TryMeElse{Label: "L_ite_else_53", Arity: 2},
        &PutValue{Xn: 200, Ai: 0},
        &PutValue{Xn: 201, Ai: 1},
        &BuiltinCall{Op: "=:=/2", Arity: 2},
        &PutValue{Xn: 202, Ai: 0},
        &PutValue{Xn: 203, Ai: 1},
        &BuiltinCall{Op: "</2", Arity: 2},
        &Cut{Reg: 207},
        &BuiltinCall{Op: "true/0", Arity: 0},
        &Jump{Label: "L_ite_cont_53"},
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
}

var sharedWamLabels = map[string]int{
        "acc_conflicts/4": 0,
        "L_ite_else_1": 16,
        "L_ite_cont_1": 22,
        "alias_list/2": 24,
        "L_alias_list_2_2": 35,
        "L_alias_list_2_2_body": 36,
        "alias_lookup/3": 48,
        "L_alias_lookup_3_2": 53,
        "L_alias_lookup_3_2_body": 54,
        "L_ite_else_2": 73,
        "L_ite_cont_2": 78,
        "audit_holds/4": 80,
        "L_audit_holds_4_2": 86,
        "L_audit_holds_4_2_body": 87,
        "L_ite_else_4": 118,
        "L_ite_cont_4": 124,
        "L_ite_else_3": 125,
        "L_ite_cont_3": 133,
        "base_holds/2": 141,
        "base_list/2": 156,
        "L_base_list_2_2": 167,
        "L_base_list_2_2_body": 168,
        "base_name/2": 180,
        "base_reason/3": 188,
        "base_ver/3": 200,
        "blocked_acc/5": 219,
        "L_blocked_acc_5_2": 234,
        "L_blocked_acc_5_2_body": 235,
        "L_ite_else_6": 257,
        "L_ite_cont_6": 259,
        "L_ite_else_5": 274,
        "L_ite_cont_5": 278,
        "L_ite_else_7": 300,
        "L_ite_cont_7": 304,
        "blocked_acc_list/5": 306,
        "L_blocked_acc_list_5_2": 313,
        "L_blocked_acc_list_5_2_body": 314,
        "blocked_from/4": 335,
        "L_ite_else_8": 351,
        "L_ite_cont_8": 353,
        "L_ite_else_9": 365,
        "L_ite_cont_9": 367,
        "L_blocked_from_4_2": 379,
        "L_blocked_from_4_2_body": 380,
        "L_ite_else_10": 395,
        "L_ite_cont_10": 397,
        "candidates_high_first/4": 418,
        "L_ite_else_11": 431,
        "L_ite_cont_11": 433,
        "canonicalize_name/3": 452,
        "close_moving/3": 464,
        "L_ite_else_12": 490,
        "L_ite_else_13": 515,
        "L_ite_cont_13": 526,
        "L_ite_cont_12": 526,
        "collect_deps/4": 528,
        "conflicts_in/4": 542,
        "conflicts_list/2": 558,
        "L_conflicts_list_2_2": 569,
        "L_conflicts_list_2_2_body": 570,
        "dep_breaks/5": 582,
        "L_ite_else_15": 615,
        "L_ite_cont_15": 617,
        "L_ite_else_14": 622,
        "L_ite_cont_14": 629,
        "dep_breaks_moving/5": 631,
        "dependents/3": 647,
        "dependents_installed/3": 669,
        "depends_in/5": 688,
        "depends_list/2": 706,
        "L_depends_list_2_2": 717,
        "L_depends_list_2_2_body": 718,
        "direct_on/4": 730,
        "L_direct_on_4_2": 736,
        "L_direct_on_4_2_body": 737,
        "L_ite_else_16": 764,
        "L_ite_cont_16": 768,
        "exclude_name/3": 774,
        "L_exclude_name_3_2": 779,
        "L_exclude_name_3_2_body": 780,
        "L_exclude_name_3_3": 792,
        "L_exclude_name_3_3_body": 793,
        "excluded_list/2": 806,
        "L_excluded_list_2_2": 817,
        "L_excluded_list_2_2_body": 818,
        "excluded_name/2": 830,
        "explain_blocked/3": 841,
        "explain_blocked_list/3": 855,
        "first_broken/4": 875,
        "L_first_broken_4_2": 881,
        "L_first_broken_4_2_body": 882,
        "L_ite_else_17": 906,
        "L_ite_else_18": 923,
        "L_ite_cont_18": 929,
        "L_ite_cont_17": 929,
        "freeze_audit/2": 931,
        "hold_reason/3": 948,
        "L_ite_else_19": 968,
        "L_ite_cont_19": 973,
        "inst_closure_names/5": 975,
        "inst_walk/6": 994,
        "L_inst_walk_6_2": 1002,
        "L_inst_walk_6_2_body": 1003,
        "L_ite_else_20": 1029,
        "L_ite_cont_20": 1064,
        "installed_list/2": 1066,
        "L_installed_list_2_2": 1077,
        "L_installed_list_2_2_body": 1078,
        "installed_or_base/3": 1090,
        "L_installed_or_base_3_2": 1100,
        "L_installed_or_base_3_2_body": 1101,
        "installed_ver/3": 1114,
        "item_ver/3": 1128,
        "L_item_ver_3_2": 1141,
        "L_item_ver_3_2_body": 1142,
        "L_item_ver_3_3": 1156,
        "L_item_ver_3_3_body": 1157,
        "keep_installed_or_base/4": 1168,
        "L_keep_installed_or_base_4_2": 1174,
        "L_keep_installed_or_base_4_2_body": 1175,
        "L_ite_else_21": 1201,
        "L_ite_cont_21": 1205,
        "layer_closure/3": 1211,
        "layered_walk_ver/4": 1228,
        "L_ite_else_22": 1247,
        "L_ite_cont_22": 1253,
        "layers_list/2": 1256,
        "L_layers_list_2_2": 1267,
        "L_layers_list_2_2_body": 1268,
        "lookup_held/3": 1280,
        "L_ite_else_23": 1297,
        "L_ite_cont_23": 1302,
        "map_requests/3": 1304,
        "L_map_requests_3_2": 1309,
        "L_map_requests_3_2_body": 1310,
        "matching_deps/4": 1327,
        "L_matching_deps_4_2": 1333,
        "L_matching_deps_4_2_body": 1334,
        "L_ite_else_24": 1364,
        "L_ite_cont_24": 1368,
        "matching_versions/4": 1374,
        "L_matching_versions_4_2": 1380,
        "L_matching_versions_4_2_body": 1381,
        "L_ite_else_25": 1406,
        "L_ite_cont_25": 1410,
        "member_selected/3": 1416,
        "names_of/2": 1425,
        "L_names_of_2_2": 1429,
        "L_names_of_2_2_body": 1430,
        "needed_names/4": 1444,
        "L_needed_names_4_2": 1450,
        "L_needed_names_4_2_body": 1451,
        "no_acc_conflicts/4": 1468,
        "L_no_acc_conflicts_4_2": 1474,
        "L_no_acc_conflicts_4_2_body": 1475,
        "L_ite_else_26": 1495,
        "L_ite_cont_26": 1497,
        "L_ite_else_27": 1507,
        "L_ite_cont_27": 1509,
        "package_in/3": 1515,
        "packages/2": 1529,
        "L_packages_2_2": 1540,
        "L_packages_2_2_body": 1541,
        "pick/7": 1553,
        "L_pick_7_2": 1569,
        "L_pick_7_2_body": 1570,
        "L_ite_else_28": 1595,
        "L_ite_cont_28": 1604,
        "pick_repair/4": 1606,
        "removal_orphans/3": 1622,
        "L_ite_else_29": 1642,
        "L_ite_cont_29": 1646,
        "L_ite_else_30": 1656,
        "L_ite_else_31": 1698,
        "L_ite_cont_31": 1700,
        "L_ite_else_32": 1708,
        "L_ite_cont_32": 1710,
        "L_ite_cont_30": 1717,
        "repairs_moving/4": 1720,
        "reqs_ok_moving/2": 1734,
        "L_reqs_ok_moving_2_2": 1738,
        "L_reqs_ok_moving_2_2_body": 1739,
        "L_ite_else_33": 1758,
        "L_ite_cont_33": 1760,
        "request_to_req/3": 1764,
        "L_ite_else_34": 1783,
        "L_ite_cont_34": 1791,
        "requested_list/2": 1793,
        "L_requested_list_2_2": 1804,
        "L_requested_list_2_2_body": 1805,
        "resolve/3": 1817,
        "resolve_layered/3": 1837,
        "resolve_pending/5": 1857,
        "L_resolve_pending_5_2": 1864,
        "L_resolve_pending_5_2_body": 1865,
        "L_ite_else_35": 1893,
        "L_ite_else_36": 1924,
        "L_ite_cont_36": 1941,
        "L_ite_cont_35": 1941,
        "roots_to_pairs/3": 1943,
        "L_roots_to_pairs_3_2": 1948,
        "L_roots_to_pairs_3_2_body": 1949,
        "L_roots_to_pairs_3_3": 1971,
        "L_roots_to_pairs_3_3_body": 1972,
        "L_roots_to_pairs_3_list_dispatch": 1983,
        "safe_upgrade/4": 1983,
        "L_ite_else_38": 2003,
        "L_ite_cont_38": 2005,
        "L_ite_else_37": 2010,
        "L_ite_else_40": 2022,
        "L_ite_cont_40": 2024,
        "L_ite_else_39": 2029,
        "L_ite_cont_39": 2040,
        "L_ite_cont_37": 2040,
        "safe_upgrade_reason/5": 2043,
        "L_safe_upgrade_reason_5_2": 2051,
        "L_safe_upgrade_reason_5_2_body": 2052,
        "L_safe_upgrade_reason_5_3": 2061,
        "L_safe_upgrade_reason_5_3_body": 2062,
        "L_safe_upgrade_reason_5_4": 2071,
        "L_safe_upgrade_reason_5_4_body": 2072,
        "L_safe_upgrade_reason_5_5": 2081,
        "L_safe_upgrade_reason_5_5_body": 2082,
        "satisfies/2": 2096,
        "L_satisfies_2_2": 2100,
        "L_satisfies_2_2_body": 2101,
        "L_satisfies_2_3": 2108,
        "L_satisfies_2_3_body": 2109,
        "L_ite_else_41": 2120,
        "L_ite_cont_41": 2122,
        "L_satisfies_2_4": 2123,
        "L_satisfies_2_4_body": 2124,
        "L_satisfies_2_5": 2132,
        "L_satisfies_2_5_body": 2133,
        "L_ite_else_42": 2146,
        "L_ite_cont_42": 2148,
        "scan_base_holds/3": 2152,
        "L_scan_base_holds_3_2": 2157,
        "L_scan_base_holds_3_2_body": 2158,
        "L_ite_else_43": 2177,
        "L_ite_else_44": 2190,
        "L_ite_else_45": 2212,
        "L_ite_else_46": 2231,
        "L_ite_cont_46": 2235,
        "L_ite_cont_45": 2235,
        "L_ite_cont_44": 2235,
        "L_ite_cont_43": 2235,
        "seen_name/2": 2240,
        "L_ite_else_47": 2253,
        "L_ite_cont_47": 2257,
        "selected_ver/3": 2259,
        "L_ite_else_48": 2275,
        "L_ite_cont_48": 2280,
        "tight_base_revdep/2": 2282,
        "tight_constraint/1": 2293,
        "tight_rev_in/3": 2298,
        "L_ite_else_49": 2324,
        "L_ite_cont_49": 2329,
        "topo_all/7": 2331,
        "L_topo_all_7_2": 2340,
        "L_topo_all_7_2_body": 2341,
        "topo_one/7": 2368,
        "L_topo_one_7_2": 2383,
        "L_topo_one_7_2_body": 2384,
        "L_ite_else_50": 2435,
        "L_ite_cont_50": 2444,
        "topo_sort_sel/3": 2446,
        "L_topo_sort_sel_3_2": 2454,
        "L_topo_sort_sel_3_2_body": 2455,
        "upgrade_set/4": 2478,
        "upgrade_set_result/4": 2492,
        "L_ite_else_51": 2518,
        "L_ite_cont_51": 2522,
        "version_lt/2": 2525,
        "L_ite_else_52": 2542,
        "L_ite_else_53": 2554,
        "L_ite_cont_53": 2564,
        "L_ite_cont_52": 2564,
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
// WAM-compiled predicate: alias_lookup/3 (shared table, pc=48)
var Alias_lookupCode = sharedWamCode
var Alias_lookupLabels = sharedWamLabels
const Alias_lookupStartPC = 48

func Alias_lookup(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 48
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: audit_holds/4 (shared table, pc=80)
var Audit_holdsCode = sharedWamCode
var Audit_holdsLabels = sharedWamLabels
const Audit_holdsStartPC = 80

func Audit_holds(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 80
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_holds/2 (shared table, pc=141)
var Base_holdsCode = sharedWamCode
var Base_holdsLabels = sharedWamLabels
const Base_holdsStartPC = 141

func Base_holds(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 141
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_list/2 (shared table, pc=156)
var Base_listCode = sharedWamCode
var Base_listLabels = sharedWamLabels
const Base_listStartPC = 156

func Base_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 156
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_name/2 (shared table, pc=180)
var Base_nameCode = sharedWamCode
var Base_nameLabels = sharedWamLabels
const Base_nameStartPC = 180

func Base_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 180
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_reason/3 (shared table, pc=188)
var Base_reasonCode = sharedWamCode
var Base_reasonLabels = sharedWamLabels
const Base_reasonStartPC = 188

func Base_reason(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 188
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: base_ver/3 (shared table, pc=200)
var Base_verCode = sharedWamCode
var Base_verLabels = sharedWamLabels
const Base_verStartPC = 200

func Base_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 200
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_acc/5 (shared table, pc=219)
var Blocked_accCode = sharedWamCode
var Blocked_accLabels = sharedWamLabels
const Blocked_accStartPC = 219

func Blocked_acc(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 219
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_acc_list/5 (shared table, pc=306)
var Blocked_acc_listCode = sharedWamCode
var Blocked_acc_listLabels = sharedWamLabels
const Blocked_acc_listStartPC = 306

func Blocked_acc_list(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 306
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: blocked_from/4 (shared table, pc=335)
var Blocked_fromCode = sharedWamCode
var Blocked_fromLabels = sharedWamLabels
const Blocked_fromStartPC = 335

func Blocked_from(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 335
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: candidates_high_first/4 (shared table, pc=418)
var Candidates_high_firstCode = sharedWamCode
var Candidates_high_firstLabels = sharedWamLabels
const Candidates_high_firstStartPC = 418

func Candidates_high_first(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 418
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: canonicalize_name/3 (shared table, pc=452)
var Canonicalize_nameCode = sharedWamCode
var Canonicalize_nameLabels = sharedWamLabels
const Canonicalize_nameStartPC = 452

func Canonicalize_name(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 452
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: close_moving/3 (shared table, pc=464)
var Close_movingCode = sharedWamCode
var Close_movingLabels = sharedWamLabels
const Close_movingStartPC = 464

func Close_moving(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 464
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: collect_deps/4 (shared table, pc=528)
var Collect_depsCode = sharedWamCode
var Collect_depsLabels = sharedWamLabels
const Collect_depsStartPC = 528

func Collect_deps(a1 Value, a2 Value, a3 Value, a4 Value) bool {
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
// WAM-compiled predicate: conflicts_in/4 (shared table, pc=542)
var Conflicts_inCode = sharedWamCode
var Conflicts_inLabels = sharedWamLabels
const Conflicts_inStartPC = 542

func Conflicts_in(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 542
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: conflicts_list/2 (shared table, pc=558)
var Conflicts_listCode = sharedWamCode
var Conflicts_listLabels = sharedWamLabels
const Conflicts_listStartPC = 558

func Conflicts_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 558
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_breaks/5 (shared table, pc=582)
var Dep_breaksCode = sharedWamCode
var Dep_breaksLabels = sharedWamLabels
const Dep_breaksStartPC = 582

func Dep_breaks(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 582
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dep_breaks_moving/5 (shared table, pc=631)
var Dep_breaks_movingCode = sharedWamCode
var Dep_breaks_movingLabels = sharedWamLabels
const Dep_breaks_movingStartPC = 631

func Dep_breaks_moving(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 631
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dependents/3 (shared table, pc=647)
var DependentsCode = sharedWamCode
var DependentsLabels = sharedWamLabels
const DependentsStartPC = 647

func Dependents(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 647
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: dependents_installed/3 (shared table, pc=669)
var Dependents_installedCode = sharedWamCode
var Dependents_installedLabels = sharedWamLabels
const Dependents_installedStartPC = 669

func Dependents_installed(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 669
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: depends_in/5 (shared table, pc=688)
var Depends_inCode = sharedWamCode
var Depends_inLabels = sharedWamLabels
const Depends_inStartPC = 688

func Depends_in(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 688
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: depends_list/2 (shared table, pc=706)
var Depends_listCode = sharedWamCode
var Depends_listLabels = sharedWamLabels
const Depends_listStartPC = 706

func Depends_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 706
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: direct_on/4 (shared table, pc=730)
var Direct_onCode = sharedWamCode
var Direct_onLabels = sharedWamLabels
const Direct_onStartPC = 730

func Direct_on(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 730
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: exclude_name/3 (shared table, pc=774)
var Exclude_nameCode = sharedWamCode
var Exclude_nameLabels = sharedWamLabels
const Exclude_nameStartPC = 774

func Exclude_name(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 774
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: excluded_list/2 (shared table, pc=806)
var Excluded_listCode = sharedWamCode
var Excluded_listLabels = sharedWamLabels
const Excluded_listStartPC = 806

func Excluded_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 806
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: excluded_name/2 (shared table, pc=830)
var Excluded_nameCode = sharedWamCode
var Excluded_nameLabels = sharedWamLabels
const Excluded_nameStartPC = 830

func Excluded_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 830
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: explain_blocked/3 (shared table, pc=841)
var Explain_blockedCode = sharedWamCode
var Explain_blockedLabels = sharedWamLabels
const Explain_blockedStartPC = 841

func Explain_blocked(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 841
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: explain_blocked_list/3 (shared table, pc=855)
var Explain_blocked_listCode = sharedWamCode
var Explain_blocked_listLabels = sharedWamLabels
const Explain_blocked_listStartPC = 855

func Explain_blocked_list(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 855
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: first_broken/4 (shared table, pc=875)
var First_brokenCode = sharedWamCode
var First_brokenLabels = sharedWamLabels
const First_brokenStartPC = 875

func First_broken(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 875
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: freeze_audit/2 (shared table, pc=931)
var Freeze_auditCode = sharedWamCode
var Freeze_auditLabels = sharedWamLabels
const Freeze_auditStartPC = 931

func Freeze_audit(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 931
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: hold_reason/3 (shared table, pc=948)
var Hold_reasonCode = sharedWamCode
var Hold_reasonLabels = sharedWamLabels
const Hold_reasonStartPC = 948

func Hold_reason(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 948
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: inst_closure_names/5 (shared table, pc=975)
var Inst_closure_namesCode = sharedWamCode
var Inst_closure_namesLabels = sharedWamLabels
const Inst_closure_namesStartPC = 975

func Inst_closure_names(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 975
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: inst_walk/6 (shared table, pc=994)
var Inst_walkCode = sharedWamCode
var Inst_walkLabels = sharedWamLabels
const Inst_walkStartPC = 994

func Inst_walk(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 994
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    vm.Regs[5] = a6
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_list/2 (shared table, pc=1066)
var Installed_listCode = sharedWamCode
var Installed_listLabels = sharedWamLabels
const Installed_listStartPC = 1066

func Installed_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1066
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_or_base/3 (shared table, pc=1090)
var Installed_or_baseCode = sharedWamCode
var Installed_or_baseLabels = sharedWamLabels
const Installed_or_baseStartPC = 1090

func Installed_or_base(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1090
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: installed_ver/3 (shared table, pc=1114)
var Installed_verCode = sharedWamCode
var Installed_verLabels = sharedWamLabels
const Installed_verStartPC = 1114

func Installed_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1114
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: item_ver/3 (shared table, pc=1128)
var Item_verCode = sharedWamCode
var Item_verLabels = sharedWamLabels
const Item_verStartPC = 1128

func Item_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1128
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: keep_installed_or_base/4 (shared table, pc=1168)
var Keep_installed_or_baseCode = sharedWamCode
var Keep_installed_or_baseLabels = sharedWamLabels
const Keep_installed_or_baseStartPC = 1168

func Keep_installed_or_base(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1168
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layer_closure/3 (shared table, pc=1211)
var Layer_closureCode = sharedWamCode
var Layer_closureLabels = sharedWamLabels
const Layer_closureStartPC = 1211

func Layer_closure(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1211
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layered_walk_ver/4 (shared table, pc=1228)
var Layered_walk_verCode = sharedWamCode
var Layered_walk_verLabels = sharedWamLabels
const Layered_walk_verStartPC = 1228

func Layered_walk_ver(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1228
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: layers_list/2 (shared table, pc=1256)
var Layers_listCode = sharedWamCode
var Layers_listLabels = sharedWamLabels
const Layers_listStartPC = 1256

func Layers_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1256
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: lookup_held/3 (shared table, pc=1280)
var Lookup_heldCode = sharedWamCode
var Lookup_heldLabels = sharedWamLabels
const Lookup_heldStartPC = 1280

func Lookup_held(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1280
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: map_requests/3 (shared table, pc=1304)
var Map_requestsCode = sharedWamCode
var Map_requestsLabels = sharedWamLabels
const Map_requestsStartPC = 1304

func Map_requests(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1304
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: matching_deps/4 (shared table, pc=1327)
var Matching_depsCode = sharedWamCode
var Matching_depsLabels = sharedWamLabels
const Matching_depsStartPC = 1327

func Matching_deps(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1327
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: matching_versions/4 (shared table, pc=1374)
var Matching_versionsCode = sharedWamCode
var Matching_versionsLabels = sharedWamLabels
const Matching_versionsStartPC = 1374

func Matching_versions(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1374
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: member_selected/3 (shared table, pc=1416)
var Member_selectedCode = sharedWamCode
var Member_selectedLabels = sharedWamLabels
const Member_selectedStartPC = 1416

func Member_selected(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1416
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: names_of/2 (shared table, pc=1425)
var Names_ofCode = sharedWamCode
var Names_ofLabels = sharedWamLabels
const Names_ofStartPC = 1425

func Names_of(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1425
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: needed_names/4 (shared table, pc=1444)
var Needed_namesCode = sharedWamCode
var Needed_namesLabels = sharedWamLabels
const Needed_namesStartPC = 1444

func Needed_names(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1444
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: no_acc_conflicts/4 (shared table, pc=1468)
var No_acc_conflictsCode = sharedWamCode
var No_acc_conflictsLabels = sharedWamLabels
const No_acc_conflictsStartPC = 1468

func No_acc_conflicts(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1468
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: package_in/3 (shared table, pc=1515)
var Package_inCode = sharedWamCode
var Package_inLabels = sharedWamLabels
const Package_inStartPC = 1515

func Package_in(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1515
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: packages/2 (shared table, pc=1529)
var PackagesCode = sharedWamCode
var PackagesLabels = sharedWamLabels
const PackagesStartPC = 1529

func Packages(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1529
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: pick/7 (shared table, pc=1553)
var PickCode = sharedWamCode
var PickLabels = sharedWamLabels
const PickStartPC = 1553

func Pick(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1553
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
// WAM-compiled predicate: pick_repair/4 (shared table, pc=1606)
var Pick_repairCode = sharedWamCode
var Pick_repairLabels = sharedWamLabels
const Pick_repairStartPC = 1606

func Pick_repair(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1606
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: removal_orphans/3 (shared table, pc=1622)
var Removal_orphansCode = sharedWamCode
var Removal_orphansLabels = sharedWamLabels
const Removal_orphansStartPC = 1622

func Removal_orphans(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1622
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: repairs_moving/4 (shared table, pc=1720)
var Repairs_movingCode = sharedWamCode
var Repairs_movingLabels = sharedWamLabels
const Repairs_movingStartPC = 1720

func Repairs_moving(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1720
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: reqs_ok_moving/2 (shared table, pc=1734)
var Reqs_ok_movingCode = sharedWamCode
var Reqs_ok_movingLabels = sharedWamLabels
const Reqs_ok_movingStartPC = 1734

func Reqs_ok_moving(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1734
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: request_to_req/3 (shared table, pc=1764)
var Request_to_reqCode = sharedWamCode
var Request_to_reqLabels = sharedWamLabels
const Request_to_reqStartPC = 1764

func Request_to_req(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1764
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: requested_list/2 (shared table, pc=1793)
var Requested_listCode = sharedWamCode
var Requested_listLabels = sharedWamLabels
const Requested_listStartPC = 1793

func Requested_list(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1793
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve/3 (shared table, pc=1817)
var ResolveCode = sharedWamCode
var ResolveLabels = sharedWamLabels
const ResolveStartPC = 1817

func Resolve(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1817
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_layered/3 (shared table, pc=1837)
var Resolve_layeredCode = sharedWamCode
var Resolve_layeredLabels = sharedWamLabels
const Resolve_layeredStartPC = 1837

func Resolve_layered(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1837
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: resolve_pending/5 (shared table, pc=1857)
var Resolve_pendingCode = sharedWamCode
var Resolve_pendingLabels = sharedWamLabels
const Resolve_pendingStartPC = 1857

func Resolve_pending(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1857
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: roots_to_pairs/3 (shared table, pc=1943)
var Roots_to_pairsCode = sharedWamCode
var Roots_to_pairsLabels = sharedWamLabels
const Roots_to_pairsStartPC = 1943

func Roots_to_pairs(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1943
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: safe_upgrade/4 (shared table, pc=1983)
var Safe_upgradeCode = sharedWamCode
var Safe_upgradeLabels = sharedWamLabels
const Safe_upgradeStartPC = 1983

func Safe_upgrade(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 1983
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: safe_upgrade_reason/5 (shared table, pc=2043)
var Safe_upgrade_reasonCode = sharedWamCode
var Safe_upgrade_reasonLabels = sharedWamLabels
const Safe_upgrade_reasonStartPC = 2043

func Safe_upgrade_reason(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2043
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    vm.Regs[4] = a5
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: satisfies/2 (shared table, pc=2096)
var SatisfiesCode = sharedWamCode
var SatisfiesLabels = sharedWamLabels
const SatisfiesStartPC = 2096

func Satisfies(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2096
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: scan_base_holds/3 (shared table, pc=2152)
var Scan_base_holdsCode = sharedWamCode
var Scan_base_holdsLabels = sharedWamLabels
const Scan_base_holdsStartPC = 2152

func Scan_base_holds(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2152
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: seen_name/2 (shared table, pc=2240)
var Seen_nameCode = sharedWamCode
var Seen_nameLabels = sharedWamLabels
const Seen_nameStartPC = 2240

func Seen_name(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2240
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: selected_ver/3 (shared table, pc=2259)
var Selected_verCode = sharedWamCode
var Selected_verLabels = sharedWamLabels
const Selected_verStartPC = 2259

func Selected_ver(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2259
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_base_revdep/2 (shared table, pc=2282)
var Tight_base_revdepCode = sharedWamCode
var Tight_base_revdepLabels = sharedWamLabels
const Tight_base_revdepStartPC = 2282

func Tight_base_revdep(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2282
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_constraint/1 (shared table, pc=2293)
var Tight_constraintCode = sharedWamCode
var Tight_constraintLabels = sharedWamLabels
const Tight_constraintStartPC = 2293

func Tight_constraint(a1 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2293
    vm.Regs[0] = a1
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: tight_rev_in/3 (shared table, pc=2298)
var Tight_rev_inCode = sharedWamCode
var Tight_rev_inLabels = sharedWamLabels
const Tight_rev_inStartPC = 2298

func Tight_rev_in(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2298
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: topo_all/7 (shared table, pc=2331)
var Topo_allCode = sharedWamCode
var Topo_allLabels = sharedWamLabels
const Topo_allStartPC = 2331

func Topo_all(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2331
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
// WAM-compiled predicate: topo_one/7 (shared table, pc=2368)
var Topo_oneCode = sharedWamCode
var Topo_oneLabels = sharedWamLabels
const Topo_oneStartPC = 2368

func Topo_one(a1 Value, a2 Value, a3 Value, a4 Value, a5 Value, a6 Value, a7 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2368
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
// WAM-compiled predicate: topo_sort_sel/3 (shared table, pc=2446)
var Topo_sort_selCode = sharedWamCode
var Topo_sort_selLabels = sharedWamLabels
const Topo_sort_selStartPC = 2446

func Topo_sort_sel(a1 Value, a2 Value, a3 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2446
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: upgrade_set/4 (shared table, pc=2478)
var Upgrade_setCode = sharedWamCode
var Upgrade_setLabels = sharedWamLabels
const Upgrade_setStartPC = 2478

func Upgrade_set(a1 Value, a2 Value, a3 Value, a4 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2478
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    vm.Regs[2] = a3
    vm.Regs[3] = a4
    return vm.Run()
}


// Strategy: wam
// WAM-compiled predicate: upgrade_set_result/4 (shared table, pc=2492)
var Upgrade_set_resultCode = sharedWamCode
var Upgrade_set_resultLabels = sharedWamLabels
const Upgrade_set_resultStartPC = 2492

func Upgrade_set_result(a1 Value, a2 Value, a3 Value, a4 Value) bool {
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
// WAM-compiled predicate: version_lt/2 (shared table, pc=2525)
var Version_ltCode = sharedWamCode
var Version_ltLabels = sharedWamLabels
const Version_ltStartPC = 2525

func Version_lt(a1 Value, a2 Value) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = 2525
    vm.Regs[0] = a1
    vm.Regs[1] = a2
    return vm.Run()
}

