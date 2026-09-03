package wam

// Auto-generated atom intern table.
// Shared by lib.go (WAM bytecode literals) and lowered.go (lowered
// predicate functions). Pointer-identity equality on these vars is
// O(1); duplicate inline `&Atom{Name:"..."}` literals fall back to
// string compare in Atom.Equals.

// Interned atom literals (compile-time deduplicated)
var (
    wamAtom____0 = internAtom("[]")
    wamAtom_blanket_1 = internAtom("blanket")
    wamAtom_abi_anchor_2 = internAtom("abi_anchor")
    wamAtom_over_frozen_3 = internAtom("over_frozen")
    wamAtom_none_4 = internAtom("none")
    wamAtom_classic_5 = internAtom("classic")
    wamAtom_layered_6 = internAtom("layered")
    wamAtom_from_catalog_7 = internAtom("from_catalog")
    wamAtom_from_base_8 = internAtom("from_base")
    wamAtom_any_9 = internAtom("any")
    wamAtom_no_candidate_10 = internAtom("no_candidate")
    wamAtom_modified_11 = internAtom("modified")
    wamAtom_footprint_12 = internAtom("footprint")
    wamAtom_layer_shadow_13 = internAtom("layer_shadow")
    wamAtom_base_14 = internAtom("base")
)

// atomInternMap is the single source of pointer identity for atoms.
// internAtom(name) returns the shared pointer for that name, allocating
// and caching one if the name is new. The wamAtom_ vars above are
// themselves initialised through internAtom, so they populate this map
// as a side effect of package initialisation.
//
// Deliberately NOT an init() that assigns into the map: Go runs every
// package-level variable initialiser before any init(), so a var in
// another file of this package that calls internAtom during its own
// initialisation (state.go's emptyListAtom) would win the map slot and
// then be overwritten here, leaving two atoms with the same name and
// different pointers. Atom.Equals is pointer-only, so that silently
// broke equality — `get_constant []` against a real empty list.
//
// Bench drivers should construct atoms via internAtom rather than
// `&Atom{Name: x}` for the same reason: SwitchOnConstant in the WAM
// bytecode matches in O(1) on pointer identity.
var atomInternMap = make(map[string]*Atom)

func internAtom(name string) *Atom {
    if a, ok := atomInternMap[name]; ok {
        return a
    }
    a := &Atom{Name: name}
    atomInternMap[name] = a
    return a
}

// InternAtom is the exported form. Drivers and embedders must build
// atoms through this rather than &Atom{Name: x}: Atom.Equals is pointer
// identity only, so a fresh literal with the same name compares unequal
// to the interned one the bytecode carries.
func InternAtom(name string) *Atom {
    return internAtom(name)
}

