:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module(library(filesex)).

:- begin_tests(go_wam_builtins).

:- dynamic user:test_builtins/1.
:- dynamic user:test_arithmetic_expr_builtin/0.
:- dynamic user:test_term_builtins/0.
:- dynamic user:test_member_collect/0.
:- dynamic user:test_memberchk_builtin/0.
:- dynamic user:test_select_builtin/0.
:- dynamic user:test_delete_builtin/0.
:- dynamic user:test_append_builtin/0.
:- dynamic user:test_subtract_builtin/0.
:- dynamic user:test_intersection_builtin/0.
:- dynamic user:test_union_builtin/0.
:- dynamic user:test_permutation_builtin/0.
:- dynamic user:test_reverse_builtin/0.
:- dynamic user:test_last_builtin/0.
:- dynamic user:test_nth_builtin/0.
:- dynamic user:test_numlist_builtin/0.
:- dynamic user:test_sort_builtin/0.
:- dynamic user:test_keysort_builtin/0.
:- dynamic user:test_term_order_builtin/0.
:- dynamic user:test_ground_builtin/0.
:- dynamic user:test_sub_atom_builtin/0.
:- dynamic user:test_char_type_builtin/0.
:- dynamic user:test_string_code_builtin/0.
:- dynamic user:test_split_string_builtin/0.
:- dynamic user:test_output_builtin/0.
:- dynamic user:test_format_builtin/0.
:- dynamic user:test_tab_builtin/0.
:- dynamic user:test_env_builtin/0.
:- dynamic user:test_succ_builtin/0.
:- dynamic user:test_between_builtin/0.
:- dynamic user:test_list_numeric_builtin/0.
:- dynamic user:test_list_to_set_builtin/0.
:- dynamic user:test_atom_number_builtin/0.
:- dynamic user:test_atom_case_builtin/0.
:- dynamic user:test_atom_concat_builtin/0.
:- dynamic user:test_atom_string_length_builtin/0.
:- dynamic user:test_char_code_builtin/0.
:- dynamic user:test_atom_codes_builtin/0.
:- dynamic user:test_atom_chars_builtin/0.
:- dynamic user:test_string_list_builtin/0.
:- dynamic user:test_number_list_builtin/0.
:- dynamic user:test_atom_string_builtin/0.
:- dynamic user:test_set_aggregate/0.
:- dynamic user:test_unify_builtin/0.
:- dynamic user:test_neg_fact/1.
:- dynamic user:test_neg_goal/0.
:- dynamic user:test_neg_goal_fail/0.

test(builtins_execution) :-
    get_time(T),
    format(atom(TmpDir), 'tmp_wam_builtins_~w', [T]),
    setup_call_cleanup(
        ( assertz(user:test_builtins(X) :-
            (   X is 1 + 2,
                X < 5,
                X =:= 3,
                X =< 3,
                is_list([a,b]),
                display(ok),
                nl,
                atom(foo),
                \+ atom(5)
            )),
          assertz(user:test_arithmetic_expr_builtin :-
            (   Abs is abs(-7),
                Abs =:= 7,
                Neg is -5,
                Neg =:= -5,
                SignNeg is sign(-3),
                SignNeg =:= -1,
                SignZero is sign(0),
                SignZero =:= 0,
                SignPos is sign(3),
                SignPos =:= 1,
                FloatVal is float(3),
                FloatVal =:= 3,
                Trunc is truncate(3.9),
                Trunc =:= 3,
                IntegerVal is integer(4.8),
                IntegerVal =:= 4,
                FloatIntegerPart is float_integer_part(5.7),
                FloatIntegerPart =:= 5,
                Sqrt is sqrt(9),
                Sqrt =:= 3,
                Sin is sin(0),
                Sin =:= 0,
                Cos is cos(0),
                Cos =:= 1,
                Tan is tan(0),
                Tan =:= 0,
                Asin is asin(0),
                Asin =:= 0,
                Acos is acos(1),
                Acos =:= 0,
                Atan is atan(0),
                Atan =:= 0,
                Floor is floor(3.9),
                Floor =:= 3,
                Ceiling is ceiling(3.1),
                Ceiling =:= 4,
                Round is round(3.5),
                Round =:= 4,
                Div is 7 / 2,
                Div =:= 3.5,
                IntDiv is 7 // 2,
                IntDiv =:= 3,
                Mod is 7 mod 3,
                Mod =:= 1,
                Max is max(4, 9),
                Max =:= 9,
                Min is min(4, 9),
                Min =:= 4,
                PowA is 2 ** 3,
                PowA =:= 8,
                PowB is 2 ^ 4,
                PowB =:= 16,
                And is 6 /\ 3,
                And =:= 2,
                Or is 4 \/ 1,
                Or =:= 5,
                Xor is 6 xor 3,
                Xor =:= 5,
                ShiftRight is 16 >> 2,
                ShiftRight =:= 4,
                ShiftLeft is 3 << 2,
                ShiftLeft =:= 12,
                \+ (_ is 1 / 0),
                \+ (_ is 1 // 0),
                \+ (_ is 1 mod 0),
                \+ (_ is sqrt(-1)),
                \+ (_ is asin(2)),
                \+ (_ is acos(2)),
                \+ (_ is _ + 1),
                \+ (_ is unknown(1))
            )),
          assertz(user:test_term_builtins :-
            (   functor(f(a, 7), F, A),
                F == f,
                A =:= 2,
                arg(2, f(a, 7), V),
                V =:= 7,
                f(a, 7) =.. L,
                length(L, 3),
                member(f, L),
                member(a, L),
                member(7, L),
                copy_term(f(X, X), C),
                arg(1, C, Y),
                arg(2, C, Z),
                Y == Z
            )),
          assertz(user:test_member_collect :-
            (   findall(X, member(X, [a,b]), L),
                length(L, 2),
                member(a, L),
                member(b, L)
            )),
          assertz(user:test_memberchk_builtin :-
            (   memberchk(X, [a,b,c]),
                X == a,
                \+ memberchk(z, [a,b,c])
            )),
          assertz(user:test_select_builtin :-
            (   select(b, [a,b,c], R),
                R = [a,c],
                select(X, [a,b], [b]),
                X == a,
                select(a, [a,b,c], RA),
                RA = [b,c],
                select(c, [a,b,c], RC),
                RC = [a,b],
                findall(Y, select(Y, [a,b,c], _), Ys),
                Ys = [a,b,c],
                findall(Rest, select(_, [a,b,c], Rest), Rests),
                member([b,c], Rests),
                member([a,c], Rests),
                \+ select(z, [a,b,c], _),
                \+ select(_, [], _),
                \+ select(_, [a|b], _)
            )),
          assertz(user:test_delete_builtin :-
            (   delete([a,b,c,b], b, R),
                R = [a,c],
                delete([a,b,c], z, Same),
                Same = [a,b,c],
                delete([a,a,a], a, Empty),
                Empty = [],
                \+ delete([a|b], a, _)
            )),
          assertz(user:test_append_builtin :-
            (   append([a,b], [c,d], R),
                R = [a,b,c,d],
                append([], [z], EmptyLeft),
                EmptyLeft = [z],
                append([x], [], EmptyRight),
                EmptyRight = [x],
                append([1,2], [3], [1,2,3]),
                append([a], [b], Bound),
                Bound = [a,b],
                \+ append([a], [b], [a,c]),
                \+ append([a|b], [c], _),
                \+ append([a], [b|c], _),
                \+ append(_, [b], _),
                \+ append([a], _, _)
            )),
          assertz(user:test_subtract_builtin :-
            (   subtract([a,b,c,b], [b,d], R),
                R = [a,c],
                subtract([a,b,c], [], Same),
                Same = [a,b,c],
                subtract([], [a], Empty),
                Empty = [],
                subtract([1,2,1,3,1], [1], Nums),
                Nums = [2,3],
                subtract([10,5,20,5,30], [5], Ordered),
                Ordered = [10,20,30],
                subtract([a,b,c], [a,b,c], AllRemoved),
                AllRemoved = [],
                \+ subtract([a|b], [a], _),
                \+ subtract([a,b], [a|b], _)
            )),
          assertz(user:test_intersection_builtin :-
            (   intersection([a,b,c,b], [b,d], R),
                R = [b,b],
                intersection([a,b,c], [d,e], Disjoint),
                Disjoint = [],
                intersection([], [a], EmptyLeft),
                EmptyLeft = [],
                intersection([a,b], [], EmptyRight),
                EmptyRight = [],
                intersection([1,2,1,3,1], [1], Nums),
                Nums = [1,1,1],
                intersection([10,5,20,5,30], [5,20], Ordered),
                Ordered = [5,20,5],
                intersection([a,b,c], [c,b,a], AllKept),
                AllKept = [a,b,c],
                \+ intersection([a|b], [a], _),
                \+ intersection([a,b], [a|b], _)
            )),
          assertz(user:test_union_builtin :-
            (   union([a,b,c], [b,d,e], R),
                R = [a,b,c,d,e],
                union([a,a,b], [c], DupesKept),
                DupesKept = [a,a,b,c],
                union([], [x,y], EmptyLeft),
                EmptyLeft = [x,y],
                union([x,y], [], EmptyRight),
                EmptyRight = [x,y],
                union([], [], BothEmpty),
                BothEmpty = [],
                union([1,2,3], [2,4], Overlap),
                Overlap = [1,2,3,4],
                union([1,2,3], [1,2,3], Identical),
                Identical = [1,2,3],
                union([1,2,3,4], [3,4,5,6], U),
                intersection([1,2,3,4], [3,4,5,6], I),
                length(U, NU),
                length(I, NI),
                Total is NU + NI,
                Total =:= 8,
                \+ union([a|b], [a], _),
                \+ union([a,b], [a|b], _)
            )),
          assertz(user:test_permutation_builtin :-
            (   permutation([a,b,c], [c,a,b]),
                permutation([3,1,2,1], [1,1,2,3]),
                permutation([foo,1,bar], [bar,foo,1]),
                permutation([x,y], Same),
                Same = [x,y],
                \+ permutation([a,b,c], [a,b,d]),
                \+ permutation([a,b], [a,b,b]),
                \+ permutation([a|b], _),
                \+ permutation([a,b], [a|b])
            )),
          assertz(user:test_reverse_builtin :-
            (   reverse([a,b,c], R),
                R = [c,b,a],
                reverse(L, [z,y]),
                L = [y,z]
            )),
          assertz(user:test_last_builtin :-
            (   last([a,b,c], X),
                X == c,
                last([only], Y),
                Y == only,
                \+ last([], _)
            )),
          assertz(user:test_nth_builtin :-
            (   nth0(0, [a,b,c], A),
                A == a,
                nth0(1, [a,b,c], B),
                B == b,
                nth1(1, [a,b,c], C),
                C == a,
                nth1(3, [a,b,c], D),
                D == c,
                \+ nth0(-1, [a,b,c], _),
                \+ nth0(3, [a,b,c], _),
                \+ nth1(0, [a,b,c], _),
                \+ nth1(4, [a,b,c], _)
            )),
          assertz(user:test_numlist_builtin :-
            (   numlist(2, 5, L),
                L = [2,3,4,5],
                numlist(3, 3, S),
                S = [3],
                \+ numlist(5, 2, _)
            )),
          assertz(user:test_between_builtin :-
            (   between(1, 3, 1),
                between(1, 3, 3),
                \+ between(1, 3, 4),
                \+ between(5, 2, _),
                findall(N, between(2, 5, N), Ns),
                Ns = [2,3,4,5],
                findall(One, between(7, 7, One), Ones),
                Ones = [7]
            )),
          assertz(user:test_list_numeric_builtin :-
            (   sum_list([1,2,3,4], Sum),
                Sum =:= 10,
                sum_list([], Zero),
                Zero =:= 0,
                sum_list([1,2.5,3], MixedSum),
                MixedSum =:= 6.5,
                max_list([3,1,4,1,5,9,2,6], Max),
                Max =:= 9,
                min_list([3,1,4,1,5,9,2,6], Min),
                Min =:= 1,
                max_list([7], OneMax),
                OneMax =:= 7,
                min_list([7], OneMin),
                OneMin =:= 7,
                \+ max_list([], _),
                \+ min_list([], _),
                \+ sum_list([1,a], _),
                \+ sum_list([1|2], _)
            )),
          assertz(user:test_list_to_set_builtin :-
            (   list_to_set([a,b,c,a,b,d,a], Set),
                Set = [a,b,c,d],
                list_to_set([], Empty),
                Empty = [],
                list_to_set([x], One),
                One = [x],
                list_to_set([1,2,1,3,2], Nums),
                Nums = [1,2,3],
                \+ list_to_set([a|b], _)
            )),
          assertz(user:test_sort_builtin :-
            (   sort([3,1,2,1,3], S),
                S = [1,2,3],
                msort([3,1,2,1,3], M),
                M = [1,1,2,3,3],
                sort([foo,1,bar,2], Mixed),
                Mixed = [bar,foo,1,2],
                \+ sort([a|b], _),
                \+ msort([a|b], _)
            )),
          assertz(user:test_keysort_builtin :-
            (   keysort([3-a,1-b,2-c], IntSorted),
                IntSorted = [1-b,2-c,3-a],
                keysort([5-x,2-y,8-z,1-w], LastSorted),
                LastSorted = [1-w,2-y,5-x,8-z],
                keysort([1-a,2-b,3-c], AlreadySorted),
                AlreadySorted = [1-a,2-b,3-c],
                keysort([5-a,4-b,3-c,2-d,1-e], ReverseSorted),
                ReverseSorted = [1-e,2-d,3-c,4-b,5-a],
                keysort([], Empty),
                Empty = [],
                keysort([42-only], Singleton),
                Singleton = [42-only],
                keysort([2-first,1-x,2-second,1-y], StableDupes),
                StableDupes = [1-x,1-y,2-first,2-second],
                keysort([banana-2,apple-1,cherry-3], AtomSorted),
                AtomSorted = [apple-1,banana-2,cherry-3],
                keysort([3.5-a,1.5-b,2.5-c], FloatSorted),
                FloatSorted = [1.5-b,2.5-c,3.5-a],
                keysort([2-a,1.5-b,1-c,2.5-d], MixedSorted),
                MixedSorted = [1-c,1.5-b,2-a,2.5-d],
                \+ keysort([a|b], _),
                \+ keysort([not_a_pair], _)
            )),
          assertz(user:test_term_order_builtin :-
            (   bar @< foo,
                foo @=< foo,
                2 @> 1,
                2 @>= 2,
                bar @< 1,
                \+ 1 @< bar,
                compare(<, bar, foo),
                compare(=, foo, foo),
                compare(>, 2, 1)
            )),
          assertz(user:test_ground_builtin :-
            (   ground(hello),
                ground(42),
                ground(foo(1, [2, 3], bar)),
                ground([a,b,c]),
                ground([]),
                \+ ground(_),
                \+ (Scratch = [z], ground(_), Scratch = [z]),
                \+ ground(foo(1, _, 3)),
                \+ ground([a, _])
            )),
          assertz(user:test_sub_atom_builtin :-
            (   sub_atom(hello, 1, 3, A, S),
                A =:= 1,
                S == ell,
                sub_atom(hello, 0, 3, 2, hel),
                sub_atom(hello, 2, 3, 0, llo),
                sub_atom(abc, 0, 3, 0, abc),
                sub_atom(abc, 0, 0, 3, ''),
                sub_atom(12345, 1, 3, 1, '234'),
                \+ sub_atom(abc, 1, 99, _, _),
                \+ sub_atom(abc, -1, 1, _, _),
                \+ sub_atom(abc, 0, -1, _, _),
                \+ sub_atom(_, 0, 1, _, _),
                \+ sub_atom(abc, _, 1, _, _),
                \+ sub_atom(abc, 0, _, _, _),
                \+ sub_atom(abc, 0, 2, 0, ab),
                \+ sub_atom(abc, 0, 2, _, ac)
            )),
          assertz(user:test_char_type_builtin :-
            (   char_type(a, alpha),
                char_type('5', alnum),
                char_type('5', digit),
                char_code(Space, 32),
                char_type(Space, space),
                char_type(Space, white),
                char_type('Z', upper),
                char_type(z, lower),
                char_type('!', punct),
                char_type('A', ascii),
                char_type('_', csym),
                char_type('_', csymf),
                char_code(Newline, 10),
                char_type(Newline, newline),
                \+ char_type('1', alpha),
                \+ char_type('Z', lower),
                \+ char_type(z, upper),
                \+ char_type(ab, alpha),
                \+ char_type(_, alpha),
                \+ char_type(a, _),
                \+ char_type(a, unknown)
            )),
          assertz(user:test_string_code_builtin :-
            (   string_code(1, abc, C1),
                C1 =:= 97,
                string_code(2, abc, 98),
                string_code(3, abc, C3),
                C3 =:= 99,
                \+ string_code(0, abc, _),
                \+ string_code(4, abc, _),
                \+ string_code(-1, abc, _),
                \+ string_code(_, abc, _),
                \+ string_code(1, _, _),
                \+ string_code(1, 42, _),
                \+ string_code(1, abc, 98)
            )),
          assertz(user:test_split_string_builtin :-
            (   char_code(Comma, 44),
                char_code(Semi, 59),
                atom_concat(Comma, Semi, CommaSemi),
                split_string('a,b,c', Comma, '', [a,b,c]),
                split_string('', Comma, '', ['']),
                split_string(hello, Comma, '', [hello]),
                split_string('a,,b', Comma, '', [a,'',b]),
                char_code(Space, 32),
                atom_concat(Space, hello, Padded0),
                atom_concat(Padded0, Space, Padded),
                split_string(Padded, '', Space, [hello]),
                atom_concat(a, Comma, SP0),
                atom_concat(SP0, Space, SP1),
                atom_concat(SP1, b, SepPadInput),
                split_string(SepPadInput, Comma, Space, [a,b]),
                split_string('a,b;c,d', CommaSemi, '', [a,b,c,d]),
                atom_concat(abc, Space, Trail0),
                atom_concat(Trail0, Space, Trail),
                split_string(Trail, '', Space, [abc]),
                split_string(123, '', '', ['123']),
                \+ split_string(_, Comma, '', _),
                \+ split_string('a,b', _, '', _),
                \+ split_string('a,b', Comma, _, _),
                \+ split_string('a,b', Comma, '', [a,b,c])
            )),
          assertz(user:test_tab_builtin :-
            (   tab(3),
                write(tabbed),
                tab(0),
                \+ tab(-1),
                \+ tab(_),
                \+ tab(foo)
            )),
          assertz(user:test_env_builtin :-
            (   setenv('UW_GO_WAM_ENV_TEST', go_wam_value),
                getenv('UW_GO_WAM_ENV_TEST', V),
                V == go_wam_value,
                setenv('UW_GO_WAM_ENV_TEST', overwritten),
                getenv('UW_GO_WAM_ENV_TEST', V2),
                V2 == overwritten,
                setenv('UW_GO_WAM_EMPTY', ''),
                getenv('UW_GO_WAM_EMPTY', Empty),
                atom_length(Empty, Len),
                Len =:= 0,
                \+ getenv('UW_GO_WAM_DEFINITELY_UNSET', _),
                \+ getenv(_, _),
                \+ setenv(_, value),
                \+ setenv('UW_GO_WAM_ENV_TEST', _)
            )),
          assertz(user:test_succ_builtin :-
            (   succ(0, 1),
                succ(2, X),
                X =:= 3,
                succ(Y, 4),
                Y =:= 3,
                \+ succ(1, 1),
                \+ succ(-1, _),
                \+ succ(_, 0),
                \+ succ(_, _)
            )),
          assertz(user:test_atom_number_builtin :-
            (   atom_number('42', N),
                N =:= 42,
                atom_number(A, 42),
                A == '42',
                atom_number(C, '42'),
                C == '42',
                atom_number('3.5', F),
                F =:= 3.5,
                atom_number(B, 3.5),
                B == '3.5',
                atom_number(42, 42),
                \+ atom_number(foo, _),
                \+ atom_number(_, _)
            )),
          assertz(user:test_atom_case_builtin :-
            (   upcase_atom(hello, U),
                U == 'HELLO',
                upcase_atom(hello, 'HELLO'),
                \+ upcase_atom(hello, hello),
                downcase_atom('HELLO', D),
                D == hello,
                downcase_atom('HELLO', hello),
                \+ downcase_atom('HELLO', 'HELLO'),
                \+ upcase_atom(_, _),
                \+ upcase_atom(42, _)
            )),
          assertz(user:test_atom_concat_builtin :-
            (   atom_concat(fo, o, A),
                A == foo,
                atom_concat(fo, o, foo),
                \+ atom_concat(fo, o, bar),
                \+ atom_concat(_, o, foo),
                \+ atom_concat(fo, _, foo),
                \+ atom_concat(42, o, _),
                \+ atom_concat(fo, 42, _)
            )),
          assertz(user:test_atom_string_length_builtin :-
            (   atom_length(foo, AL),
                AL =:= 3,
                atom_length(foo, 3),
                \+ atom_length(foo, 2),
                string_length(bar, SL),
                SL =:= 3,
                string_length(bar, 3),
                \+ string_length(bar, 2),
                \+ atom_length(_, _),
                \+ string_length(_, _),
                \+ atom_length(42, _),
                \+ string_length(42, _)
            )),
          assertz(user:test_char_code_builtin :-
            (   char_code(a, C),
                C =:= 97,
                char_code(a, 97),
                \+ char_code(a, 98),
                char_code(A, 65),
                A == 'A',
                \+ char_code(ab, _),
                \+ char_code(_, _),
                \+ char_code(_, -1),
                \+ char_code(_, 65536)
            )),
          assertz(user:test_atom_codes_builtin :-
            (   atom_codes(foo, Codes),
                Codes = [102,111,111],
                atom_codes(foo, [102,111,111]),
                \+ atom_codes(foo, [102,111]),
                atom_codes(A2, [98,97,114]),
                A2 == bar,
                atom_codes('', Empty),
                Empty = [],
                \+ atom_codes(_, _),
                \+ atom_codes(_, [-1]),
                \+ atom_codes(_, [65536]),
                \+ atom_codes(_, [foo]),
                \+ atom_codes(_, [1.5])
            )),
          assertz(user:test_atom_chars_builtin :-
            (   atom_chars(foo, Chars),
                Chars = [f,o,o],
                atom_chars(foo, [f,o,o]),
                \+ atom_chars(foo, [f,o]),
                atom_chars(A2, [b,a,r]),
                A2 == bar,
                atom_chars('', Empty),
                Empty = [],
                \+ atom_chars(_, _),
                \+ atom_chars(_, [ab]),
                \+ atom_chars(_, [1])
            )),
          assertz(user:test_string_list_builtin :-
            (   string_codes(foo, Codes),
                Codes = [102,111,111],
                string_codes(foo, [102,111,111]),
                \+ string_codes(foo, [102,111]),
                string_codes(A2, [98,97,114]),
                A2 == bar,
                string_codes('', EmptyCodes),
                EmptyCodes = [],
                \+ string_codes(_, _),
                \+ string_codes(_, [-1]),
                \+ string_codes(_, [foo]),
                string_chars(baz, Chars),
                Chars = [b,a,z],
                string_chars(baz, [b,a,z]),
                \+ string_chars(baz, [b,a]),
                string_chars(A3, [q,u,x]),
                A3 == qux,
                string_chars('', EmptyChars),
                EmptyChars = [],
                \+ string_chars(_, _),
                \+ string_chars(_, [ab]),
                \+ string_chars(_, [1])
            )),
          assertz(user:test_number_list_builtin :-
            (   number_codes(42, Codes),
                Codes = [52,50],
                number_codes(42, [52,50]),
                \+ number_codes(42, [52]),
                number_codes(N2, [45,51]),
                N2 =:= -3,
                number_codes(F2, [51,46,53]),
                F2 =:= 3.5,
                \+ number_codes(_, _),
                \+ number_codes(_, [foo]),
                \+ number_codes(_, [65536]),
                \+ number_codes(_, [97,98,99]),
                number_chars(42, Chars),
                Chars = ['4','2'],
                number_chars(42, ['4','2']),
                \+ number_chars(42, ['4']),
                number_chars(N3, ['-','3']),
                N3 =:= -3,
                number_chars(F3, ['3','.','5']),
                F3 =:= 3.5,
                \+ number_chars(_, _),
                \+ number_chars(_, [ab]),
                \+ number_chars(_, [a,b,c])
            )),
          assertz(user:test_atom_string_builtin :-
            (   atom_string(hello, S),
                S == hello,
                atom_string(hello, hello),
                \+ atom_string(hello, world),
                atom_string(A2, world),
                A2 == world,
                string_to_atom(hello, A3),
                A3 == hello,
                string_to_atom(hello, hello),
                \+ string_to_atom(hello, world),
                \+ atom_string(_, _),
                \+ string_to_atom(_, _)
            )),
          assertz(user:test_output_builtin :-
            (   write(go_write),
                writeln(go_writeln),
                print(go_print),
                nl
            )),
          assertz(user:test_format_builtin :-
            (   format('fmt_one ~~ ok~n'),
                format('fmt_two ~w ~w~~~n', [go, 42]),
                \+ format('fmt_missing ~w', [])
            )),
          assertz(user:test_set_aggregate :-
            (   aggregate_all(set(X), member(X, [a,b,a]), S),
                length(S, 2),
                member(a, S),
                member(b, S)
            )),
          assertz(user:test_unify_builtin :-
            (   =(f(a, X), f(a, b)),
                X == b,
                \+ =(a, b)
            )),
          assertz(user:test_neg_fact(a)),
          assertz(user:test_neg_fact(b)),
          assertz(user:test_neg_goal :-
            (   \+ test_neg_fact(c)
            )),
          assertz(user:test_neg_goal_fail :-
            (   \+ test_neg_fact(a)
            ))
        ),
        run_builtins_test(TmpDir),
        ( retractall(user:test_builtins(_)),
          retractall(user:test_arithmetic_expr_builtin),
          retractall(user:test_term_builtins),
          retractall(user:test_member_collect),
          retractall(user:test_memberchk_builtin),
          retractall(user:test_select_builtin),
          retractall(user:test_delete_builtin),
          retractall(user:test_append_builtin),
          retractall(user:test_subtract_builtin),
          retractall(user:test_intersection_builtin),
          retractall(user:test_union_builtin),
          retractall(user:test_permutation_builtin),
          retractall(user:test_reverse_builtin),
          retractall(user:test_last_builtin),
          retractall(user:test_nth_builtin),
          retractall(user:test_numlist_builtin),
          retractall(user:test_sort_builtin),
          retractall(user:test_keysort_builtin),
          retractall(user:test_term_order_builtin),
          retractall(user:test_ground_builtin),
          retractall(user:test_sub_atom_builtin),
          retractall(user:test_char_type_builtin),
          retractall(user:test_string_code_builtin),
          retractall(user:test_split_string_builtin),
          retractall(user:test_tab_builtin),
          retractall(user:test_env_builtin),
          retractall(user:test_succ_builtin),
          retractall(user:test_between_builtin),
          retractall(user:test_list_numeric_builtin),
          retractall(user:test_list_to_set_builtin),
          retractall(user:test_atom_number_builtin),
          retractall(user:test_atom_case_builtin),
          retractall(user:test_atom_concat_builtin),
          retractall(user:test_atom_string_length_builtin),
          retractall(user:test_char_code_builtin),
          retractall(user:test_atom_codes_builtin),
          retractall(user:test_atom_chars_builtin),
          retractall(user:test_string_list_builtin),
          retractall(user:test_number_list_builtin),
          retractall(user:test_atom_string_builtin),
          retractall(user:test_output_builtin),
          retractall(user:test_format_builtin),
          retractall(user:test_set_aggregate),
          retractall(user:test_unify_builtin),
          retractall(user:test_neg_fact(_)),
          retractall(user:test_neg_goal),
          retractall(user:test_neg_goal_fail),
          delete_directory_and_contents(TmpDir) )
    ).

run_builtins_test(TmpDir) :-
    Predicates = [test_builtins/1, test_arithmetic_expr_builtin/0, test_term_builtins/0, test_member_collect/0, test_memberchk_builtin/0, test_select_builtin/0, test_delete_builtin/0, test_append_builtin/0, test_subtract_builtin/0, test_intersection_builtin/0, test_union_builtin/0, test_permutation_builtin/0, test_reverse_builtin/0, test_last_builtin/0, test_nth_builtin/0, test_numlist_builtin/0, test_between_builtin/0, test_list_numeric_builtin/0, test_list_to_set_builtin/0, test_sort_builtin/0, test_keysort_builtin/0, test_term_order_builtin/0, test_ground_builtin/0, test_sub_atom_builtin/0, test_char_type_builtin/0, test_string_code_builtin/0, test_split_string_builtin/0, test_output_builtin/0, test_format_builtin/0, test_tab_builtin/0, test_env_builtin/0, test_succ_builtin/0, test_atom_number_builtin/0, test_atom_case_builtin/0, test_atom_concat_builtin/0, test_atom_string_length_builtin/0, test_char_code_builtin/0, test_atom_codes_builtin/0, test_atom_chars_builtin/0, test_string_list_builtin/0, test_number_list_builtin/0, test_atom_string_builtin/0, test_set_aggregate/0, test_unify_builtin/0, test_neg_fact/1, test_neg_goal/0, test_neg_goal_fail/0],
    Options = [module_name(builtin_test), prefer_wam(true)],

    write_wam_go_project(Predicates, Options, TmpDir),

    % Verify generated code has our fixes
    directory_file_path(TmpDir, 'value.go', ValuePath),
    read_file_to_string(ValuePath, ValueCode, []),
    assertion(sub_string(ValueCode, _, _, _, "type Structure struct")),
    directory_file_path(TmpDir, 'state.go', StatePath),
    read_file_to_string(StatePath, StateCode, []),
    assertion(sub_string(StateCode, _, _, _, 'IndexedClausePCs []int')),
    assertion(sub_string(StateCode, _, _, _, 'ForeignResults   []Value')),
    assertion(sub_string(StateCode, _, _, _, 'MemberTail       Value')),
    assertion(sub_string(StateCode, _, _, _, 'func (vm *WamState) backtrack() bool')),
    assertion(sub_string(StateCode, _, _, _, 'if len(cp.IndexedClausePCs) > 0')),
    assertion(sub_string(StateCode, _, _, _, 'if cp.MemberTail != nil')),
    assertion(sub_string(StateCode, _, _, _, 'if len(cp.SelectResults) > 0')),
    assertion(sub_string(StateCode, _, _, _, 'if len(cp.ForeignResults) > 0')),
    assertion(sub_string(StateCode, _, _, _, 'func (vm *WamState) runNegationParallel(targetPC int, args []Value) bool')),
    assertion(sub_string(StateCode, _, _, _, 'func raceToTrue(tasks []func() bool) bool')),
    assertion(sub_string(StateCode, _, _, _, 'return !vm.runNegationParallel(pc, args)')),
    directory_file_path(TmpDir, 'lib.go', LibPath),
    read_file_to_string(LibPath, LibCode, []),
    assertion(sub_string(LibCode, _, _, _, 'Op: "=</2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "is_list/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "display/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "functor/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "arg/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "=../2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "copy_term/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "=/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "memberchk/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "select/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "delete/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "append/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "subtract/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "intersection/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "union/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "permutation/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "reverse/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "last/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "nth0/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "nth1/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "numlist/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "between/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "sum_list/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "min_list/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "max_list/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "list_to_set/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "sort/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "msort/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "keysort/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "@</2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "@=</2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "@>/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "@>=/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "compare/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "ground/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "sub_atom/5"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "char_type/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "string_code/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "split_string/4"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "tab/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "getenv/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "setenv/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "writeln/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "print/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "format/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "format/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "succ/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "atom_number/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "upcase_atom/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "downcase_atom/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "atom_concat/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "atom_length/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "string_length/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "char_code/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "atom_codes/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "atom_chars/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "string_codes/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "string_chars/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "number_codes/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "number_chars/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "atom_string/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "string_to_atom/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "fail/0"')),
    assertion(sub_string(LibCode, _, _, _, 'AggType: "set"')),

    % Add a main.go to run the test in a separate cmd directory
    directory_file_path(TmpDir, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'test', TestDir),
    make_directory_path(TestDir),
    directory_file_path(TestDir, 'main.go', MainPath),
    write_file(MainPath,
'package main

import (
	"fmt"
	wam "builtin_test"
)

func main() {
	vm := wam.NewWamState(wam.Test_builtinsCode, wam.Test_builtinsLabels)
	vm.PC = wam.Test_builtinsStartPC
	// A1 will hold X — use named unbound so we can deref after execution
	x := &wam.Unbound{Name: "X", Idx: 0}
	vm.Regs[0] = x

	if vm.Run() {
		fmt.Printf("SUCCESS: X=%v\\n", vm.Deref(x))
	} else {
		fmt.Println("FAILURE")
	}

	arithVM := wam.NewWamState(wam.Test_arithmetic_expr_builtinCode, wam.Test_arithmetic_expr_builtinLabels)
	arithVM.PC = wam.Test_arithmetic_expr_builtinStartPC
	if arithVM.Run() {
		fmt.Println("ARITH_EXPR_SUCCESS")
	} else {
		fmt.Println("ARITH_EXPR_FAILURE")
	}

	termVM := wam.NewWamState(wam.Test_term_builtinsCode, wam.Test_term_builtinsLabels)
	termVM.PC = wam.Test_term_builtinsStartPC
	if termVM.Run() {
		fmt.Println("TERM_SUCCESS")
	} else {
		fmt.Println("TERM_FAILURE")
	}

	memberVM := wam.NewWamState(wam.Test_member_collectCode, wam.Test_member_collectLabels)
	memberVM.PC = wam.Test_member_collectStartPC
	if memberVM.Run() {
		fmt.Println("MEMBER_SUCCESS")
	} else {
		fmt.Println("MEMBER_FAILURE")
	}

	memberchkVM := wam.NewWamState(wam.Test_memberchk_builtinCode, wam.Test_memberchk_builtinLabels)
	memberchkVM.PC = wam.Test_memberchk_builtinStartPC
	if memberchkVM.Run() {
		fmt.Println("MEMBERCHK_SUCCESS")
	} else {
		fmt.Println("MEMBERCHK_FAILURE")
	}

	selectVM := wam.NewWamState(wam.Test_select_builtinCode, wam.Test_select_builtinLabels)
	selectVM.PC = wam.Test_select_builtinStartPC
	if selectVM.Run() {
		fmt.Println("SELECT_SUCCESS")
	} else {
		fmt.Println("SELECT_FAILURE")
	}

	deleteVM := wam.NewWamState(wam.Test_delete_builtinCode, wam.Test_delete_builtinLabels)
	deleteVM.PC = wam.Test_delete_builtinStartPC
	if deleteVM.Run() {
		fmt.Println("DELETE_SUCCESS")
	} else {
		fmt.Println("DELETE_FAILURE")
	}

	appendVM := wam.NewWamState(wam.Test_append_builtinCode, wam.Test_append_builtinLabels)
	appendVM.PC = wam.Test_append_builtinStartPC
	if appendVM.Run() {
		fmt.Println("APPEND_SUCCESS")
	} else {
		fmt.Println("APPEND_FAILURE")
	}

	subtractVM := wam.NewWamState(wam.Test_subtract_builtinCode, wam.Test_subtract_builtinLabels)
	subtractVM.PC = wam.Test_subtract_builtinStartPC
	if subtractVM.Run() {
		fmt.Println("SUBTRACT_SUCCESS")
	} else {
		fmt.Println("SUBTRACT_FAILURE")
	}

	intersectionVM := wam.NewWamState(wam.Test_intersection_builtinCode, wam.Test_intersection_builtinLabels)
	intersectionVM.PC = wam.Test_intersection_builtinStartPC
	if intersectionVM.Run() {
		fmt.Println("INTERSECTION_SUCCESS")
	} else {
		fmt.Println("INTERSECTION_FAILURE")
	}

	unionVM := wam.NewWamState(wam.Test_union_builtinCode, wam.Test_union_builtinLabels)
	unionVM.PC = wam.Test_union_builtinStartPC
	if unionVM.Run() {
		fmt.Println("UNION_SUCCESS")
	} else {
		fmt.Println("UNION_FAILURE")
	}

	permutationVM := wam.NewWamState(wam.Test_permutation_builtinCode, wam.Test_permutation_builtinLabels)
	permutationVM.PC = wam.Test_permutation_builtinStartPC
	if permutationVM.Run() {
		fmt.Println("PERMUTATION_SUCCESS")
	} else {
		fmt.Println("PERMUTATION_FAILURE")
	}

	reverseVM := wam.NewWamState(wam.Test_reverse_builtinCode, wam.Test_reverse_builtinLabels)
	reverseVM.PC = wam.Test_reverse_builtinStartPC
	if reverseVM.Run() {
		fmt.Println("REVERSE_SUCCESS")
	} else {
		fmt.Println("REVERSE_FAILURE")
	}

	lastVM := wam.NewWamState(wam.Test_last_builtinCode, wam.Test_last_builtinLabels)
	lastVM.PC = wam.Test_last_builtinStartPC
	if lastVM.Run() {
		fmt.Println("LAST_SUCCESS")
	} else {
		fmt.Println("LAST_FAILURE")
	}

	nthVM := wam.NewWamState(wam.Test_nth_builtinCode, wam.Test_nth_builtinLabels)
	nthVM.PC = wam.Test_nth_builtinStartPC
	if nthVM.Run() {
		fmt.Println("NTH_SUCCESS")
	} else {
		fmt.Println("NTH_FAILURE")
	}

	numlistVM := wam.NewWamState(wam.Test_numlist_builtinCode, wam.Test_numlist_builtinLabels)
	numlistVM.PC = wam.Test_numlist_builtinStartPC
	if numlistVM.Run() {
		fmt.Println("NUMLIST_SUCCESS")
	} else {
		fmt.Println("NUMLIST_FAILURE")
	}

	sortVM := wam.NewWamState(wam.Test_sort_builtinCode, wam.Test_sort_builtinLabels)
	sortVM.PC = wam.Test_sort_builtinStartPC
	if sortVM.Run() {
		fmt.Println("SORT_SUCCESS")
	} else {
		fmt.Println("SORT_FAILURE")
	}

	keysortVM := wam.NewWamState(wam.Test_keysort_builtinCode, wam.Test_keysort_builtinLabels)
	keysortVM.PC = wam.Test_keysort_builtinStartPC
	if keysortVM.Run() {
		fmt.Println("KEYSORT_SUCCESS")
	} else {
		fmt.Println("KEYSORT_FAILURE")
	}

	termOrderVM := wam.NewWamState(wam.Test_term_order_builtinCode, wam.Test_term_order_builtinLabels)
	termOrderVM.PC = wam.Test_term_order_builtinStartPC
	if termOrderVM.Run() {
		fmt.Println("TERM_ORDER_SUCCESS")
	} else {
		fmt.Println("TERM_ORDER_FAILURE")
	}

	groundVM := wam.NewWamState(wam.Test_ground_builtinCode, wam.Test_ground_builtinLabels)
	groundVM.PC = wam.Test_ground_builtinStartPC
	if groundVM.Run() {
		fmt.Println("GROUND_SUCCESS")
	} else {
		fmt.Println("GROUND_FAILURE")
	}

	subAtomVM := wam.NewWamState(wam.Test_sub_atom_builtinCode, wam.Test_sub_atom_builtinLabels)
	subAtomVM.PC = wam.Test_sub_atom_builtinStartPC
	if subAtomVM.Run() {
		fmt.Println("SUB_ATOM_SUCCESS")
	} else {
		fmt.Println("SUB_ATOM_FAILURE")
	}

	charTypeVM := wam.NewWamState(wam.Test_char_type_builtinCode, wam.Test_char_type_builtinLabels)
	charTypeVM.PC = wam.Test_char_type_builtinStartPC
	if charTypeVM.Run() {
		fmt.Println("CHAR_TYPE_SUCCESS")
	} else {
		fmt.Println("CHAR_TYPE_FAILURE")
	}

	stringCodeVM := wam.NewWamState(wam.Test_string_code_builtinCode, wam.Test_string_code_builtinLabels)
	stringCodeVM.PC = wam.Test_string_code_builtinStartPC
	if stringCodeVM.Run() {
		fmt.Println("STRING_CODE_SUCCESS")
	} else {
		fmt.Println("STRING_CODE_FAILURE")
	}

	splitStringVM := wam.NewWamState(wam.Test_split_string_builtinCode, wam.Test_split_string_builtinLabels)
	splitStringVM.PC = wam.Test_split_string_builtinStartPC
	if splitStringVM.Run() {
		fmt.Println("SPLIT_STRING_SUCCESS")
	} else {
		fmt.Println("SPLIT_STRING_FAILURE")
	}

	tabVM := wam.NewWamState(wam.Test_tab_builtinCode, wam.Test_tab_builtinLabels)
	tabVM.PC = wam.Test_tab_builtinStartPC
	if tabVM.Run() {
		fmt.Println("TAB_SUCCESS")
	} else {
		fmt.Println("TAB_FAILURE")
	}

	envVM := wam.NewWamState(wam.Test_env_builtinCode, wam.Test_env_builtinLabels)
	envVM.PC = wam.Test_env_builtinStartPC
	if envVM.Run() {
		fmt.Println("ENV_SUCCESS")
	} else {
		fmt.Println("ENV_FAILURE")
	}

	succVM := wam.NewWamState(wam.Test_succ_builtinCode, wam.Test_succ_builtinLabels)
	succVM.PC = wam.Test_succ_builtinStartPC
	if succVM.Run() {
		fmt.Println("SUCC_SUCCESS")
	} else {
		fmt.Println("SUCC_FAILURE")
	}

	betweenVM := wam.NewWamState(wam.Test_between_builtinCode, wam.Test_between_builtinLabels)
	betweenVM.PC = wam.Test_between_builtinStartPC
	if betweenVM.Run() {
		fmt.Println("BETWEEN_SUCCESS")
	} else {
		fmt.Println("BETWEEN_FAILURE")
	}

	listNumericVM := wam.NewWamState(wam.Test_list_numeric_builtinCode, wam.Test_list_numeric_builtinLabels)
	listNumericVM.PC = wam.Test_list_numeric_builtinStartPC
	if listNumericVM.Run() {
		fmt.Println("LIST_NUMERIC_SUCCESS")
	} else {
		fmt.Println("LIST_NUMERIC_FAILURE")
	}

	listToSetVM := wam.NewWamState(wam.Test_list_to_set_builtinCode, wam.Test_list_to_set_builtinLabels)
	listToSetVM.PC = wam.Test_list_to_set_builtinStartPC
	if listToSetVM.Run() {
		fmt.Println("LIST_TO_SET_SUCCESS")
	} else {
		fmt.Println("LIST_TO_SET_FAILURE")
	}

	atomNumberVM := wam.NewWamState(wam.Test_atom_number_builtinCode, wam.Test_atom_number_builtinLabels)
	atomNumberVM.PC = wam.Test_atom_number_builtinStartPC
	if atomNumberVM.Run() {
		fmt.Println("ATOM_NUMBER_SUCCESS")
	} else {
		fmt.Println("ATOM_NUMBER_FAILURE")
	}

	atomCaseVM := wam.NewWamState(wam.Test_atom_case_builtinCode, wam.Test_atom_case_builtinLabels)
	atomCaseVM.PC = wam.Test_atom_case_builtinStartPC
	if atomCaseVM.Run() {
		fmt.Println("ATOM_CASE_SUCCESS")
	} else {
		fmt.Println("ATOM_CASE_FAILURE")
	}

	atomConcatVM := wam.NewWamState(wam.Test_atom_concat_builtinCode, wam.Test_atom_concat_builtinLabels)
	atomConcatVM.PC = wam.Test_atom_concat_builtinStartPC
	if atomConcatVM.Run() {
		fmt.Println("ATOM_CONCAT_SUCCESS")
	} else {
		fmt.Println("ATOM_CONCAT_FAILURE")
	}

	atomStringLengthVM := wam.NewWamState(wam.Test_atom_string_length_builtinCode, wam.Test_atom_string_length_builtinLabels)
	atomStringLengthVM.PC = wam.Test_atom_string_length_builtinStartPC
	if atomStringLengthVM.Run() {
		fmt.Println("ATOM_STRING_LENGTH_SUCCESS")
	} else {
		fmt.Println("ATOM_STRING_LENGTH_FAILURE")
	}

	charCodeVM := wam.NewWamState(wam.Test_char_code_builtinCode, wam.Test_char_code_builtinLabels)
	charCodeVM.PC = wam.Test_char_code_builtinStartPC
	if charCodeVM.Run() {
		fmt.Println("CHAR_CODE_SUCCESS")
	} else {
		fmt.Println("CHAR_CODE_FAILURE")
	}

	atomCodesVM := wam.NewWamState(wam.Test_atom_codes_builtinCode, wam.Test_atom_codes_builtinLabels)
	atomCodesVM.PC = wam.Test_atom_codes_builtinStartPC
	if atomCodesVM.Run() {
		fmt.Println("ATOM_CODES_SUCCESS")
	} else {
		fmt.Println("ATOM_CODES_FAILURE")
	}

	atomCharsVM := wam.NewWamState(wam.Test_atom_chars_builtinCode, wam.Test_atom_chars_builtinLabels)
	atomCharsVM.PC = wam.Test_atom_chars_builtinStartPC
	if atomCharsVM.Run() {
		fmt.Println("ATOM_CHARS_SUCCESS")
	} else {
		fmt.Println("ATOM_CHARS_FAILURE")
	}

	stringListVM := wam.NewWamState(wam.Test_string_list_builtinCode, wam.Test_string_list_builtinLabels)
	stringListVM.PC = wam.Test_string_list_builtinStartPC
	if stringListVM.Run() {
		fmt.Println("STRING_LIST_SUCCESS")
	} else {
		fmt.Println("STRING_LIST_FAILURE")
	}

	numberListVM := wam.NewWamState(wam.Test_number_list_builtinCode, wam.Test_number_list_builtinLabels)
	numberListVM.PC = wam.Test_number_list_builtinStartPC
	if numberListVM.Run() {
		fmt.Println("NUMBER_LIST_SUCCESS")
	} else {
		fmt.Println("NUMBER_LIST_FAILURE")
	}

	atomStringVM := wam.NewWamState(wam.Test_atom_string_builtinCode, wam.Test_atom_string_builtinLabels)
	atomStringVM.PC = wam.Test_atom_string_builtinStartPC
	if atomStringVM.Run() {
		fmt.Println("ATOM_STRING_SUCCESS")
	} else {
		fmt.Println("ATOM_STRING_FAILURE")
	}

	outputVM := wam.NewWamState(wam.Test_output_builtinCode, wam.Test_output_builtinLabels)
	outputVM.PC = wam.Test_output_builtinStartPC
	if outputVM.Run() {
		fmt.Println("OUTPUT_SUCCESS")
	} else {
		fmt.Println("OUTPUT_FAILURE")
	}

	formatVM := wam.NewWamState(wam.Test_format_builtinCode, wam.Test_format_builtinLabels)
	formatVM.PC = wam.Test_format_builtinStartPC
	if formatVM.Run() {
		fmt.Println("FORMAT_SUCCESS")
	} else {
		fmt.Println("FORMAT_FAILURE")
	}

	setVM := wam.NewWamState(wam.Test_set_aggregateCode, wam.Test_set_aggregateLabels)
	setVM.PC = wam.Test_set_aggregateStartPC
	if setVM.Run() {
		fmt.Println("SET_SUCCESS")
	} else {
		fmt.Println("SET_FAILURE")
	}

	unifyVM := wam.NewWamState(wam.Test_unify_builtinCode, wam.Test_unify_builtinLabels)
	unifyVM.PC = wam.Test_unify_builtinStartPC
	if unifyVM.Run() {
		fmt.Println("UNIFY_SUCCESS")
	} else {
		fmt.Println("UNIFY_FAILURE")
	}

	negVM := wam.NewWamState(wam.Test_neg_goalCode, wam.Test_neg_goalLabels)
	negVM.PC = wam.Test_neg_goalStartPC
	if negVM.Run() {
		fmt.Println("NEG_SUCCESS")
	} else {
		fmt.Println("NEG_FAILURE")
	}

	negFailVM := wam.NewWamState(wam.Test_neg_goal_failCode, wam.Test_neg_goal_failLabels)
	negFailVM.PC = wam.Test_neg_goal_failStartPC
	if negFailVM.Run() {
		fmt.Println("NEG_FAIL_UNEXPECTED_SUCCESS")
	} else {
		fmt.Println("NEG_FAIL_EXPECTED")
	}
}
'),

    % Update go.mod to include local replace for the module
    directory_file_path(TmpDir, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace builtin_test => ../../\n"], GoModNew),
    write_file(GoModPath, GoModNew),

    % Verify it compiles and runs
    (   catch(process_create(path(go), ['version'], [stdout(null), stderr(null)]), _, fail)
    ->  format(string(RunCmd), "cd ~w && go run main.go 2>&1", [TestDir]),
        process_create(path(sh), ['-c', RunCmd], [stdout(pipe(Out)), process(Pid)]),
        read_string(Out, _, FullOutput),
        process_wait(Pid, Exit),
        format('Full output from Go: ~s~n', [FullOutput]),
        assertion(Exit == exit(0)),
        assertion(sub_string(FullOutput, _, _, _, "ok")),
        assertion(sub_string(FullOutput, _, _, _, "SUCCESS: X=3")),
        assertion(sub_string(FullOutput, _, _, _, "ARITH_EXPR_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "TERM_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "MEMBER_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "MEMBERCHK_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "SELECT_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "DELETE_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "APPEND_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "SUBTRACT_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "INTERSECTION_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "UNION_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "PERMUTATION_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "REVERSE_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "LAST_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "NTH_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "NUMLIST_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "SORT_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "KEYSORT_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "TERM_ORDER_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "GROUND_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "SUB_ATOM_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "CHAR_TYPE_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "STRING_CODE_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "SPLIT_STRING_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "   tabbedTAB_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "ENV_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "SUCC_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "BETWEEN_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "LIST_NUMERIC_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "LIST_TO_SET_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "ATOM_NUMBER_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "ATOM_CASE_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "ATOM_CONCAT_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "ATOM_STRING_LENGTH_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "CHAR_CODE_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "ATOM_CODES_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "ATOM_CHARS_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "STRING_LIST_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "NUMBER_LIST_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "ATOM_STRING_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "OUTPUT_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "fmt_one ~ ok")),
        assertion(sub_string(FullOutput, _, _, _, "fmt_two go 42~")),
        assertion(sub_string(FullOutput, _, _, _, "FORMAT_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "SET_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "UNIFY_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "NEG_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "NEG_FAIL_EXPECTED"))
    ;   format("Go not found, skipping execution test.~n")
    ).

:- end_tests(go_wam_builtins).

write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).
