/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Executable boundary checks on the emitted WAM runtime and C JSON shim. */
#include "term_heap.h"
#include "term_build.h"
#include "json.h"
#include "test_helpers.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static WamState vm;
static TermHeap th;
static int fail_realloc, fail_calloc, fail_malloc;
void *__real_realloc(void *, size_t);
void *__real_calloc(size_t, size_t);
void *__real_malloc(size_t);
void *__wrap_realloc(void *p, size_t n) { return fail_realloc ? NULL : __real_realloc(p, n); }
void *__wrap_calloc(size_t n, size_t s) { return fail_calloc ? NULL : __real_calloc(n, s); }
void *__wrap_malloc(size_t n) { return fail_malloc ? NULL : __real_malloc(n); }

static void reset(void) {
    fail_malloc = fail_calloc = fail_realloc = 0;
    wam_free_state(&vm);
    wam_state_init(&vm);
    term_heap_init(&th, &vm);
}

static void compare(WamValue left, WamValue right, const char *want, const char *label) {
    vm.A[0] = wam_make_ref(&vm);
    vm.A[1] = left;
    vm.A[2] = right;
    check(wam_execute_builtin(&vm, "compare/3", 3), label);
    WamValue *out = wam_deref_ptr(&vm, &vm.A[0]);
    check(out->tag == VAL_ATOM && strcmp(out->data.atom, want) == 0, label);
}

static void bad_json(const char *input, const char *label) {
    JsonParseError error = {0};
    Json j = json_parse(input, &error);
    check(error.message != NULL, label);
    json_free(&j);
}

int main(void) {
    wam_state_init(&vm);
    term_heap_init(&th, &vm);
    compare(val_int(1), val_int(2), "<", "integers");
    compare(val_float(1.0), val_int(1), "<", "float before equal integer");
    compare(val_int(1), val_float(1.0), ">", "integer after equal float");
    compare(val_float(-0.0), val_float(0.0), "<", "negative zero");
    compare(val_float(0.0), val_float(-0.0), ">", "positive zero");
    compare(val_float(-0.0), val_float(-0.0), "=", "same negative zero");
    compare(val_float(NAN), val_float(1.0), "<", "NaN before finite");
    compare(val_float(1.0), val_float(NAN), ">", "finite after NaN");
    compare(val_float(NAN), val_float(-INFINITY), "<", "NaN before infinity");
    compare(val_float(NAN), val_float(NAN), "=", "NaN equality");
    compare(val_float(INFINITY), val_int(INT_MAX), ">", "infinity");
    compare(val_atom("a"), val_atom("b"), "<", "atom order");
    WamValue x = wam_make_ref(&vm), y = wam_make_ref(&vm);
    compare(x, x, "=", "same variable");
    compare(x, y, "<", "variable allocation order");
    compare(y, x, ">", "reverse variable order");
    compare(x, val_int(0), "<", "variable before number");
    WamValue aa[2] = {val_int(1),val_int(2)};
    WamValue f = term_heap_compound(&th,"z",1,aa);
    WamValue g = term_heap_compound(&th,"a",2,aa);
    compare(f,g,"<","arity before functor name");
    vm.A[0]=val_atom(">");vm.A[1]=val_int(1);vm.A[2]=val_int(2);
    check(!wam_execute_builtin(&vm,"compare/3",3),"bound order mismatch fails");
    check(vm.error==0,"bound mismatch is logical failure");
    WamValue deep1 = val_int(1), deep2 = val_int(1);
    for(int i=0;i<260;i++) { deep1=term_heap_compound(&th,"f",1,&deep1);deep2=term_heap_compound(&th,"f",1,&deep2); }
    vm.A[0]=wam_make_ref(&vm);vm.A[1]=deep1;vm.A[2]=deep2;
    check(!wam_execute_builtin(&vm,"compare/3",3) && vm.error==WAM_ERR_UNSUPPORTED,"comparison depth fails explicitly");

    reset();
    term_heap_int(&th, INT_MIN);term_heap_int(&th,INT_MAX);
    check(!th.error,"WAM int endpoints");
    term_heap_int(&th,(int64_t)INT_MAX+1);
    check(th.error!=NULL,"reject WAM integer overflow");
    reset();
    term_heap_int(&th,(int64_t)INT_MIN-1);
    check(th.error!=NULL,"reject WAM integer underflow");
    reset();
    JsonParseError err={0};Json ver=json_parse("[4294967296,0,0]",&err);
    tb_ver_term(&th,&ver);check(th.error!=NULL,"version conversion rejects truncation");json_free(&ver);
    reset();
    ver=json_parse("{\"deb\":[0,[[\"\",4294967296]],[]]}",&err);
    tb_ver_term(&th,&ver);check(th.error!=NULL,"Debian segment rejects truncation");json_free(&ver);

    reset();
    term_heap_intern(&th,"f/0");
    vm.H=vm.H_cap;
    WamValue *old_heap=vm.H_array;int old_h=vm.H,old_cap=vm.H_cap;
    fail_realloc=1;
    WamValue result=term_heap_compound(&th,"f",0,NULL);
    check(th.error && result.tag==VAL_UNBOUND,"allocation failure propagates");
    check(vm.H_array==old_heap && vm.H==old_h && vm.H_cap==old_cap,"allocation failure leaves heap untouched");
    reset();
    vm.H=1;old_h=vm.H;
    check(!term_heap_ensure(&th,INT_MAX) && th.error,"oversized allocation fails");
    check(vm.H==old_h,"capacity failure leaves heap untouched");
    reset();
    fail_calloc=1;
    check(!term_heap_calloc(&th,4,sizeof(WamValue)) && th.error,"temporary allocation failure propagates");
    reset();
    term_heap_intern(&th,"already_owned");fail_malloc=1;
    result=term_heap_atom(&th,"fresh_atom");
    check(th.error && result.tag==VAL_UNBOUND,"atom allocation failure cannot borrow transient input");
    reset();
    const char *owned=term_heap_intern(&th,"already_owned");
    check(term_heap_intern(&th,owned)==owned && !th.error,"already owned atom remains valid");

    bad_json("9223372036854775808","JSON positive overflow");
    bad_json("-9223372036854775809","JSON negative overflow");
    bad_json("18446744073709551616","JSON unsigned wrap");
    Json j=json_parse("-9223372036854775808",&err);
    check(!err.message && json_as_int(&j)==INT64_MIN,"JSON minimum integer");json_free(&j);
    j=json_parse("9223372036854775807",&err);
    check(!err.message && json_as_int(&j)==INT64_MAX,"JSON maximum integer");json_free(&j);
    j=json_parse("\"\\ud83d\\ude00\"",&err);
    check(!err.message && strcmp(json_as_string(&j),"\xf0\x9f\x98\x80")==0,"surrogate pair decodes to UTF8");json_free(&j);
    j=json_parse("\"12345678901234567890123456789\\ud83d\\ude00\"",&err);
    check(!err.message && strlen(json_as_string(&j))==33,"surrogate pair at buffer boundary");json_free(&j);
    bad_json("\"\\ud800\"","unpaired high surrogate");
    bad_json("\"\\ud800\\u0041\"","invalid low surrogate");
    bad_json("\"\\udc00\"","unpaired low surrogate");
    bad_json("\"p\\u0000q\"","NUL rejected");
    bad_json("{\"p\\u0000q\":1}","NUL object key rejected");
    bad_json("\"\\u1\"","short Unicode escape");
    wam_free_state(&vm);
    return finish_tests();
}
