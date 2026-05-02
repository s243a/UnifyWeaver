#include <stdio.h>
#define D(name) extern int run_##name(void);
D(probe_s1g_v0) D(probe_s1g_v3) D(probe_s1g_v5) D(probe_s1g_v6) D(probe_s1g_v7) D(probe_s1g_v8)
D(probe_smed_v6) D(probe_smed_v7) D(probe_smed_v9) D(probe_smed_v10)
int main(void) {
    printf("s1g(0): r=%d\n", run_probe_s1g_v0()); fflush(stdout);
    printf("s1g(3): r=%d\n", run_probe_s1g_v3()); fflush(stdout);
    printf("s1g(5): r=%d\n", run_probe_s1g_v5()); fflush(stdout);
    printf("s1g(6): r=%d\n", run_probe_s1g_v6()); fflush(stdout);
    printf("s1g(7): r=%d\n", run_probe_s1g_v7()); fflush(stdout);
    printf("s1g(8): r=%d\n", run_probe_s1g_v8()); fflush(stdout);
    printf("smed(6): r=%d\n", run_probe_smed_v6()); fflush(stdout);
    printf("smed(7): r=%d\n", run_probe_smed_v7()); fflush(stdout);
    printf("smed(9): r=%d\n", run_probe_smed_v9()); fflush(stdout);
    printf("smed(10): r=%d\n", run_probe_smed_v10()); fflush(stdout);
    return 0;
}
