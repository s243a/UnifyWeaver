/* rpyc_bridge.h - C header for Rust FFI Bridge */
#ifndef RPYC_BRIDGE_H
#define RPYC_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Python runtime. Call once at startup. */
int rpyc_init(void);

/* Connect to RPyC server. Returns 0 on success, -1 on error. */
int rpyc_connect(const char* host, int port);

/* Disconnect from RPyC server. */
void rpyc_disconnect(void);

/* Call a function on a remote module.
 * Returns JSON string (caller must free with rpyc_free_string).
 * Example: rpyc_call("math", "sqrt", "[16]") -> "4.0"
 */
char* rpyc_call(const char* module, const char* func, const char* args_json);

/* Get attribute from remote module.
 * Returns JSON string (caller must free with rpyc_free_string).
 */
char* rpyc_getattr(const char* module, const char* attr);

/* Free a string returned by rpyc_call or rpyc_getattr. */
void rpyc_free_string(char* s);

/* Check if connected to RPyC server. Returns 1 if connected, 0 otherwise. */
int rpyc_is_connected(void);

/* Get last error message (may return NULL). */
const char* rpyc_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* RPYC_BRIDGE_H */
