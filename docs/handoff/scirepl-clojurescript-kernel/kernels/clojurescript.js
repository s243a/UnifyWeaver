/**
 * kernels/clojurescript.js — ClojureScript kernel using Scittle (SCI).
 *
 * Loads scittle.js — borkdude's Small Clojure Interpreter compiled to JS —
 * and provides a persistent SCI eval context. This is the browser analog of
 * the Lua (Fengari) kernel: JS-loadable, no build step, offline-capable when
 * the runtime is bundled locally (see kernel_config / build-profiles).
 *
 * It runs the output of UnifyWeaver's `clojurescript_target` (Prolog → CLJS),
 * closing the "prolog-generates-clojurescript" loop in-app.
 *
 * NOTE (hand-off): drafted by reading the SciREPL tree but not yet run inside
 * SciREPL. Two spots to validate against your Scittle version + kernel test
 * harness:
 *   1. The global name/shape: `window.scittle.core.eval_string`.
 *   2. stdout capture via binding `cljs.core/*print-fn*` (the standard CLJS
 *      REPL technique). If your Scittle build routes prints differently,
 *      adjust the wrapper in execute().
 */

class ClojureScriptKernel {
    constructor() {
        this._ready = false;
    }

    static displayName = 'ClojureScript';

    async init() {
        if (this._ready) return;

        // Load Scittle if not already present (bundle locally for offline use;
        // see build-profiles.json `clojurescript` sources).
        if (!window.scittle) {
            const km = window.kernelManager;
            if (km) {
                km.updateProgress('Downloading Scittle (ClojureScript) runtime…');
            }
            const primary = 'https://cdn.jsdelivr.net/npm/scittle@0.6.22/dist/scittle.js';
            if (km && km.loadKernelSource) {
                await km.loadKernelSource('clojurescript', primary, (url) => km._loadScript(url));
            } else {
                await new Promise((resolve, reject) => {
                    const script = document.createElement('script');
                    script.src = primary;
                    script.onload = resolve;
                    script.onerror = () => reject(new Error('Failed to load Scittle from CDN'));
                    document.head.appendChild(script);
                });
            }
        }

        if (window.kernelManager) {
            window.kernelManager.updateProgress('Initializing ClojureScript…');
        }

        if (!window.scittle || !window.scittle.core || !window.scittle.core.eval_string) {
            throw new Error('Scittle loaded but scittle.core.eval_string is unavailable');
        }

        this._ready = true;

        if (window.kernelManager) {
            window.kernelManager.hideDownloadModal();
        }
    }

    isReady() {
        return this._ready;
    }

    getName() {
        return 'ClojureScript (Scittle)';
    }

    getLanguage() {
        return 'clojurescript';
    }

    async execute(code) {
        if (!this._ready) {
            throw new Error('ClojureScript kernel not initialized');
        }

        const trimmed = (code || '').trim();
        if (!trimmed) {
            return { stdout: '', result: null, error: null };
        }

        // Capture println/print output while keeping the value of the last
        // form. SCI keeps a single global context across eval_string calls, so
        // defs / namespaces persist between cells (the proposal's open question
        // about REPL state — Scittle keeps one eval context per page).
        const wrapped =
            '(let [_out (atom "")]\n' +
            '  (let [_ret (binding [cljs.core/*print-fn* (fn [s] (swap! _out str s))]\n' +
            '               (do\n' + trimmed + '\n))]\n' +
            '    (cljs.core/array (deref _out) (pr-str _ret))))';

        try {
            const res = window.scittle.core.eval_string(wrapped);
            // res is a JS array: [stdout, printed-result-string]
            const stdout = (res && res[0] != null) ? String(res[0]) : '';
            const resultStr = (res && res[1] != null) ? String(res[1]) : 'nil';
            let result = null;
            if (resultStr !== 'nil' && resultStr !== '') {
                result = { type: 'text', content: resultStr };
            }
            return { stdout, result, error: null };
        } catch (e) {
            return { stdout: '', result: null, error: (e && e.message) ? e.message : String(e) };
        }
    }

    getMemoryUsage() {
        return 0; // SCI runs on the JS heap, no separate WASM allocation
    }

    destroy() {
        // Scittle's SCI context is a page global; nothing to free explicitly.
        this._ready = false;
    }
}

// Register with kernel manager
if (window.kernelManager) {
    window.kernelManager.register('clojurescript', ClojureScriptKernel);
}
