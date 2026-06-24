/**
 * Test: Prolog Generates ClojureScript workbook
 *
 * Simulates the workbook flow:
 * 1. Load UnifyWeaver compiler modules (Prolog)
 * 2. Define family tree facts (Prolog)
 * 3. Compile ancestor/2 to ClojureScript (Prolog)
 * 4. Execute generated ClojureScript code (Scittle kernel)
 * 5. Run queries via find-all / check-path
 *
 * Mirrors test_prolog_generates_typr.mjs. Run from the SciREPL repo root with a
 * static server on :8085 and the clojurescript kernel installed:
 *   node test_prolog_generates_clojurescript.mjs
 */
import { chromium } from 'playwright';

const BASE = 'http://localhost:8085';

let _counter = 0;
async function domExec(page, code, { timeout = 60000 } = {}) {
    const id = `_de_${++_counter}`;
    const rAttr = `data-${id}-r`;
    const eAttr = `data-${id}-e`;
    await page.addScriptTag({ content: `
        (async () => {
            try {
                const __r = await (async () => { ${code} })();
                document.body.setAttribute('${rAttr}',
                    __r === undefined ? '__undef__' : JSON.stringify(__r));
            } catch(e) {
                document.body.setAttribute('${eAttr}', e.message || String(e));
            }
        })();
    `});
    await page.waitForFunction(
        ([r, e]) => document.body.hasAttribute(r) || document.body.hasAttribute(e),
        [rAttr, eAttr], { timeout, polling: 1000 }
    );
    const err = await page.getAttribute('body', eAttr);
    if (err) throw new Error(err);
    const raw = await page.getAttribute('body', rAttr);
    if (raw === '__undef__') return undefined;
    return JSON.parse(raw);
}

async function ensureKernel(page, name, timeout = 300000) {
    const attr = `data-k-${name}`;
    const eAttr = `data-k-${name}-e`;
    await page.addScriptTag({ content: `
        window.kernelManager.ensureReady('${name}')
            .then(() => document.body.setAttribute('${attr}', '1'))
            .catch(e => document.body.setAttribute('${eAttr}', e.message));
    `});
    await page.waitForFunction(
        ([a, e]) => document.body.hasAttribute(a) || document.body.hasAttribute(e),
        [attr, eAttr], { timeout, polling: 2000 }
    );
    const err = await page.getAttribute('body', eAttr);
    if (err) throw new Error(`${name} kernel failed: ${err}`);
}

async function kernelExec(page, kernel, code, { timeout = 60000 } = {}) {
    const escaped = JSON.stringify(code);
    return await domExec(page, `
        const r = await window.kernelManager.execute(${escaped}, '${kernel}');
        return { stdout: r.stdout || '', error: r.error || '' };
    `, { timeout });
}

let _failures = 0;
function check(label, cond) {
    if (cond) { console.log(`   ok  - ${label}`); }
    else { console.log(`   FAIL - ${label}`); _failures++; }
}

(async () => {
    const browser = await chromium.launch({
        headless: true,
        args: ['--disable-dev-shm-usage', '--disable-gpu', '--no-sandbox']
    });
    const page = await browser.newPage();
    page.on('console', msg => {
        const t = msg.text();
        if (t.includes('rror') || t.includes('ClojureScript') || t.includes('Prolog') || t.includes('compile'))
            console.log('  [page]', t.substring(0, 200));
    });

    console.log('1. Loading SciREPL...');
    await page.addInitScript(() => {
        localStorage.setItem('scirepl_privacy_accepted', '1');
        localStorage.setItem('scirepl_auto_download', '1');
        localStorage.removeItem('scirepl_enabled_languages');
    });
    await page.goto(BASE, { waitUntil: 'networkidle', timeout: 30000 });
    await page.evaluate(async () => {
        const regs = await navigator.serviceWorker.getRegistrations();
        for (const r of regs) await r.unregister();
    });
    await page.reload({ waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForSelector('#run-btn:not([disabled])', { timeout: 10000 });

    // Step 1: Load Prolog kernel
    console.log('\n2. Loading Prolog kernel...');
    await ensureKernel(page, 'prolog', 300000);
    console.log('   Prolog ready');

    // Step 2: Load compiler modules
    console.log('\n3. Loading UnifyWeaver compiler...');
    const loadResult = await kernelExec(page, 'prolog',
        "[\'../init\'].\n:- use_module(unifyweaver(targets/clojurescript_target)).\n:- use_module(unifyweaver(core/recursive_compiler)).",
        { timeout: 60000 });
    if (loadResult.error) console.log('   ERROR:', loadResult.error.substring(0, 200));

    // Step 3: Define family tree
    console.log('\n4. Defining family tree...');
    const factsResult = await kernelExec(page, 'prolog',
        ":- dynamic parent/2, ancestor/2.\nparent(alice, bob).\nparent(bob, charlie).\nparent(bob, diana).\nparent(charlie, eve).\nparent(diana, frank).\nancestor(X, Y) :- parent(X, Y).\nancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).",
        { timeout: 30000 });
    if (factsResult.error) console.log('   ERROR:', factsResult.error.substring(0, 200));

    // Step 4: Compile to ClojureScript
    console.log('\n5. Compiling ancestor/2 to ClojureScript...');
    const compileResult = await kernelExec(page, 'prolog',
        "recursive_compiler:compile_recursive(ancestor/2, [target(clojurescript)], Code), write(Code).",
        { timeout: 60000 });
    console.log('   Compile stdout length:', compileResult.stdout.length);
    if (compileResult.error) console.log('   COMPILE ERROR:', compileResult.error.substring(0, 300));
    check('generated CLJS is non-trivial', compileResult.stdout.length > 100);
    check('CLJS banner present', compileResult.stdout.includes('Target: ClojureScript'));
    check('find-all defined', compileResult.stdout.includes('find-all'));
    check('no JVM interop leaked', !compileResult.stdout.includes('java.io') &&
          !compileResult.stdout.includes('System/exit'));

    // Step 5: Execute the generated ClojureScript code
    if (compileResult.stdout.length > 100) {
        console.log('\n6. Loading ClojureScript kernel...');
        await ensureKernel(page, 'clojurescript', 300000);
        console.log('   ClojureScript ready');

        console.log('\n7. Executing generated ClojureScript code...');
        const cljsResult = await kernelExec(page, 'clojurescript', compileResult.stdout, { timeout: 60000 });
        if (cljsResult.error) console.log('   CLJS ERROR:', cljsResult.error.substring(0, 300));

        // Step 6: Query (relies on Scittle keeping state across executions)
        console.log('\n8. Running queries...');
        const queryResult = await kernelExec(page, 'clojurescript',
            '(println "desc:" (sort (find-all "alice")))\n(println "alice->eve:" (check-path "alice" "eve"))\n(println "alice->zzz:" (check-path "alice" "zzz"))',
            { timeout: 30000 });
        console.log('   Query stdout:', queryResult.stdout.trim());
        if (queryResult.error) console.log('   Query ERROR:', queryResult.error.substring(0, 200));
        check('descendants include eve', queryResult.stdout.includes('eve'));
        check('descendants include frank', queryResult.stdout.includes('frank'));
        check('alice is ancestor of eve', /alice->eve:\s*true/.test(queryResult.stdout));
        check('alice is NOT ancestor of zzz', /alice->zzz:\s*false/.test(queryResult.stdout));
    } else {
        console.log('\n   Skipping CLJS execution — no code generated');
        _failures++;
    }

    console.log('\n' + '='.repeat(50));
    console.log(_failures === 0 ? 'PASS' : `FAIL (${_failures} check(s) failed)`);
    console.log('='.repeat(50));

    await browser.close();
    process.exit(_failures === 0 ? 0 : 1);
})().catch(err => {
    console.error('FATAL:', err.message);
    process.exit(1);
});
