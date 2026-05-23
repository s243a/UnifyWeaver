module Program

// NAF-focused micro-benchmark for runNegationParallel.
//
// Two scenarios — each measures wall time for 3 runs of
// `runNegationParallel` against a hand-constructed 5-branch Par* chain.
//
// Scenario A — "fast-succeed + 4 slow-fail":
//   Branch 1 is a trivial Proceed (succeeds immediately).  Branches
//   2-5 have busy decrement loops, then Fail.  Async.Choice from
//   PR #2353 returns on first Some, so wall time is bounded by the
//   FAST branch's time + thread setup overhead.  This is the case
//   where soft-cancel already delivers the wall-time win.
//
// Scenario B — "all 5 slow-fail":
//   No branch succeeds.  Async.Choice has nothing to short-circuit on
//   — it must wait for the slowest branch to return None.  Wall time
//   = slowest branch's natural runtime.  Hard-cancel (future work)
//   wouldn't help wall time here either, since we need all branches'
//   None to conclude.
//
// Output contract:
//   - Asserts correctness: scenario A returns true, scenario B
//     returns false.
//   - Asserts wall time < 30s (sanity bound).
//   - Prints `wall_ms_A=N wall_ms_B=N` lines for future comparison
//     (e.g. when hard-cancel lands — though for these scenarios the
//     wall-time delta should be ≈ 0).

open WamTypes
open WamRuntime

let mkSlowFailBody (iterations: int) : Instruction list =
    // Synthetic busy work: each iteration emits 2 instructions
    // (PutValue + PutConstant on register 102, which doesn't
    // affect the eventual Fail).  Tune `iterations` to control
    // per-branch wall time.
    let body =
        [ for _ in 1 .. iterations ->
            [ PutValue (101, 102)
              PutConstant (Integer 0, 102) ]
        ]
        |> List.concat
    body @ [ Fail ]

// Build a 5-branch Par* chain.  `branch1Body` is the first branch
// (either Proceed for fast-succeed or slow-fail).
let mkContext (branch1Body: Instruction list) (slowSize: int) =
    let slowFail () = mkSlowFailBody slowSize
    let branch1 = branch1Body
    let branch2 = slowFail ()
    let branch3 = slowFail ()
    let branch4 = slowFail ()
    let branch5 = slowFail ()
    let len1 = List.length branch1
    let len2 = List.length branch2
    let len3 = List.length branch3
    let len4 = List.length branch4
    let pcParTry        = 1
    let pcBranch1Entry  = 2
    let pcParRetry2     = pcBranch1Entry + len1
    let pcBranch2Entry  = pcParRetry2 + 1
    let pcParRetry3     = pcBranch2Entry + len2
    let pcBranch3Entry  = pcParRetry3 + 1
    let pcParRetry4     = pcBranch3Entry + len3
    let pcBranch4Entry  = pcParRetry4 + 1
    let pcParTrust      = pcBranch4Entry + len4
    let instructions =
        [ Proceed ]                       // PC 0 — halt sentinel
      @ [ ParTryMeElsePc pcParRetry2 ]    // PC 1 — entry
      @ branch1                            // PC 2..
      @ [ ParRetryMeElsePc pcParRetry3 ]
      @ branch2
      @ [ ParRetryMeElsePc pcParRetry4 ]
      @ branch3
      @ [ ParRetryMeElsePc pcParTrust ]
      @ branch4
      @ [ ParTrustMe ]
      @ branch5
    let code = List.toArray instructions
    let ctx =
      { WcCode              = code
        WcLabels            = Map.empty
        WcForeignFacts      = Map.empty
        WcFfiFacts          = Map.empty
        WcFfiWeightedFacts  = Map.empty
        WcAtomIntern        = Map.empty
        WcAtomDeintern      = Map.empty
        WcForeignConfig     = Map.empty
        WcLoweredPredicates = Map.empty
        WcCancellationToken = None }
    ctx, pcParTry, pcParRetry2

let mkState () =
    let r = Array.create MaxRegs (Unbound -1)
    r.[101] <- Integer 0
    r.[102] <- Integer 0
    { WsPC         = 0
      WsRegs       = r
      WsStack      = []
      WsHeap       = []
      WsHeapLen    = 0
      WsTrail      = []
      WsTrailLen   = 0
      WsCP         = 0
      WsCPs        = []
      WsCPsLen     = 0
      WsBindings   = Map.empty
      WsCutBar     = 0
      WsVarCounter = 1000
      WsBuilder    = None
      WsBuilderStack = []
      WsAggAccum   = []
      WsB0Stack    = [] }

let runScenario (name: string) (branch1: Instruction list) (expectTrue: bool) (slowSize: int) : int64 * int64 * bool =
    let ctx, parPC, elsePC = mkContext branch1 slowSize
    let s0 = mkState ()
    // Warm-up to amortize JIT.
    let _ = runNegationParallel ctx s0 parPC elsePC
    // Timed runs.  Capture both wall time and CPU time across all
    // threads.  After hard-cancel landing, scenario A (fast-succeed
    // + slow-fail siblings) should show CPU time drop sharply —
    // siblings get cancelled instead of running to completion in the
    // background.  Scenario B (all-fail) should be roughly unchanged
    // since every branch needs to fail before negation succeeds.
    let runs = 3
    let proc = System.Diagnostics.Process.GetCurrentProcess()
    let cpuBefore = proc.TotalProcessorTime
    let sw = System.Diagnostics.Stopwatch.StartNew()
    let mutable allMatch = true
    for _ in 1 .. runs do
        let result = runNegationParallel ctx s0 parPC elsePC
        if result <> expectTrue then allMatch <- false
    sw.Stop()
    let cpuAfter = proc.TotalProcessorTime
    let wallMs = sw.ElapsedMilliseconds
    let cpuMs = int64 (cpuAfter - cpuBefore).TotalMilliseconds
    if allMatch then
        printfn "[PASS] %s: runNegationParallel returns %b" name expectTrue
    else
        printfn "[FAIL] %s: runNegationParallel returned wrong value (expected %b)" name expectTrue
    wallMs, cpuMs, allMatch

[<EntryPoint>]
let main _argv =
    // Slow loop size chosen for ~5-50ms per slow branch on typical
    // hardware.  Smaller = noisier numbers; larger = longer CI runs.
    let slowSize = 5000
    // Scenario A: 1 fast-succeed + 4 slow-fail.
    let wallA, cpuA, okA =
        runScenario "scenario A (fast-succeed + 4 slow-fail)"
                    [ Proceed ] true slowSize
    // Scenario B: all 5 slow-fail.
    let wallB, cpuB, okB =
        runScenario "scenario B (all 5 slow-fail)"
                    (mkSlowFailBody slowSize) false slowSize
    // Sanity timing.
    let sanityBound = 30000L
    let timingOK = wallA < sanityBound && wallB < sanityBound
    if timingOK then
        printfn "[PASS] both scenarios complete under %dms sanity bound" sanityBound
    else
        printfn "[FAIL] scenario(s) exceeded sanity bound (A=%dms B=%dms)" wallA wallB
    printfn "wall_ms_A=%d wall_ms_B=%d cpu_ms_A=%d cpu_ms_B=%d slow_size=%d runs=3"
            wallA wallB cpuA cpuB slowSize
    let passes =
        (if okA then 1 else 0)
      + (if okB then 1 else 0)
      + (if timingOK then 1 else 0)
    printfn "RESULT %d/3" passes
    if passes = 3 then 0 else 1
