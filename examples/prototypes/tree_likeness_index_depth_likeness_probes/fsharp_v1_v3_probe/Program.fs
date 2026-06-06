// V1 depth-likeness probe — F# port of test_depth_likeness.py
//
// For each seed node from seeds.tsv:
//   - Compute d_wPow(seed, root) using the bidirectional kernel with B = depth(seed).
//   - Compare to depth + 1 (the carrot baseline).
//
// Outputs: TSV (seed_id, depth, n_paths, d_wPow, diff)
//
// Usage:
//   dotnet run -c Release -- <lmdb-dir> <root-id> <seeds-tsv> <output-tsv>

open System
open System.IO
open System.Collections.Generic
open LightningDB
open Kernel

// ============================================================
// LMDB lookup via LightningDB cursors
// ============================================================

let openEnv (path: string) =
    let cfg = EnvironmentConfiguration(MaxDatabases = 8)
    cfg.MapSize <- 12L * 1024L * 1024L * 1024L
    let env = new LightningEnvironment(path, cfg)
    env.Open(EnvironmentOpenFlags.ReadOnly ||| EnvironmentOpenFlags.NoLock)
    env

let encodeI32 (v: int) : byte[] = BitConverter.GetBytes(v)
let decodeI32 (b: byte[]) : int = BitConverter.ToInt32(b, 0)

type LmdbLookup(env: LightningEnvironment, dbName: string) =
    let db =
        use tx = env.BeginTransaction(TransactionBeginFlags.ReadOnly)
        let d = tx.OpenDatabase(dbName, DatabaseConfiguration(Flags = DatabaseOpenFlags.DuplicatesSort))
        tx.Commit() |> ignore
        d
    member _.Lookup (key: int) : int list =
        use tx = env.BeginTransaction(TransactionBeginFlags.ReadOnly)
        use cursor = tx.CreateCursor(db)
        let struct (rc, _, _) = cursor.SetKey(encodeI32 key)
        if rc <> MDBResultCode.Success then []
        else
            let mutable result = []
            let struct (_, _, v) = cursor.GetCurrent()
            let arr = v.CopyToNewArray()
            result <- [decodeI32 arr]
            let mutable ok = true
            while ok do
                let struct (rc2, _, v2) = cursor.NextDuplicate()
                if rc2 = MDBResultCode.Success then
                    let arr2 = v2.CopyToNewArray()
                    result <- decodeI32 arr2 :: result
                else
                    ok <- false
            result

// ============================================================
// d_wPow from paths
// ============================================================

let computeDwPow (paths: (int * int * int) list) (D: float) (bD: float) (nExp: float) : float option =
    if List.isEmpty paths then None
    else
        let mutable sumNum = 0.0
        let mutable sumW = 0.0
        for (h, n, m) in paths do
            let w = (1.0 / D) ** float n * (1.0 / bD) ** float m
            sumNum <- sumNum + w * (float (h + 1)) ** -nExp
            sumW <- sumW + w
        if sumW <= 0.0 then None
        else
            let ratio = sumNum / sumW
            if ratio <= 0.0 then None
            else Some (ratio ** (-1.0 / nExp))

// ============================================================
// Main
// ============================================================

[<EntryPoint>]
let main argv =
    if argv.Length < 4 then
        eprintfn "usage: program <lmdb-dir> <root-id> <seeds-tsv> <output-tsv> [--variant v1|v3]"
        eprintfn "  v1 (default): B = depth(v)        — only child shortcuts admissible"
        eprintfn "  v3:           B = max_parent_dist — also admits parent-direction shortcuts"
        1
    else
        let lmdbDir = argv.[0]
        let rootId = int argv.[1]
        let seedsFile = argv.[2]
        let outputFile = argv.[3]
        let variant =
            let rec find i =
                if i >= argv.Length then "v1"
                elif argv.[i] = "--variant" && i + 1 < argv.Length then argv.[i + 1]
                else find (i + 1)
            find 4
        if variant <> "v1" && variant <> "v3" then
            eprintfn "unknown variant: %s (use v1 or v3)" variant
            exit 1

        printfn "=== V1 depth-likeness F# probe ==="
        printfn "LMDB: %s" lmdbDir
        printfn "Root: %d" rootId
        printfn "Seeds: %s" seedsFile
        printfn "Output: %s" outputFile

        let env = openEnv lmdbDir
        let lookupParents = LmdbLookup(env, "category_parent")
        let lookupChildren = LmdbLookup(env, "category_child")
        let lookupP nd = lookupParents.Lookup nd
        let lookupC nd = lookupChildren.Lookup nd

        printfn ""
        printfn "Calibrating graph..."
        let t0 = DateTime.Now
        let dist = Dictionary<int, int>()
        dist.[rootId] <- 0
        let mutable frontier = [rootId]
        let mutable depth = 0
        let mutable sumChildDeg = 0.0
        let mutable sumChildDeg2 = 0.0
        let mutable childNodes = 0
        let mutable sumParentDeg = 0.0
        let mutable sumParentDeg2 = 0.0
        let mutable parentNodes = 0
        while not (List.isEmpty frontier) do
            depth <- depth + 1
            let mutable next = []
            for nd in frontier do
                let children = lookupC nd
                if not (List.isEmpty children) then
                    let d = float children.Length
                    sumChildDeg <- sumChildDeg + d
                    sumChildDeg2 <- sumChildDeg2 + d * d
                    childNodes <- childNodes + 1
                let parents = lookupP nd
                if not (List.isEmpty parents) then
                    let d = float parents.Length
                    sumParentDeg <- sumParentDeg + d
                    sumParentDeg2 <- sumParentDeg2 + d * d
                    parentNodes <- parentNodes + 1
                for c in children do
                    if not (dist.ContainsKey(c)) then
                        dist.[c] <- depth
                        next <- c :: next
            frontier <- next

        let D = if childNodes > 0 then max 1.5 (sumChildDeg / float childNodes) else 3.0
        let edChild = if childNodes > 0 then sumChildDeg / float childNodes else 1.0
        let edParent = if parentNodes > 0 then sumParentDeg / float parentNodes else 1.0
        let ed2Child = if childNodes > 0 then sumChildDeg2 / float childNodes else 1.0
        let ed2Parent = if parentNodes > 0 then sumParentDeg2 / float parentNodes else 1.0
        // friendship-paradox-corrected b_eff
        let bEff =
            if edChild > 0.0 && edParent > 0.0 && ed2Parent > 0.0 then
                max 1.0 ((ed2Child / edChild) / (ed2Parent / edParent))
            else 1.0
        let nExp = 2.0
        let tcal = (DateTime.Now - t0).TotalSeconds
        printfn "  Reachable=%d  D=%.3f  b_eff=%.3f  b_eff*D=%.2f  (%.1fs)" dist.Count D bEff (bEff * D) tcal

        // For V3: compute max acyclic parent distance to root via DP
        // in children-BFS depth order (all parents of n have smaller min-depth)
        let maxDist = Dictionary<int, int>()
        if variant = "v3" then
            printfn "Computing max-parent-distance DP..."
            let t1 = DateTime.Now
            maxDist.[rootId] <- 0
            // Group nodes by their min-depth so we can iterate in order
            let nodesAtDepth = Dictionary<int, List<int>>()
            for kv in dist do
                let d = kv.Value
                if not (nodesAtDepth.ContainsKey(d)) then
                    nodesAtDepth.[d] <- List<int>()
                nodesAtDepth.[d].Add(kv.Key)
            let maxBfsDepth =
                let mutable m = 0
                for kv in nodesAtDepth do
                    if kv.Key > m then m <- kv.Key
                m
            for d in 1 .. maxBfsDepth do
                if nodesAtDepth.ContainsKey(d) then
                    for n in nodesAtDepth.[d] do
                        // Find parents of n that already have max_dist set
                        let parents = lookupP n
                        let mutable best = -1
                        for p in parents do
                            match maxDist.TryGetValue(p) with
                            | true, pd ->
                                if pd > best then best <- pd
                            | _ -> ()
                        if best >= 0 then
                            maxDist.[n] <- 1 + best
            let tdp = (DateTime.Now - t1).TotalSeconds
            printfn "  max-dist computed for %d nodes (%.1fs)" maxDist.Count tdp

        // Read seeds
        let seedLines =
            File.ReadAllLines(seedsFile)
            |> Array.filter (fun l -> not (l.StartsWith("#") || l.StartsWith("seed_id") || String.IsNullOrWhiteSpace(l)))
            |> Array.map (fun l ->
                let parts = l.Split('\t')
                if parts.Length >= 2 then (int parts.[0], int parts.[1])
                else (int parts.[0], -1))
        printfn "Read %d seeds from %s" seedLines.Length seedsFile

        use out = new StreamWriter(outputFile)
        if variant = "v3" then
            out.WriteLine("seed_id\tmin_dist\tmax_dist\tbudget\tn_paths\td_wPow\tdiff_vs_min\tdiff_vs_max")
        else
            out.WriteLine("seed_id\tdepth\tbudget\tn_paths\td_wPow\tdiff")
        printfn ""
        if variant = "v3" then
            printfn "  %5s %10s %6s %6s %4s %6s %10s %8s %8s" "min" "seed" "max" "B" "p" "" "d_wPow" "-min" "-max"
        else
            printfn "  %6s %10s %6s %8s %10s %8s  %s" "depth" "seed" "B" "paths" "d_wPow" "diff" "elapsed"

        let pc = 1.0
        let cc = 5.0
        let bD = bEff * D
        // Bound path enumeration so V3 deep nodes don't explode
        Kernel.maxPaths <- 100000

        let mutable nDone = 0
        for (seed, _) in seedLines do
            let d =
                match dist.TryGetValue(seed) with
                | true, v -> v
                | _ -> -1
            if d < 0 then
                if variant = "v3" then
                    out.WriteLine(sprintf "%d\t-1\t-1\t-1\t0\t\t\t" seed)
                else
                    out.WriteLine(sprintf "%d\t-1\t-1\t0\t\t" seed)
            else
                // Choose budget
                let B =
                    if variant = "v3" then
                        match maxDist.TryGetValue(seed) with
                        | true, m -> float m
                        | _ -> float d
                    else
                        float d
                let mxD =
                    match maxDist.TryGetValue(seed) with
                    | true, m -> m
                    | _ -> d
                let ts = DateTime.Now
                let paths = nativeKernel_bidirectional_ancestor_withMinDist lookupP lookupC dist seed rootId pc cc B
                let dt = (DateTime.Now - ts).TotalSeconds
                let dwpOpt = computeDwPow paths D bD nExp
                match dwpOpt with
                | Some dwp ->
                    if variant = "v3" then
                        let diffMin = dwp - float d
                        let diffMax = dwp - float mxD
                        out.WriteLine(sprintf "%d\t%d\t%d\t%g\t%d\t%g\t%g\t%g" seed d mxD B (List.length paths) dwp diffMin diffMax)
                        printfn "  %5d %10d %6d %6g %4d        %10.4f %+8.3f %+8.3f  (%.2fs)" d seed mxD B (List.length paths) dwp diffMin diffMax dt
                    else
                        let diff = dwp - float d
                        out.WriteLine(sprintf "%d\t%d\t%g\t%d\t%g\t%g" seed d B (List.length paths) dwp diff)
                        printfn "  %6d %10d %6g %8d %10.4f %+8.3f  (%.2fs)" d seed B (List.length paths) dwp diff dt
                | None ->
                    if variant = "v3" then
                        out.WriteLine(sprintf "%d\t%d\t%d\t%g\t%d\t\t\t" seed d mxD B (List.length paths))
                    else
                        out.WriteLine(sprintf "%d\t%d\t%g\t%d\t\t" seed d B (List.length paths))
            nDone <- nDone + 1

        out.Flush()
        out.Close()
        env.Dispose()
        printfn ""
        printfn "Processed %d seeds. Output: %s" nDone outputFile
        0
