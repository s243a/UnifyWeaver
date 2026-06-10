{-# LANGUAGE BangPatterns #-}
-- | Runtime smoke for the Haskell-target dispatch fixes from PR #2356.
--
-- The Prolog test file test_wam_haskell_target.pl asserts the
-- generated Haskell *source* contains the expected case-expression
-- branches.  This smoke goes further: it cabal-builds the real
-- WamTypes + WamRuntime modules and drives `step` directly against
-- crafted instructions to confirm the runtime semantics match.
--
-- Two bug classes verified:
--   Bug A.1 (SwitchOnConstantPc/SwitchOnConstant fall-through on miss).
--     The WAM compiler drops `tom:default` labels, so a SwitchOnConstantPc
--     table that doesn't match A1 must fall through to the next
--     instruction (typically TryMeElse).  Previously returned Nothing.
--   Bug A.2 (Retry/Trust no-op-advance on empty wsCPs).
--     When indexed dispatch jumps directly to a clause beginning with
--     Retry/Trust, there's no CP to pop.  Empty wsCPs must be a no-op
--     PC advance, not failure.
--
-- Each case is a single `step` call.  Asserts:
--   * Successful path returns Just s' with the right wsPC (and wsCPs
--     correctly mutated where applicable).
--   * The previously-Nothing cases now also return Just s' with PC + 1.
--
-- Output protocol: RESULT <ok>/<total> line at EOF, parsed by the
-- Prolog driver.
module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.Array (listArray)
import WamTypes
import WamRuntime (step)

mkCtx :: InternTable -> Map.Map String Int -> WamContext
mkCtx tbl labels = WamContext
  { wcCode = listArray (0, -1) []
  , wcLabels = labels
  , wcForeignFacts = Map.empty
  , wcForeignConfig = Map.empty
  , wcLoweredPredicates = Map.empty
  , wcInternTable = tbl
  , wcFfiFacts = Map.empty
  , wcFfiWeightedFacts = Map.empty
  , wcInlineFacts = Map.empty
  , wcFactSources = Map.empty
  , wcEdgeLookups = Map.empty
  }

mkState :: [(Int, Value)] -> [ChoicePoint] -> WamState
mkState regs cps = WamState
  { wsPC = 10
  , wsRegs = IM.fromList regs
  , wsStack = []
  , wsHeap = []
  , wsHeapLen = 0
  , wsTrail = []
  , wsTrailLen = 0
  , wsCP = 0
  , wsCPs = cps
  , wsCPsLen = length cps
  , wsBindings = IM.empty
  , wsCutBar = 0
  , wsBuilder = NoBuilder
  , wsBuilderStack = []
  , wsVarCounter = 1000
  , wsAggAccum = []
  }

dummyCP :: ChoicePoint
dummyCP = ChoicePoint
  { cpNextPC   = 999
  , cpRegs     = IM.empty
  , cpStack    = []
  , cpCP       = 0
  , cpTrailLen = 0
  , cpHeapLen  = 0
  , cpBindings = IM.empty
  , cpCutBar   = 0
  , cpAggFrame = Nothing
  , cpBuiltin  = Nothing
  }

check :: String -> Bool -> IO Bool
check name ok = do
  putStrLn ((if ok then "PASS " else "FAIL ") ++ name)
  return ok

main :: IO ()
main = do
  let (fooId, t1) = internAtom emptyInternTable "foo"
      (barId, t2) = internAtom t1 "bar"
      (bazId, tbl) = internAtom t2 "baz"
      ctx = mkCtx tbl (Map.fromList [("L_retry", 200)])

  -- ====================================================================
  -- Bug A.2: empty wsCPs cases (Retry/Trust)
  -- ====================================================================
  -- All three should advance PC (was Nothing pre-fix).

  r1 <- check "TrustMe with empty wsCPs => advance PC (no-op)" $
    case step ctx (mkState [] []) TrustMe of
      Just s' -> wsPC s' == 11 && null (wsCPs s') && wsCPsLen s' == 0
      Nothing -> False

  r2 <- check "RetryMeElse 'L_retry' with empty wsCPs => advance PC" $
    case step ctx (mkState [] []) (RetryMeElse "L_retry") of
      Just s' -> wsPC s' == 11 && null (wsCPs s')
      Nothing -> False

  r3 <- check "RetryMeElsePc 200 with empty wsCPs => advance PC" $
    case step ctx (mkState [] []) (RetryMeElsePc 200) of
      Just s' -> wsPC s' == 11 && null (wsCPs s')
      Nothing -> False

  -- Sanity: same instructions WITH a CP behave classically.
  r4 <- check "TrustMe with one CP => pop CP and advance PC" $
    case step ctx (mkState [] [dummyCP]) TrustMe of
      Just s' -> wsPC s' == 11 && null (wsCPs s') && wsCPsLen s' == 0
      Nothing -> False

  r5 <- check "RetryMeElsePc 200 with one CP => mutate cpNextPC, advance PC" $
    case step ctx (mkState [] [dummyCP]) (RetryMeElsePc 200) of
      Just s' ->
        wsPC s' == 11 &&
        length (wsCPs s') == 1 &&
        cpNextPC (head (wsCPs s')) == 200
      Nothing -> False

  -- ====================================================================
  -- Bug A.1: SwitchOnConstantPc miss cases (fall through, not fail)
  -- ====================================================================
  -- Table maps foo->20, bar->30.  A1=baz, A1=Integer 99, A1=Unbound
  -- with no binding, A1=missing register, A1=Float — all should
  -- fall through (PC+1 = 11), not Nothing.

  let switchTbl = IM.fromList [(fooId, 20), (barId, 30)]

  r6 <- check "SwitchOnConstantPc Atom miss => fall through (PC+1)" $
    case step ctx (mkState [(1, Atom bazId)] []) (SwitchOnConstantPc switchTbl) of
      Just s' -> wsPC s' == 11
      Nothing -> False

  r7 <- check "SwitchOnConstantPc Atom hit => jump to target PC" $
    case step ctx (mkState [(1, Atom fooId)] []) (SwitchOnConstantPc switchTbl) of
      Just s' -> wsPC s' == 20
      Nothing -> False

  let intTbl = IM.fromList [(7, 40), (8, 50)]

  r8 <- check "SwitchOnConstantPc Integer miss => fall through (PC+1)" $
    case step ctx (mkState [(1, Integer 99)] []) (SwitchOnConstantPc intTbl) of
      Just s' -> wsPC s' == 11
      Nothing -> False

  r9 <- check "SwitchOnConstantPc Integer hit => jump to target PC" $
    case step ctx (mkState [(1, Integer 7)] []) (SwitchOnConstantPc intTbl) of
      Just s' -> wsPC s' == 40
      Nothing -> False

  r10 <- check "SwitchOnConstantPc Float in A1 => fall through (PC+1)" $
    case step ctx (mkState [(1, Float 3.14)] []) (SwitchOnConstantPc switchTbl) of
      Just s' -> wsPC s' == 11
      Nothing -> False

  -- ====================================================================
  -- Bug A.1 mirror: SwitchOnConstant (the Map-keyed Value variant)
  -- ====================================================================
  let valTbl = Map.fromList
        [ (Atom fooId, "L_retry")  -- maps to 200 via wcLabels
        ]

  r11 <- check "SwitchOnConstant Value miss => fall through (PC+1)" $
    case step ctx (mkState [(1, Atom bazId)] []) (SwitchOnConstant valTbl) of
      Just s' -> wsPC s' == 11
      Nothing -> False

  r12 <- check "SwitchOnConstant Value hit => jump to mapped PC" $
    case step ctx (mkState [(1, Atom fooId)] []) (SwitchOnConstant valTbl) of
      Just s' -> wsPC s' == 200
      Nothing -> False

  let oks = length (filter id [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12])
  putStrLn ("RESULT " ++ show oks ++ "/12")
