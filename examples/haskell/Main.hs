-- Haskell Integration Test
-- Tests calling UnifyWeaver-generated Prolog predicates

import PrologMath

main :: IO ()
main = do
    putStrLn "==========================================="
    putStrLn "Haskell Integration Test"
    putStrLn "==========================================="
    putStrLn ""
    
    let tests = 
            [ ("sumTo 10 0", sumTo 10 0, 55)
            , ("sumTo 100 0", sumTo 100 0, 5050)
            , ("factorial 5", factorial 5, 120)
            , ("factorial 10", factorial 10, 3628800)
            , ("fib 10", fib 10, 55)
            , ("fib 15", fib 15, 610)
            ]
    
    results <- mapM runTest tests
    let (passed, failed) = foldr countResults (0, 0) results
    
    putStrLn ""
    putStrLn $ "Results: " ++ show passed ++ " passed, " ++ show failed ++ " failed"
    putStrLn "==========================================="
    
    if failed > 0
        then putStrLn "INTEGRATION TEST FAILED"
        else putStrLn "INTEGRATION TEST PASSED"

runTest :: (String, Int, Int) -> IO Bool
runTest (name, result, expected) = do
    if result == expected
        then do
            putStrLn $ "[PASS] " ++ name ++ " = " ++ show result
            return True
        else do
            putStrLn $ "[FAIL] " ++ name ++ " = " ++ show result ++ " (expected " ++ show expected ++ ")"
            return False

countResults :: Bool -> (Int, Int) -> (Int, Int)
countResults True (p, f) = (p + 1, f)
countResults False (p, f) = (p, f + 1)
