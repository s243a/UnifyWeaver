# Add Database Persistence to Go Generator Mode

## Summary

Implements `db_backend(bbolt)` option for saving/loading fixpoint state, enabling incremental Datalog computation.

## Usage

```prolog
compile_predicate_to_go(ancestor/2, [mode(generator), db_backend(bbolt)], Code)

% With custom file/bucket:
compile_predicate_to_go(ancestor/2, [mode(generator), db_backend(bbolt), 
                                      db_file('my.db'), db_bucket(results)], Code)
```

## How It Works

1. **On startup:** Load existing facts from bbolt bucket
2. **Compute fixpoint:** Including loaded facts
3. **After fixpoint:** Save all facts back to bbolt

## Incremental Workflow

```bash
# First run - computes and saves
./ancestor

# Add more parent facts via json_input, continue from saved state
echo '{"relation":"parent","args":{"arg0":"alice","arg1":"bob"}}' | ./ancestor
```

## Generated Code

```go
// Load facts from database
db, err := bolt.Open("facts.db", 0600, nil)
if err == nil {
    defer db.Close()
    db.View(func(tx *bolt.Tx) error {
        b := tx.Bucket([]byte("facts"))
        // ... load facts
    })
}

// ... fixpoint iteration ...

// Save all facts to database  
db2.Update(func(tx *bolt.Tx) error {
    b, _ := tx.CreateBucketIfNotExists([]byte("facts"))
    for key, fact := range total {
        v, _ := json.Marshal(fact)
        b.Put([]byte(key), v)
    }
    return nil
})
```

## Verification

- All 6 tests pass
