package main

import (
	"encoding/json"
	"fmt"
	"os"

	bolt "go.etcd.io/bbolt"
)

func main() {
	// Open database (read-only)
	db, err := bolt.Open("test_users.db", 0600, &bolt.Options{ReadOnly: true})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	// Read all records from bucket
	err = db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte("users"))
		if bucket == nil {
			return fmt.Errorf("bucket 'users' not found")
		}

		return bucket.ForEach(func(k, v []byte) error {
			// Deserialize JSON record
			var data map[string]interface{}
			if err := json.Unmarshal(v, &data); err != nil {
				fmt.Fprintf(os.Stderr, "Error unmarshaling record: %v\n", err)
				return nil // Continue with next record
			}

			// Output as JSON
			output, err := json.Marshal(data)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error marshaling output: %v\n", err)
				return nil // Continue with next record
			}

			fmt.Println(string(output))
			return nil
		})
	})

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading database: %v\n", err)
		os.Exit(1)
	}
}
