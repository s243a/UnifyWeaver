package storage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"time"

	bolt "go.etcd.io/bbolt"
)

type Store struct {
	db *bolt.DB
}

func NewStore(path string) (*Store, error) {
	db, err := bolt.Open(path, 0600, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, err
	}

	// Init buckets
	err = db.Update(func(tx *bolt.Tx) error {
		_, _ = tx.CreateBucketIfNotExists([]byte("objects"))
		_, _ = tx.CreateBucketIfNotExists([]byte("embeddings"))
		return nil
	})

	if err != nil {
		db.Close()
		return nil, err
	}

	return &Store{db: db}, nil
}

func (s *Store) Close() error {
	return s.db.Close()
}

func (s *Store) UpsertObject(id string, data map[string]interface{}) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("objects"))
		val, err := json.Marshal(data)
		if err != nil {
			return err
		}
		return b.Put([]byte(id), val)
	})
}

func (s *Store) UpsertEmbedding(id string, vector []float32) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("embeddings"))
		
		// Serialize []float32 to []byte
		buf := make([]byte, len(vector)*4)
		for i, v := range vector {
			bits := math.Float32bits(v)
			binary.LittleEndian.PutUint32(buf[i*4:], bits)
		}
		
		return b.Put([]byte(id), buf)
	})
}

func (s *Store) GetEmbedding(id string) ([]float32, error) {
	var vector []float32
	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("embeddings"))
		v := b.Get([]byte(id))
		if v == nil {
			return fmt.Errorf("not found")
		}
		
		count := len(v) / 4
		vector = make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(v[i*4:])
			vector[i] = math.Float32frombits(bits)
		}
		return nil
	})
	return vector, err
}

// IterateEmbeddings iterates over all embeddings
func (s *Store) IterateEmbeddings(fn func(id string, vector []float32) error) error {
	return s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("embeddings"))
		return b.ForEach(func(k, v []byte) error {
			count := len(v) / 4
			vector := make([]float32, count)
			for i := 0; i < count; i++ {
				bits := binary.LittleEndian.Uint32(v[i*4:])
				vector[i] = math.Float32frombits(bits)
			}
			return fn(string(k), vector)
		})
	})
}

// GetObject retrieves an object by ID
func (s *Store) GetObject(id string) (map[string]interface{}, error) {
	var data map[string]interface{}
	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("objects"))
		v := b.Get([]byte(id))
		if v == nil {
			return fmt.Errorf("not found: %s", id)
		}
		return json.Unmarshal(v, &data)
	})
	return data, err
}

// CountObjects returns the number of objects in the store
func (s *Store) CountObjects() (int, error) {
	var count int
	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("objects"))
		count = b.Stats().KeyN
		return nil
	})
	return count, err
}

// CountEmbeddings returns the number of embeddings in the store
func (s *Store) CountEmbeddings() (int, error) {
	var count int
	err := s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte("embeddings"))
		count = b.Stats().KeyN
		return nil
	})
	return count, err
}
