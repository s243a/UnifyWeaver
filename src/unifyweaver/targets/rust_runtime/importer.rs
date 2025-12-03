use redb::{Database, Error, ReadableTable, TableDefinition, WriteTransaction};
use serde_json::Value;
use std::sync::Arc;

// Table Definitions
// Objects: ID -> JSON String
const OBJECTS: TableDefinition<&str, &str> = TableDefinition::new("objects");
// Embeddings: ID -> Byte Array (f32 vector)
const EMBEDDINGS: TableDefinition<&str, &[u8]> = TableDefinition::new("embeddings");
// Links: Source -> Target (Note: Redb doesn't support multimap natively easily in 1 table without composite keys)
// For 1-to-N links, we'll use a composite key: "source|target" -> ""
const LINKS: TableDefinition<&str, &str> = TableDefinition::new("links");

pub struct PtImporter {
    db: Arc<Database>,
}

impl PtImporter {
    pub fn new(path: &str) -> Result<Self, Error> {
        let db = Database::create(path)?;
        let db_arc = Arc::new(db);
        
        // Initialize tables
        let write_txn = db_arc.begin_write()?;
        {
            write_txn.open_table(OBJECTS)?;
            write_txn.open_table(EMBEDDINGS)?;
            write_txn.open_table(LINKS)?;
        }
        write_txn.commit()?;

        Ok(Self { db: db_arc })
    }

    pub fn upsert_object(&self, id: &str, _obj_type: &str, data: &Value) -> Result<(), Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(OBJECTS)?;
            table.insert(id, data.to_string().as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn upsert_embedding(&self, id: &str, vector: &[f32]) -> Result<(), Error> {
        // Convert f32 slice to u8 slice
        let byte_len = vector.len() * 4;
        let byte_ptr = vector.as_ptr() as *const u8;
        let byte_slice = unsafe { std::slice::from_raw_parts(byte_ptr, byte_len) };

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(EMBEDDINGS)?;
            table.insert(id, byte_slice)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn upsert_link(&self, source: &str, target: &str) -> Result<(), Error> {
        let key = format!("{}|{}", source, target);
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(LINKS)?;
            table.insert(key.as_str(), "")?;
        }
        write_txn.commit()?;
        Ok(())
    }
}
