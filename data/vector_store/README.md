# Generated vector store

Run the repository build script to create:

- `index.faiss` — cosine-similarity FAISS index
- `chunks.jsonl` — chunk text and source metadata aligned with the index
- `store_manifest.json` — embedding, chunking, and source-checksum configuration

These files are generated artifacts and should not be edited manually.

```bash
python scripts/build_vector_store.py --dry-run
export OPENAI_API_KEY="your-key"
python scripts/build_vector_store.py
```
