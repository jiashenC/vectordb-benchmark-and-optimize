# Sift 1M Benchmark

This study profiles vector similarity search performance through EvaDB.

## How to Run

Install package requirements

```bash
pip install -r requirements.txt
```

Download Sift 1M dataset.

```bash
wget https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

Make needed CSV file to help loading hdf5 data.

```bash
python make_csv.py
```

Run benchmark study.

```bash
python main.py --index-type [FAISS | CHROMADB | QDRANT | PGVECTOR]
```