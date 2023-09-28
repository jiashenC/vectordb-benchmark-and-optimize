import h5py
import evadb

from tqdm import tqdm
from time import perf_counter
from argparse import ArgumentParser


def profile_pgvector(index_type="PGVECTOR"):
    cur = evadb.connect().cursor()

    cur.query("CREATE FUNCTION IF NOT EXISTS HDF5Reader IMPL 'hdf5_reader.py'").df()

    postgres_params = {
        "user": "eva",
        "password": "password",
        "host": "127.0.0.1",
        "port": "5432",
        "database": "evadb",
    }
    cur.query(f"CREATE DATABASE IF NOT EXISTS postgres WITH ENGINE = 'postgres', PARAMETERS = {postgres_params}").df()

    table_df = cur.query("USE postgres { SELECT * FROM pg_catalog.pg_tables }").df()
    table_list = []
    if not table_df.empty:
        table_list = table_df["tablename"].to_list()

    f = h5py.File("./sift-128-euclidean.hdf5")
    if "trainvector" not in table_list:
        print("Start table creation ...")
        cur.query("USE postgres { CREATE TABLE trainVector (feature vector(128), num INT) }").df()
        for i in tqdm(range(1000000)):
            vector = f["train"][i]
            vector = vector.reshape(-1).tolist()
            cur.query(f"""
                USE postgres {{ INSERT INTO trainVector (feature, num) VALUES ('{vector}', {i}) }}
            """).df()

    ##################################
    # Build index
    ##################################
    print("Start building ...")
    st = perf_counter()
    cur.query(f"CREATE INDEX IF NOT EXISTS train{index_type}Index ON postgres.trainvector (feature) USING {index_type}").df()
    print(f"{index_type} build time: {perf_counter() - st:.3f}")

    # ##################################
    # # Search index
    # ##################################
    print("Start searching ...")
    tp = 0
    st = perf_counter()
    ITER = 10
    for i in range(ITER):
        res = cur.query(f"""
                SELECT * FROM postgres.trainvector
                ORDER BY Similarity(HDF5Reader('test', '{i}'), feature)
                LIMIT 100
                """).df()
        res = set(res["trainvector.num"].to_list())
        gt = set(f["neighbors"][i].tolist())
        tp += len(gt & res)
    print(tp / (ITER * 100))


def profile_other(index_type):
    cur = evadb.connect().cursor()

    cur.query("CREATE FUNCTION IF NOT EXISTS HDF5Reader IMPL 'hdf5_reader.py'").df()

    table_df = cur.query("SHOW TABLES").df()
    table_list = []
    if not table_df.empty:
        table_list = table_df["name"].to_list()
        print("Existing table", table_list)

    if "trainVector" not in table_list:
        print("Start table creation ...")
        if "trainIndex" not in table_list:
            cur.query("CREATE TABLE trainIndex (index INTEGER)").df()
            cur.query("LOAD CSV 'trainIndex.csv' INTO trainIndex").df()
        cur.query("CREATE TABLE trainVector AS SELECT HDF5Reader('train', index) FROM trainIndex").df()


    ##################################
    # Build index
    ##################################
    print("Start building ...")
    st = perf_counter()
    cur.query(f"CREATE INDEX IF NOT EXISTS train{index_type}Index ON trainVector (feature) USING {index_type}").df()
    print(f"{index_type} build time: {perf_counter() - st:.3f}")

    ##################################
    # Search index
    ##################################
    f = h5py.File("./sift-128-euclidean.hdf5")
    print("Start searching ...")
    tp = 0
    ITER = 10
    st = perf_counter()
    for i in range(ITER):
        res = cur.query(f"""
                SELECT * FROM trainVector
                ORDER BY Similarity(HDF5Reader('test', '{i}'), feature)
                LIMIT 100
                """).df()
        res = set((res["trainvector._row_id"] - 1).to_list())
        gt = set(f["neighbors"][i].tolist())
        tp += len(gt & res)
    print(tp / (ITER * 100))
    print(f"Search time: {perf_counter() - st:.3f}")


def main():
    parser = ArgumentParser("--index-type", type=str, required=True)
    args = parser.parse_args()

    if args.index_type in ["FAISS", "CHROMADB", "QDRANT"]:
        profile_other(args.index_type)
    elif args.index_type == "PGVECTOR":
        profile_pgvector()
    else:
        raise Exception(f"Index type {args.index_type} is not supported in this benchmark.")
