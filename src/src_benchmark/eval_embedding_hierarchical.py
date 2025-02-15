import os
import dask.dataframe as da
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_benchmark.utils_eval import (
    load_benchmark,
    cosine_similarity,
    compare_similarities,
    get_embeddings_from_json,
    get_embeddings_snomed2vec,
)


def eval_benchmark(df_benchmark, id_to_embedding):
    # Prepare the embedding triples
    embedding_triples = []
    for _, row in df_benchmark.iterrows():
        embedding_A = id_to_embedding[row.sctid]
        embedding_B = id_to_embedding[row.close_sctid]
        embedding_C = id_to_embedding[row.far_sctid]
        embedding_triples.append((embedding_A, embedding_B, embedding_C))

    # Compute similarities between embeddings
    sims_list = [
        [cosine_similarity(vec1, vec2), cosine_similarity(vec1, vec3)]
        for vec1, vec2, vec3 in embedding_triples
    ]

    # Extract the results
    results = [compare_similarities(sims) for sims in sims_list]
    misses = sum(results)
    total = len(results)
    found = total - misses
    accuracy = found / total
    print(
        f"Accuracy = {accuracy} ({found}/{total}) - making a total of {misses} errors."
    )
    return accuracy


def main(path_benchmark, typemodel=None, snomed2vec_path=None, neurips_path=None):
    # Load benchmark
    df_benchmark = load_benchmark(path_benchmark)

    # Load embedding from parquets
    if typemodel is not None:
        if typemodel.endswith("_abtt"):
            tmp = typemodel[: len(typemodel) - 5]
            emcol = f"{tmp}_embedding"
        else:
            emcol = f"{typemodel}_embedding"
        path_parquet = os.path.join(".", "embeddings", "parquets", typemodel)
        if not os.path.exists(path_parquet):
            print("parquet directory of that model does not exist")
            exit()
        df = da.read_parquet(path_parquet, engine="pyarrow")
        df = df.compute()
        id_to_embedding = dict(zip(df["sct_id"], df[emcol]))
        print("Loaded embedding:", typemodel)
    elif snomed2vec_path is not None:
        id_to_embedding = get_embeddings_snomed2vec(snomed2vec_path)
        print("Loaded embedding:", os.path.basename(snomed2vec_path))
    elif neurips_path is not None:
        id_to_embedding = get_embeddings_from_json(neurips_path)
        print("Loaded embedding:", os.path.basename(neurips_path))

    # Evaluate
    accuracy = eval_benchmark(df_benchmark, id_to_embedding)


path_benchmark = os.path.join(".", "data", "graph", "snomed_benchmark.tsv")
typemodel = "qwen2"

# SNOMED2VEC
path_node2vec_s2v = os.path.join(
    ".",
    "embeddings",
    "baseline",
    "Snomed2vec",
    "Node2Vec",
    "snomed.emb.p1.q1.w20.l40.e200.graph_format.txt",
)
path_poincare = os.path.join(
    ".",
    "embeddings",
    "baseline",
    "Snomed2vec",
    "Poincare",
    "SNOMEDCT_isa.txt.emb_dims_200.nthreads_1.txt",
)


# NEURIPS
path_avgemb = os.path.join(
    ".",
    "embeddings",
    "baseline",
    "Neurips_baseline",
    "snomed_embeddings",
    "avgemb.json",
)
path_bert = os.path.join(
    ".", "embeddings", "baseline", "Neurips_baseline", "snomed_embeddings", "bert.json"
)
path_deepwalk = os.path.join(
    ".",
    "embeddings",
    "baseline",
    "Neurips_baseline",
    "snomed_embeddings",
    "deepwalk.json",
)
path_elmo = os.path.join(
    ".", "embeddings", "baseline", "Neurips_baseline", "snomed_embeddings", "elmo.json"
)
path_harp = os.path.join(
    ".", "embeddings", "baseline", "Neurips_baseline", "snomed_embeddings", "harp.json"
)
path_line = os.path.join(
    ".", "embeddings", "baseline", "Neurips_baseline", "snomed_embeddings", "line.json"
)
path_node2vec_nips = os.path.join(
    ".",
    "embeddings",
    "baseline",
    "Neurips_baseline",
    "snomed_embeddings",
    "node2vec.json",
)
path_use = os.path.join(
    ".", "embeddings", "baseline", "Neurips_baseline", "snomed_embeddings", "use.json"
)

## Our case
main(path_benchmark, typemodel=typemodel)

## Snomed2vec
# main(path_benchmark, snomed2vec_path=path_poincare)

## NeurIPS
# main(path_benchmark, neurips_path=path_node2vec_nips)
