import os
import dask.dataframe as da
import numpy as np
import ast
from tqdm import tqdm
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_benchmark.utils_eval import (
    load_benchmark,
    cosine_similarity,
    get_statistics,
    get_embeddings_from_json,
    get_embeddings_snomed2vec,
    elementwise_add_embeddings,
    compare_similarities_multiple,
    get_rank_of_index,
)


def eval_benchmark(df_benchmark, id_to_embedding, topk=5):
    # Prepare the embedding triples
    embedding_pairs = []
    id_labels = []
    parent_id_list = []
    for _, row in df_benchmark.iterrows():
        if row.id_node not in id_to_embedding:
            continue
        embedding_target = id_to_embedding[row.id_node]
        parent_ids = ast.literal_eval(row.parents_ids)
        if parent_ids[0] not in id_to_embedding or parent_ids[1] not in id_to_embedding:
            continue
        embedding_A = id_to_embedding[parent_ids[0]]
        embedding_B = id_to_embedding[parent_ids[1]]
        comb = elementwise_add_embeddings(embedding_A, embedding_B)
        embedding_pairs.append((embedding_target, comb))
        id_labels.append(row.id_node)
        parent_id_list.append([parent_ids[0], parent_ids[1]])

    # Compute similarities between embeddings
    sims_list = [cosine_similarity(vec1, vec2) for vec1, vec2 in embedding_pairs]

    # Extract the results S1
    losses = [abs(sim - 1) for sim in sims_list]
    stats_losses = get_statistics(losses)
    print("Loss statistics")
    print(stats_losses)

    badids = []
    for concept_id in list(id_to_embedding.keys()):
        if id_to_embedding[concept_id][0] == 0:
            badids.append(concept_id)

    print(badids)

    concept_ids = list(id_to_embedding.keys())
    result_tuples = []
    all_ranks = []
    for id_label, embedding_pair, parent_ids in tqdm(
        zip(id_labels, embedding_pairs, parent_id_list)
    ):
        comb = embedding_pair[1]
        # Compute 1: top 5 concept ids
        my_concepts_noparents = [id for id in concept_ids if id not in parent_ids]
        sims_snomed = [
            cosine_similarity(comb, id_to_embedding[id], id)
            for id in my_concepts_noparents
        ]
        top_x_indices = compare_similarities_multiple(sims_snomed, n=topk)
        ordered_ids = [my_concepts_noparents[idx] for idx in top_x_indices]

        # Compute 2: rank of our concept
        c_ids = np.array(my_concepts_noparents)
        target_original_index = np.argwhere(c_ids == id_label)[0]
        target_rank = get_rank_of_index(sims_snomed, target_original_index) + 1
        result_tuples.append((id_label, ordered_ids, target_rank))
        all_ranks.append(target_rank)

    stats_ranks = get_statistics(all_ranks)
    print("Rank statistics")
    print(stats_ranks)
    for loss, (id_label, top_k_ids, target_rank) in zip(losses, result_tuples):
        df_benchmark.loc[df_benchmark["id_node"] == id_label, "error_margin"] = loss
        df_benchmark.loc[df_benchmark["id_node"] == id_label, "top_k_ids"] = str(
            top_k_ids
        )
        df_benchmark.loc[df_benchmark["id_node"] == id_label, "target_rank"] = (
            target_rank
        )

    return (stats_losses, stats_ranks), df_benchmark


def main(
    path_benchmark,
    output_path,
    sample_size=None,
    typemodel=None,
    snomed2vec_path=None,
    neurips_path=None,
):
    # Load benchmark
    df_benchmark = load_benchmark(path_benchmark)
    output_name = ""
    # Samplify if needed
    if type(sample_size) == int:
        # Set the random seed
        np.random.seed(42)
        # Assuming your dataframe is called 'df'
        df_benchmark = df_benchmark.sample(n=sample_size, random_state=42)

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
        output_name = typemodel
    elif snomed2vec_path is not None:
        id_to_embedding = get_embeddings_snomed2vec(snomed2vec_path)
        print("Loaded embedding:", os.path.basename(snomed2vec_path))
        output_name = os.path.basename(snomed2vec_path)
    elif neurips_path is not None:
        id_to_embedding = get_embeddings_from_json(neurips_path)
        print("Loaded embedding:", os.path.basename(neurips_path))
        output_name = os.path.basename(neurips_path)

    # Evaluate
    score, df_res = eval_benchmark(df_benchmark, id_to_embedding)
    output_tsv = output_path + output_name + ".tsv"
    output_txt = output_path + output_name + ".txt"
    df_res.to_csv(output_tsv, sep="\t", index=False, encoding="utf8")

    # Save to file as a string
    with open(output_txt, "w") as file:
        file.write(str(score[0]))
        file.write(str(score[1]))


path_benchmark = os.path.join(".", "data", "graph", "polyhierarchy_benchmark.tsv")
output_path = os.path.join(".", "data", "graph", "results", "polyhierarchy_benchmark_")
typemodel = "stellabig"

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
main(path_benchmark, output_path, typemodel=typemodel)

## Snomed2vec
# main(path_benchmark, output_path, snomed2vec_path=path_poincare)

## NeurIPS
# main(path_benchmark, output_path, neurips_path=path_bert)
