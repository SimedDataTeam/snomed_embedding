import pandas as pd
import numpy as np
import json
from tqdm import tqdm


def get_embeddings_from_json(path_file):
    with open(path_file, "r", encoding="utf8") as f:
        myjsondict = json.load(f)
    myfinaldict = dict()
    for k, v in myjsondict.items():
        better_key = int(k)
        myfinaldict[better_key] = v
    if path_file.endswith("node2vec.json"):
        # remove stats
        del myfinaldict[415295]
    if path_file.endswith("line.json"):
        # remove stats
        del myfinaldict[394051]
    if path_file.endswith("avgemb.json"):
        with open(path_file + ".txt", "r") as f:
            content = f.read().strip()
            # Remove brackets and split by commas
            empty_ids = [int(x.strip()) for x in content.strip("[]").split(",")]
        # remove stats
        for emid in empty_ids:
            del myfinaldict[emid]
    return myfinaldict


def get_embeddings_snomed2vec(path_file):
    count = 0
    mydico = dict()
    with open(path_file, "r", encoding="utf8") as f:
        for line in tqdm(f):
            if count == 0:
                # ignore first line
                count += 1
                continue
            mydata = line.strip().split()
            concept = int(mydata[0])
            embedding = mydata[1:]
            mydico[concept] = embedding
    return mydico


def cosine_similarity(vec1, vec2, id2=4587):
    """
    Compute cosine similarity between two vectors.

    Args:
    vec1 (array-like): First embedding vector
    vec2 (array-like): Second embedding vector

    Returns:
    float: Cosine similarity between the two vectors
    """
    vec1, vec2 = np.array(vec1, dtype=float), np.array(vec2, dtype=float)
    try:
        assert len(vec1) == len(vec2)
    except Exception as e:
        print(id2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_rank_of_index(lst, index):
    """
    Get the rank of a value at a specific index (0 = highest value).

    Args:
        lst (list): Input list
        index (int): Index to find rank for

    Returns:
        int: Rank of the value at given index
    """
    # Convert to numpy array if it isn't already
    arr = np.array(lst)

    # Get the sorted indices in descending order
    sorted_indices = np.argsort(arr)[::-1]

    # Find where our index appears in the sorted indices
    rank = np.where(sorted_indices == index)[0][0]

    return int(rank)


def compare_similarities_multiple(similarities, n=5):
    """
    Compare cosine similarities and return top n indices of closest pairs (n-ary version).

    Args:
    similarities (list): List of cosine similarity scores
    n (int): Number of top values to return (default: 5)

    Returns:
    list: Indices of pairs with highest similarities
    """
    # Convert to numpy array if it isn't already
    arr = np.array(similarities)

    # Get indices of top N values
    top_n_indices = np.argsort(arr)[-n:][::-1]

    return top_n_indices.tolist()


def compare_similarities(similarities):
    """
    Compare cosine similarities and return indices of closest pairs (binary version).

    Args:
    similarities (list): List of cosine similarity scores

    Returns:
    list: Indices of pairs with highest similarities
    """
    max_similarity = max(similarities)
    # 0 = SUCCESS, 1 = FAIL
    index = 1 if similarities[1] == max_similarity else 0
    return index


def load_benchmark(path_benchmark):
    df_benchmark = pd.read_csv(path_benchmark, sep="\t", encoding="utf8")
    print(f"Loaded benchmark ! Total samples = {len(df_benchmark)}")
    return df_benchmark


def elementwise_add_embeddings(embedding1, embedding2):
    """
    Perform element-wise addition of two embedding vectors.

    Args:
        embedding1 (np.ndarray or list): First embedding vector
        embedding2 (np.ndarray or list): Second embedding vector

    Returns:
        np.ndarray: Sum of the embedding vectors

    Raises:
        ValueError: If embeddings have different lengths or are empty
    """
    # Convert to numpy arrays if they aren't already
    emb1 = np.array(embedding1, dtype=np.float64)
    emb2 = np.array(embedding2, dtype=np.float64)

    # Check if embeddings have the same length
    if emb1.shape != emb2.shape:
        raise ValueError("Embeddings must have the same dimensions")

    # Check if embeddings are empty
    if emb1.size == 0:
        raise ValueError("Embeddings cannot be empty")

    # Calculate element-wise sum
    sum_embedding = emb1 + emb2

    return sum_embedding


def get_statistics(values):
    """
    Calculate various statistical measures for a list of values.

    Args:
        values (list): List of values

    Returns:
        dict: Dictionary containing various statistical measures
    """
    values = np.array(values)

    statistics = {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "variance": np.var(values),
        "rmse": np.sqrt(
            np.mean(np.square(values))
        ),  # Root Mean Square (for errors, RMSE)
    }

    return statistics
