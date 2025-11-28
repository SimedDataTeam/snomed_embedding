import os

import dask.dataframe as da

from snomed_embedding.src_benchmark.utils_eval import get_embeddings_from_json, get_embeddings_snomed2vec


def load_embedding_baseline(embedding_path, modelname):
    """
    Load baseline embedding from a given path.

    Args:
        embedding_path (str): Path to the embedding file.
        modelname (str): Name of the embedding model to load.

    Returns:
        dict: Dictionary containing the loaded embedding.

    """
    if modelname in ["node2vec", "deepwalk", "avgemb", "bert", "use", "elmo", "harp"]:
        embedding_dict = get_embeddings_from_json(os.path.join(embedding_path, f"{modelname}.json"))
    elif modelname == "poincare" or modelname == "node2vec_s2v":
        embedding_dict = get_embeddings_snomed2vec(embedding_path)
        # convert each embedding being list of str to list of float
        for k, v in embedding_dict.items():
            embedding_dict[k] = [float(x) for x in v]
    else:
        raise ValueError(f"Unknown modelname: {modelname}")
    print(f"Loaded baseline embedding with {len(embedding_dict)} entries")
    # convert keys to str
    embedding_dict = {str(k): v for k, v in embedding_dict.items()}
    return embedding_dict


def load_embedding(embedding_path, modelname):
    """
    Load embedding from a given path.

    Args:
        embedding_path (str): Path to the embedding file.
        modelname (str): Name of the embedding model to load.

    Returns:
        dict: Dictionary containing the loaded embedding.

    """
    print("Reading parquets...")
    dask_dataframe = da.read_parquet(os.path.join(embedding_path, modelname), engine="pyarrow")
    print("Finished reading parquets.")
    # lowercase modelname for column access
    modelname = modelname.lower()
    # drop all columns except 'sct_id' and '{modelname}_embedding'
    dask_dataframe = dask_dataframe[["sct_id", f"{modelname}_embedding"]]
    # convert to pandas
    df = dask_dataframe.compute()
    # ensure sct_id is str
    df["sct_id"] = df["sct_id"].astype(str)
    # get diconary mapping sct_id to {modelname}_embedding
    embedding_dict = dict(zip(df["sct_id"], df[f"{modelname}_embedding"], strict=True))
    print(f"Loaded embedding with {len(embedding_dict)} entries")
    return embedding_dict
