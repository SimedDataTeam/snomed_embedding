import os
import ast
import pandas as pd
import re
import torch
from tqdm import tqdm
import dask.dataframe as da
from dask import config
import pyarrow as pa
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from concept_embedding.models import get_models

tqdm.pandas(desc="My bar!")  # lots of cool paramiters you can pass here.


def get_relevant_offsets(target_start, target_end, token_offsets_list):
    """Given the target_start index, target_end index, and a list of offsets, keep offsets overlapping with target"""
    relevant_indices = []

    for i, (token_start, token_end) in enumerate(token_offsets_list):
        if token_end == 0:  # Skip special tokens with no offset
            continue

        # Check if the token overlaps with or is adjacent to the target offset
        if token_end >= target_start and token_start <= target_end:
            relevant_indices.append(i)

    # Return the adjusted range
    return relevant_indices

def fsn_without_semtag(fsn):
    """Get fsn without semtag"""
    # Define a regular expression pattern to match text inside parenthesis
    # pattern = r"\s*\([^)]*\)"
    pattern = r"\s*\([^)]*\)(?!.*\([^)]*\))"
    # Use re.sub() to replace the matched pattern with an empty string
    cleaned_text = re.sub(pattern, "", fsn)
    return cleaned_text


def extract_semtag(fsn):
    """Extract the semtag (text within the last set of parentheses) from an FSN."""
    # Define a regular expression pattern to capture the text inside the LAST set of parentheses
    match = re.search(r"\(([^)]*)\)(?!.*\()", fsn)

    # If a match is found, return the matched group (semtag without parentheses)
    if match:
        return match.group(1).strip()

    # Return None if no semtag is found
    print(f"Weird, {fsn} has no semtag...")
    return None


def check_fsn_in_desc_returning_fsn(fsn, description, synonyms):
    output = check_fsn_in_desc(fsn, description, synonyms)
    if output != -1:
        return output
    else:
        return output


def check_fsn_in_desc(str_to_get, description, synonyms, with_offsets=False):
    description = description.lower()
    # solution 1 check if fsn is repeated (80-90% of the time true)
    if str_to_get.lower() not in description:
        # solution 2 check if "" "" exists and takes the expression inside first sentence
        match = re.search(r'""(.*?)""', description)
        if match:
            if with_offsets == True:
                offset_start = match.start(1)
                offset_end = match.end(1)
                return match.group(1), (offset_start, offset_end)
            else:
                return match.group(1)
        first_sentence = description.split(". ")[0]
        match = re.search(r'"(.*?)"', first_sentence)
        if match:
            if with_offsets == True:
                offset_start = match.start(1)
                offset_end = match.end(1)
                return match.group(1), (offset_start, offset_end)
            else:
                return match.group(1)
        else:
            # solution 3 check synonym
            if type(synonyms) == str:
                synonyms = synonyms.strip("[]")
                synonyms = synonyms.split(", ")
            for syn in synonyms:
                if syn.lower() in description:
                    str_to_get = syn.lower()
                    break
            else:
                print(f"Not found for {str_to_get} - From Description: {description}")
                return -1
    if with_offsets == True:
        offset_start = description.find(str_to_get.lower())
        offset_end = offset_start + len(str_to_get.lower())
        return str_to_get, (offset_start, offset_end)
    else:
        return str_to_get


def get_embedding_snomed_description(
    fsn_no_semtag, description, synonyms, tokenizer, model, device, typemodel
):
    attention_mask = None
    ## NO DESCRIPTIONS: embed the FSN directly and return the value
    if description is None:
        if (
            typemodel == "stella"
            or typemodel == "kalm"
            or typemodel == "robertabi"
            or typemodel == "jina"
            or typemodel == "gte"
            or typemodel == "stellabig"
            or typemodel == "sapbert"
            or typemodel == "e5"
            or typemodel == "modernbert"
            or typemodel == "qwen2"
        ):
            tokens = tokenizer(fsn_no_semtag)
            indexed_tokens = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            maxlen = tokenizer.model_max_length
            tok_length = len(indexed_tokens)
        else:
            maxlen = tokenizer.model_max_length
            tokens = tokenizer.tokenize(fsn_no_semtag)
            tok_length = len(tokens)
            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        if tok_length >= maxlen:
            indexed_tokens = indexed_tokens[:maxlen]
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        if attention_mask:
            attention_mask_tensor = torch.tensor([attention_mask]).to(device)
        # Run the text through the model, and get the hidden states
        with torch.no_grad():
            if attention_mask:
                outputs = model(tokens_tensor, attention_mask=attention_mask_tensor)
            else:
                outputs = model(tokens_tensor)

        if len(outputs[0][0]) > 1:
            tensor_ent = torch.mean(outputs[0][0][:], dim=0)
        else:
            tensor_ent = outputs[0][0][:]
        if typemodel == "jina":
            ent_embedding = tensor_ent.flatten().cpu().float().numpy().tolist()
        else:
            ent_embedding = tensor_ent.flatten().cpu().numpy().tolist()
        return ent_embedding

    ### FOR DESCRIPTIONS
    token_to_get = fsn_no_semtag.lower()
    description = description.lower()

    token_to_get, (offset_start_to_get, offset_end_to_get) = check_fsn_in_desc(
        str_to_get=token_to_get,
        description=description,
        synonyms=synonyms,
        with_offsets=True,
    )

    if (
        typemodel == "stella"
        or typemodel == "kalm"
        or typemodel == "robertabi"
        or typemodel == "jina"
        or typemodel == "gte"
        or typemodel == "stellabig"
        or typemodel == "sapbert"
        or typemodel == "e5"
        or typemodel == "modernbert"
        or typemodel == "qwen2"
    ):
        tokens = tokenizer(description, truncation=True, return_offsets_mapping=True)
        indexed_tokens = tokens["input_ids"]
        offsets = tokens["offset_mapping"]
        attention_mask = tokens["attention_mask"]
    else:
        maxlen = tokenizer.model_max_length
        tokens = tokenizer(description, truncation=True, return_offsets_mapping=True)
        indexed_tokens = tokens["input_ids"]
        offsets = tokens["offset_mapping"]
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    if attention_mask:
        attention_mask_tensor = torch.tensor([attention_mask]).to(device)
    # Run the text through the model, and get the hidden states
    with torch.no_grad():
        if attention_mask:
            outputs = model(tokens_tensor, attention_mask=attention_mask_tensor)
        else:
            outputs = model(tokens_tensor)

    # Trouver les indices de la séquence `token_to_get_tokens` dans `tokens`
    word_indices = get_relevant_offsets(
        target_start=offset_start_to_get,
        target_end=offset_end_to_get,
        token_offsets_list=offsets,
    )

    # Making sure the token(s) is/are found
    if len(word_indices) == 0:
        print("NOT FOUND !")
        breakpoint()

    if len(word_indices) > 1:
        tensor_ent = torch.mean(outputs[0][0][word_indices], dim=0)
    else:
        tensor_ent = outputs[0][0][word_indices]
    if typemodel == "jina":
        ## Cannot convert BFloat16 to numpy, so we cast first
        ent_embedding = tensor_ent.flatten().cpu().float().numpy().tolist()
    else:
        ent_embedding = tensor_ent.flatten().cpu().numpy().tolist()
    return ent_embedding


def compute_embeddings_inside_df(df, model, tokenizer, typemodel, with_description):
    print("Model:", typemodel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU:", device)
    model.to(device)
    model.eval()
    if with_description:
        df[typemodel + "_embedding"] = df.progress_apply(
            lambda x: get_embedding_snomed_description(
                x.FSN_nosemtag,
                x.description,
                x.english_synonyms,
                tokenizer,
                model,
                device,
                typemodel,
            ),
            axis=1,
        )
    else:
        df[typemodel + "_embedding"] = df.progress_apply(
            lambda x: get_embedding_snomed_description(
                x.FSN_nosemtag,
                None,
                x.english_synonyms,
                tokenizer,
                model,
                device,
                typemodel,
            ),
            axis=1,
        )
    return df


def build_concept_embeddings(df, with_description=True, type_model=None) -> list[dict]:
    # get fsn w/o semtag
    df["FSN_nosemtag"] = df["FSN"].apply(fsn_without_semtag)

    # Get models
    models, tokenizers, types_models = get_models(typem=type_model)
    for model, tokenizer, typemodel in zip(models, tokenizers, types_models):
        # Compute embeddings and update the df
        df = compute_embeddings_inside_df(
            df, model, tokenizer, typemodel, with_description
        )
        print(f"{typemodel} embeddings saved successfully !")
    return df, models, tokenizers, types_models


def tsv_embeddings_to_parquet(
    snomed_path=os.path.join(".", "data", "descriptions_FR_25_11.tsv"),
    with_description=True,
    type_models=["fralbert"],
):
    config.set({"dataframe.convert-string": False})

    # 1) load tsv file
    df = pd.read_csv(snomed_path, sep="\t", encoding="utf8")

    # 2) add semtag in the tsv file
    df["semtag"] = df["fsn"].apply(extract_semtag)

    # 3) rename columns and remove one unecessary column
    df = df.rename(columns={"fsn": "FSN"})
    df = df.rename(columns={"term": "french_synonyms"})
    df = df.rename(columns={"synonym": "english_synonyms"})
    df = df.rename(columns={"nosemtag": "fsnnosemtag"})
    df = df.drop("prompt", axis=1)
    print("Getting embedding step")
    df, _, _, typemodels = build_concept_embeddings(
        df, with_description=with_description, type_model=type_models
    )
    print("Getting embedding: DONE")

    # 4) format embedding and save to parquet
    for typemodel in typemodels:
        save_dir = os.path.join(".", "embeddings", "parquets", f"{typemodel}")
        # ignore this embedding if parquet already exists
        if os.path.exists(save_dir):
            continue
        else:
            os.makedirs(save_dir, exist_ok=True)
        # Define the schema for the Parquet file (PyArrow types)
        schema = pa.schema(
            [
                ("sct_id", pa.int64()),
                ("description", pa.string()),
                ("english_synonyms", pa.string()),
                ("FSN", pa.string()),
                ("children", pa.string()),
                ("parents", pa.string()),
                ("french_synonyms", pa.string()),
                ("file", pa.string()),
                ("siblings", pa.string()),
                ("fsnnosemtag", pa.string()),
                ("semtag", pa.string()),
                (
                    f"{typemodel}_embedding",
                    pa.list_(pa.float64()),
                ),  # Define column as list of doubles
            ]
        )
        df[f"{typemodel}_embedding"] = df[f"{typemodel}_embedding"].progress_apply(
            lambda x: ast.literal_eval(x.strip()) if isinstance(x, str) else x
        )
        # save as parquet
        tdf = df[
            [
                "sct_id",
                "description",
                "english_synonyms",
                "FSN",
                "children",
                "parents",
                "french_synonyms",
                "file",
                "siblings",
                "fsnnosemtag",
                "semtag",
                f"{typemodel}_embedding",
            ]
        ]
        ddf = da.from_pandas(tdf, npartitions=20)
        ddf.to_parquet(save_dir, engine="pyarrow", schema=schema)
        del tdf, ddf
