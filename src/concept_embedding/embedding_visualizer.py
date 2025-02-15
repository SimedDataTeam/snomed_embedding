import plotly.express as px
import umap
import pandas as pd
import os
import dask.dataframe as da
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from concept_embedding.models import TYPEMODELS

def visualize_umap(df, emcol="Embedding", labelcol="semtag", save_path="figure.png"):
    features = pd.DataFrame(
        df[emcol].to_list()
    )  # Converts the list of lists into a DataFrame

    # Apply UMAP
    n_components = 2
    n_neighbors = 15
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding = umap_model.fit_transform(features)

    # Convert the embedding into a DataFrame with proper column names
    umap_df = pd.DataFrame(
        embedding, columns=[f"UMAP {i+1}" for i in range(n_components)]
    )

    # Add the labels to the UMAP data for coloring the points
    umap_df[labelcol] = df[labelcol].values

    # Create a scatter plot of the UMAP components
    fig = px.scatter(
        umap_df, x="UMAP 1", y="UMAP 2", color=labelcol, title="UMAP Visualization"
    )

    # Output
    fig.write_image(save_path, scale=1, width=1000, height=800)
    print("UMAP DONE")


def main():
    # run the three visualizers for each model
    for typemodel in TYPEMODELS:
        parquet_dir = os.path.join(".", "embeddings", "parquets", f"{typemodel}")
        print("Model:", typemodel)
        if not os.path.exists(parquet_dir):
            print("folder does not exist...")
            continue
        df = da.read_parquet(parquet_dir, engine="pyarrow")
        df = df.compute()
        print("df loaded !")
        emcol = f"{typemodel}_embedding"
        # filter semantic tags
        filtered_df = df[
            df["semtag"].isin(
                ["body structure", "substance", "finding", "disorder", "procedure"]
            )
        ]
        df = filtered_df.reset_index(drop=True)

        # Perform stratified sampling
        sample_size = 5000

        # Get the proportion of each group in the strat_column
        grouped = df.groupby("semtag")

        # Perform stratified sampling
        df = grouped.apply(
            lambda x: x.sample(frac=sample_size / len(df), random_state=42)
        ).reset_index(drop=True)

        print(f"{len(df)} concepts to plot.")

        visualize_umap(
            df,
            emcol=emcol,
            save_path=os.path.join(".", "figures", f"{typemodel}_UMAP.png"),
        )
