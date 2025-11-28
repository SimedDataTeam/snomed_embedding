import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from snomed_embedding.los_prediction.load_embedding import load_embedding, load_embedding_baseline

dico_concept_mapping_july24 = {"15777000": "714628002", "26763009": "40617009", "108290001": "1287742003", "241055006": "71651007"}
dico_concept_mapping_harp_deepwalk = {"15777000": "714628002", "840539006": "186747009"}
dico_concept_mapping_n2vneurips = {
    "15777000": "714628002",
    "840539006": "186747009",
    "97331000119101": "769220000",
    "124171000119105": "423279000",
    "132281000119108": "128053003",
    "1551000119108": "390834004",
    "90781000119102": "236499007",
    "368581000119106": "230572002",
    "16335031000119103": "169069000",
    "1571000087109": "47079000",
}
dico_concept_mapping_avgemb = {
    "15777000": "207272004",
    "840539006": "186747009",
    "389087006": "207551003",
    "302870006": "267499005",
    "55822004": "154739000",
    "127783003": "258058009",
}
dico_concept_mapping_s2v = {
    "840539006": "186747009",
    "770349000": "91302008",
    "434363004": "426329006",
}


def prepare_ml_data(dico_concept_mapping=dico_concept_mapping_july24, modelname="sapbert"):
    # Load embedding dictionary
    print("=" * 80)
    print("Loading embedding dictionary...")
    print("=" * 80)
    embedding_dict = load_embedding("/workspaces/project/embeddings/parquets", modelname=modelname)

    # path_nips = os.path.join(
    #     ".",
    #     "embeddings",
    #     "baseline",
    #     "Neurips_baseline",
    #     "snomed_embeddings",
    # )
    # path_node2vec_s2v = os.path.join(
    #     ".",
    #     "embeddings",
    #     "baseline",
    #     "Snomed2vec",
    #     "Node2Vec",
    #     "snomed.emb.p1.q1.w20.l40.e200.graph_format.txt",
    # )
    # path_poincare = os.path.join(
    #     ".",
    #     "embeddings",
    #     "baseline",
    #     "Snomed2vec",
    #     "Poincare",
    #     "SNOMEDCT_isa.txt.emb_dims_200.nthreads_1.txt",
    # )
    # embedding_dict = load_embedding_baseline(path_node2vec_s2v, modelname=modelname)

    # Load encounter data
    print("\n" + "=" * 80)
    print("Loading encounter data from TSV...")
    print("=" * 80)
    encounter_data = pd.read_csv(
        "/workspaces/project/data/synthea/encounter_data.tsv",
        sep="\t",
    )
    print(f"Loaded {len(encounter_data)} encounters")
    print(f"Columns: {list(encounter_data.columns)}")

    # Extract all unique SNOMED CT codes from conditions and procedures
    print("\n" + "=" * 80)
    print("Extracting SNOMED CT codes...")
    print("=" * 80)

    # Get all condition codes
    all_condition_codes = set()
    for conditions in encounter_data["CONDITIONS"].dropna():
        codes = str(conditions).split()
        all_condition_codes.update(codes)

    # Get all procedure codes
    all_procedure_codes = set()
    for procedures in encounter_data["PROCEDURES"].dropna():
        codes = str(procedures).split()
        all_procedure_codes.update(codes)

    # Map condition codes using the provided dictionary
    all_condition_codes = {dico_concept_mapping.get(code, code) for code in all_condition_codes}

    # Map procedure codes using the provided dictionary
    all_procedure_codes = {dico_concept_mapping.get(code, code) for code in all_procedure_codes}

    # Calculate coverage
    print(f"\nUnique CONDITION codes in dataset: {len(all_condition_codes)}")
    print(f"Unique PROCEDURE codes in dataset: {len(all_procedure_codes)}")

    # Check coverage for conditions
    condition_codes_in_embedding = all_condition_codes & set(embedding_dict.keys())
    condition_coverage = len(condition_codes_in_embedding) / len(all_condition_codes) * 100 if all_condition_codes else 0

    # Check coverage for procedures
    procedure_codes_in_embedding = all_procedure_codes & set(embedding_dict.keys())
    procedure_coverage = len(procedure_codes_in_embedding) / len(all_procedure_codes) * 100 if all_procedure_codes else 0

    # Combined coverage
    all_codes = all_condition_codes | all_procedure_codes
    all_codes_in_embedding = all_codes & set(embedding_dict.keys())
    total_coverage = len(all_codes_in_embedding) / len(all_codes) * 100 if all_codes else 0

    # Print results
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)
    print("\nCONDITIONS:")
    print(f"  Total unique codes: {len(all_condition_codes)}")
    print(f"  Codes in embedding: {len(condition_codes_in_embedding)}")
    print(f"  Coverage: {condition_coverage:.2f}%")

    print("\nPROCEDURES:")
    print(f"  Total unique codes: {len(all_procedure_codes)}")
    print(f"  Codes in embedding: {len(procedure_codes_in_embedding)}")
    print(f"  Coverage: {procedure_coverage:.2f}%")

    print("\nTOTAL (CONDITIONS + PROCEDURES):")
    print(f"  Total unique codes: {len(all_codes)}")
    print(f"  Codes in embedding: {len(all_codes_in_embedding)}")
    print(f"  Coverage: {total_coverage:.2f}%")

    # Show some examples of missing codes
    missing_condition_codes = all_condition_codes - condition_codes_in_embedding
    missing_procedure_codes = all_procedure_codes - procedure_codes_in_embedding

    if missing_condition_codes:
        print("\nExample missing CONDITION codes (showing up to 10):")
        for code in list(missing_condition_codes)[:10]:
            print(f"  - {code}")

    if missing_procedure_codes:
        print("\nExample missing PROCEDURE codes (showing up to 10):")
        for code in list(missing_procedure_codes)[:10]:
            print(f"  - {code}")

    print("\n" + "=" * 80)

    # Prepare training set with embeddings
    print("Preparing dataset with embeddings...")
    dataset = []
    for _, row in encounter_data.iterrows():
        encounter_id = row["ENCOUNTER"]
        condition_codes = str(row["CONDITIONS"]).split() if pd.notna(row["CONDITIONS"]) else []
        procedure_codes = str(row["PROCEDURES"]).split() if pd.notna(row["PROCEDURES"]) else []

        # Map codes using the provided dictionary
        condition_codes = [dico_concept_mapping.get(code, code) for code in condition_codes]
        procedure_codes = [dico_concept_mapping.get(code, code) for code in procedure_codes]

        # Get embeddings for condition codes
        condition_embeddings = [embedding_dict[code] for code in condition_codes if code in embedding_dict]

        # Get embeddings for procedure codes
        procedure_embeddings = [embedding_dict[code] for code in procedure_codes if code in embedding_dict]

        dataset.append(
            {
                "ENCOUNTER": encounter_id,
                "CONDITION_EMBEDDINGS": condition_embeddings,
                "PROCEDURE_EMBEDDINGS": procedure_embeddings,
                "Y": row["LONG_STAY"],
            },
        )
    return dataset


class EncounterDataset(Dataset):
    """PyTorch Dataset for encounter embeddings"""

    def __init__(self, data, use_procedures=False):
        """
        Initialize the dataset.

        Args:
            data: List of dictionaries with CONDITION_EMBEDDINGS, PROCEDURE_EMBEDDINGS, and Y
            use_procedures: If True, use both conditions and procedures. If False, use only conditions (default)

        """
        self.data = data
        self.use_procedures = use_procedures

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Get condition embeddings
        condition_emb = item["CONDITION_EMBEDDINGS"]

        # Optionally add procedure embeddings
        if self.use_procedures:
            procedure_emb = item["PROCEDURE_EMBEDDINGS"]
            all_embeddings = condition_emb + procedure_emb
        else:
            all_embeddings = condition_emb

        # Average pooling over embeddings
        if len(all_embeddings) > 0:
            # Stack and average
            embedding_tensor = torch.stack([torch.tensor(emb, dtype=torch.float32) for emb in all_embeddings])
            avg_embedding = torch.mean(embedding_tensor, dim=0)
        else:
            # If no embeddings, create zero vector (assuming 768 dimensions for embeddings)
            avg_embedding = torch.zeros(768, dtype=torch.float32)

        label = torch.tensor(item["Y"], dtype=torch.float32)

        return avg_embedding, label


class FeedForwardNN(nn.Module):
    """Feed Forward Neural Network for binary classification"""

    def __init__(self, input_dim, hidden_dims=None):
        if hidden_dims is None:
            hidden_dims = [512, 256]
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_binary_classifier(
    dataset,
    batch_size=64,
    epochs=40,
    learning_rate=1e-5,
    hidden_dims=None,
    use_procedures=True,
):
    """
    Train a feed-forward neural network for binary classification.

    Args:
        dataset: List of dictionaries with CONDITION_EMBEDDINGS, PROCEDURE_EMBEDDINGS, and Y
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        hidden_dims: List of hidden layer dimensions
        use_procedures: If True, use both conditions and procedures. If False, use only conditions (default)

    Returns:
        dict: Dictionary containing trained model and results

    """
    if hidden_dims is None:
        hidden_dims = [512, 256]

    print("\n" + "=" * 80)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 80)
    print(f"Using embeddings from: {'CONDITIONS + PROCEDURES' if use_procedures else 'CONDITIONS ONLY'}")

    # Split dataset into train (80%), dev (10%), test (10%)
    np.random.seed(42)  # noqa: NPY002
    indices = np.random.permutation(len(dataset))  # noqa: NPY002

    train_size = int(0.8 * len(dataset))
    dev_size = int(0.1 * len(dataset))

    train_indices = indices[:train_size]
    dev_indices = indices[train_size : train_size + dev_size]
    test_indices = indices[train_size + dev_size :]

    train_data = [dataset[i] for i in train_indices]
    dev_data = [dataset[i] for i in dev_indices]
    test_data = [dataset[i] for i in test_indices]

    print(f"Train size: {len(train_data)}")
    print(f"Dev size: {len(dev_data)}")
    print(f"Test size: {len(test_data)}")

    # print positive/negative samples in each split
    def count_labels(data):
        pos = sum(1 for item in data if item["Y"] == 1)
        neg = sum(1 for item in data if item["Y"] == 0)
        return pos, neg

    train_pos, train_neg = count_labels(train_data)
    dev_pos, dev_neg = count_labels(dev_data)
    test_pos, test_neg = count_labels(test_data)

    print(f"Train positive samples: {train_pos}, Train negative samples: {train_neg}")
    print(f"Dev positive samples: {dev_pos}, Dev negative samples: {dev_neg}")
    print(f"Test positive samples: {test_pos}, Test negative samples: {test_neg}")

    # Create PyTorch datasets
    train_dataset = EncounterDataset(train_data, use_procedures=use_procedures)
    dev_dataset = EncounterDataset(dev_data, use_procedures=use_procedures)
    test_dataset = EncounterDataset(test_data, use_procedures=use_procedures)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get input dimension from first sample
    sample_embedding, _ = train_dataset[0]
    input_dim = sample_embedding.shape[0]
    print(f"Input dimension: {input_dim}")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FeedForwardNN(input_dim, hidden_dims).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)

    # Training loop
    best_dev_accuracy = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        dev_correct = 0
        dev_total = 0

        with torch.no_grad():
            for embeddings, labels in dev_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)

                outputs = model(embeddings).squeeze()
                predictions = (outputs >= 0.5).float()
                dev_correct += (predictions == labels).sum().item()
                dev_total += labels.size(0)

        dev_accuracy = 100 * dev_correct / dev_total

        # Save best model
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_model_state = model.state_dict().copy()

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Dev Acc: {dev_accuracy:.2f}%")

    # Load best model
    model.load_state_dict(best_model_state)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Evaluate on all splits
    def evaluate(loader, split_name):
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for embeddings, labels in loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)

                outputs = model(embeddings).squeeze()
                predictions = (outputs >= 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Convert to numpy and ensure it's always a list
                pred_numpy = predictions.cpu().numpy()
                label_numpy = labels.cpu().numpy()

                # Handle both single samples and batches
                all_predictions.extend(pred_numpy.tolist() if pred_numpy.ndim > 0 else [pred_numpy.item()])
                all_labels.extend(label_numpy.tolist() if label_numpy.ndim > 0 else [label_numpy.item()])

        accuracy = 100 * correct / total
        print(f"{split_name} Accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy, all_predictions, all_labels

    train_acc, _, _ = evaluate(train_loader, "Train")
    dev_acc, _, _ = evaluate(dev_loader, "Dev")
    test_acc, test_preds, test_labels = evaluate(test_loader, "Test")

    print("\n" + "=" * 80)

    results = {
        "model": model,
        "train_accuracy": train_acc,
        "dev_accuracy": dev_acc,
        "test_accuracy": test_acc,
        "test_predictions": test_preds,
        "test_labels": test_labels,
        "device": device,
    }

    return results


if __name__ == "__main__":
    dataset = prepare_ml_data()
    results = train_binary_classifier(dataset, use_procedures=True)
