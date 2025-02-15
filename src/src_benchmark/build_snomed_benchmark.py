import pandas as pd
import os
import ast
import random
import json
from collections import defaultdict, deque

ROOT_ID = 138875005

def has_relation_in_ancestors(concept_id, relation, reverse_graph, mydict_relations):
    # Get all ancestors of the current concept
    ancestors = set()
    queue = [concept_id]
    visited = set()

    while queue:
        current = queue.pop(0)
        current = str(current)
        if current in visited:
            continue
        visited.add(current)

        # Add parents to queue
        if current in reverse_graph:
            parents = reverse_graph[current]
            queue.extend(parents)
            ancestors.update(parents)

    # Check if any ancestor has this relation
    for ancestor_id in ancestors:
        if ancestor_id in mydict_relations:
            ancestor_relations = mydict_relations[int(ancestor_id)]
            if relation in ancestor_relations:
                return True
    return False


def get_descendant_at_distance(
    node_id: str, graph: dict[str, list[str]], distance: int
) -> str:
    """
    Helper function to get a random descendant at a specific distance.

    Args:
        node_id: Starting node
        graph: Graph with child relationships
        distance: How many steps down to go

    Returns:
        ID of the descendant at specified distance

    Raises:
        ValueError: If no descendant exists at specified distance
    """
    if distance == 0:
        return node_id

    if node_id not in graph:
        raise ValueError(f"Cannot find node {node_id} in graph.")
    if not graph[node_id]:
        # print(f"Cannot find descendant at distance {distance} for node {node_id}.")
        return None

    # Get random child
    child = str(random.choice(graph[node_id]))

    # Recursively get descendant at remaining distance
    return get_descendant_at_distance(child, graph, distance - 1)


def get_gold_standard_by_immediate_children(
    node_id: str, graph: dict[str, list[str]], child_distance: int = 1
) -> tuple[tuple[str, str], tuple[str, str]]:
    """
    Get random child and distant descendant relations for a given node.

    Args:
        node_id: The ID of the node to find relations for
        graph: Dictionary where keys are nodes and values are lists of child nodes
        child_distance: How many steps down to go for the second relation (1 for grandchild, 2 for great-grandchild, etc.)

    Returns:
        Tuple of two relations: ((node_id, child), (node_id, descendant))

    Raises:
        KeyError: If node_id is not in graph
        ValueError: If node has no children or if can't reach descendant at specified distance
    """
    assert child_distance >= 1

    # Check if node exists and has children
    if node_id not in graph:
        raise KeyError(f"Node {node_id} not found in graph")
    if not graph[node_id]:
        # raise ValueError(f"Node {node_id} has no children")
        # print(f"Node {node_id} has no children")
        return None

    # Get immediate child
    immediate_child = str(random.choice(graph[node_id]))

    # Get descendant at specified distance
    try:
        distant_descendant = get_descendant_at_distance(
            immediate_child, graph, child_distance
        )
    except ValueError as e:
        # raise ValueError(
        #     f"Could not find descendant at distance {child_distance}: {str(e)}"
        # )
        print(f"Could not find descendant at distance {child_distance}: {str(e)}")
        return None

    # If descendant not found, simply return None
    if not distant_descendant:
        return None

    # Return the relations
    return ((node_id, immediate_child), (node_id, distant_descendant))


def get_ancestor_at_distance(
    node_id: str, reverse_graph: dict[str, list[str]], distance: int
) -> str:
    """
    Helper function to get a random ancestor at a specific distance.

    Args:
        node_id: Starting node
        reverse_graph: Graph with parent relationships
        distance: How many steps up to go

    Returns:
        ID of the ancestor at specified distance

    Raises:
        ValueError: If no ancestor exists at specified distance
    """
    if distance == 0:
        return node_id

    if node_id not in reverse_graph or not reverse_graph[node_id]:
        raise ValueError(
            f"Cannot find ancestor at distance {distance} for node {node_id}"
        )

    # Get random parent
    parent = str(random.choice(reverse_graph[node_id]))

    # Recursively get ancestor at remaining distance
    return get_ancestor_at_distance(parent, reverse_graph, distance - 1)


def get_gold_standard_by_immediate_parents(
    node_id: str, reverse_graph: dict[str, list[str]], parent_distance: int = 1
) -> tuple[tuple[str, str], tuple[str, str]]:
    """
    Get random parent and grandparent relations for a given node.

    Args:
        node_id: The ID of the node to find relations for
        reverse_graph: Dictionary where keys are nodes and values are lists of parent nodes
        parent_distance: How many steps up to go for the second relation (1 for grandparent, 2 for great-grandparent, etc.)

    Returns:
        Tuple of two relations: ((node_id, parent), (node_id, grandparent))

    Raises:
        KeyError: If node_id is not in reverse_graph
        ValueError: If node has no parents or parent has no parents (grandparents)
    """
    assert parent_distance >= 1

    # Check if node exists and has parents
    if node_id not in reverse_graph:
        raise KeyError(f"Node {node_id} not found in reverse graph")
    if not reverse_graph[node_id]:
        raise ValueError(f"Node {node_id} has no parents")

    # Get immediate parent
    immediate_parent = str(random.choice(reverse_graph[node_id]))

    # Get ancestor at specified distance
    try:
        distant_ancestor = get_ancestor_at_distance(
            immediate_parent, reverse_graph, parent_distance
        )
    except ValueError as e:
        breakpoint()
        raise ValueError(
            f"Could not find ancestor at distance {parent_distance}: {str(e)}"
        )

    # Return the relations
    return ((node_id, immediate_parent), (node_id, distant_ancestor))


def stratified_sample_by_depth(
    depth_dict: dict[str, int], samples_per_depth: int
) -> dict[int, list[str]]:
    """
    Sample IDs while stratifying by depth values.

    Args:
        depth_dict: Dictionary with IDs as keys and depth values as values
        samples_per_depth: Number of samples to take for each depth value

    Returns:
        Dictionary with depth values as keys and lists of sampled IDs as values
    """

    # Group IDs by depth
    depth_groups = defaultdict(list)
    for id_, depth in depth_dict.items():
        depth_groups[depth].append(id_)

    # Sample from each depth group
    sampled_ids = {}
    for depth, ids in depth_groups.items():
        # If we have fewer IDs than requested samples, take all IDs
        n_samples = min(samples_per_depth, len(ids))
        sampled_ids[depth] = random.sample(ids, n_samples)

    return sampled_ids


def check_and_load_json(filename):
    if os.path.exists(filename):
        # File exists, try to load it
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("File exists but is not valid JSON")
            return None
    else:
        print(f"File {filename} does not exist")
        return None


def build_graph(df):
    """Builds a graph structure from the parent-child relationships."""
    print("Building graph SNOMED")
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    COUNTER = 0
    for _, row in df.iterrows():
        print(COUNTER)
        COUNTER += 1
        concept = row["id"]
        row_children = row["children"]
        if type(row_children) == str:
            if row_children.startswith("[") and row_children.endswith("]"):
                children = ast.literal_eval(row_children)
                for child in children:
                    graph[concept].append(child)
                    reverse_graph[child].append(concept)
    return graph, reverse_graph


def compute_depth(graph):
    """Computes depth of each concept using BFS."""
    depth_result = {ROOT_ID: 0}  # Root node has depth 0
    queue = deque([ROOT_ID])  # Start BFS from the root

    while queue:
        current_node = queue.popleft()
        current_node_depth = depth_result[current_node]

        children_nodes = graph[current_node]
        for child in children_nodes:
            if child not in depth_result:  # If depth is not assigned yet
                depth_result[child] = current_node_depth + 1
                queue.append(child)
            else:
                # Multiple parents case: take the minimum depth
                depth_result[child] = min(depth_result[child], current_node_depth + 1)

    return depth_result


def build_snomed_graph(df, filename_path):
    """Builds the snomed graph and compute the depths."""
    graph, reverse_graph = build_graph(df)
    depth = compute_depth(graph=graph)
    snomed_to_save = {"graph": graph, "reverse_graph": reverse_graph, "depth": depth}
    with open(filename_path, "w") as fw:
        json.dump(snomed_to_save, fw, indent=4)

    return graph, reverse_graph, depth


def load_snomed_graph(graph_path, snomed_path):
    # Build the SNOMED graph and extract the depths
    print("Loading SNOMED graph...")
    json_output = check_and_load_json(graph_path)
    df_snomed = pd.read_csv(snomed_path, sep="\t", encoding="utf8")
    if json_output is None:
        graph, reverse_graph, depth = build_snomed_graph(df_snomed, graph_path)
    else:
        graph, reverse_graph, depth = (
            json_output["graph"],
            json_output["reverse_graph"],
            json_output["depth"],
        )
    print("SNOMED graph loaded...")
    return df_snomed, graph, reverse_graph, depth


def hierarchy_benchmark_build(
    snomed_path,
    graph_path,
    output_path,
    n_round_bottom=150,  # 50
    n_round_top=70000,  # 14000
    seed=42,
    included_filter=True,
):
    if seed is not None:
        random.seed(seed)

    # Build the SNOMED graph and extract the depths
    df_snomed, graph, reverse_graph, depth = load_snomed_graph(
        graph_path=graph_path, snomed_path=snomed_path
    )
    print("Compute random samples...")
    # Randomly stratify samples
    dico_depth_samples = stratified_sample_by_depth(depth, samples_per_depth=100)
    print("Random samples found...")
    print("Build dataset now...")
    dataset_tuple_pairs = []
    dataset_type = []
    for _ in range(n_round_bottom):  # number of round to get a bigger dataset
        ## FROM_BOTTOM_SEARCH
        # Starting from depth = 3, the max parent dist is 1. Until depth = 18, max parent dist is 16.
        for k, v in dico_depth_samples.items():
            ## print for debug purpose
            # print(f"For depth = {k}, we got {len(v)} concepts.")
            # ignore depths of size 0, 1 and 2
            if k < 3:
                continue
            for n in range(k - 2, 0, -1):
                my_node = random.choice(v)
                my_data_sample = get_gold_standard_by_immediate_parents(
                    node_id=my_node, reverse_graph=reverse_graph, parent_distance=n
                )
                dataset_tuple_pairs.append(my_data_sample)
                type_data = f"{k}_depth_{n}_parentdistance"
                dataset_type.append(type_data)
    num_from_bottom = len(dataset_type)
    for _ in range(n_round_top):  # number of round to get a bigger dataset
        ## FROM_TOP_SEARCH
        # Starting from depth = 2, the max parent dist is 16. Until depth = 16, max parent is 1.
        for k, v in dico_depth_samples.items():
            # ignore depths of size 0, 1 and 2
            if k < 2 or k > 16:
                continue
            for n in range(1, 19 - k):
                my_node = random.choice(v)
                my_data_sample = get_gold_standard_by_immediate_children(
                    node_id=my_node, graph=graph, child_distance=n
                )
                if my_data_sample is None:
                    continue
                dataset_tuple_pairs.append(my_data_sample)
                type_data = f"{k}_depth_{n}_childdistance"
                dataset_type.append(type_data)
    num_from_top = len(dataset_type) - num_from_bottom
    print(
        f"My dataset is for size {len(dataset_type)}, with {num_from_bottom} from bottom and {num_from_top} from top."
    )

    # Create indices list of elements to keep
    indices_to_keep = []
    unique_pairs = set()

    # Track indices of unique elements
    for i, pair in enumerate(dataset_tuple_pairs):
        if pair not in unique_pairs:
            unique_pairs.add(pair)
            indices_to_keep.append(i)

    # Filter both lists using indices
    dataset_tuple_pairs = [dataset_tuple_pairs[i] for i in indices_to_keep]
    dataset_type = [dataset_type[i] for i in indices_to_keep]
    print(f"We got {len(dataset_tuple_pairs)} if we deduplicate.")

    if included_filter:
        df_snomed = df_snomed[df_snomed["is_included"] == True]

        # Keep track of valid pairs and their indices
        valid_pairs = []
        for i, pair in enumerate(dataset_tuple_pairs):
            if all(
                int(sct_id) in df_snomed["id"].values
                for tuple_pair in pair
                for sct_id in tuple_pair
            ):
                valid_pairs.append(i)

        # Filter both lists using valid indices
        dataset_tuple_pairs = [dataset_tuple_pairs[i] for i in valid_pairs]
        dataset_type = [dataset_type[i] for i in valid_pairs]
    fromtop, frombot = 0, 0
    for dtype in dataset_type:
        if "child" in dtype:
            fromtop += 1
        elif "parent" in dtype:
            frombot += 1
    print(
        f"We got {len(dataset_tuple_pairs)} if we remove filter out the bad ones, with {frombot} from bottom and {fromtop} from top."
    )

    # Save to dataframe, include fsn for readability
    rows = []
    id_to_fsn = dict(zip(df_snomed["id"], df_snomed["fsn"]))
    for k in id_to_fsn.keys():
        temp = id_to_fsn[k]
        if temp.startswith("[") and temp.endswith("]"):
            temp = temp.strip("[]")
        if temp.startswith("'") and temp.endswith("'"):
            temp = temp.strip("'")
        if temp.startswith('"') and temp.endswith('"'):
            temp = temp.strip('"')
        id_to_fsn[k] = temp

    for data_tuple, data_type in zip(dataset_tuple_pairs, dataset_type):
        ((sctid, close_sctid), (_, far_sctid)) = data_tuple
        sctid, close_sctid, far_sctid = int(sctid), int(close_sctid), int(far_sctid)
        fsn, close_fsn, far_fsn = (
            id_to_fsn[sctid],
            id_to_fsn[close_sctid],
            id_to_fsn[far_sctid],
        )
        rows.append([fsn, close_fsn, far_fsn, sctid, close_sctid, far_sctid, data_type])
    df = pd.DataFrame(
        rows,
        columns=[
            "fsn",
            "close_fsn",
            "far_fsn",
            "sctid",
            "close_sctid",
            "far_sctid",
            "distance_type",
        ],
    )
    # Save the DataFrame to a new TSV file
    df.to_csv(output_path, sep="\t", index=False, encoding="utf8")


def find_unique_multiparent_nodes(reverse_graph):
    """
    Find nodes that have multiple parents and whose set of parents is unique to them.

    Args:
        reverse_graph (dict): A dictionary where keys are node IDs and values are lists of parent IDs

    Returns:
        list: List of node IDs that have multiple parents and whose parent set is unique
    """
    # Store parent sets for each node
    parent_sets = {}

    # Create a set of parents for each node
    for node_id, parents in reverse_graph.items():
        # Convert parents list to frozenset for hashability
        parent_set = frozenset(parents)
        if len(parent_set) > 1:  # Only consider nodes with multiple parents
            if parent_set in parent_sets:
                parent_sets[parent_set].append(node_id)
            else:
                parent_sets[parent_set] = [node_id]

    # Find nodes with unique parent sets
    result = []
    for parent_set, nodes in parent_sets.items():
        if len(nodes) == 1:  # Only one node has this exact set of parents
            result.append(nodes[0])

    return result


def polyhierarchy_benchmark_build(
    snomed_path, graph_path, output_path, included_filter=True, max_parent_size=2
):

    # Build the SNOMED graph and extract the depths
    df_snomed, graph, reverse_graph, depth = load_snomed_graph(
        graph_path=graph_path, snomed_path=snomed_path
    )

    # Get ids to fsn
    id_to_fsn = dict(zip(df_snomed["id"], df_snomed["fsn"]))
    for k in id_to_fsn.keys():
        temp = id_to_fsn[k]
        if temp.startswith("[") and temp.endswith("]"):
            temp = temp.strip("[]")
        if temp.startswith("'") and temp.endswith("'"):
            temp = temp.strip("'")
        if temp.startswith('"') and temp.endswith('"'):
            temp = temp.strip('"')
        id_to_fsn[k] = temp

    # Get ids to relations
    mydict_relations = dict(zip(df_snomed["id"], df_snomed["relations"]))
    for k, v in mydict_relations.items():
        tuples_list = ast.literal_eval(v)
        if len(tuples_list) > 0:
            # Filter out tuples where first element is 116680003 - meaning isa relationship, we only want attributes, not hierarchical relationships
            filtered_tuples = [tup for tup in tuples_list if tup[0] != 116680003]
            mydict_relations[k] = filtered_tuples
        else:
            mydict_relations[k] = []

    # Get eligible IDs for the benchmark
    print("Starting to compute the IDs eligible to the polyhierarchy condition")
    id_list = find_unique_multiparent_nodes(reverse_graph=reverse_graph)
    print(f"{len(id_list)} IDs found...")

    # Eliminate IDs if: it is not included, or if one parent is not included
    # Filter IDs based on inclusion criteria
    if included_filter:
        df_snomed = df_snomed[df_snomed["is_included"] == True]
        filtered_id_list = [
            id_val
            for id_val in id_list
            if (
                int(id_val) in df_snomed["id"].values
                and all(
                    int(parent_id) in df_snomed["id"].values
                    for parent_id in reverse_graph[id_val]
                )
            )
        ]
        id_list = filtered_id_list
        print(f"{len(id_list)} IDs are found to be among the INTER concepts...")

    # Eliminate cases where the child concept is NOT fully defined
    df_snomed = df_snomed[df_snomed["status"] == "defined"]
    filtered_id_list = [
        id_val
        for id_val in id_list
        if (
            int(id_val) in df_snomed["id"].values
            and all(
                int(parent_id) in df_snomed["id"].values
                for parent_id in reverse_graph[id_val]
            )
        )
    ]
    id_list = filtered_id_list
    print(f"{len(id_list)} IDs are found to be fully defined...")

    filtered_id_list = []
    # Eliminate cases where the relations in the child is inexistant in any parent / ancestor
    for current_id in id_list:
        relations = mydict_relations[int(current_id)]
        all_relations_valid = True

        # if any relation does NOT exist above the concept, do NOT add in filtered_id_list
        for relation in relations:
            if not has_relation_in_ancestors(
                current_id, relation, reverse_graph, mydict_relations
            ):
                all_relations_valid = False
                break

        # Only keep concepts where all relations are valid
        if all_relations_valid:
            filtered_id_list.append(current_id)

    # Replace original id_list with filtered version
    id_list = filtered_id_list
    print(
        f"{len(id_list)} IDs are found to respect the attribute relation conditions..."
    )

    # Save to dataframe, include fsn for readability
    rows = []
    for id_kept in id_list:
        current_fsn = id_to_fsn[int(id_kept)]
        parents_ids = reverse_graph[id_kept]
        parents_fsn = [id_to_fsn[int(p_id)] for p_id in parents_ids]
        parent_size = len(parents_ids)
        if parent_size > max_parent_size:
            continue
        rows.append([id_kept, current_fsn, parents_ids, parents_fsn, parent_size])

    print(f"{len(rows)} rows in the benchmark! Saving now.")
    df = pd.DataFrame(
        rows,
        columns=[
            "id_node",
            "fsn_node",
            "parents_ids",
            "parents_fsn",
            "parent_len",
        ],
    )
    # Save the DataFrame to a new TSV file
    df.to_csv(output_path, sep="\t", index=False, encoding="utf8")


## This tsv file contains the SNOMED ontology, that you can extract directly from SNOMED official files. Ours was July 2024 version.
snomed_path = os.path.join(".", "data", "complete.tsv")
## This file is optional - it is a cache graph file, if you do not have, you need to put None instead
# graph_path = None
graph_path = os.path.join(".", "data", "graph", "snomed_graph.json")

## Run this to build benchmark 1 Hierarchical Similarity:
output_path = os.path.join(".", "data", "graph", "snomed_benchmark.tsv")
hierarchy_benchmark_build(
    snomed_path=snomed_path, graph_path=graph_path, output_path=output_path
)

## Run this to build benchmark 2 Semantic Composition:
output_path = os.path.join(".", "data", "graph", "polyhierarchy_benchmark.tsv")
polyhierarchy_benchmark_build(
    snomed_path=snomed_path, graph_path=graph_path, output_path=output_path
)
