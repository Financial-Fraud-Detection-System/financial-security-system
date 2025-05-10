import json
import logging
import os
from collections import defaultdict
from itertools import combinations as intelligentGrouper
from typing import Dict, List, Set, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from config import Config
from database.neo4j import Neo4jConnection
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)

neo = Neo4jConnection(
    Config.NEO4J_URI,
    Config.NEO4J_USER,
    Config.NEO4J_PASSWORD,
)


# MODEL
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index)


# feature engineering & tensor prep
def _prepare_node_features(
    data_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, torch.Tensor, Dict[str, int], Dict[int, str]]:
    """Return scaled feature tensor + mapping dicts (acc→idx)."""
    ip_trans_freq = (
        data_df.groupby(["acc_num", "ip_address"])["trans_num"]
        .count()
        .reset_index()
        .groupby("acc_num")["trans_num"]
        .sum()
        .reset_index()
        .rename(columns={"trans_num": "ip_trans_freq"})
    )

    agg = (
        data_df.groupby("acc_num")
        .agg(
            amt_mean=("amt", "mean"),
            amt_std=("amt", "std"),
            trans_count=("trans_num", "count"),
            state=(
                "state",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            ),
            zip=("zip", "first"),
            city_pop=("city_pop", "mean"),
            unix_time_mean=("unix_time", "mean"),
            time_span=("unix_time", lambda x: x.max() - x.min()),
            ip_address=(
                "ip_address",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            ),
        )
        .reset_index()
        .merge(ip_trans_freq, on="acc_num", how="left")
        .fillna({"ip_trans_freq": 0, "amt_std": 0})
        .astype({"trans_count": int})
    )

    for col in ["ip_address", "state", "zip"]:
        agg[col] = agg[col].astype("category").cat.codes

    scaler = StandardScaler()
    feature_cols = [
        "amt_mean",
        "amt_std",
        "trans_count",
        "state",
        "zip",
        "city_pop",
        "unix_time_mean",
        "time_span",
        "ip_trans_freq",
        "ip_address",
    ]
    x = torch.tensor(scaler.fit_transform(agg[feature_cols]), dtype=torch.float)

    acc_to_idx = {acc: i for i, acc in enumerate(agg["acc_num"])}
    idx_to_acc = {i: acc for acc, i in acc_to_idx.items()}
    return agg, x, acc_to_idx, idx_to_acc


# Neo4j look‑ups
def _pull_existing_graph(
    acc_to_idx: Dict[str, int],
) -> Tuple[Set[Tuple[int, int]], Dict[int, Set[str]]]:
    """Return (edges, rings) currently stored in Neo4j that involve known accounts."""
    edges: Set[Tuple[int, int]] = set()
    rings: Dict[int, Set[str]] = defaultdict(set)

    with neo.driver.session() as sess:
        for rec in sess.run(
            "MATCH (a:Account)-[:RELATED]->(b:Account) RETURN a.acc_num AS src, b.acc_num AS dst"
        ):
            if rec["src"] in acc_to_idx and rec["dst"] in acc_to_idx:
                edges.add((acc_to_idx[rec["src"]], acc_to_idx[rec["dst"]]))

        for rec in sess.run(
            "MATCH (r:FraudRing)-[:INVOLVES]->(a:Account) RETURN id(r) AS ring_id, a.acc_num AS acc"
        ):
            rings[rec["ring_id"]].add(rec["acc"])

    return edges, rings


# derive edges from batch
def _derive_edges(
    data_df: pd.DataFrame, acc_to_idx: Dict[str, int], existing: Set[Tuple[int, int]]
) -> Set[Tuple[int, int]]:
    """Combine existing edges with IP/ZIP/temporal edges extracted from the current batch."""
    edges = set(existing)

    def _add_edges(group_col: str, threshold: int = 2):
        grouped = data_df.groupby(group_col)["acc_num"].apply(list)
        for key, nodes in grouped.items():
            if len(nodes) > 1:
                counts = (
                    data_df[data_df[group_col] == key]
                    .groupby("acc_num")["trans_num"]
                    .count()
                )
                valid = [n for n in nodes if counts.get(n, 0) >= threshold]
                for s, d in intelligentGrouper(valid, 2):
                    edges.update(
                        {(acc_to_idx[s], acc_to_idx[d]), (acc_to_idx[d], acc_to_idx[s])}
                    )

    _add_edges("ip_address")
    _add_edges("zip")

    # temporal (<5‑min gap, same IP)
    for ip, g in data_df.sort_values("unix_time").groupby("ip_address"):
        times, nodes = g["unix_time"].values, g["acc_num"].values
        for i in range(len(times) - 1):
            if times[i + 1] - times[i] < 300:
                s, d = nodes[i], nodes[i + 1]
                edges.update(
                    {(acc_to_idx[s], acc_to_idx[d]), (acc_to_idx[d], acc_to_idx[s])}
                )
    return edges


# model loader
def _load_model(in_channels: int) -> GraphSAGE:
    model_path = os.path.join(os.path.dirname(__file__), "artifacts", "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found – train the model first.")
    model = GraphSAGE(in_channels, hidden_channels=128, out_channels=216)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# prediction & ring assembly
def _predict_rings(
    model: GraphSAGE, data: Data, idx_to_acc: Dict[int, str]
) -> Dict[int, List[str]]:
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        labels = logits.argmax(dim=1).cpu().numpy()

    rings = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != 0:  # 0 = non‑fraud
            rings[int(label)].append(idx_to_acc[idx])
    return rings


def _filter_novel_rings(
    predicted: Dict[int, List[str]], existing: Dict[int, Set[str]]
) -> List[Tuple[int, List[str]]]:
    existing_sets = {tuple(sorted(accs)) for accs in existing.values()}
    novel: list[Tuple[int, list[str]]] = []
    for ring_id, accounts in predicted.items():
        ordered = sorted(accounts)
        if tuple(ordered) not in existing_sets:
            novel.append((ring_id, ordered))
    return novel


# Neo4j persistence
def _persist_batch(
    agg_features: pd.DataFrame,
    edges: Set[Tuple[int, int]],
    novel_rings: List[Tuple[int, List[str]]],
    idx_to_acc: Dict[int, str],
):
    with neo.driver.session() as sess:
        for _, row in agg_features.iterrows():
            sess.execute_write(
                neo.create_account_node, acc_num=row["acc_num"], features=row.to_dict()
            )

        for s_idx, d_idx in edges:
            sess.execute_write(
                neo.create_edge,
                src_acc=idx_to_acc[s_idx],
                dst_acc=idx_to_acc[d_idx],
                edge_type="RELATED",
            )

        for ring in novel_rings:
            sess.execute_write(
                neo.create_fraud_ring,
                ring_id=ring[0],
                accounts=[{"acc_num": acc} for acc in ring[1]],
            )


# PUBLIC API
def process_and_predict(transactions: List[Dict]) -> List[Tuple[int, List[str]]]:
    """
    Predict fraud rings from a batch of transaction dicts and return **only** those
    rings not already stored in Neo4j.
    """
    data_df = pd.DataFrame(transactions, copy=False)
    data_df["acc_num"] = data_df["acc_num"].astype(str)

    # Node features & mappings
    agg_features, x, acc_to_idx, idx_to_acc = _prepare_node_features(data_df)

    # Graph context from Neo4j
    existing_edges, existing_rings = _pull_existing_graph(acc_to_idx)

    # Build full edge set
    edges = _derive_edges(data_df, acc_to_idx, existing_edges)
    if edges:
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    # Inference
    model = _load_model(in_channels=x.size(1))
    predicted_rings = _predict_rings(model, data, idx_to_acc)

    # Compare & persist
    novel_rings = _filter_novel_rings(predicted_rings, existing_rings)
    _persist_batch(agg_features, edges, novel_rings, idx_to_acc)

    return novel_rings


# Example usage
if __name__ == "__main__":
    transactions = [
        {
            "acc_num": "acc123",
            "amt": 5000.0,
            "state": "NY",
            "zip": "10001",
            "city_pop": 1000000,
            "trans_num": "t1",
            "unix_time": 1625097600,
            "ip_address": "192.168.1.1",
        },
        {
            "acc_num": "acc123",
            "amt": 6000.0,
            "state": "NY",
            "zip": "10001",
            "city_pop": 1000000,
            "trans_num": "t2",
            "unix_time": 1625097601,
            "ip_address": "192.168.1.1",
        },
        {
            "acc_num": "acc456",
            "amt": 5500.0,
            "state": "NY",
            "zip": "10001",
            "city_pop": 1000000,
            "trans_num": "t3",
            "unix_time": 1625097602,
            "ip_address": "192.168.1.1",
        },
        {
            "acc_num": "acc456",
            "amt": 7000.0,
            "state": "NY",
            "zip": "10001",
            "city_pop": 1000000,
            "trans_num": "t4",
            "unix_time": 1625097603,
            "ip_address": "192.168.1.1",
        },
        {
            "acc_num": "acc999",
            "amt": 6500.0,
            "state": "NY",
            "zip": "10001",
            "city_pop": 1000000,
            "trans_num": "t5",
            "unix_time": 1625097604,
            "ip_address": "192.168.1.1",
        },
        {
            "acc_num": "acc999",
            "amt": 8000.0,
            "state": "NY",
            "zip": "10001",
            "city_pop": 1000000,
            "trans_num": "t6",
            "unix_time": 1625097605,
            "ip_address": "192.168.1.1",
        },
        {
            "acc_num": "acc789",
            "amt": 10000.0,
            "state": "CA",
            "zip": "90001",
            "city_pop": 500000,
            "trans_num": "t7",
            "unix_time": 1625097700,
            "ip_address": "192.168.1.2",
        },
        {
            "acc_num": "acc789",
            "amt": 12000.0,
            "state": "CA",
            "zip": "90001",
            "city_pop": 500000,
            "trans_num": "t8",
            "unix_time": 1625097701,
            "ip_address": "192.168.1.2",
        },
        {
            "acc_num": "acc101",
            "amt": 11000.0,
            "state": "CA",
            "zip": "90001",
            "city_pop": 500000,
            "trans_num": "t9",
            "unix_time": 1625097702,
            "ip_address": "192.168.1.2",
        },
        {
            "acc_num": "acc102",
            "amt": 100.0,
            "state": "TX",
            "zip": "73301",
            "city_pop": 200000,
            "trans_num": "t10",
            "unix_time": 1625097800,
            "ip_address": "192.168.1.3",
        },
        {
            "acc_num": "acc102",
            "amt": 150.0,
            "state": "TX",
            "zip": "73301",
            "city_pop": 200000,
            "trans_num": "t11",
            "unix_time": 1625097900,
            "ip_address": "192.168.1.3",
        },
    ]

    predicted_rings = process_and_predict(transactions)
    print(json.dumps(predicted_rings, indent=2))
