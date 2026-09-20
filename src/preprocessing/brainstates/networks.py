from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import datasets, image

from .config import NETWORKS, TOKENS, OUTPUT_ROOT, CACHE_ROOT


def load_schaefer_labels(cache_dir: Path) -> list[str]:
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=1000, yeo_networks=7, resolution_mm=2, data_dir=str(cache_dir)
    )
    result = []
    for label in atlas.labels:
        if label not in (b"Background", "Background"):
            if isinstance(label, bytes):
                decoded_label = label.decode()
            else:
                decoded_label = str(label)
            result.append(decoded_label)
    return result


def parse_network_indices(labels: list[str]) -> dict[str, list[int]]:
    indices = dict()
    for net in NETWORKS:
        indices[net] = []

    # Matching parcel with network
    for i, label in enumerate(labels):
        matched = False
        for net, token in TOKENS.items():
            if token in label:
                indices[net].append(i)
                matched = True
                break
        if not matched:
            raise ValueError(f"Parcel {i} ({label}) did not match any network token")

    # Check absent of parcel in networks
    empty = []
    for net, idxs in indices.items():
        if not idxs:
            empty.append(net)
    if empty:
        raise ValueError(f"No parcels found for networks: {empty}")

    # Merged networks
    all_indices = []
    for net in NETWORKS:
        all_indices.extend(indices[net])

    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Overlapping parcel sets across networks")

    if len(all_indices) != len(labels):
        raise ValueError(
            f"Network union covers {len(all_indices)} parcels, expected {len(labels)}"
        )

    return indices


def network_indices(cache_dir: Path) -> dict[str, np.ndarray]:
    labels = load_schaefer_labels(cache_dir)
    idxs = parse_network_indices(labels)
    result = dict()
    for net in NETWORKS:
        result[net] = np.array(idxs[net])
    return result

def network_table(cache_dir: Path) -> pd.DataFrame:
    labels = load_schaefer_labels(cache_dir)
    indices = parse_network_indices(labels)
    network_by_parcel = {}
    for net, idxs in indices.items():
        for i in idxs:
            network_by_parcel[i] = net
    network_df = pd.DataFrame({
            "parcel_index": range(len(labels)),
            "atlas_label": labels,
            "network": [network_by_parcel[i] for i in range(len(labels))]
        })
    return network_df


if __name__ == "__main__":
    labels = load_schaefer_labels(CACHE_ROOT)
    print(len(labels))
    print(labels[0:5])

    indices = parse_network_indices(labels)
    for net, idxs in indices.items():
        print(f"{net}: {len(idxs)} parcels")

    indices = network_indices(CACHE_ROOT)
    for net, arr in indices.items():
        print(f'{net}: shape={arr.shape}, first 5 indices={arr[:5]}')

    table = network_table(CACHE_ROOT)
    print(table.head(5))