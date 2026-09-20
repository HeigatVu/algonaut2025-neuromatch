from pathlib import Path

import pandas as pd

import h5py
from preprocessing.brainstates.data import parse_h5_key

MELD_URLS = {
    "train": "https://raw.githubusercontent.com/HeigatVu/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/HeigatVu/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/HeigatVu/MELD/master/data/MELD/test_sent_emo.csv",
}


def download_meld(cache_dir: Path) -> dict[str, Path]:
    """ 
    Download MELD dataset
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in MELD_URLS.items():
        dest = cache_dir / f"{split}_sent_emo.csv"
        if not dest.exists():
            df = pd.read_csv(url)
            df.to_csv(dest, index=False)
        paths[split] = dest
    return paths


def parse_timestamp(ts: str) -> float:
    """ 
    Parse MELD timestamp 'HH:MM:SS,mmm' to seconds
    """
    ts = ts.strip()
    main, ms = ts.split(",")
    parts = main.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s + int(ms) / 1000


def load_meld(cache_dir: Path) -> pd.DataFrame:
    """
    Load MELD dataset from cache directory
    """
    paths = download_meld(cache_dir)
    frames = []
    for split, path in paths.items():
        df = pd.read_csv(path)
        df["split_name"] = split
        frames.append(df)
    meld = pd.concat(frames, ignore_index=True)
    meld["start_s"] = meld["StartTime"].apply(parse_timestamp)
    meld["end_s"] = meld["EndTime"].apply(parse_timestamp)
    return meld


def filter_friends(meld: pd.DataFrame, max_season: int = 6) -> pd.DataFrame:
    """
    Filter MELD dataset to include only episodes from the first `max_season` seasons
    """
    return meld[meld["Season"] <= max_season].copy()


def compute_split_points(data_root: Path, subject: str = "sub-01") -> dict[tuple[int, int], float]:
    """
    Compute episode split points from algonauts segment durations.
    """

    h5_path = next((data_root / subject / "func").glob("*task-friends*.h5"))
    split_points = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            seg = parse_h5_key(key)
            if seg.split == "a":
                dur_s = f[key].shape[0] * 1.49
                split_points[(seg.season, seg.episode)] = dur_s
    return split_points


def assign_segment(
    meld: pd.DataFrame,
    split_points: dict[tuple[int, int], float],
) -> pd.DataFrame:
    """Assign each MELD utterance to an algonauts segment (a or b).

    Adds columns: 'segment_split' ('a' or 'b'), 'segment_start_s' (offset within segment).
    """

    splits = []
    offsets = []
    for _, row in meld.iterrows():
        key = (int(row["Season"]), int(row["Episode"]))
        if key not in split_points:
            splits.append(None)
            offsets.append(None)
            continue
        split_t = split_points[key]
        if row["start_s"] < split_t:
            splits.append("a")
            offsets.append(row["start_s"])
        else:
            splits.append("b")
            offsets.append(row["start_s"] - split_t)
    meld = meld.copy()
    meld["segment_split"] = splits
    meld["segment_start_s"] = offsets
    return meld

if __name__ == "__main__":
    print("Loading MELD dataset...")
    meld = load_meld(Path("./MELD"))
    print(meld.head())
