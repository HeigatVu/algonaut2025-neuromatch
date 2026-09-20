from pathlib import Path
import h5py
import pandas as pd

from .config import SUBJECTS
from .data import common_segments, discover_fmri_segments, validate_fmri_array

def friends_h5(data_root:Path, subject:str) -> Path:
    return next((data_root/subject/"func").glob("*task-friends*.h5"))

def build_audit(data_root:Path) -> pd.DataFrame:
    paths = dict()
    discovered = dict()
    # Save paths
    for subject in SUBJECTS:
        paths[subject] = friends_h5(data_root, subject)
    # print(paths)
    
    # Save segment
    for subject, path in paths.items():
        discovered[subject] = discover_fmri_segments(path)
    # print(discovered)
    
    # Remove unoverlapped segment
    common = common_segments(discovered)
    # print(common)
    rows = []
    for subject, path in paths.items():
        with h5py.File(path, "r") as handle:
            for segment in common:
                name = discovered[subject][segment]
                array = handle[name][()]
                validate_fmri_array(array, f"{subject}/{segment.task}")
                rows.append(
                    {"subject":subject, 
                    "segment": segment.task,
                    "season" : segment.season,
                    "episode" : segment.episode,
                    "split" : segment.split,
                    "n_timepoints": len(array),
                    "n_parcels": array.shape[1],
                    "fmri_path": str(path),
                    "h5_key": name,
                })
    frame = pd.DataFrame(rows).sort_values(["subject", "season", "episode", "split"])
    if len(common) != 287:
        raise ValueError(f"287 common segments, found {len(common)}")
    
    return frame
                
if __name__ == "__main__":
    data_root = Path("data/algonauts_2025.competitors/fmri")
    df = build_audit(data_root)
    print(df)