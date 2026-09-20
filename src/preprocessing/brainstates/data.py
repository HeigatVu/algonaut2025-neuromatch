from multiprocessing.sharedctypes import Value
from dataclasses import dataclass
from pathlib import Path
import re
from collections.abc import Mapping
import numpy as np

import h5py

TASK = re.compile(r"_task-s(?P<season>\d{2})e(?P<episode>\d{2})(?P<split>[a-z])$")


@dataclass(frozen=True, order=True)
class SegmentKey:
    season: int
    episode: int
    split: str

    @property
    def task(self) -> str:
        return f"s{self.season:02d}e{self.episode:02d}{self.split}"


def parse_h5_key(name: str) -> SegmentKey:
    match = TASK.search(name)
    return SegmentKey(int(match["season"]), int(match["episode"]), match["split"])


def discover_fmri_segments(path: Path) -> dict[SegmentKey, str]:
    with h5py.File(path, "r") as handle:
        result = dict()
        for name in handle.keys():
            result[parse_h5_key(name)] = name
        return result

def common_segments(by_subject:Mapping[str, dict[SegmentKey, str]]) -> tuple[SegmentKey]:
    sets_data_h5 = []
    for items in by_subject.values():
        sets_data_h5.append(set(items))
    
    return tuple(sorted(set.intersection(*sets_data_h5)))
    
def validate_fmri_array(array: np.ndarray, label: str) -> None:
    if array.ndim != 2 or array.shape[1] != 1000:
        raise ValueError(f"{label}: expected timepoints x 1000, got {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{label}: no timepoints")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite values")


if __name__ == "__main__":
    a = SegmentKey(1, 1, 'a')
    b = SegmentKey(1, 1, 'b')
    print(a.task)
    print(a < b )

    key = parse_h5_key("ses-003_task-s01e01a")
    print(key)
    print(key.season)

    h5_path_subject1 = Path("data/algonauts_2025.competitors/fmri/sub-01/func/sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5")
    h5_path_subject2 = Path("data/algonauts_2025.competitors/fmri/sub-02/func/sub-02_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5")
    h5_path_subject3 = Path("data/algonauts_2025.competitors/fmri/sub-03/func/sub-03_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5")
    h5_path_subject5 = Path("data/algonauts_2025.competitors/fmri/sub-05/func/sub-05_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5")
    segments1 = discover_fmri_segments(h5_path_subject1)
    print(len(segments1.keys()))
    segments2 = discover_fmri_segments(h5_path_subject2)
    print(len(segments2.keys()))
    segments3 = discover_fmri_segments(h5_path_subject3)
    print(len(segments3.keys()))
    segments5 = discover_fmri_segments(h5_path_subject5)
    print(len(segments5.keys()))

    