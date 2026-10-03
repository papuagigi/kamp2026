#!/usr/bin/env python3
"""현재 실행 데이터의 파일 구성·분할 고정·묶음 누수를 확인한다. 원본은 수정하지 않는다."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from xray_config import DATA, ROOT, V2_SPLIT_MD5


def check_dataset(data=DATA):
    with (data / "data.yaml").open() as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or "path" not in config:
        raise ValueError("data.yaml에 데이터 경로가 없습니다")
    yaml_root = ROOT / Path(config["path"])
    if yaml_root.resolve() != data.resolve():
        raise ValueError("data.yaml이 선택한 데이터 폴더와 다릅니다")
    for split in ["train", "val", "test"]:
        if config.get(split) != f"images/{split}":
            raise ValueError(f"data.yaml의 {split} 경로가 현재 분할과 다릅니다")
    man = pd.read_csv(data / "manifest.csv", encoding="utf-8-sig")
    if man.empty or not man.stem.is_unique:
        raise ValueError("manifest가 비어 있거나 사진 이름이 중복됩니다")
    if set(man.split) != {"train", "val", "test"}:
        raise ValueError("학습/검증/평가 분할이 올바르지 않습니다")
    if (man.groupby("burst_id").split.nunique() > 1).any():
        raise ValueError("같은 촬영 묶음이 여러 분할에 있습니다")
    result = {"data_version": data.name, "images": len(man), "split_images": man.split.value_counts().to_dict()}
    if data.name == "xray_v2":
        digest = hashlib.md5((data / "split.csv").read_bytes()).hexdigest()
        if digest != V2_SPLIT_MD5:
            raise ValueError("v2 split.csv가 고정 분할과 다릅니다. 새 분할은 별도 버전으로 관리하세요")
        split = pd.read_csv(data / "split.csv")
        if not man[["stem", "burst_id", "split"]].sort_values("stem").reset_index(drop=True).equals(split.sort_values("stem").reset_index(drop=True)):
            raise ValueError("manifest와 split.csv가 일치하지 않습니다")
        if (man.groupby("md5").split.nunique() > 1).any():
            raise ValueError("같은 원본 사진이 여러 분할에 있습니다")
        result["split_csv_md5"] = digest
    for split, rows in man.groupby("split"):
        expected = set(rows.stem)
        for kind, ext in [("images", "png"), ("labels", "txt")]:
            actual = {p.stem for p in (data / kind / split).glob(f"*.{ext}")}
            if actual != expected:
                raise ValueError(f"{split}/{kind}: manifest와 파일 목록이 일치하지 않습니다")
    return result


if __name__ == "__main__":
    print(json.dumps(check_dataset(), ensure_ascii=False, indent=2))
