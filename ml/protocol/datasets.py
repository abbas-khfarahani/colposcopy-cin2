"""Image, patient-bag, and SSL pair datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Extra / unlabeled JPEGs in the SSL pool are sometimes truncated.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    try:
        image.load()
    except OSError:
        # Truncated file: keep whatever pixels PIL already decoded.
        pass
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def _resolve(root: Path, row: pd.Series) -> Path:
    rel = str(row.get("image_path_rel") or "").strip()
    if rel:
        cand = root / rel
        if cand.is_file():
            return cand
    folder = str(row.get("folder_id") or "")
    file_name = str(row.get("file") or "")
    cand = root / "patients" / folder / file_name
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"Missing image person={row.get('person_id')} file={file_name}")


class AceticImageDataset(Dataset):
    """One acetic image per item."""

    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_root: Path,
        transform: Optional[Callable] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        if sample_weights is None:
            self.weights = [1.0] * len(self.frame)
        else:
            aligned = sample_weights.reindex(self.frame.index)
            self.weights = aligned.fillna(1.0).astype(float).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float, dict[str, Any]]:
        row = self.frame.iloc[index]
        path = _resolve(self.dataset_root, row)
        image = open_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        label = float(int(row["label_cin2plus"]))
        meta = {
            "person_id": str(row["person_id"]),
            "folder_id": str(row["folder_id"]),
            "file": str(row["file"]),
            "path": str(path),
            "image_path_rel": str(row.get("image_path_rel") or ""),
            "person_label": int(row["person_label"]),
            "weight": float(self.weights[index]),
            "n_images": int(row.get("n_images", 0) or 0),
            "hpv_bucket": str(row.get("hpv_bucket") or "missing"),
            "cyto_bucket": str(row.get("cyto_bucket") or "missing"),
            "age": None if pd.isna(row.get("age")) else float(row.get("age")),
        }
        return image, label, meta


class PatientBagDataset(Dataset):
    """One patient (variable-length acetic bag) per item."""

    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_root: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.groups: list[pd.DataFrame] = [
            grp.reset_index(drop=True) for _, grp in frame.groupby("person_id", sort=True)
        ]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        grp = self.groups[index]
        tensors = []
        rels = []
        for _, row in grp.iterrows():
            path = _resolve(self.dataset_root, row)
            image = open_rgb(path)
            if self.transform is not None:
                image = self.transform(image)
            tensors.append(image)
            rels.append(str(row.get("image_path_rel") or row.get("file")))
        images = torch.stack(tensors, dim=0)
        person_label = int(grp["person_label"].max())
        row0 = grp.iloc[0]
        return {
            "images": images,
            "label": float(person_label),
            "person_id": str(row0["person_id"]),
            "n_images": int(len(grp)),
            "image_path_rel": rels,
            "hpv_bucket": str(row0.get("hpv_bucket") or "missing"),
            "cyto_bucket": str(row0.get("cyto_bucket") or "missing"),
            "age": None if pd.isna(row0.get("age")) else float(row0.get("age")),
        }


def collate_images(batch):
    images, labels, metas = zip(*batch)
    return {
        "x": torch.stack(list(images), dim=0),
        "y": torch.tensor(labels, dtype=torch.float32),
        "w": torch.tensor([m["weight"] for m in metas], dtype=torch.float32),
        "meta": list(metas),
    }


def collate_bags(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bags": [b["images"] for b in batch],
        "y": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "meta": batch,
    }


class SSLPairDataset(Dataset):
    """Two augmented views; MICLe may pair a second image from the same person."""

    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_root: Path,
        transform: Callable,
        patient_aware: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.patient_aware = bool(patient_aware)
        self.by_person: dict[str, list[int]] = {}
        for i, pid in enumerate(self.frame["person_id"].astype(str)):
            self.by_person.setdefault(pid, []).append(i)

    def __len__(self) -> int:
        return len(self.frame)

    def _open_row(self, row: pd.Series):
        try:
            image = open_rgb(_resolve(self.dataset_root, row))
            return self.transform(image)
        except (OSError, FileNotFoundError, ValueError) as exc:
            print(
                f"  skip unreadable SSL image person={row.get('person_id')} "
                f"file={row.get('file')}: {exc}",
                flush=True,
            )
            return None

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        pid = str(row["person_id"])
        view1 = self._open_row(row)
        if view1 is None:
            # Walk the person's other files, then the rest of the pool.
            order = list(self.by_person.get(pid, [])) + list(range(len(self.frame)))
            view1 = None
            for j in order:
                view1 = self._open_row(self.frame.iloc[j])
                if view1 is not None:
                    index = j
                    row = self.frame.iloc[j]
                    pid = str(row["person_id"])
                    break
            if view1 is None:
                raise OSError(f"no readable SSL image near index={index}")
        other_idx = index
        if self.patient_aware:
            cands = [j for j in self.by_person[pid] if j != index]
            if cands:
                other_idx = cands[index % len(cands)]
        view2 = self._open_row(self.frame.iloc[other_idx])
        if view2 is None:
            view2 = view1
        return view1, view2, pid


def collate_ssl(batch):
    v1, v2, pids = zip(*batch)
    return torch.stack(list(v1)), torch.stack(list(v2)), list(pids)
