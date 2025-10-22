from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss  # type: ignore
import numpy as np

from . import utils


@dataclass
class LatentRecord:
    """Stores a single training latent and associated answers."""
    idx: int
    hidden_state: np.ndarray
    base_answer: str
    base_canonical: str
    gold_answer: str
    gold_canonical: str
    token_count: int


@dataclass
class Neighbor:
    """Holds neighbor metadata returned from kNN search."""
    idx: int
    similarity: float
    base_canonical: str
    gold_canonical: str
    base_answer: str
    gold_answer: str


class LatentIndex:
    """Wraps a FAISS index over latent hidden states."""
    def __init__(
        self,
        records: List[LatentRecord],
        normalize: bool = True,
    ) -> None:
        """Builds the FAISS index from cached records."""
        if not records:
            raise ValueError("Cannot build index with no records.")
        self.normalize = normalize
        self.records = records
        self.dimension = records[0].hidden_state.shape[0]
        self.ids = np.array([rec.idx for rec in records], dtype=np.int64)
        vectors = np.stack([rec.hidden_state for rec in records], axis=0).astype(
            np.float32
        )
        if normalize:
            vectors = np.stack(
                [utils.normalize_vector(vec) for vec in vectors], axis=0
            )
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)

    @staticmethod
    def _records_from_cache(path: Path) -> List[LatentRecord]:
        """Loads latent records from a JSONL cache."""
        records: List[LatentRecord] = []
        for row in utils.read_jsonl(path):
            hidden = np.array(row["hidden_state"], dtype=np.float32)
            records.append(
                LatentRecord(
                    idx=int(row["id"]),
                    hidden_state=hidden,
                    base_answer=row["base_answer"],
                    base_canonical=row["canonical_base"],
                    gold_answer=row["gold_answer"],
                    gold_canonical=row["gold_canonical"],
                    token_count=int(row.get("tokens", 0)),
                )
            )
        return records

    @classmethod
    def from_cache(
        cls, path: Path, normalize: bool = True
    ) -> "LatentIndex":
        """Loads latent records from disk and constructs an index."""
        records = cls._records_from_cache(path)
        return cls(records, normalize=normalize)

    @classmethod
    def load(
        cls,
        cache_path: Path,
        index_path: Path,
        metadata_path: Path,
    ) -> "LatentIndex":
        """Restores a FAISS index and metadata from disk."""
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        metadata = utils.load_json(metadata_path)
        records = cls._records_from_cache(cache_path)
        if not records:
            raise ValueError("No cached records available for index reconstruction.")

        ids = metadata.get("ids")
        if ids is None:
            ids = [rec.idx for rec in records]
        id_to_record = {rec.idx: rec for rec in records}
        ordered_records: List[LatentRecord] = []
        for idx in ids:
            if idx not in id_to_record:
                raise KeyError(f"Record id {idx} referenced in metadata not found.")
            ordered_records.append(id_to_record[idx])

        instance: "LatentIndex" = cls.__new__(cls)
        instance.normalize = bool(metadata.get("normalize", True))
        instance.records = ordered_records
        instance.dimension = int(metadata.get("dimension", ordered_records[0].hidden_state.shape[0]))
        instance.ids = np.array(ids, dtype=np.int64)
        instance.index = faiss.read_index(str(index_path))
        if instance.index.ntotal != len(instance.records):
            raise ValueError(
                f"FAISS index size ({instance.index.ntotal}) does not match records ({len(instance.records)})."
            )
        return instance

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Serializes the FAISS index and metadata to disk."""
        utils.ensure_directory(index_path)
        utils.ensure_directory(metadata_path)
        faiss.write_index(self.index, str(index_path))
        meta_payload = {
            "normalize": self.normalize,
            "dimension": self.dimension,
            "ids": self.ids.tolist(),
        }
        utils.write_json(metadata_path, meta_payload)

    def search(
        self,
        query: np.ndarray,
        k: int,
        exclude_idx: Optional[int] = None,
    ) -> List[Neighbor]:
        """Finds the top-k neighbors for a query latent."""
        if query.ndim != 1:
            raise ValueError("Query vector must be 1-D.")
        query = query.astype(np.float32)
        if self.normalize:
            query = utils.normalize_vector(query)
        # request extra neighbors in case we have to drop the query itself
        search_k = min(len(self.records), k + (1 if exclude_idx is not None else 0))
        distances, indices = self.index.search(query.reshape(1, -1), search_k)
        result: List[Neighbor] = []
        for rank, idx in enumerate(indices[0].tolist()):
            if idx == -1:
                continue
            record = self.records[idx]
            if exclude_idx is not None and record.idx == exclude_idx:
                continue
            sim = float(distances[0][rank])
            if not self.normalize:
                sim = -sim  # convert L2 distance into similarity proxy
            result.append(
                Neighbor(
                    idx=record.idx,
                    similarity=sim,
                    base_canonical=record.base_canonical,
                    gold_canonical=record.gold_canonical,
                    base_answer=record.base_answer,
                    gold_answer=record.gold_answer,
                )
            )
            if len(result) >= k:
                break
        return result

    def __len__(self) -> int:
        """Returns the number of stored latent records."""
        return len(self.records)
