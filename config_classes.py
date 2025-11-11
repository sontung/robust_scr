from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Type, Iterator

import torch
from torch.utils.data import Sampler


class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)


@dataclass
class CoordLossConfig(PrintableConfig):

    target_name: str = "gt_coords"
    tolerance: float = 0.1
    step_ratio: float = 0.5
    max_weight: float = 1.0


@dataclass
class BatchRandomSamplerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: BatchRandomSampler)

    batch_size: int = 5120


class BatchRandomSampler(Sampler[torch.Tensor]):
    dataset_size: int
    batch_size: int
    generator: torch.Generator

    def __init__(
        self,
        config: BatchRandomSamplerConfig,
        dataset_size: int,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> None:
        self.dataset_size = dataset_size
        self.batch_size = config.batch_size
        self.generator = generator if generator is not None else torch.Generator()

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield from torch.randperm(
                self.dataset_size,
                generator=self.generator,
                device=self.generator.device,
            ).split(self.batch_size)[:-1]


class RepeatSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.
        Good for caching.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, dataset_size, num_repeats) -> None:

        self.dataset_size = dataset_size
        self.num_repeats = num_repeats

    def __iter__(self) -> Iterator[int]:
        for i in range(self.dataset_size):
            for _ in range(self.num_repeats):
                yield i

    def __len__(self) -> int:
        return self.dataset_size * self.num_repeats


class PQKNN:
    """KNN search using product quantization."""

    def __init__(self, pq, codes, n_neighbors=10, device="cuda"):
        self.n_neighbors = n_neighbors
        self.device = device
        self.M = pq.M
        self.Ds = pq.Ds
        self.codes = torch.from_numpy(codes).to(torch.int64).to(device)
        self.codewords = torch.from_numpy(pq.codewords).to(device)
        self.row_idx = (
            torch.arange(self.M, dtype=torch.int64, device=self.codes.device)
            .unsqueeze(0)
            .expand_as(self.codes)
        )

    def kneighbors(self, query):
        query_gpu = torch.from_numpy(query).to(self.device)
        diff = query_gpu.view(self.M, 1, self.Ds) - self.codewords
        dtable = torch.einsum("ijk,ijk->ij", diff, diff)
        dists = dtable[self.row_idx, self.codes]
        dists = torch.sum(dists.reshape(-1, self.M), dim=1)
        _, indices = torch.topk(dists, self.n_neighbors, largest=False)
        return indices
