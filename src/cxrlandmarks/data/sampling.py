"""Train sample ORDER of the paper run, replicated without detectron2.

What detectron2 0.6 ``build_detection_train_loader`` did for the paper run
(``DATALOADER.SAMPLER_TRAIN="TrainingSampler"``, ``SEED=42``, ``IMS_PER_BATCH=8``,
``NUM_WORKERS=16``, ``ASPECT_RATIO_GROUPING=True``, world size 1):

1. ``TrainingSampler(N, seed=42)``: an infinite stream ``randperm(N) ++ randperm(N) ++ ...``
   drawn from ONE ``torch.Generator`` seeded with 42 (epochs continue the generator).
2. ``ToIterableDataset(shard_chunk_size=8)``: worker ``w`` of ``W`` takes stream positions
   ``[w*8 + c :: W*8]`` for ``c in 0..7`` round-robin, i.e. contiguous chunks of 8.
3. ``torch DataLoader`` (batch_size=1, in-order) returns ONE element per worker in turn:
   output ``j`` = worker ``j % W``'s ``(j // W)``-th element.
4. ``AspectRatioGroupedDataset(batch=8)``: buckets by ``width > height``; PAXRay++ images
   are all 512x512 (one bucket), so a batch is 8 consecutive outputs.

Hence batch ``b`` holds the images at stream positions
``pos(j) = (j // (8W)) * 8W + (j % W) * 8 + (j // W) % 8`` for ``j = 8b .. 8b+7``
(see ``d2_output_positions``; positions permute within blocks of 8W).

Augmentation RNG: each loader worker ran ``seed_all_rng((base_seed + w) % 2**31 + w)``
(python ``random``, numpy, torch), ``base_seed`` being one int64 drawn from the MAIN
process torch RNG when the iterator is created -- after model construction, so it is not
recoverable from the config. ``worker_init_fn`` below reproduces that seeding rule for a
torch ``DataLoader`` over ``PaxRayLandmarkDataset``. CutMix ran in the main process with
the main-process python ``random`` (never seeded by the paper script) and numpy (seeded 42
at start, then consumed by nothing else in the data path).
"""
from __future__ import annotations

import itertools
import random
from typing import Iterator

import numpy as np
import torch


def training_index_stream(n: int, seed: int = 42) -> Iterator[int]:
    """detectron2 ``TrainingSampler(n, shuffle=True, seed=seed)`` (world size 1)."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    while True:
        yield from torch.randperm(n, generator=g).tolist()


def d2_output_positions(num_workers: int, chunk: int) -> Iterator[int]:
    """Stream positions in the order the paper's DataLoader emitted them."""
    W = max(1, int(num_workers))
    if W == 1:
        yield from itertools.count()
        return
    for j in itertools.count():
        yield (j // (chunk * W)) * chunk * W + (j % W) * chunk + (j // W) % chunk


def paper_batch_indices(n: int, num_batches: int, *, seed: int = 42, batch_size: int = 8,
                        num_workers: int = 16) -> list[list[int]]:
    """Dataset indices (into the FILTERED train records) of the first ``num_batches`` batches."""
    total = num_batches * batch_size
    positions = list(itertools.islice(d2_output_positions(num_workers, batch_size), total))
    stream = list(itertools.islice(training_index_stream(n, seed), max(positions) + 1))
    flat = [stream[p] for p in positions]
    return [flat[i:i + batch_size] for i in range(0, total, batch_size)]


class PaperOrderSampler(torch.utils.data.Sampler):
    """Infinite sampler yielding dataset indices in the paper's emitted order. Use with a
    ``DataLoader(batch_size=8, num_workers=k)`` -- torch's map-style loader preserves
    sampler order, so batches equal the paper's batches regardless of k."""

    def __init__(self, n: int, *, seed: int = 42, batch_size: int = 8, num_workers: int = 16):
        self.n, self.seed, self.batch_size, self.num_workers = n, seed, batch_size, num_workers

    def __iter__(self):
        span = self.batch_size * max(1, self.num_workers)
        stream = training_index_stream(self.n, self.seed)
        order = list(itertools.islice(d2_output_positions(self.num_workers, self.batch_size), span))
        while True:
            block = [next(stream) for _ in range(span)]
            for pos in order:
                yield block[pos]

    def __len__(self) -> int:  # infinite stream; len is the per-epoch size used for scheduling
        return self.n


def worker_init_fn(worker_id: int) -> None:
    """detectron2 ``worker_init_reset_seed``: reseed python/numpy/torch per worker."""
    seed = torch.initial_seed() % 2 ** 31 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
