"""Shared deterministic schedule and sample-exact accumulation for both arms."""
import hashlib
import json
import random


def epoch_slots(rows: int, seed: int, epoch: int) -> list[int]:
    slots = list(range(rows))
    random.Random(f"alpha-paired-v1:{seed}:{epoch}").shuffle(slots)
    return slots


def schedule_hash(slots: list[int], source_indices: list[int]) -> str:
    payload = [[slot, int(source_indices[slot])] for slot in slots]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def window_samples(microbatch_index: int, rows: int, batch: int, accumulation: int) -> int:
    if min(rows, batch, accumulation) < 1 or not 0 <= microbatch_index * batch < rows:
        raise ValueError("Invalid optimizer-window arguments")
    first = (microbatch_index // accumulation) * accumulation * batch
    return min(rows - first, batch * accumulation)
