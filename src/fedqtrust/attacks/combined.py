"""Combined attack client allocation."""

from __future__ import annotations


def combined_attack_sets(total_clients: int) -> tuple[set[int], set[int]]:
    first = int(0.15 * total_clients)
    second = int(0.15 * total_clients)
    label_flip = set(range(first))
    free_rider = set(range(first, first + second))
    return label_flip, free_rider

