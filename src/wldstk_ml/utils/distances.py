"""This module provides distance functions."""

import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculates the l2 distance between two vectors"""
    return np.linalg.norm(x1 - x2)
