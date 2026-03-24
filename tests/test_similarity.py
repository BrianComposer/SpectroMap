import numpy as np

from spectromap.similarity import compare_representations, symmetric_pseudodistance


def test_zero_distance_for_identical_representations():
    a = np.column_stack([np.linspace(0, 1, 8), np.linspace(-2, 2, 8)])
    d = symmetric_pseudodistance(a, a)
    assert abs(d) < 1e-12


def test_similarity_range():
    a = np.column_stack([np.linspace(0, 1, 8), np.linspace(-2, 2, 8)])
    b = np.column_stack([np.linspace(0, 1, 8), np.linspace(2, -2, 8)])
    distance, similarity = compare_representations(a, b)
    assert distance >= 0
    assert 0 <= similarity <= 1
