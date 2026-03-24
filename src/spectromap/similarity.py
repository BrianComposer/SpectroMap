from __future__ import annotations

import math
import numpy as np


def distance_function(point_a: np.ndarray, point_b: np.ndarray) -> float:
    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)

    if point_a.shape != point_b.shape:
        raise ValueError("Points must have the same dimensionality.")

    diff = point_a - point_b
    # return float(np.dot(diff, diff))  # squared euclidean
    return float(np.linalg.norm(diff))


def sorting_function(
    num_points: int,
    num_centroids: int,
    i: int,
    j: int,
    sigma_ratio: float = 1 / 6,
) -> float:
    if num_points <= 1 and num_centroids <= 1:
        return 1.0

    sigma = max(1e-6, sigma_ratio * max(num_points, num_centroids))

    if num_centroids <= 1:
        expected = 0.0
    else:
        expected = (num_points - 1) * j / (num_centroids - 1)

    exponent = -0.5 * ((i - expected) ** 2) / (sigma ** 2)
    return math.exp(exponent)


def calculate_pseudodistance(
    points: np.ndarray,
    centroids: np.ndarray,
    sigma_ratio: float = 1 / 6,
) -> float:
    """
    Stable, normalized pseudodistance.
    """

    points = np.asarray(points, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)

    if points.ndim != 2 or centroids.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")

    if points.shape[1] != centroids.shape[1]:
        raise ValueError("Dimensionality mismatch.")

    n = len(points)
    m = len(centroids)

    if n == 0 or m == 0:
        return float("inf")

    weighted_sum = 0.0
    weight_total = 0.0

    for i in range(n):
        for j in range(m):
            w = sorting_function(n, m, i, j, sigma_ratio)
            d = distance_function(points[i], centroids[j])
            weighted_sum += w * d
            weight_total += w

    if weight_total < 1e-12:
        return float("inf")

    #NORMALIZACIÓN CRÍTICA
    distance = weighted_sum / weight_total
    # normalización adicional
    distance = distance / (1.0 + distance)

    return float(distance)


def symmetric_pseudodistance(
    a: np.ndarray,
    b: np.ndarray,
    sigma_ratio: float = 1 / 6,
) -> float:
    d_ab = calculate_pseudodistance(a, b, sigma_ratio)
    d_ba = calculate_pseudodistance(b, a, sigma_ratio)

    return 0.5 * (d_ab + d_ba)


# mapping robusto a [0,1]
def distance_to_similarity(distance: float, gamma: float = 1.0) -> float:
    if math.isinf(distance) or math.isnan(distance):
        return 0.0

    similarity = 1.0 / (1.0 + gamma * distance)

    # clamp defensivo
    return max(0.0, min(1.0, similarity))


def compare_representations(
    a: np.ndarray,
    b: np.ndarray,
    sigma_ratio: float = 1 / 6,
    gamma: float = 1.0,
) -> tuple[float, float]:

    distance = symmetric_pseudodistance(a, b, sigma_ratio)

    similarity = distance_to_similarity(distance, gamma)

    return distance, similarity


# from __future__ import annotations
#
# import math
# import numpy as np
#
#
# def distance_function(point_a: np.ndarray, point_b: np.ndarray) -> float:
#     point_a = np.asarray(point_a, dtype=np.float64)
#     point_b = np.asarray(point_b, dtype=np.float64)
#     if point_a.shape != point_b.shape:
#         raise ValueError("Points must have the same dimensionality.")
#     diff = point_a - point_b
#     return float(np.dot(diff, diff))
#
#
# def sorting_function(num_points: int, num_centroids: int, i: int, j: int, sigma_ratio: float = 1 / 6) -> float:
#     if num_points <= 1 and num_centroids <= 1:
#         return 1.0
#
#     sigma = max(1e-6, sigma_ratio * max(num_points, 1))
#     if num_centroids <= 1:
#         expected = 0.0
#     else:
#         expected = (num_points - 1) * j / (num_centroids - 1)
#
#     exponent = -0.5 * ((i - expected) ** 2) / (sigma ** 2)
#     return math.exp(exponent)
#
#
# def calculate_pseudodistance(
#     points: np.ndarray,
#     centroids: np.ndarray,
#     sigma_ratio: float = 1 / 6,
# ) -> float:
#     """
#     Revised version of the original custom pseudodistance.
#
#     The original idea is preserved:
#     - compare every point to every centroid
#     - weight comparisons with a Gaussian positional kernel
#
#     Improvements:
#     - all points are used
#     - all dimensions are used
#     - empty and one-point edge cases are handled
#     - normalization is numerically stable
#     """
#     points = np.asarray(points, dtype=np.float64)
#     centroids = np.asarray(centroids, dtype=np.float64)
#
#     if points.ndim != 2 or centroids.ndim != 2:
#         raise ValueError("Inputs must be 2D arrays.")
#     if points.shape[1] != centroids.shape[1]:
#         raise ValueError("Point dimensionality mismatch.")
#
#     num_points = len(points)
#     num_centroids = len(centroids)
#
#     if num_points == 0 or num_centroids == 0:
#         return float("inf")
#
#     weighted_sum = 0.0
#     weight_total = 0.0
#
#     for i in range(num_points):
#         for j in range(num_centroids):
#             weight = sorting_function(num_points, num_centroids, i, j, sigma_ratio=sigma_ratio)
#             weighted_sum += weight * distance_function(points[i], centroids[j])
#             weight_total += weight
#
#     if weight_total == 0:
#         return float("inf")
#
#     return weighted_sum / weight_total
#
#
# def symmetric_pseudodistance(
#     a: np.ndarray,
#     b: np.ndarray,
#     sigma_ratio: float = 1 / 6,
# ) -> float:
#     d_ab = calculate_pseudodistance(a, b, sigma_ratio=sigma_ratio)
#     d_ba = calculate_pseudodistance(b, a, sigma_ratio=sigma_ratio)
#     return 0.5 * (d_ab + d_ba)
#
#
# def distance_to_similarity(distance: float, gamma: float = 1.0) -> float:
#     if math.isinf(distance):
#         return 0.0
#     return 1.0 / (1.0 + gamma * distance)
#
#
# def compare_representations(
#     a: np.ndarray,
#     b: np.ndarray,
#     sigma_ratio: float = 1 / 6,
#     gamma: float = 1.0,
# ) -> tuple[float, float]:
#     distance = symmetric_pseudodistance(a, b, sigma_ratio=sigma_ratio)
#     similarity = distance_to_similarity(distance, gamma=gamma)
#     return distance, similarity
