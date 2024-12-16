"""Geospatial utilities for location clustering and distance calculations."""
from math import radians, sin, cos, sqrt, atan2
from typing import List, Any
import numpy as np
from sklearn.cluster import DBSCAN


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers using the haversine formula.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Distance between points in kilometers
    """
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def dbscan_cluster(locations: List[Any], eps_km: float = 100, min_samples: int = 5) -> np.ndarray:
    """Perform DBSCAN clustering on locations within specified radius.

    Args:
        locations: List of location objects with coordinates attribute
        eps_km: Maximum distance between points in same cluster (kilometers)
        min_samples: Minimum number of samples in a cluster

    Returns:
        Array of cluster labels (-1 indicates noise points)
    """
    # Extract coordinates
    coords = np.array([[float(loc.coordinates.latitude), float(loc.coordinates.longitude)]
                      for loc in locations])

    # Create distance matrix using haversine
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i,j] = haversine_distance(
                coords[i,0], coords[i,1],
                coords[j,0], coords[j,1]
            )

    # Perform DBSCAN clustering
    clustering = DBSCAN(
        eps=eps_km,
        min_samples=min_samples,
        metric='precomputed'
    )
    labels = clustering.fit_predict(distances)

    return labels
