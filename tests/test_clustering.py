"""Tests for geographic clustering analysis features."""
import pytest
import numpy as np
from datetime import datetime, timedelta

from app.utils import geo, analysis, visualization

# Mock the folium module to avoid web dependencies
class MockMap:
    def __init__(self, *args, **kwargs):
        pass

    def add_to(self, *args, **kwargs):
        return self

class MockHeatMap:
    def __init__(self, *args, **kwargs):
        pass

    def add_to(self, *args, **kwargs):
        return self

# Patch folium for testing
visualization.folium.Map = MockMap
visualization.plugins.HeatMap = MockHeatMap

def test_haversine_distance():
    """Test distance calculation between coordinates."""
    # Test known distance between two points
    dist = geo.haversine_distance(0, 0, 1, 1)
    assert abs(dist - 157.2) < 0.1  # ~157.2 km

    # Test zero distance
    dist = geo.haversine_distance(10, 20, 10, 20)
    assert dist == 0

    # Test antipodal points
    dist = geo.haversine_distance(0, 0, 0, 180)
    assert abs(dist - 20015.1) < 0.1  # Half Earth's circumference


def test_dbscan_clustering():
    """Test DBSCAN clustering of locations."""
    class MockLocation:
        def __init__(self, lat, lon, cases=100):
            self.coordinates = type('Coordinates', (), {'latitude': lat, 'longitude': lon})
            self.latest = type('Latest', (), {'confirmed': cases})

    # Create test locations
    locations = [
        MockLocation(0, 0),      # Cluster 1
        MockLocation(0.1, 0.1),  # Cluster 1 (~15.7 km from 0,0)
        MockLocation(0.2, 0.2),  # Cluster 1 (~31.4 km from 0,0)
        MockLocation(10, 10),    # Noise point (far from others)
        MockLocation(10.1, 10.1) # Noise point (close to previous but < min_samples)
    ]

    # Test clustering with default parameters
    labels = geo.dbscan_cluster(locations, eps_km=50, min_samples=3)
    assert len(labels) == 5
    assert labels[0] == labels[1] == labels[2]  # First three points in same cluster
    assert labels[3] == labels[4] == -1         # Last two points are noise


def test_spread_vector_calculation():
    """Test spread vector calculation using historical data."""
    class MockTimeline:
        def __init__(self, start_cases, end_cases, days=14):
            self.timeline = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            self.timeline[start_date.isoformat() + 'Z'] = start_cases
            self.timeline[end_date.isoformat() + 'Z'] = end_cases

    class MockLocation:
        def __init__(self, id, start_cases, end_cases):
            self.id = id
            self.timelines = type('Timelines', (), {
                'confirmed': MockTimeline(start_cases, end_cases)
            })

    # Test normal case
    loc = MockLocation(1, 100, 200)  # 100 case increase over 14 days
    vector = analysis.calculate_spread_vector(loc)
    assert vector is not None
    assert vector['start_cases'] == 100
    assert vector['end_cases'] == 200
    assert vector['total_increase'] == 100
    assert abs(vector['daily_increase'] - 7.14) < 0.1  # ~7.14 cases per day


def test_heatmap_generation():
    """Test heatmap visualization generation."""
    class MockLocation:
        def __init__(self, lat, lon, cases):
            self.coordinates = type('Coordinates', (), {'latitude': lat, 'longitude': lon})
            self.latest = type('Latest', (), {'confirmed': cases})

    # Create test locations
    locations = [
        MockLocation(0, 0, 100),
        MockLocation(1, 1, 200),
        MockLocation(-1, -1, 300)
    ]

    # Test with default center
    m = visualization.create_heatmap(locations)
    assert m is not None
    assert isinstance(m, MockMap)

    # Test with custom center
    m = visualization.create_heatmap(locations, center_lat=0, center_lon=0)
    assert m is not None
    assert isinstance(m, MockMap)
