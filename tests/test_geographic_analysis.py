import pytest
from datetime import datetime, timedelta
import numpy as np
from app.services.geographic_analysis import GeographicAnalyzer, GeoLocation

@pytest.fixture
def analyzer():
    return GeographicAnalyzer(eps_km=100, min_samples=2)

@pytest.fixture
def sample_locations():
    """Create a sample dataset of locations"""
    base_time = datetime.now()
    return [
        GeoLocation(40.7128, -74.0060, 100, base_time, "nyc"),  # New York
        GeoLocation(40.7614, -73.9776, 150, base_time, "manhattan"),  # Manhattan
        GeoLocation(34.0522, -118.2437, 200, base_time, "la"),  # Los Angeles
        GeoLocation(51.5074, -0.1278, 80, base_time, "london"),  # London
    ]

@pytest.fixture
def historical_data():
    """Create historical data for testing spread vectors"""
    locations = []
    base_time = datetime.now()
    
    # Add data for multiple days
    for days in range(7):
        current_time = base_time - timedelta(days=days)
        locations.extend([
            GeoLocation(40.7128, -74.0060, 100 + days*10, current_time, "nyc"),
            GeoLocation(40.7614, -73.9776, 150 + days*15, current_time, "manhattan"),
            GeoLocation(34.0522, -118.2437, 200 + days*20, current_time, "la"),
        ])
    
    return locations

def test_haversine_distance(analyzer):
    """Test the haversine distance calculation"""
    # New York to Los Angeles
    distance = analyzer._haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3935 <= distance <= 3945  # Approximately 3940 km

def test_identify_clusters_empty(analyzer):
    """Test cluster identification with empty input"""
    clusters = analyzer.identify_clusters([])
    assert clusters == {}

def test_identify_clusters(analyzer, sample_locations):
    """Test cluster identification with sample data"""
    clusters = analyzer.identify_clusters(sample_locations)
    
    # New York and Manhattan should be in the same cluster
    ny_cluster = None
    for cluster_id, locations in clusters.items():
        if any(loc.location_id == "nyc" for loc in locations):
            ny_cluster = cluster_id
            break
    
    assert ny_cluster is not None
    cluster_locations = clusters[ny_cluster]
    assert any(loc.location_id == "manhattan" for loc in cluster_locations)
    
    # Los Angeles and London should be in different clusters
    la_cluster = None
    london_cluster = None
    for cluster_id, locations in clusters.items():
        if any(loc.location_id == "la" for loc in locations):
            la_cluster = cluster_id
        if any(loc.location_id == "london" for loc in locations):
            london_cluster = cluster_id
    
    assert la_cluster != london_cluster

def test_calculate_spread_vectors(analyzer, historical_data):
    """Test spread vector calculation"""
    vectors = analyzer.calculate_spread_vectors(historical_data)
    
    assert len(vectors) > 0
    for vector in vectors:
        assert len(vector) == 3  # lat, lon, magnitude
        assert isinstance(vector[0], float)  # latitude
        assert isinstance(vector[1], float)  # longitude
        assert isinstance(vector[2], float)  # magnitude

def test_generate_risk_heatmap(analyzer, sample_locations):
    """Test heatmap generation"""
    heatmap = analyzer.generate_risk_heatmap(sample_locations)
    assert heatmap is not None
    
    # Test with empty locations
    with pytest.raises(ValueError):
        analyzer.generate_risk_heatmap([])
    
    # Test with custom center
    custom_center = (0.0, 0.0)
    heatmap = analyzer.generate_risk_heatmap(sample_locations, center=custom_center)
    assert heatmap is not None

def test_edge_cases(analyzer):
    """Test edge cases and error handling"""
    # Single location
    single_location = [GeoLocation(0.0, 0.0, 100, datetime.now(), "single")]
    clusters = analyzer.identify_clusters(single_location)
    assert len(clusters) == 1
    
    # Locations at same point
    same_point_locations = [
        GeoLocation(0.0, 0.0, 100, datetime.now(), "point1"),
        GeoLocation(0.0, 0.0, 200, datetime.now(), "point2"),
    ]
    clusters = analyzer.identify_clusters(same_point_locations)
    assert len(clusters) == 1 