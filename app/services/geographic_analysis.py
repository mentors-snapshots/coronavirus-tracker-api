from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from folium import plugins
import pandas as pd
from datetime import datetime, timedelta

@dataclass
class GeoLocation:
    latitude: float
    longitude: float
    cases: int
    timestamp: datetime
    location_id: str

class GeographicAnalyzer:
    """Analyzes geographic patterns and clusters of COVID-19 cases"""
    
    def __init__(self, eps_km: float = 100, min_samples: int = 5):
        """
        Initialize the Geographic Analyzer
        
        Args:
            eps_km: The maximum distance (in km) between two points for them to be considered neighbors
            min_samples: The minimum number of points required to form a dense region
        """
        self.eps_km = eps_km
        self.min_samples = min_samples
        
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth
        
        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def identify_clusters(self, locations: List[GeoLocation]) -> Dict[int, List[GeoLocation]]:
        """
        Identify geographic clusters using DBSCAN algorithm
        
        Args:
            locations: List of GeoLocation objects
            
        Returns:
            Dictionary mapping cluster IDs to lists of locations
        """
        if not locations:
            return {}
            
        # Extract coordinates
        coordinates = np.array([(loc.latitude, loc.longitude) for loc in locations])
        
        # Convert eps from km to coordinates (approximately)
        eps = self.eps_km / 111.0  # 1 degree ≈ 111 km
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=self.min_samples, metric='haversine')
        labels = db.fit_predict(coordinates)
        
        # Group locations by cluster
        clusters = {}
        for label, location in zip(labels, locations):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(location)
            
        return clusters
    
    def calculate_spread_vectors(self, 
                               historical_data: List[GeoLocation], 
                               days: int = 7) -> List[Tuple[float, float, float]]:
        """
        Calculate spread vectors based on changes in case concentrations
        
        Args:
            historical_data: List of historical GeoLocation objects
            days: Number of days to analyze
            
        Returns:
            List of (latitude, longitude, magnitude) tuples representing spread vectors
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([
            {
                'latitude': loc.latitude,
                'longitude': loc.longitude,
                'cases': loc.cases,
                'timestamp': loc.timestamp
            }
            for loc in historical_data
        ])
        
        # Calculate daily changes
        vectors = []
        end_date = df['timestamp'].max()
        start_date = end_date - timedelta(days=days)
        
        # Group by location and calculate case changes
        df['date'] = df['timestamp'].dt.date
        start_cases = df[df['timestamp'].dt.date == start_date.date()].set_index(['latitude', 'longitude'])['cases']
        end_cases = df[df['timestamp'].dt.date == end_date.date()].set_index(['latitude', 'longitude'])['cases']
        
        case_changes = end_cases - start_cases
        
        # Create vectors based on case changes
        for (lat, lon) in case_changes.index:
            magnitude = case_changes.get((lat, lon), 0)
            if magnitude != 0:
                vectors.append((lat, lon, magnitude))
                
        return vectors
    
    def generate_risk_heatmap(self, 
                             locations: List[GeoLocation], 
                             center: Optional[Tuple[float, float]] = None) -> folium.Map:
        """
        Generate a risk heatmap based on case density
        
        Args:
            locations: List of GeoLocation objects
            center: Optional center point for the map
            
        Returns:
            Folium map object with heatmap layer
        """
        if not locations:
            raise ValueError("No locations provided for heatmap generation")
            
        # Calculate center point if not provided
        if center is None:
            center = (
                np.mean([loc.latitude for loc in locations]),
                np.mean([loc.longitude for loc in locations])
            )
        
        # Create base map
        m = folium.Map(location=center, zoom_start=4)
        
        # Prepare heatmap data
        heat_data = [
            [loc.latitude, loc.longitude, loc.cases]
            for loc in locations
        ]
        
        # Add heatmap layer
        plugins.HeatMap(heat_data).add_to(m)
        
        return m 