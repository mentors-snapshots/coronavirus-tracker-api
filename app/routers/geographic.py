from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
import tempfile
from pathlib import Path

from app.services.geographic_analysis import GeographicAnalyzer, GeoLocation

router = APIRouter()
analyzer = GeographicAnalyzer()

@router.get("/clusters")
async def get_clusters(min_cases: int = 0):
    """Get geographic clusters of COVID-19 cases"""
    try:
        locations = []  # Placeholder for locations
        
        # Filter by minimum cases if specified
        if min_cases > 0:
            locations = [loc for loc in locations if loc.cases >= min_cases]
        
        clusters = analyzer.identify_clusters(locations)
        
        # Convert to serializable format
        return {
            str(cluster_id): [
                {
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "cases": loc.cases,
                    "timestamp": loc.timestamp.isoformat(),
                    "location_id": loc.location_id
                }
                for loc in cluster_locations
            ]
            for cluster_id, cluster_locations in clusters.items()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/spread-vectors")
async def get_spread_vectors(days: int = 7):
    """Get virus spread vectors"""
    try:
        # Fetch historical data
        historical_data = []  # Replace with actual data fetching
        
        vectors = analyzer.calculate_spread_vectors(historical_data, days=days)
        
        return [
            {
                "latitude": lat,
                "longitude": lon,
                "magnitude": mag
            }
            for lat, lon, mag in vectors
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/heatmap")
async def get_heatmap(
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None
):
    """Get COVID-19 risk heatmap"""
    try:
        # Fetch locations
        locations = []  # Replace with actual data fetching
        
        # Set center coordinates if provided
        center = None
        if center_lat is not None and center_lon is not None:
            center = (center_lat, center_lon)
        
        # Generate heatmap
        heatmap = analyzer.generate_risk_heatmap(locations, center=center)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            heatmap.save(tmp.name)
            return {"heatmap_path": tmp.name}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 