"""Utilities for visualizing COVID-19 data using Folium."""
from typing import List, Any, Optional
import folium
from folium import plugins


def create_heatmap(
    locations: List[Any],
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None
) -> folium.Map:
    """Generate heatmap visualization of COVID-19 cases.

    Args:
        locations: List of location objects with coordinates and case data
        center_lat: Optional latitude for map center
        center_lon: Optional longitude for map center

    Returns:
        Folium Map object with heatmap layer
    """
    # Use provided center or calculate from data
    if center_lat is None or center_lon is None:
        center_lat = sum(float(loc.coordinates.latitude) for loc in locations) / len(locations)
        center_lon = sum(float(loc.coordinates.longitude) for loc in locations) / len(locations)

    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    # Prepare heatmap data: [lat, lon, weight]
    # Weight is based on confirmed cases
    heat_data = [
        [
            float(loc.coordinates.latitude),
            float(loc.coordinates.longitude),
            loc.latest.confirmed
        ]
        for loc in locations
        if hasattr(loc, 'latest') and hasattr(loc.latest, 'confirmed')
    ]

    # Add heatmap layer with appropriate radius
    plugins.HeatMap(
        heat_data,
        min_opacity=0.4,
        radius=25,
        blur=15,
        max_zoom=1
    ).add_to(m)

    return m
