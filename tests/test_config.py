"""Test configuration."""
from fastapi import FastAPI
import folium
from folium import plugins
import numpy as np
from sklearn.cluster import DBSCAN

# Create minimal test app without monitoring
app = FastAPI(
    title="Coronavirus Tracker API",
    description="API for tracking Coronavirus (COVID-19) cases",
    version="2.0.0"
)

# Import utility modules
from app.utils import geo, analysis, visualization

# Initialize test routes
@app.get("/test")
async def test_endpoint():
    return {"status": "ok"}
