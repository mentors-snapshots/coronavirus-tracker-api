"""app.routers.v2"""
import enum
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query

from ..data import DATA_SOURCES
from ..models import LatestResponse, LocationResponse, LocationsResponse
from ..utils import geo, analysis, visualization

V2 = APIRouter()


class Sources(str, enum.Enum):
    """
    A source available for retrieving data.
    """

    JHU = "jhu"
    CSBS = "csbs"
    NYT = "nyt"


@V2.get("/latest", response_model=LatestResponse)
async def get_latest(
    request: Request, source: Sources = Sources.JHU
):  # pylint: disable=unused-argument
    """
    Getting latest amount of total confirmed cases, deaths, and recoveries.
    """
    locations = await request.state.source.get_all()
    return {
        "latest": {
            "confirmed": sum(map(lambda location: location.confirmed, locations)),
            "deaths": sum(map(lambda location: location.deaths, locations)),
            "recovered": sum(map(lambda location: location.recovered, locations)),
        }
    }


# pylint: disable=unused-argument,too-many-arguments,redefined-builtin
@V2.get("/locations", response_model=LocationsResponse, response_model_exclude_unset=True)
async def get_locations(
    request: Request,
    source: Sources = "jhu",
    country_code: str = None,
    province: str = None,
    county: str = None,
    timelines: bool = False,
):
    """
    Getting the locations.
    """
    # All query paramameters.
    params = dict(request.query_params)

    # Remove reserved params.
    params.pop("source", None)
    params.pop("timelines", None)

    # Retrieve all the locations.
    locations = await request.state.source.get_all()

    # Attempt to filter out locations with properties matching the provided query params.
    for key, value in params.items():
        # Clean keys for security purposes.
        key = key.lower()
        value = value.lower().strip("__")

        # Do filtering.
        try:
            locations = [
                location
                for location in locations
                if str(getattr(location, key)).lower() == str(value)
            ]
        except AttributeError:
            pass
        if not locations:
            raise HTTPException(
                404, detail=f"Source `{source}` does not have the desired location data.",
            )

    # Return final serialized data.
    return {
        "latest": {
            "confirmed": sum(map(lambda location: location.confirmed, locations)),
            "deaths": sum(map(lambda location: location.deaths, locations)),
            "recovered": sum(map(lambda location: location.recovered, locations)),
        },
        "locations": [location.serialize(timelines) for location in locations],
    }


# pylint: disable=invalid-name
@V2.get("/locations/{id}", response_model=LocationResponse)
async def get_location_by_id(
    request: Request, id: int, source: Sources = Sources.JHU, timelines: bool = True
):
    """
    Getting specific location by id.
    """
    location = await request.state.source.get(id)
    
    return {"location": location.serialize(timelines)}


@V2.get("/sources")
async def sources():
    """
    Retrieves a list of data-sources that are availble to use.
    """
    return {"sources": list(DATA_SOURCES.keys())}


@V2.get("/clusters")
async def get_clusters(
    request: Request,
    min_cases: int = Query(5, ge=5),
    radius_km: int = Query(100, ge=1),
):
    """Get location clusters using DBSCAN algorithm.

    Args:
        min_cases: Minimum number of cases for a location to be considered (min: 5)
        radius_km: Maximum distance between points in same cluster (min: 1)
    """
    locations = await request.state.source.get_all()

    # Filter locations with minimum cases
    locations = [loc for loc in locations if loc.latest.confirmed >= min_cases]

    if not locations:
        raise HTTPException(404, detail="No locations found with specified minimum cases")

    # Perform clustering
    labels = geo.dbscan_cluster(locations, eps_km=radius_km, min_samples=min_cases)

    # Group locations by cluster
    clusters = {}
    for loc, label in zip(locations, labels):
        if label >= 0:  # Ignore noise points (-1)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(loc.serialize(timelines=False))

    return {"clusters": clusters}


@V2.get("/spread-vectors")
async def get_spread_vectors(
    request: Request,
    days: int = Query(14, ge=1),
):
    """Calculate spread vectors using historical data.

    Args:
        days: Number of days to analyze (min: 1, default: 14)
    """
    locations = await request.state.source.get_all()

    vectors = {}
    for loc in locations:
        vector = analysis.calculate_spread_vector(loc, days=days)
        if vector:
            vectors[loc.id] = {
                "location": loc.serialize(timelines=False),
                "vector": vector
            }

    if not vectors:
        raise HTTPException(404, detail="No spread vectors could be calculated with specified parameters")

    return {"spread_vectors": vectors}


@V2.get("/heatmap")
async def get_heatmap(
    request: Request,
    center_lat: Optional[float] = Query(None),
    center_lon: Optional[float] = Query(None),
):
    """Generate heatmap visualization of COVID-19 cases.

    Args:
        center_lat: Optional latitude for map center
        center_lon: Optional longitude for map center
    """
    locations = await request.state.source.get_all()

    if not locations:
        raise HTTPException(404, detail="No location data available")

    # Create heatmap
    m = visualization.create_heatmap(locations, center_lat, center_lon)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        m.save(f.name)
        return {"heatmap_path": f.name}
