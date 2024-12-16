"""Utilities for analyzing COVID-19 spread patterns and historical data."""
from datetime import datetime, timedelta
from typing import Dict, Optional, Any


def calculate_spread_vector(location: Any, days: int = 14) -> Optional[Dict]:
    """Calculate spread vector using historical data over specified time period.

    Args:
        location: Location object containing timeline data
        days: Number of days to analyze (default: 14)

    Returns:
        Dictionary containing spread vector data including:
        - start_date: ISO formatted start date
        - end_date: ISO formatted end date
        - start_cases: Number of cases at start date
        - end_cases: Number of cases at end date
        - total_increase: Total case increase over period
        - daily_increase: Average daily case increase
        Returns None if timeline data is insufficient
    """
    try:
        timeline = location.timelines.confirmed.timeline
        dates = sorted(timeline.keys())

        if not dates:
            return None

        # Get data points for specified days
        end_date = dates[-1]
        start_date = (datetime.fromisoformat(end_date.rstrip('Z')) -
                     timedelta(days=days)).isoformat() + 'Z'

        # Check if we have enough historical data
        if start_date not in timeline:
            return None

        start_cases = timeline[start_date]
        end_cases = timeline[end_date]

        # Calculate spread metrics
        total_increase = end_cases - start_cases
        daily_increase = total_increase / days

        return {
            'start_date': start_date,
            'end_date': end_date,
            'start_cases': start_cases,
            'end_cases': end_cases,
            'total_increase': total_increase,
            'daily_increase': daily_increase,
            'location_id': location.id
        }
    except (AttributeError, KeyError, ValueError):
        # Handle cases where timeline data is missing or malformed
        return None
