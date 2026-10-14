"""ALFRED date-level history through the FRED API; no invented publication time."""

from pfe.data.fred import collect_fred_series, normalize_fred_observations


def normalize_alfred_observations(
    rows, series_metadata, *, received_at, vintage_timezone="America/New_York"
):
    """Use explicit next-calendar-day vintage availability, reconstructed/PIT False."""
    return normalize_fred_observations(
        rows,
        series_metadata,
        received_at=received_at,
        availability="vintage_next_day",
        vintage_timezone=vintage_timezone,
    )


def collect_alfred_series(transport, store, **options):
    """Collect explicitly bounded real-time history with date-level reconstruction."""
    if options.pop("availability", "vintage_next_day") != "vintage_next_day":
        raise ValueError("ALFRED history requires explicit vintage_next_day convention")
    return collect_fred_series(transport, store, availability="vintage_next_day", **options)
