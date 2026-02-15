"""Recommendation engine — re-exports from summary module.

Recommendations are generated as part of the summary module.
This file exists for organizational clarity.
"""

from afterburn.report.summary import generate_recommendations

__all__ = ["generate_recommendations"]
