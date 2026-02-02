"""
Data utilities for UnifyWeaver.

Modules:
    wikipedia_categories: Wikipedia category lookup and hierarchy bridge
"""

from .wikipedia_categories import (
    WikipediaCategoryBridge,
    get_category_bridge,
)

__all__ = [
    'WikipediaCategoryBridge',
    'get_category_bridge',
]
