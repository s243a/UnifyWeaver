"""Shared utility functions for Python runtime."""


def get_local(data, local_name):
    """Get value from data dict ignoring namespace prefixes.

    Tries: exact match, @prefix, and any namespace:localname pattern.
    Examples: 'title', '@title', 'dcterms:title', '@dcterms:title'

    Args:
        data: Dictionary containing attributes
        local_name: The local name to search for (without namespace)

    Returns:
        The value if found, None otherwise
    """
    if local_name in data:
        return data[local_name]
    if f'@{local_name}' in data:
        return data[f'@{local_name}']
    suffix = f':{local_name}'
    for key in data:
        if key.endswith(suffix):
            return data[key]
    return None
