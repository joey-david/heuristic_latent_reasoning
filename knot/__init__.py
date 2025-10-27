"""knot subpackage providing only lightweight FAISS retrieval utilities.

This trimmed __init__ intentionally avoids importing optional modules so that
downstream code can import `knot.retrieval` without pulling in unused deps.
"""

__all__: list[str] = []
