"""WorkItem: typed data packet that flows through the pipeline.

A WorkItem replaces the raw dict as the unit of data flowing through the
pipeline, enabling fan-out/fan-in and subprocess serialization.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkItem:
    """Typed data packet that flows through a pipeline independently.

    Attributes:
        id:           Unique identifier (e.g., filename stem).
        attributes:   Key-value data (replaces the raw context dict).
        input_files:  Tagged input file paths.
        output_files: Tagged output file paths.
        parent_id:    For fan-out provenance — ID of the item that generated this one.
        meta:         Scheduler hints, timing, etc.
    """

    id: str
    attributes: Dict[str, Any]
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate_serializable(self) -> None:
        """Raise ValueError if attributes cannot be serialized to JSON."""
        try:
            json.dumps(self.attributes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"WorkItem attributes must be JSON-serializable: {exc}"
            ) from exc

    def to_json(self) -> str:
        """Serialize the entire WorkItem to a JSON string.

        Non-serializable attribute values are converted via ``str()``.
        """
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, data: str) -> WorkItem:
        """Deserialize a WorkItem from a JSON string."""
        d = json.loads(data)
        return cls(**d)

    @staticmethod
    def generate_id(prefix: str = "") -> str:
        """Generate a short unique ID, optionally prefixed."""
        short = uuid.uuid4().hex[:8]
        return f"{prefix}_{short}" if prefix else short
