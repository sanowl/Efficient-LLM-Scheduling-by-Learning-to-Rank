from dataclasses import dataclass
from typing import Optional

@dataclass
class Request:
    prompt: str
    arrival_time: float
    score: Optional[float] = None
    priority: bool = False
    starvation_count: int = 0
    quantum: int = 0
    output_length: Optional[int] = None  # Actual output length, used for training
