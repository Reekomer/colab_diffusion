from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptAttributes:
  creation_date: datetime
  prompt: str
  negative_prompt: Optional[str]
  process_type: str
  init_image: str
  mask_image: Optional[str]
  strength: Optional[float]
  num_inference_steps:int
  guidance_scale:float
  seed: int
  width: int
  height: int