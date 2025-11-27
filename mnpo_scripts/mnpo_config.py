from dataclasses import dataclass, field
from scripts.simpo_config import SimPOConfig
from typing import List, Optional, Tuple, Union


@dataclass
class MNPOConfig(SimPOConfig):
    """
    Configuration class for MNPOTrainer.
    """
    # The mixing ratio between the reference model and the historical models.
    ratio: float = 0.3333
    # The eta parameter for the MNPO loss.
    eta: float = 0.0075
    beta: float = 10
    # The maximum number of historical models to consider.
    max_history_t: int = 2
    # The weights for historical models.
    weights: List[float] = field(default_factory=lambda: [1.0, 0.0]) # [t-1, t-2 ....]
    het_loss_weight: float = 0.0