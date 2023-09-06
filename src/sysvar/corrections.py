from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Correction:

    values: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    def build_strings(self) -> List[str]:
        return [
            f"[ {low} - {up} ]" for low, up in zip(self.lower_bounds, self.upper_bounds)
        ]
