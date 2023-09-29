from dataclasses import dataclass
from typing import List, Iterable

import numpy as np

from sysvar.utils import read_yaml


class MissingInformationError(Exception):
    pass


@dataclass()
class Correction:

    dependant_variable: str
    custom: bool = False
    systematic: str = None
    MC_production: str = None
    corrections: Iterable = None
    lower_bounds: Iterable = None
    upper_bounds: Iterable = None

    def __post_init__(self):
        if self.custom:
            # If this is a custom correction populate the fields from arguments
            self.values = np.array(self.corrections)
            self.lower_bounds = np.array(self.lower_bounds)
            self.upper_bounds = np.array(self.upper_bounds)
        else:
            # If this is a normal Belle II correction load it from configs
            try:
                info = read_yaml(self.systematic, self.MC_production)
            except TypeError:
                raise MissingInformationError(
                    "Need to specify the systematic effect and the MC production in the positional arguments"
                )

            self.values = info["correction"]
            self.lower_bounds = info["min"]
            self.upper_bounds = info["max"]

    @property
    def value_edges(self) -> np.ndarray:
        return np.unique(np.concatenate((self.lower_bounds, self.upper_bounds)))

    @property
    def value_mids(self) -> np.ndarray:
        return (self.value_edges[1:] + self.value_edges[:-1]) / 2

    @property
    def queries(self):
        return [
            f"{low} <= {self.dependant_variable} < {up}"
            for low, up in zip(self.lower_bounds, self.upper_bounds)
        ]

    @property
    def strings(self) -> List[str]:
        return [
            f"[ {low} - {up} ]" for low, up in zip(self.lower_bounds, self.upper_bounds)
        ]
