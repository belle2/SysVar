from dataclasses import dataclass, field
from typing import List, Iterable

import numpy as np

from sysvar.uncertainties import (
    Uncertainty,
    FullyCorrelatedUncertainty,
    UncorrelatedUncertainty,
)
from sysvar.utils import read_yaml


class MissingInformationError(Exception):
    pass


class UncertaintyWithSameNameExists(Exception):
    pass


@dataclass()
class Correction:

    dependant_variable: str
    custom: bool = False
    systematic: str = None
    MC_production: str = None
    central_values: Iterable = None
    lower_bounds: Iterable = None
    upper_bounds: Iterable = None
    uncertainties: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.custom:
            # If this is a custom correction populate the fields from arguments
            self.values = np.array(self.centra_values)
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
            # Add the fully correlated uncertainties
            for unc_name, unc_values in info["fully_correlated"].items():
                self.uncertainties.update(
                    {
                        unc_name: FullyCorrelatedUncertainty(
                            unc_name, unc_values, self.strings
                        )
                    }
                )
            for unc_name, unc_values in info["uncorrelated"].items():
                self.uncertainties.update(
                    {
                        unc_name: UncorrelatedUncertainty(
                            unc_name, unc_values, self.strings
                        )
                    }
                )

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

    def add_uncertainty(self, unc: Uncertainty) -> None:

        """
        Add an uncertainty to the Variator.

        Args:
            unc (Uncertainty): The uncertainty to be added.

        Raises:
            UncertaintyWithSameNameExists: If uncertainty with the same name has already been added to the variator.

        """
        if unc.name in self.uncertainties.keys():
            raise UncertaintyWithSameNameExists(
                f"An uncertainty with the name {unc.name} already exist in the set of uncertainties that that have been added to the Correction. Make sure that you add a specific uncertainty only once, and that there are no duplicate names"
            )
        else:
            self.uncertainties.update({unc.name: unc})
