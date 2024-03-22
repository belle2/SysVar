from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Iterable, Union

from particle import Particle

import itertools
import numpy as np
from pandas import DataFrame
from uncertainties import unumpy as unp, ufloat

from sysvar.uncertainties import (
    Uncertainty,
    FullyCorrelatedUncertainty,
    FullyCorrelatedUncertaintyInParts,
    UncorrelatedUncertainty,
)
from sysvar.utils import read_yaml

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class MissingInformationError(Exception):
    pass


class UncertaintyWithSameNameExists(Exception):
    pass


@dataclass
class BaseCorrection(ABC):
    systematic: str = None
    MC_production: str = None
    uncertainties: dict = field(default_factory=dict)

    def __post_init__(self):

        try:
            self.info = read_yaml(self.systematic, self.MC_production)
        except TypeError:
            raise MissingInformationError(
                "Need to specify the systematic effect and the MC production in the positional arguments"
            )

    @property
    @abstractmethod
    def strings(self):
        pass

    @property
    @abstractmethod
    def queries(self):
        pass

    @property
    def N(self) -> int:
        return len(self.central_values)

    @property
    def total_error(self) -> np.ndarray:
        if len(self.uncertainties) > 1:
            return np.sqrt(
                np.sum(
                    [np.power(x.errors, 2) for x in self.uncertainties.values()], axis=0
                )
            )
        else:
            return unp.std_devs(self.central_values)

    def add_uncertainty(self, unc: Uncertainty) -> None:

        """
        Add an uncertainty to the Correction.

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


@dataclass
class Correction1D(BaseCorrection):

    dependant_variable: Union[str, None] = None
    central_values: Iterable = None
    lower_bounds: Iterable = None
    upper_bounds: Iterable = None

    def __post_init__(self):
        super().__post_init__()

        self.central_values = self.info["correction"]
        self.dependant_variable = self.info["dependant_variable"]
        self.lower_bounds = self.info["min"]
        self.upper_bounds = self.info["max"]

        # Add the fully correlated uncertainties
        if "fully_correlated" in self.info.keys():
            for unc_name, unc_values in self.info["fully_correlated"].items():
                self.uncertainties.update(
                    {
                        unc_name: FullyCorrelatedUncertainty(
                            unc_name, unc_values, self.strings
                        )
                    }
                )

        if "uncorrelated" in self.info.keys():
            for unc_name, unc_values in self.info["uncorrelated"].items():
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
    def strings(self) -> List[str]:
        return [
            f"[ {low} - {up} ]" for low, up in zip(self.lower_bounds, self.upper_bounds)
        ]

    @property
    def queries(self):
        return [
            f"{low} <= {self.dependant_variable} < {up}"
            for low, up in zip(self.lower_bounds, self.upper_bounds)
        ]


@dataclass
class Correction2DCategorical(BaseCorrection):

    categorical_variable: Union[str, None] = None
    continuus_variable: Union[str, None] = None
    central_values: Iterable = None
    continuus_edges: Iterable = None
    categorical_values: Iterable = None
    categorical_label: str = None

    def __post_init__(self):
        super().__post_init__()

        self.central_values = []
        part_dimensions = []
        for correction in self.info["corrections"]:
            self.central_values.extend(correction)
            part_dimensions.append(len(correction))

        self.categorical_variable = self.info["categorical_variable"]
        self.categorical_values = self.info["categorical_values"]
        self.categorical_label = self.info["categorical_label"]

        self.continuus_variable = self.info["continuus_variable"]
        self.continuus_edges = self.info["continuus_edges"]

        # Add the fully correlated uncertainties
        if "fully_correlated" in self.info.keys():
            for unc_name, unc_lists in self.info["fully_correlated"].items():

                self.uncertainties.update(
                    {
                        unc_name: FullyCorrelatedUncertaintyInParts(
                            unc_name,
                            list(itertools.chain.from_iterable(unc_lists)),
                            self.strings,
                            part_dimensions,
                        )
                    }
                )

        if "uncorrelated" in self.info.keys():
            for unc_name, unc_lists in self.info["uncorrelated"].items():
                self.uncertainties.update(
                    {
                        unc_name: UncorrelatedUncertainty(
                            unc_name,
                            list(itertools.chain.from_iterable(unc_lists)),
                            self.strings,
                        )
                    }
                )

    @property
    def iterator(self):
        return itertools.product(
            self.categorical_values, zip(self.continuus_edges, self.continuus_edges[1:])
        )

    @property
    def strings(self) -> List[str]:
        return [
            f"{self.categorical_label}: {cv} [ {low} - {up} ]"
            for cv, (low, up) in self.iterator
        ]

    @property
    def queries(self):
        return [
            f"{self.categorical_variable} == {cv} & {low} <= {self.continuus_variable} < {up}"
            for cv, (low, up) in self.iterator
        ]


@dataclass
class BFCorrection:

    dependant_variable: Union[str, None] = None
    central_values: Iterable = None
    strings: Iterable = None
    uncertainties: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self.central_values = self._calculate_scaling_ratios()
        self.dependant_variable = info["general"]["dependant_variable"]
        self.string = self._create_strings()
        # Add the uncertainties as fully uncorrelated. This is a choice.
        # Can be very complicated as some of these modes may have been measured
        # by the same experiments, with complicated correlations
        self.uncertainties.update(
            {
                "BF uncertainty": UncorrelatedUncertainty(
                    "BF uncertainty", list(unp.std_devs(corrections)), self.strings
                )
            }
        )

    def _create_strings(self) -> List[str]:

        mother = Particle.from_pdgid(info["general"]["mother_particle"]).latex_name
        daughter_pdgs = [
            x for mode in info["modes"].values() for x in mode["daughters"]
        ]
        strings = []
        for daughter_set in daughter_pdgs:
            daughter_names = [Particle.from_pdgid(x).latex_name for x in daughter_set]
            strings.append(
                rf"${mother} \rightarrow {' '.join(str(x) for x in daughter_names)}$"
            )

        return strings

    def _calculate_scaling_ratios(self):
        pdg_BFs = unp.uarray(
            [x["pdg_live"][0] for x in self.info["modes"].values()],
            [x["pdg_live"][1] for x in self.info["modes"].values()],
        )

        decaydec_BFs = unp.uarray(
            [x["decay_dec"] for x in self.info["modes"].values()],
            [0 for x in self.info["modes"].values()],
        )

        # Safe 0 division. Returns 1+- 0 for the ones where decay.dec = 0
        corrections = np.divide(
            pdg_BFs,
            decaydec_BFs,
            out=unp.uarray(np.ones_like(pdg_BFs), np.zeros_like(pdg_BFs)),
            where=decaydec_BFs != 0,
        )

        return list(unp.nominal_values(corrections))

    @property
    def queries(self):
        return [
            f"{self.dependant_variable} == '{mode['dmID']}'"
            for mode in self.info["modes"].values()
        ]


# #######################################################################################
def add_weights_to_dataframe(
    df: DataFrame,
    correction: type(BaseCorrection),
    weightname: str,
    overwrite: bool = False,
):

    """
    Add weights to a DataFrame based on a correction object.

    Args:
        df (pd.DataFrame): The DataFrame to which weights should be added.
        correction: The correction object containing central values and queries.
        weightname (str): The name of the weight column to add.
        overwrite (bool, optional): Whether to overwrite the weight column if it already exists.

    Returns:
        None

    """

    def _add_weights(df, correction, weightname):

        df.loc[:, weightname] = 1
        for v, q in zip(correction.central_values, correction.queries):
            mask = df.eval(q)
            df.loc[mask, weightname] = v

    if weightname in df.columns and overwrite:
        logging.info("%s exists but it will be overwriten", weightname)

        _add_weights(df, correction, weightname)

    elif weightname in df.columns and not overwrite:

        logging.warning(
            "%s exists but it not will be ovewritten. Skipping. No weights are added. If you want to change this behaviour set the overwrite argument to True",
            weightname,
        )
    elif weightname not in df.columns:
        logging.info("%s does not exist. Adding it to dataframe", weightname)
        _add_weights(df, correction, weightname)


def combine_weights(
    df: DataFrame, new_weight: str, weights: List[str], overwrite: bool = False
):

    if new_weight in df.columns and overwrite:
        logging.info("%s exists but it will be overwritten", new_weight)
        df.loc[:, new_weight] = df[weights].prod(axis=1)

    elif new_weight in df.columns and not overwrite:
        logging.warning(
            "%s exists but it not will be ovewritten. Skipping. No weights are combined If you want to change this behaviour set the overwrite argument to True",
            new_weight,
        )
    elif new_weight not in df.columns:
        logging.info("%s does not exist. Adding it to dataframe", new_weight)
        df.loc[:, new_weight] = df[weights].prod(axis=1)
