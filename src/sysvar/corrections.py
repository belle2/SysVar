from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Iterable, Union, Optional
from os import path

from particle import Particle

import itertools
import numpy as np
from pandas import DataFrame, concat, read_csv
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

        self.extra_variables = self.info["extra_variables"]

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
            f"{self.categorical_variable} == {cv} & {low} <= {self.continuus_variable} < {up} & {self._get_extra_cut()}"
            for cv, (low, up) in self.iterator
        ]

    def _get_extra_cut(self):
        return self.info["extra_cut"]


@dataclass
class CorrectionBF:

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


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Iterable, Union, Optional
from os import path

from particle import Particle

import itertools
import numpy as np
from pandas import DataFrame, concat, read_csv
from uncertainties import unumpy as unp, ufloat


class CorrectionTableFinder:

    """
    Factory method class to get correction tables for kaons,  pions, electrons and muons
    """

    def __init__(
        self,
        particle_species,
        online_cut,
        base_table_path,
        variable
    ):
        self.particle_species = particle_species
        self.online_cut = online_cut
        self.base_table_path = base_table_path
        self.variable = variable

        self.true_pdg = (self.particle_species_settings[particle_species]["true_pdgs"],)
        self.fake_pdg = (self.particle_species_settings[particle_species]["fake_pdgs"],)

        self.value = self.get_cut_value()
        self.cut_type = self.get_cut_type()

        efficiency_table_names = self.build_table_name(
            self.particle_species_settings[self.particle_species]["eff_table_ids"]
        )
        fake_rate_table_names = self.build_table_name(
            self.particle_species_settings[self.particle_species]["fake_rate_table_ids"]
        )

        self.eff_table = self.get_table(efficiency_table_names)
        self.fake_table = self.get_table(fake_rate_table_names)


    @classmethod
    def kaons(cls, external_info):

        particle_species = "kaon"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable = None
        )

    @classmethod
    def kaons(cls, external_info):

        particle_species = "kaon"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable = None
        )

    @classmethod
    def pions(cls, external_info):

        particle_species = "pion"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable = None
        )

    @classmethod
    def electrons(cls, external_info):

        particle_species = "elec"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable = external_info["variable"]
        )

    @classmethod
    def muons(cls, external_info):

        particle_species = "muon"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable = external_info["variable"]
        )

    @property
    def particle_species_settings(self) -> dict:

        return {
            "kaon": {
                "true_pdgs": [321],
                "fake_pdgs": [211],
                "eff_table_ids": ["keff"],
                "fake_rate_table_ids": ["piFk"],
            },
            "pion": {
                "true_pdgs": 211,
                "fake_pdgs": 321,
                "eff_table_ids": ["pieff"],
                "fake_rate_table_ids": ["kFpi"],
            },
            "elec": {
                "true_pdgs": [11],
                "fake_pdgs": [321, 211],
                "eff_table_ids": ["e_efficiency"],
                "fake_rate_table_ids": [
                    "K_e_fakeRate",
                    "pi_e_fakeRate",
                ],
            },
            "muon": {
                "true_pdgs": [13],
                "fake_pdgs": [321, 211],
                "eff_table_ids": ["mu_efficiency"],
                "fake_rate_table_ids": [
                    "K_mu_fakeRate",
                    "pi_mu_fakeRate",
                ],
            },
        }

    def get_cut_type(
        self,
    ) -> str:
        """Reads the yaml configuration file and extracts the cut type that have been applied in the online reconstuction

        Args:
        species: Particle species, should be K+ or pi+

        Returns:
        the cut type that has been applied online. Binary or global
        """
        # Read the online selections that have been applied on the online reconstruction
        # TODO update this to the config file of each experiment to avoid making the mistake of
        # changing the value during the offline preproccesing

        if self.particle_species in ["muon", "elec"]:
            cut_type = self.online_cut

        elif self.particle_species in ["kaon", "pion"]:
            if "binaryPID" in self.online_cut:
                cut_type = "B"

            elif "ID" in self.online_cut:
                cut_type = "G"
            else:
                logging.warning(
                    "Cut type, neither global, nor binary HID selection has been applied online"
                )
        else:
            raise ValueError("Wrong particle species")

        return cut_type

    def get_cut_value(self) -> str:
        """Reads the yaml configuration file and extracts the cut type that have been applied in the online reconstuction

        Args:
        species: Particle species, should be K+ or pi+

        Returns:
        the cut type that has been applied online. Binary or global
        """
        # Read the online selections that have been applied on the online reconstruction
        # TODO update this to the config file of each experiment to avoid making the mistake of
        # changing the value during the offline preproccesing
        if self.particle_species in ["muon", "elec"]:
            cut_value = self.online_cut[-1]

        elif self.particle_species in ["kaon", "pion"]:
            cut_value = self.online_cut[-3:]

        return self.online_cut[-1]

    def build_table_name(
        self,
        table_ids: str,
    ) -> list:
        """Builds the efficiency and fake rate tables path names

        Args:
        table_ids: efficiency or fake table id

        Returns:
        list with the efficiency or fake rate table file names
        """
        # Create the file names.
        # These are both for plus and minus
        if self.particle_species in ["kaon", "pion"]:
            # First build the names for positive charge
            file_names = [self.build_hid_table_name(x, "p") for x in table_ids]
            # Now add thhe names for negative charge
            file_names.extend([self.build_hid_table_name(x, "m") for x in table_ids])

        elif self.particle_species in ["elec", "muon"]:
            file_names = ["_".join((x, "table.csv")) for x in table_ids]

        return [path.join(self.base_table_path, x) for x in file_names]

    def build_hid_table_name(self, table_id, charge):
        return "_".join(
            (
                "Rdtmc",
                table_id,
                charge,
                self.cut_type + "0-" + str(self.value)[-1],
                "all.log",
            )
        )

    def get_table(self, table_names):

        if self.particle_species in ["kaon", "pion"]:
            table = concat([read_csv(x) for x in table_names])
            table = table.reset_index(drop=True)
            self.make_pidvar_compatible(table, max_uncertainty=10)

        elif self.particle_species in ["elec", "muon"]:
            table = concat([read_csv(x) for x in table_names])
            table = table.query(self.get_lid_queries())

        return table

    @staticmethod
    def make_pidvar_compatible(
        table: DataFrame, max_uncertainty: Optional[float] = 1e2
    ):
        """
        Convert the pandas dataframes obtained Hadron ID CSV tables via into a
        format consistent with the format of the lepton ID tables which ``PIDvar``
        understands.

        In particular, convert the ``charge`` column from ``1``/``-1`` integer
        entries to ``+``/``-`` string entries and calculate the
        ``theta_min``/``theta_max`` columns.

        :param table: Pandas dataframe obtained from ``pandas.read_csv`` on HID table

        :param inplace: Whether to modify the existing dataframe in place.
            Otherwise, a copy of the existing dataframe will be returned.

        :param max_uncertainty: Drop rows in HID tables where any of the data-MC
            uncertainties (sys/stat up/down) exceed this value. Rationale: The HID
            tables contain rows with nonsense uncertainties > 10⁸, so it is meant to
            remove those entries. Therefore, the exact value is not important. Set
            to ``None`` to disable dropping any columns.

        :return: Modified dataframe that can be used by ``PIDvar``.
        """

        # Some checks that table has expected format of Hadron ID tables
        if not set(table["charge"]).issubset({1, -1}):
            raise ValueError(
                "Expected that the ``charge`` entries of the original Hadron ID dataframe consists"
                + "only of ``1`` and ``-1``, but it contains {}".format(
                    set(table["charge"])
                )
            )

        if "theta_min" in table or "theta_max" in table:
            raise ValueError("Dataframe already has ``theta_…`` columns")

        if max_uncertainty is not None:
            unc_cols = [
                "data_MC_uncertainty_stat_up",
                "data_MC_uncertainty_stat_dn",
                "data_MC_uncertainty_sys_up",
                "data_MC_uncertainty_sys_dn",
            ]
            table = table[table[unc_cols].max(axis=1) <= max_uncertainty]
            # for θ in [0, π], cos(θ) is strictly decreasing, so we have invert min and max when inverting the cosine
        table.loc[:, "theta_min"] = np.arccos(table["cos_max"].copy(deep=True))
        table.loc[:, "theta_max"] = np.arccos(table["cos_min"].copy(deep=True))

        # PIDvar expects charge columns to contain + or -
        table.loc[:, "charge"] = np.where(table["charge"] == +1, "+", "-")

    def get_lid_queries(self):

        working_point = f"(working_point == '{self.cut_type}')"

        best_available = "(is_best_available == True)"

        if self.particle_species == "elec":
            exclude_bins = "(not ((theta_min == 0.56 and theta_max == 2.23) or (theta_min == 0.22 and theta_max == 2.71) or (p_min == 0.2 and p_max == 7) or (p_min == 0.2 and p_max == 5)))"

        elif self.particle_species == "muon":

            exclude_bins = "(not ((theta_min == 0.82 and theta_max == 2.22) or (theta_min == 0.4 and theta_max == 0.82) or (theta_min == 0.4 and theta_max == 2.6) or (p_min == 0.2 and p_max == 5)))"

        variable = f"(variable == '{self.variable}')"

        return " and ".join((working_point, best_available, exclude_bins, variable))
