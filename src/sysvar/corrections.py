from __future__ import annotations

from functools import cached_property

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from typing import List, Iterable, Optional
from os import path

from particle import Particle

import itertools
import numpy as np
from pandas import DataFrame, concat, read_csv
from uncertainties import unumpy as unp, ufloat

from .uncertainties import (
    Uncertainty,
    FullyCorrelatedUncertainty,
    FullyCorrelatedUncertaintyInParts,
    UncorrelatedUncertainty,
    get_uncertainty_types,
)
from sysvar.utils import SavableAttributesObject, read_yaml
from sysvar.visualize import CorrectionVisualizer, UncertaintyVisualizer

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class MissingInformationError(Exception):
    pass


class UncertaintyWithSameNameExists(Exception):
    pass


class UnkownUncertaintyType(Exception):
    pass


class NotValidRateType(Exception):
    pass


@dataclass()
class BaseCorrection(ABC, SavableAttributesObject):
    uncertainties: dict = field(default_factory=dict)

    @property
    @abstractmethod
    def visual_labels(self):
        pass

    @abstractmethod
    def build_queries(self) -> list:
        pass

    @staticmethod
    def _build_column_name(prefix: str | None, variable: str) -> str:
        """
        Constructs a column name by combining a prefix and a variable name.

        This method takes an optional prefix and a mandatory variable name to create a
        column name string. If the prefix is provided, the resulting column name will be
        a concatenation of the prefix and the variable, separated by an underscore. If
        the prefix is `None`, the column name will simply be the variable name.

        Args:
            prefix (str | None): An optional string to prepend to the variable name. If `None`, no prefix is added.
           variable (str): The variable name to use for constructing the column name.

        Returns:
            str: The constructed column name, either as "prefix_variable" or just "variable" if no prefix is provided.

        Examples:
            >>> obj = MyClass()
            >>> obj._build_column_name("prefix", "variable")
            'prefix_variable'
            >>> obj._build_column_name(None, "variable")
            'variable'
        """
        if not isinstance(variable, str):
            raise ValueError(
                f"{variable} is expected to be str by you passed {type(variable)}"
            )
        if not isinstance(prefix, str) and prefix is not None:
            raise ValueError(
                f"{prefix} is expected to be str by you passed {type(prefix)}"
            )
        elif prefix is None:
            column_name = variable
        else:
            column_name = "_".join([prefix, variable])

        return column_name

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

    def add_uncertainty(self, unc_name, unc_values, unc_obj: Uncertainty) -> None:
        """
        Add an uncertainty to the Correction.

        Args:
            unc (Uncertainty): The uncertainty to be added.

        Raises:
            UncertaintyWithSameNameExists: If uncertainty with the same name has already been added to the variator.

        """
        if unc_name in self.uncertainties.keys():
            raise UncertaintyWithSameNameExists(
                f"An uncertainty with the name {unc_name} already exist in the set of uncertainties that that have been added to the Correction. Make sure that you add a specific uncertainty only once, and that there are no duplicate names"
            )
        else:
            self.uncertainties.update(
                {unc_name: unc_obj(unc_name, unc_values, self.visual_labels)}
            )

    def populate_uncertainties(self):

        # Load the implemented uncertainty types
        sysvar_uncertainties = get_uncertainty_types()

        for unc_ctgy, uncertainty_dictionary in self.info["uncertainties"].items():
            if unc_ctgy not in sysvar_uncertainties.keys():
                raise UnkownUncertaintyType(
                    f"Unkown type of uncertainty is declared in the yaml file. Available uncertainty types are: ({', '.join(list(sysvar_uncertainties.keys()))}) but '{unc_ctgy}' was found in the yaml file"
                )
            else:
                for unc_name, unc_values in uncertainty_dictionary.items():
                    self.add_uncertainty(
                        unc_name=unc_name,
                        unc_values=unc_values,
                        unc_obj=sysvar_uncertainties[unc_ctgy],
                    )

    def plot_error_comparison(self, save: bool = False, filename: str = ""):

        self.visualizer = CorrectionVisualizer(self)
        self.visualizer.plot_error_comparison(save=save, filename=filename)

    def plot_uncertainty(
        self, unc_name: str | None = None, save: bool = False, filename: str = ""
    ):
        if unc_name is None:
            for unc_obj in self.uncertainties.values():
                self.visualizer = UncertaintyVisualizer(unc_obj)
                self.visualizer.plot_cov_and_corr(save=save, filename=filename)
        else:
            self.visualizer = UncertaintyVisualizer(self.uncertainties[unc_name])
            self.visualizer.plot_cov_and_corr(save=save, filename=filename)


@dataclass()
class BaseCorrectionFromYaml(BaseCorrection):
    systematic: str = None
    MC_production: str = None

    def __post_init__(self):

        super().__init__()
        try:
            self.info = read_yaml(self.systematic, self.MC_production)
        except TypeError:
            raise MissingInformationError(
                f"Need to specify the systematic effect and the MC production in the positional arguments. You passed {self.systematic} and {self.MC_production}"
            )

        self.visualizer = None
        self.title = self.info["title"]

    def add_extra_cuts(self, queries: str, prefix: str) -> str:
        if self._get_extra_cut_info() is not None:
            # Add the prefix to the extra cut info
            for var, values in self._get_extra_cut_info().items():
                extra_cut = self._build_column_name(prefix, f"{var} in {values}")
                queries = self._extend_queries_with_extra_cut(queries, extra_cut)

            return queries
        else:
            return queries

    @staticmethod
    def _extend_queries_with_extra_cut(queries: list, extra_cut: str) -> list:
        """Extends a list of queries by appending an extra condition to each query.

        Args:
            queries (list): A list of query strings.
            extra_cut (str): An additional condition to be appended to each query.

        Returns:
            list: A new list of query strings with the extra condition appended.

        Example:
            >>> _extend_queries_with_extra_cut(['query1', 'query2'], 'extra_condition')
            ['query1 & extra_condition', 'query2 & extra_condition']
        """
        return [" & ".join([q, extra_cut]) for q in queries]

    def _get_extra_cut_info(self):
        return self.info["extra_cuts"]


@dataclass
class Correction1D(BaseCorrectionFromYaml):

    dependant_variable: str | None = None
    central_values: Iterable = None
    lower_bounds: Iterable = None
    upper_bounds: Iterable = None

    def __post_init__(self):

        super().__post_init__()

        self.central_values = self.info["correction"]
        self.dependant_variable = self.info["dependant_variable"]
        self.unit = self.info["unit"]

        self.lower_bounds = self.info["min"]
        self.upper_bounds = self.info["max"]

        self.populate_uncertainties()

    @property
    def value_edges(self) -> np.ndarray:
        return np.unique(np.concatenate((self.lower_bounds, self.upper_bounds)))

    @property
    def value_mids(self) -> np.ndarray:
        return (self.value_edges[1:] + self.value_edges[:-1]) / 2

    @property
    def visual_labels(self) -> List[str]:
        return [
            f"{low} < {self.dependant_variable} < {up} {self.unit}"
            for low, up in zip(self.lower_bounds, self.upper_bounds)
        ]

    def build_queries(self, prefix: str | None = None) -> list:

        column_name = self._build_column_name(prefix, self.dependant_variable)
        queries = [
            f"{low} <= {column_name} < {up}"
            for low, up in zip(self.lower_bounds, self.upper_bounds)
        ]
        queries = self.add_extra_cuts(queries, prefix)

        return queries


@dataclass
class Correction2D(BaseCorrectionFromYaml):

    uncertainties: dict = field(default_factory=dict)

    def __post_init__(self):

        super().__post_init__()

        self.dependant_variable_1 = self.info["dependant_variable_1"]
        self.dependant_variable_2 = self.info["dependant_variable_2"]
        self.unit_1 = self.info["unit_1"]
        self.unit_2 = self.info["unit_2"]

        self.central_values = self._extract_central_values()
        self.populate_uncertainties()

    # Add an iterator to ensure that we'll loop over the corrections and bins
    # consistently
    @property
    def iterator(self):
        rows, columns, momenta, angles = [], [], [], []

        for i, column_name in enumerate(self.central_values_table.columns):
            for j, row_name in enumerate(self.central_values_table.index):
                # clean the pi0 tables.... What a format...
                if i == 0:
                    column_name = column_name.replace("  row:p column:t    ", "")
                # strip the strings and extract the momentum range
                column_name = column_name.replace("p=", "")
                ps = [float(x) / 10 for x in column_name.split("_")]

                # strip the strings and extract the theta range
                row_name = row_name.replace("cost=", "")
                ts = [float(x) for x in row_name.split("_")]

                rows.append(j)
                columns.append(i)
                momenta.append(ps)
                angles.append(ts)

        # Return a generator. Now can access all the central values and
        # errors using iloc and the rows/colums
        return zip(rows, columns, momenta, angles)

    @cached_property
    def central_values_table(self) -> DataFrame:
        table_path = self.build_table_path("nom")
        return read_csv(table_path)

    @cached_property
    def stat_error_table(self) -> DataFrame:
        table_path = self.build_table_path("stat")
        # Add column names when reading to skip creation of index column
        return read_csv(table_path, names=[f"p bin {i}" for i in range(8)])

    @cached_property
    def sys_error_table(self) -> DataFrame:
        table_path = self.build_table_path("sys")
        # Add column names when reading to skip creation of index column
        return read_csv(table_path, names=[f"p bin {i}" for i in range(8)])

    def build_table_path(self, suffix: str) -> str:

        table_dir = self.info["table_dir"]
        table_name = self.info["table_name"]

        filename = ".".join(("_".join((table_name, suffix)), "txt"))
        return path.join(table_dir, filename)

    @property
    def visual_labels(self) -> List[str]:
        return [
            f"{momenta[0]} <= {self.dependant_variable_1} < {momenta[1]} {self.unit_1} & {angles[0]} <= {self.dependant_variable_2} < {angles[1]} {self.unit_2}"
            for r, c, momenta, angles in self.iterator
        ]

    def _extract_central_values(self):
        return [
            self.central_values_table.iloc[row, column]
            for row, column, ps, ths in self.iterator
        ]

    def _extract_errors(self, table: DataFrame):
        return [table.iloc[row, column] for row, column, ps, ths in self.iterator]

    def build_queries(self, prefix: str | None = None) -> list:

        column_name_1 = self._build_column_name(prefix, self.dependant_variable_1)
        column_name_2 = self._build_column_name(prefix, self.dependant_variable_2)

        queries = [
            f"{momenta[0]} <= {column_name_1} < {momenta[1]} & {angles[0]} <= {column_name_2} < {angles[1]}"
            for r, c, momenta, angles in self.iterator
        ]
        queries = self.add_extra_cuts(queries, prefix)

        return queries

    def populate_uncertainties(self):
        sysvar_uncertainties = get_uncertainty_types()

        for unc_name, unc_ctgy in self.info["error_correlations"].items():

            if unc_name == "stat":
                unc_values = self._extract_errors(self.stat_error_table)
            elif unc_name == "sys":
                unc_values = self._extract_errors(self.sys_error_table)
            else:
                raise NotImplementedError(
                    "Only stat and sys error have been implemented now"
                )

            self.add_uncertainty(
                unc_name=unc_name,
                unc_values=unc_values,
                unc_obj=sysvar_uncertainties[unc_ctgy],
            )


@dataclass
class Correction2DCategorical(BaseCorrectionFromYaml):

    categorical_variable: str | None = None
    continuus_variable: str | None = None
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


@dataclass
class CorrectionBF(BaseCorrectionFromYaml):

    dependant_variable: str | None = None
    central_values: Iterable = None
    visual_labels: Iterable = None
    uncertainties: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        central_values, error_amplitudes = self._calculate_scaling_ratios()
        self.central_values = central_values
        # Visual labels needs to be defined  before we populate the uncertainties
        # Otherwise the uncertainty object does not have visual_labels
        self.visual_labels = self._create_strings()
        self.populate_uncertainties(error_amplitudes)
        self.dependant_variable = self.info["dependant_variable"]

    def _create_strings(self) -> List[str]:

        mother = Particle.from_pdgid(self.info["mother_particle"]).latex_name
        daughter_pdgs = [
            x for mode in self.info["modes"].values() for x in mode["daughters"]
        ]
        strings = []
        for daughter_set in daughter_pdgs:
            daughter_names = []
            for x in daughter_set:
                try:
                    daughter_names.append(Particle.from_pdgid(x).latex_name)
                except:
                    daughter_names.append(x)

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

        # Safe 0 division. Returns 1+- 0 for the ones where
        # decay.dec = 0 or pdg = 0
        corrections = np.divide(
            pdg_BFs,
            decaydec_BFs,
            out=unp.uarray(np.ones_like(pdg_BFs), np.zeros_like(pdg_BFs)),
            where=((decaydec_BFs != 0) & (pdg_BFs != 0)),
        )

        return list(unp.nominal_values(corrections)), list(unp.std_devs(corrections))

    def populate_uncertainties(self, error_amplitudes: list):
        """
        Overrides the method of the base class method as the error amplitutes are calculated dynamically from the calculate_scaling_ratios method

        """

        # Add the uncertainties as fully uncorrelated. This is a choice.
        # Can be very complicated as some of these modes may have been measured
        # by the same experiments, with complicated correlatiosn
        unc_type = "fully_correlated"

        sysvar_uncertainties = get_uncertainty_types()
        self.add_uncertainty(
            unc_name="BF_unc",
            unc_values=error_amplitudes,
            unc_obj=sysvar_uncertainties[unc_type],
        )

    def build_queries(self, prefix: str | None = None) -> list:

        column_name = self._build_column_name(prefix, self.dependant_variable)
        queries = [
            (
                f"{column_name} == '{mode['dmID']}'"
                if isinstance(mode["dmID"], str)
                else f"{column_name} in {mode['dmID']}"
            )
            for mode in self.info["modes"].values()
        ]

        queries = self.add_extra_cuts(queries, prefix)

        return queries


@dataclass
class CustomCorrection(BaseCorrection):

    info: InitVar[dict] = None
    dependant_variable: str | None = None
    central_values: Iterable = None
    uncertainties: dict = field(default_factory=dict)
    query_targets: Iterable = None

    def __post_init__(self, info):
        self.info = info

        self.dependant_variable = self.info["dependant_variable"]
        self.central_values = self.info["central_values"]
        self.query_targets = self.info["query_targets"]

        self.unit = self.info["unit"]
        self.title = self.info["title"]

        self.populate_uncertainties()

    @property
    def value_edges(self) -> np.ndarray:
        return np.arange(len(self.central_values) + 1)

    @property
    def value_mids(self) -> np.ndarray:
        return (self.value_edges[1:] + self.value_edges[:-1]) / 2

    @property
    def visual_labels(self) -> List[str]:
        return [
            f"{self.dependant_variable} = {target}" for target in self.query_targets
        ]

    def build_queries(self, prefix: str | None = None) -> list:

        column_name = self._build_column_name(prefix, self.dependant_variable)
        queries = [f"{column_name} == {target}" for target in self.query_targets]
        return queries


@dataclass
class CorrectionPID(BaseCorrectionFromYaml):

    uncertainties: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        rate = self.info["rate"]

        self.check_valid_rate(rate)
        self.table = self.get_table(rate)
        self.central_values = self._extract_central_values()

        self.p = self.info["momentum_variable"]
        self.theta = self.info["theta_variable"]
        self.PDG = self.info["PDG_variable"]
        self.mcPDG = self.info["mcPDG_variable"]

        self.momentum_unit = self.info["momentum_unit"]

        # Add uncertainties as fully uncorrelated. This is a conservative choice
        error_id = "stat"
        self.uncertainties.update(
            {
                f"{error_id} uncertainty": UncorrelatedUncertainty(
                    f"{error_id} uncertainty",
                    self._extract_errors(f"{error_id}"),
                    self.visual_labels,
                )
            }
        )
        error_id = "sys"
        self.uncertainties.update(
            {
                f"{error_id} uncertainty": UncorrelatedUncertainty(
                    f"{error_id} uncertainty",
                    self._extract_errors(f"{error_id}"),
                    self.visual_labels,
                )
            }
        )

    @staticmethod
    def check_valid_rate(rate):

        valid_rates = ["eff", "fake"]

        if rate not in valid_rates:
            raise NotValidRateType(
                f"Valid rate arguments are {*valid_rates,} but you passed {rate}"
            )

    def get_table(self, rate):

        table_finders = []
        if "eID" in self.systematic:
            table_finders.append(CorrectionTableFinder.electrons(self.info))
        if "muID" in self.systematic:
            table_finders.append(CorrectionTableFinder.muons(self.info))
        elif "kID" in self.systematic:
            table_finders.append(CorrectionTableFinder.kaons(self.info))
        elif "piID" in self.systematic:
            table_finders.append(CorrectionTableFinder.pions(self.info))

        eff_table = concat([x.eff_table for x in table_finders])
        fake_rate_table = concat([x.fake_rate_table for x in table_finders])

        if rate == "eff":
            table = eff_table
            # PATCH
            # This has to be read somewhere else somehow
            self._true_pdg = table_finders[0].true_pdg
        elif rate == "fake":
            table = fake_rate_table
            # PATCH
            # This has to be read somewhere else somehow
            self._true_pdg = table_finders[0].fake_pdg

        return table

    @property
    def true_pdg(self) -> list:
        return self._true_pdg

    @true_pdg.setter
    def true_pdg(self, true_pdg):
        self._true_pdg = true_pdg

    @property
    def iterator(self):
        return self.table.iterrows()

    @property
    def queries(self):
        # PATCH
        # This just "implements" the property to satisfy the parent class
        pass

    def build_queries(
        self, prefix: str | None = None, extra_cut: str | None = None
    ) -> List[str]:

        # Pre-compute column names to avoid repeated function calls
        p_column_name = self._build_column_name(prefix, self.p)
        theta_column_name = self._build_column_name(prefix, self.theta)
        PDG_column_name = self._build_column_name(prefix, self.PDG)
        mcPDG_column_name = self._build_column_name(prefix, self.mcPDG)

        # Create a local reference for self._true_pdg to avoid repeated attribute access
        true_pdg = self._true_pdg

        # Use a list comprehension with cached lookups to improve performance
        queries = []
        append_query = queries.append  # Local function assignment for faster append

        # Use local variable access within the loop to speed up string formatting
        for _, row in self.iterator:
            # Access row[1] once and cache its values in local variables
            p_min = row["p_min"]
            p_max = row["p_max"]
            theta_min = row["theta_min"]
            theta_max = row["theta_max"]
            mcPDG = row["mcPDG"]

            # Construct the query string with reduced overhead
            query = (
                f"({p_min} <= {p_column_name} < {p_max} & "
                f"{theta_min} <= {theta_column_name} < {theta_max} & "
                f"{PDG_column_name} == {mcPDG} & "
                f"{mcPDG_column_name} in {true_pdg})"
            )

            # Append the constructed query string to the list
            append_query(query)

        # Add any extra cuts to the queries if needed
        queries = self.add_extra_cuts(queries, prefix)

        return queries

    @property
    def visual_labels(self) -> List[str]:
        return [
            rf"{row[1]['p_min']} <= p < {row[1]['p_max']} {self.momentum_unit} & {row[1]['theta_min']} <= $\theta$ < {row[1]['theta_max']} & q = {row[1]['charge']}"
            for row in self.iterator
        ]

    def _extract_central_values(self):
        return [row[1]["data_MC_ratio"] for row in self.iterator]

    def _extract_errors(self, error_type):

        # this assumes symmetric errors and takes the maximum out of the two.
        return [
            row[1][
                [
                    f"data_MC_uncertainty_{error_type}_up",
                    f"data_MC_uncertainty_{error_type}_dn",
                ]
            ].max()
            for row in self.iterator
        ]


# #######################################################################################


def create_correction_object(syst_effect: str, MC_prod: str) -> BaseCorrection:
    """Retrieves amd creates the appropriate correction object based on the systematic effect and MC production type.

    Args:
        syst_effect (str): The systematic effect identifier.
        MC_prod (str): The Monte Carlo production type identifier.

    Returns:
        BaseCorrection: The appropriate correction object based on the provided systematic effect and MC production type.

    Raises:
        NotImplementedError: If the correction type specified in the configuration is not implemented.

    Example:
        >>> correction = get_correction_object("syst1", "MC1")
        >>> isinstance(correction, BaseCorrection)
        True
    """
    correction_types = {
        "1D": Correction1D,
        "2D": Correction2D,
        "2DCategorical": Correction2DCategorical,
        "BF": CorrectionBF,
        "PID": CorrectionPID,
    }

    if isinstance(syst_effect, str):
        corr_type = read_yaml(syst_effect, MC_prod)["correction_type"]

        try:
            return correction_types[corr_type](
                systematic=syst_effect, MC_production=MC_prod
            )
        except KeyError:
            raise NotImplementedError(
                f"Available corrections are: {list(correction_types.keys())} but you passed {corr_type}"
            )
    elif isinstance(syst_effect, dict):
        return CustomCorrection(info=syst_effect)

    else:
        raise ValueError(
            "Pass a string for existing standard systematic to create a correction object from yaml files or a dictionary to create a custom correction object"
        )


class CorrectionTableFinder:
    """
    Factory method class to get correction tables for kaons,  pions, electrons and muons
    """

    def __init__(self, particle_species, online_cut, base_table_path, variable):
        self.particle_species = particle_species
        self.online_cut = online_cut
        self.base_table_path = base_table_path
        self.variable = variable

        self.true_pdg = self.particle_species_settings[particle_species]["true_pdgs"]
        self.fake_pdg = self.particle_species_settings[particle_species]["fake_pdgs"]

        self.value = self.get_cut_value()
        self.cut_type = self.get_cut_type()

        efficiency_table_names = self.build_table_name(
            self.particle_species_settings[self.particle_species]["eff_table_ids"]
        )
        fake_rate_table_names = self.build_table_name(
            self.particle_species_settings[self.particle_species]["fake_rate_table_ids"]
        )

        self.eff_table = self.get_table(efficiency_table_names)
        self.fake_rate_table = self.get_table(fake_rate_table_names)

    @classmethod
    def kaons(cls, external_info):

        particle_species = "kaon"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable=None,
        )

    @classmethod
    def pions(cls, external_info):

        particle_species = "pion"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable=None,
        )

    @classmethod
    def electrons(cls, external_info):

        particle_species = "elec"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable=external_info["variable"],
        )

    @classmethod
    def muons(cls, external_info):

        particle_species = "muon"

        return cls(
            particle_species=particle_species,
            online_cut=external_info["online_cut"],
            base_table_path=external_info["table_paths"],
            variable=external_info["variable"],
        )

    @property
    def particle_species_settings(self) -> dict:

        return {
            "kaon": {
                "true_pdgs": [321, -321],
                "fake_pdgs": [211, -211],
                "eff_table_ids": ["keff"],
                "fake_rate_table_ids": ["piFk"],
            },
            "pion": {
                "true_pdgs": [211, -211],
                "fake_pdgs": [321, -321],
                "eff_table_ids": ["pieff"],
                "fake_rate_table_ids": ["kFpi"],
            },
            "elec": {
                "true_pdgs": [11, -11],
                "fake_pdgs": [321, 211, -321, -211],
                "eff_table_ids": ["e_efficiency"],
                "fake_rate_table_ids": [
                    "K_e_fakeRate",
                    "pi_e_fakeRate",
                ],
            },
            "muon": {
                "true_pdgs": [13, -13],
                "fake_pdgs": [321, 211, -321, -211],
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
            self.make_pidvar_compatible(table, max_uncertainty=10)

        elif self.particle_species in ["elec", "muon"]:
            table = concat([read_csv(x) for x in table_names])
            table.query(self.get_lid_queries(), inplace=True)

        self.add_mcPDG_to_table(table)

        return table

    def add_mcPDG_to_table(self, table):

        table.loc[:, "mcPDG"] = -9999

        table.loc[table["charge"] == "-", "mcPDG"] = self.true_pdg[0]
        table.loc[table["charge"] == "+", "mcPDG"] = self.true_pdg[1]

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
            # table = table[table[unc_cols].max(axis=1) <= max_uncertainty]
            # for θ in [0, π], cos(θ) is strictly decreasing, so we have invert min and max when inverting the cosine
        table["theta_min"] = -9999
        table["theta_max"] = -9999
        table.loc[:, "theta_min"] = np.arccos(table["cos_max"].copy(deep=True))
        table.loc[:, "theta_max"] = np.arccos(table["cos_min"].copy(deep=True))

        # PIDvar expects charge columns to contain + or -
        table.loc[:, "charge"] = np.where(table["charge"] == +1, "+", "-")

        return table

    def get_lid_queries(self):

        working_point = f"(working_point == '{self.cut_type}')"

        best_available = "(is_best_available == True)"

        if self.particle_species == "elec":
            exclude_bins = "(not ((theta_min == 0.56 and theta_max == 2.23) or (theta_min == 0.22 and theta_max == 2.71) or (p_min == 0.2 and p_max == 7) or (p_min == 0.2 and p_max == 5)))"

        elif self.particle_species == "muon":

            exclude_bins = "(not ((theta_min == 0.82 and theta_max == 2.22) or (theta_min == 0.4 and theta_max == 0.82) or (theta_min == 0.4 and theta_max == 2.6) or (p_min == 0.2 and p_max == 5)))"

        variable = f"(variable == '{self.variable}')"

        return " and ".join((working_point, best_available, exclude_bins, variable))
