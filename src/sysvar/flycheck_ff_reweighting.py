import logging
import numpy as np

from pandas import DataFrame
from collections.abc import MutableMapping
from typing import Union, Iterable, Dict, List, Iterator

from hammer.hammerlib import Hammer, Process, Particle, FourMomentum

from sysvar.ff_models import FFModel, BGL, BGLB2, CLN, BLPRXP, BLR, print_model_info
from sysvar.utils import get_varied_FF_central_values

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class HammerRateContainer(MutableMapping):
    def __init__(self) -> None:
        self._rates_dict: Dict[str, float] = dict()

    def __getitem__(self, item: str) -> float:
        return self._rates_dict[item]

    def __setitem__(self, key: str, value: float):
        self._rates_dict[key] = value

    def __delitem__(self, key) -> None:
        del self._rates_dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rates_dict.keys())

    def __len__(self) -> int:
        return len(self._rates_dict)

    def get_or_calculate_rate(
        self,
        scheme_name: str,
        hammer_id_str: str,
        hammer_instance: Hammer,
    ) -> float:
        key: str = f"{scheme_name}_{hammer_id_str}"
        try:
            return self._rates_dict[key]
        except KeyError:
            rate: float = hammer_instance.get_rate(hammer_id_str, scheme_name)
            assert isinstance(rate, float), (
                rate,
                type(rate),
                hammer_id_str,
                scheme_name,
                key,
            )
            self._rates_dict[key] = rate
            return self._rates_dict[key]

    def get_or_calculate_input_rate(
        self,
        hammer_id_str: str,
        hammer_instance: Hammer,
    ) -> float:
        key: str = f"INPUT_{hammer_id_str}"
        try:
            return self._rates_dict[key]
        except KeyError:
            input_rate: float = hammer_instance.get_denominator_rate(hammer_id_str)
            assert isinstance(input_rate, float), (
                input_rate,
                type(input_rate),
                hammer_id_str,
                key,
            )
            self._rates_dict[key] = input_rate
            return self._rates_dict[key]


class HammerStudy:
    def __init__(self, mode=str, default_names: bool = True):

        if mode == "B2":
            self.input_models = [
                BGL.BtoDstEllNu_dec(),
                ####
                BGL.BtoDEllNu_dec_outdated(),
                ####
                BLR.BtoD1EllNu_dec(),
                BLR.BtoD0stEllNu_dec(),
                BLR.BtoD1prEllNu_dec(),
                BLR.BtoD2stEllNu_dec(),
            ]
        else:
            raise NotImplementedError(
                "Not having default options for Belle and LHCb MC yet"
            )

        if default_names:
            self._generations = {
                "B": "B_anc",
                "B_daughters": [
                    "B_d0",
                    "B_d1",
                    "B_d2",
                ],
                "Xc_daughters": ["B_g_d0", "B_g_d1"],
                "tau_daughters": ["B_g_d10", "B_g_d11", "B_g_d12", "B_g_d13"],
                "Xc_grand_daughters": ["B_g_g_d0", "B_g_g_d1"],
                "Xc_great_grand_daughters": ["B_g_g_g_d0", "B_g_g_g_d1"],
            }
            self._four_momentum_vars = ["mcE", "mcPX", "mcPY", "mcPZ"]
            self._pdg_var = "mcPDG"

        self.hammer = Hammer()
        self.rate_container = HammerRateContainer()

        self._print_information_setup()

    def _print_information_setup(self):

        logging.info("Set up Hammer study")
        logging.info("The input variables have been specified as follows:")
        logging.info("PDG variable: %s", self._pdg_var)
        logging.info(
            "Four momentum variables: %s, %s, %s, %s,",
            *[var for var in self._four_momentum_vars],
        )
        logging.info(
            "B meson prefix: %s ",
            self._generations["B"],
        )
        logging.info(
            "B meson daughter prefixes: %s ",
            [var for var in self._generations["B_daughters"]],
        )
        logging.info(
            "Xc meson daughter prefixes: %s ",
            [var for var in self._generations["Xc_daughters"]],
        )
        logging.info(
            "The HammerStudy class will take care of building the full variable names"
        )
        logging.warning(
            "If this does not align with the variable names in your tuples set default_names = False in the object constructor and modify the following properties: %s",
            [
                key
                for key, attr in HammerStudy.__dict__.items()
                if (isinstance(attr, property) and attr.fset is not None)
            ],
        )

    @property
    def four_momentum_vars(self):
        return self._four_momentum_vars

    @four_momentum_vars.setter
    def four_momentum_vars(self, variables: Iterable):
        self._four_momentum_vars = variables

    @property
    def pdg_var(self):
        return self._pdg_var

    @pdg_var.setter
    def pdg_var(self, pdg: str):
        self._pdg_var = pdg

    @property
    def decays(self):
        return [self._get_decay_name(model) for model in self.input_models]

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, dictionary: dict):
        self._generations = dictionary

    @staticmethod
    def _get_ff_scheme_dictionary(model: FFModel, suffix: Union[None, str] = None):

        if suffix is None:
            dictionary = {f"B{model.Xc}": model.name}
        else:
            dictionary = {f"B{model.Xc}": f"{model.name}_{suffix}"}

        return dictionary

    @staticmethod
    def _get_decay_name(model: FFModel, to: bool = False):

        return f"Bto{model.Xc}{model.lep}Nu" if to else f"B{model.Xc}{model.lep}Nu"

    def set_ff_input_scheme(self, verbose=False):

        """
        This has the same name as the hammer API to ensure compatibility
        """

        logging.info("Defaulting to Belle II decay.dec parameters")
        if not verbose:
            logging.info(
                "Enable the verbose option when calling set_ff_input_scheme to know more about the models that were use as an input"
            )

        input_schemes_for_hammer = {}
        for model in self.input_models:

            # What enters in these strings is important.
            # However there are a lot of inconsistencies across the HAMMER API.
            model_scheme = self._get_ff_scheme_dictionary(model)

            # Now update the dictionary and the list
            input_schemes_for_hammer.update(model_scheme)

            # Add the D** ff models
            self._add_Dstst_ff_models(model, input_schemes_for_hammer)

        # Set the input info for hammer now using the HAMMER API
        self.hammer.set_ff_input_scheme(input_schemes_for_hammer)

        for model in self.input_models:
            param_string = self._create_param_string(model)
            if verbose:
                logging.info(
                    "Adding transition and model: %s as input for HAMMER", model_scheme
                )
                print_model_info(model)
                logging.info(
                    "The parameters of the model have been set as: %s: ", param_string
                )

            self.hammer.set_options(param_string)

    def _create_param_string(
        self,
        model: FFModel,
        suffix: Union[None, str] = None,
        parameters: Union[None, Iterable] = None,
    ):
        # For the input FF schemes we don't need a suffix identifier
        if suffix is None:
            # Notice the inconsistancy of the HAMMER API here.
            # e.g. BtoD* instead of BD*
            # If "to" is not added tho, hammer errors
            template = f"Bto{model.Xc}{model.name}: {model.param_init}"
        else:
            # But for new models we want to add a suffix
            template = f"Bto{model.Xc}{model.name}_{suffix}: {model.param_init}"

        if parameters is None:
            # here loop over the parameter names and values
            string = template % tuple(
                [
                    param.nominal_value
                    if name != "DelMbc"
                    else param["mc"].nominal_value
                    for name, param in model.params.items()
                ]
            )

        else:
            # here loop over the values e.g. for eigenvariations
            string = f"Bto{model.Xc}{model.name}_{suffix}: {model.param_init}" % tuple(
                [param for param in parameters]
            )
        return string

    def add_ff_scheme(self, models, scheme_name, verbose=False):

        """
        This has the same name as the hammer API to ensure compatibility
        """

        scheme_dict = {scheme_name: {}}
        scheme_options = {scheme_name: []}

        for i, model in enumerate(models):
            ####################################################################
            if not isinstance(model, FFModel):
                raise ValueError(
                    f"Model number {i} in the arguments is not an instance of FFModel"
                )
            decay = self._get_decay_name(model)
            # Include the decay that is described by the particular model
            self.hammer.include_decay(decay)

            # Create the string containing the FF model parameters
            param_string = self._create_param_string(model=model, suffix=scheme_name)
            if verbose:
                logging.info("Adding %s model for %s decay", model.name, decay)
                print_model_info(model)
                logging.info("The parameters are set to: %s", param_string)

            scheme_dict[scheme_name].update(self._get_ff_scheme_dictionary(model))
            scheme_options[scheme_name].append(param_string)

            self._add_Dstst_ff_models(model, scheme_dict[scheme_name])
            ###################################################################

            eigenvalue_variations = get_varied_FF_central_values(model)
            if verbose:
                logging.info(
                    "Now creating eigenvariations of the parameters %s",
                    eigenvalue_variations,
                )
            for i, eigenvar in enumerate(eigenvalue_variations):

                # Create the tokens that will be added to the variation schemes
                up_var, down_var = f"up{i}", f"down{i}"

                # Now update the dictionaries with the up variations
                self._add_eigenvariation_to_scheme_setup(
                    scheme_dict, scheme_options, scheme_name, model, up_var, eigenvar[0]
                )
                # And now with the down variations
                self._add_eigenvariation_to_scheme_setup(
                    scheme_dict,
                    scheme_options,
                    scheme_name,
                    model,
                    down_var,
                    eigenvar[1],
                )

            ###################################################################

        if verbose:
            logging.info(
                "The scheme and it's eigenvariations have been set to: %s", scheme_dict
            )
            logging.info(
                "Now will pad the models with missing eigenvariations to ensure that hammer runs"
            )

        self._pad_missing_eigenvariations(
            models, scheme_dict, scheme_options, scheme_name
        )

        # And now we want to add all schemes to HAMMER
        for (scheme_n1, names), options in zip(
            scheme_dict.items(), scheme_options.values()
        ):
            # make sure that all models have the D** decay form factors
            # This is added for the eigenvariations that are padded just before
            for model in models:
                self._add_Dstst_ff_models(model, names)

            self.hammer.add_ff_scheme(scheme_n1, names)
            for option in options:
                self.hammer.set_options(option)

    def _pad_missing_eigenvariations(
        self, models, scheme_dict: dict, scheme_options: dict, scheme_name: str
    ):

        for model in models:
            # Check if all the models have the same number of variations

            for (scheme_n1, model_names), options in zip(
                scheme_dict.items(), scheme_options.values()
            ):

                if f"B{model.Xc}" not in model_names.keys():
                    # If some model has less eigenvariations than the maximum number
                    # Then pad them with the nominal value
                    logging.warning(
                        "Will pad model %s for variation %s with the nominal values",
                        model.name,
                        scheme_n1,
                    )

                    model_names.update(self._get_ff_scheme_dictionary(model))
                    options.append(
                        self._create_param_string(model=model, suffix=scheme_name)
                    )

    def _add_eigenvariation_to_scheme_setup(
        self,
        scheme_dict: dict,
        scheme_options: dict,
        scheme_name: str,
        model,
        var_token: str,
        parameters: np.ndarray,
    ):

        # Create a placeholder if the scheme name has not been defined yet
        if scheme_name + "_" + var_token not in scheme_dict.keys():
            self._add_placeholder_for_variation(
                scheme_dict, scheme_options, scheme_name, var_token
            )

        # Now update the scheme dictionaries
        scheme_dict[scheme_name + "_" + var_token].update(
            self._get_ff_scheme_dictionary(model, var_token)
        )
        scheme_options[scheme_name + "_" + var_token].append(
            self._create_param_string(
                model=model,
                suffix=var_token,
                parameters=parameters,
            )
        )

    @staticmethod
    def _add_placeholder_for_variation(
        scheme_dict: dict,
        scheme_options: dict,
        scheme_name: str,
        var_token: str,
    ):
        """
        This will create a subdictionary for a new variation
        """

        scheme_dict.update({scheme_name + "_" + var_token: {}})
        scheme_options.update({scheme_name + "_" + var_token: []})

    @staticmethod
    def _add_Dstst_ff_models(model, dictionary):

        if "D**" in model.Xc:
            dictionary.update({f"{model.Xc}D*Pi": "PW"})
            dictionary.update({f"{model.Xc}DPi": "PW"})

    def process_events(self, df: DataFrame, scheme: str, verbose=False):

        self.hammer.set_units("GeV")  # this is also the default
        self.hammer.init_run()

        df[self._get_new_df_columnnames(scheme)] = df.apply(
            lambda row: self._event_wise(row, scheme, verbose),
            axis=1,
            result_type="expand",
        )

    def _get_new_df_columnnames(self, scheme: str):

        new_cols = [scheme for scheme in self.hammer.get_ff_scheme_names()]
        new_cols.append(f"denom_rate_{scheme}")
        new_cols.append(f"new_rate_{scheme}")
        new_cols.append(f"hammer_reweighted_{scheme}")
        new_cols.append(f"hammer_found_rates_{scheme}")

        return new_cols

    def _event_wise(self, row, schemename: str, verbose: bool):

        # If this is not a semileptonic event return zeros for everything
        if row["B_d1_mcPDG"] not in self._get_particle_pdg_codes("leptons"):
            return self._add_unweighted_results()

        self.hammer.init_event()
        process = Process()

        # We will adopt a Breadth First Search approach

        # First add the Bmeson
        Bmeson = self._add_hammer_particle(process, row, self._generations["B"])

        # Now let's first add the first generation
        B_daughters = [
            self._add_hammer_particle(process, row, prefix)
            for prefix in self.generations["B_daughters"]
        ]
        # And add the decay vertex
        process.add_vertex(Bmeson, {*B_daughters})

        # Now let's first add the first generation if we have a D* or a D**meson
        if row[f"{self._generations['B_daughters'][0]}_{self._pdg_var}"] in [
            *self._get_particle_pdg_codes("D*"),
            *self._get_particle_pdg_codes("D**"),
        ]:
            Xc_daughters = [
                self._add_hammer_particle(process, row, prefix)
                for prefix in self.generations["Xc_daughters"]
            ]
            # And add the decay vertex. The Xc should have been added first
            process.add_vertex(B_daughters[0], {*Xc_daughters})

        # Now let's first add the second generation if we have a D**meson
        if row[f"{self._generations['B_daughters'][0]}_{self._pdg_var}"] in [
            *self._get_particle_pdg_codes("D**"),
        ]:
            Xc_grand_daughters = [
                self._add_hammer_particle(process, row, prefix)
                for prefix in self.generations["Xc_grand_daughters"]
            ]
            # And add the decay vertex. The Xc should have been added first
            process.add_vertex(Xc_daughters[0], {*Xc_grand_daughters})

        self.hammer.add_process(process)

        self.hammer.process_event()

        # If the event was actually a semileptonic one show info if need be
        if self._get_vertex_decay(row) != "" and verbose:
            logging.info("#################################################")
            logging.info("Processing event: %s", row["__event__"])
            logging.info("Vertex decay %s", self._get_vertex_decay(row))

            for scheme in self.hammer.get_ff_scheme_names():
                try:
                    logging.info("%s , %s", scheme, self.hammer.get_weight(scheme))
                except:
                    logging.info("Probably didn't find eigenvariation")

        return self._collect_results(process, row, schemename)

    def _collect_results(self, p, row, scheme):
        hammer_id_string: str = self._get_vertex_decay(row)
        hammer_scheme_names = self.hammer.get_ff_scheme_names()

        weights = [self.hammer.get_weight(scheme) for scheme in hammer_scheme_names]
        try:
            denom_rate: float = self.rate_container.get_or_calculate_input_rate(
                hammer_id_str=hammer_id_string,
                hammer_instance=self.hammer,
            )
        except OverflowError:
            logging.warning(
                "Event %s threw an OverflowError. No rates were found", row["__event__"]
            )
            denom_rate = -1.0

        try:
            new_rates: List[float] = [
                self.rate_container.get_or_calculate_rate(
                    scheme_name=this_scheme_name,
                    hammer_id_str=hammer_id_string,
                    hammer_instance=self.hammer,
                )
                for this_scheme_name in hammer_scheme_names
            ]
            new_rate: float = new_rates[0]
        except OverflowError:
            logging.warning(
                "Event %s threw an OverflowError. No rates were found", row["__event__"]
            )
            new_rate = -1.0
            new_rates = [new_rate] * len(hammer_scheme_names)

        hammer_reweighted = 0 if (np.array(weights) == 1).all() else 1
        hammer_found_rates = (
            0 if denom_rate in [0.0, 1.0] or new_rate in [0.0, -1.0] else 1
        )

        scaled_weights: List[float] = list(
            np.array(weights) * denom_rate / np.array(new_rates)
        )

        return (
            *scaled_weights,
            denom_rate,
            new_rate,
            hammer_reweighted,
            hammer_found_rates,
        )

    def _add_unweighted_results(self):

        weights = [0 for scheme in self.hammer.get_ff_scheme_names()]
        denom_rate, new_rate, hammer_reweighted, hammer_found_rates = 0, 0, 0, 0

        return *weights, denom_rate, new_rate, hammer_reweighted, hammer_found_rates

    def _add_hammer_particle(self, process, row, prefix):

        return process.add_particle(
            Particle(
                FourMomentum(*[row[var] for var in self._build_column_names(prefix)]),
                row[f"{prefix}_{self._pdg_var}"],
            )
        )

    def _build_column_names(self, prefix):
        return [f"{prefix}_{var}" for var in self._four_momentum_vars]

    def _get_vertex_decay(self, row):

        pdg_to_string_dict = {
            511: "B0",
            -511: "B0bar",
            521: "B+",
            -521: "B-",  # B
            411: "D+",
            -411: "D-",
            421: "D0",
            -421: "D0bar",  # D
            413: "D*+",
            -413: "D*-",
            423: "D*0",
            -423: "D*0bar",  # D*
            10411: "D**0*+",
            -10411: "D**0*-",
            10421: "D**0*0",
            -10421: "D**0*0bar",  # D**: D_0*
            10413: "D**1+",
            -10413: "D**1-",
            10423: "D**10",
            -10423: "D**10bar",  # D**: D_1
            20413: "D**1*+",
            -20413: "D**1*-",
            20423: "D**1*0",
            -20423: "D**1*0bar",  # D**: D_1'
            415: "D**2*+",
            -415: "D**2*-",
            425: "D**2*0",
            -425: "D**2*0bar",  # D**: D_2*
            11: "E",
            -11: "E",
            13: "Mu",
            -13: "Mu",
            15: "Tau",
            -15: "Tau",  # Charged leptons
            12: "Nu",
            -12: "Nu",
            14: "Nu",
            -14: "Nu",
            16: "Nu",
            -16: "Nu",
        }

        try:
            decay_string = pdg_to_string_dict[row[f"{self._generations['B']}_mcPDG"]]
            for prefix in self._generations["B_daughters"]:
                decay_string += pdg_to_string_dict[row[f"{prefix}_mcPDG"]]
        except KeyError:
            decay_string = ""

        return decay_string

    @staticmethod
    def _get_particle_pdg_codes(particle_string):

        if particle_string == "D":
            pdg_codes = [411, -411, 421, -421]
        elif particle_string == "D*":
            pdg_codes = [413, -413, 423, -423]
        elif particle_string == "D**":
            pdg_codes = [
                10411,
                -10411,
                10421,
                -10421,
                10413,
                -10413,
                10423,
                -10423,
                20413,
                -20413,
                20423,
                -20423,
                415,
                -415,
                425,
                -425,
            ]
        elif particle_string == "tau":
            pdg_codes = [15, -15]
        elif particle_string == "ell":
            pdg_codes = [11, -11, 13, -13]
        elif particle_string == "leptons":
            pdg_codes = [11, -11, 13, -13, 15, -15]
        else:
            raise ValueError("Wrong kind of Xc string was passed")

        return pdg_codes
