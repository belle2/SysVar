import numpy as np
import uproot
from sysvar.utils import read_yaml

from sysvar.templates import Template2D
from sysvar.corrections import Correction, BFCorrection
from sysvar.variations import Variator
from sysvar.eigendecomposer import EigenDecomposer
from sysvar.visualize import EigenDecomposerVisualizer

import logging

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


def save_nominal_templates(df, analysis: str):

    settings = read_yaml("template_setup", analysis)

    regions = settings["regions"]
    region_trees = settings["tree_names"]
    fit_ctgies = settings["fit_ctgies"]

    region_id_column = settings["region_id_column"]
    ctgy_id_column = settings["ctgy_id_column"]

    with uproot.recreate(settings["filename"], compression=None) as newfile:

        logging.info("Recreate file with uproot: %s", settings["filename"])

        for region, tree in zip(regions, region_trees):
            for ctgy in fit_ctgies:
                q = f"{ctgy_id_column} == '{ctgy}' and {region_id_column} in @region"

                if len(df.query(q)) > 0:
                    t = Template2D(df.query(q), settings["bins"], settings["weight"])
                    newfile[f"{tree}/{ctgy}/Nominal"] = t.make_hist()
                    logging.info(
                        "Computing template in region: %s for fit ctgy: %s",
                        region,
                        ctgy,
                    )
                else:
                    logging.info(
                        "Skipping template in region: %s for fit ctgy: %s", region, ctgy
                    )
                    continue

            logging.info("Adding empty Data for region: %s", region)
            # Save empty data now as we work only on Asimov
            newfile[f"{tree}/Data/Nominal"] = np.array([0, 0, 0]), np.array(
                [0, 1, 2, 3]
            )


def save_template_variation(df, analysis: str, systematic: str):

    settings = read_yaml("template_setup", analysis)

    eigen = EigenDecomposer(df, settings, systematic)

    eigen.precision = 0.01
    eigen.find_important_eigendimension_indices()

    ev = EigenDecomposerVisualizer(eigen, ["test"], "./")
    ev.plot_eigenvalues()

    regions = settings["regions"]
    region_trees = settings["tree_names"]
    fit_ctgies = settings["fit_ctgies"]

    region_id_column = settings["region_id_column"]
    ctgy_id_column = settings["ctgy_id_column"]

    variations = eigen._get_unrolled_variations()
    nominals = eigen._get_unrolled_nominals()

    with uproot.update(settings["filename"]) as newfile:

        logging.info("Updating file with uproot: %s", settings["filename"])

        for i_rm, reco_mode in enumerate(eigen.decay_modes):
            for j_ctgy, ctgy in enumerate(fit_ctgies):

                for k_var in range(eigen.N_important_dims):
                    logging.info(
                        "Computing template in region: %s for fit ctgy template: %s and variation #%s",
                        reco_mode[1],
                        ctgy,
                        k_var + 1,
                    )
                    newfile[
                        f"{reco_mode[1]}/{ctgy}/{eigen.syst_effect}_var{k_var+1}_up"
                    ] = (
                        nominals[i_rm, j_ctgy, :] + variations[i_rm, j_ctgy, :, k_var],
                        np.linspace(0, 1, eigen.Nbins + 1),
                    )
                    newfile[
                        f"{reco_mode[1]}/{ctgy}/{eigen.syst_effect}_var{k_var+1}_down"
                    ] = (
                        nominals[i_rm, j_ctgy, :] - variations[i_rm, j_ctgy, :, k_var],
                        np.linspace(0, 1, eigen.Nbins + 1),
                    )


def save_existing_eigenvariations(df, analysis: str, systematic: str):

    settings = read_yaml("template_setup", analysis)

    regions = settings["regions"]
    region_trees = settings["tree_names"]
    fit_ctgies = settings["fit_ctgies"]

    region_id_column = settings["region_id_column"]
    ctgy_id_column = settings["ctgy_id_column"]

    N_eigen = settings["systematics"][systematic]["N_eigen"]

    weight = settings["weight"]
    syst_weight = settings["systematics"][systematic]["weight"]

    with uproot.update(settings["filename"]) as newfile:

        logging.info("Updating file with uproot: %s", settings["filename"])
        for region, tree in zip(regions, region_trees):
            for ctgy in fit_ctgies:

                q = f"{ctgy_id_column} == '{ctgy}' and {region_id_column} in @region"

                tmp_df = df.query(q)
                if len(tmp_df) > 0:
                    logging.info(
                        "Computing templates in region: %s for fit ctgy: %s",
                        region,
                        ctgy,
                    )

                    for variation in range(N_eigen):

                        hist_up = np.histogramdd(
                            np.array(tmp_df[[*settings["bins"].keys()]]),
                            bins=[bins for bins in settings["bins"].values()],
                            weights=np.array(
                                tmp_df[weight]
                                / tmp_df[syst_weight]
                                * tmp_df[f"{syst_weight}_up{variation}"].fillna(1)
                            ),
                        )

                        hist_down = np.histogramdd(
                            np.array(tmp_df[[*settings["bins"].keys()]]),
                            bins=[bins for bins in settings["bins"].values()],
                            weights=tmp_df[weight]
                            / tmp_df[syst_weight]
                            * tmp_df[f"{syst_weight}_down{variation}"].fillna(1),
                        )

                        newfile[f"{tree}/{ctgy}/{systematic}_up{variation}"] = hist_up[
                            0
                        ].flatten(), np.linspace(
                            0, 1, hist_up[0].flatten().shape[0] + 1
                        )
                        newfile[
                            f"{tree}/{ctgy}/{systematic}_down{variation}"
                        ] = hist_down[0].flatten(), np.linspace(
                            0, 1, hist_down[0].flatten().shape[0] + 1
                        )

                else:
                    logging.info(
                        "Skipping template in region: %s for fit ctgy: %s", region, ctgy
                    )
                    continue
