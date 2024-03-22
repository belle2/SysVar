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
