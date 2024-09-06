#!/usr/bin/env python

from __future__ import annotations

import uproot
import logging
import typing as t
import numpy as np
import pandas as pd

from sysvar.utils import read_yaml

__all__ = [
    "save_existing_eigenvariations",
]

logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


def save_existing_eigenvariations(
    df: pd.DataFrame,
    analysis: str,
    systematic: str,
) -> None:
    settings = read_yaml("template_setup", analysis)

    root_file_path = settings["output_filepath"]

    reco_channel_id_column: str = settings["reco_channel_id_column"]
    assert reco_channel_id_column in df.columns, reco_channel_id_column
    reco_channel_info: t.Dict[str, t.List[int]] = settings["reco_channels"]

    ctgy_id_column: str = settings["template_id_column"]
    assert ctgy_id_column in df.columns, ctgy_id_column
    template_names: t.List[str] = settings["templates"]

    total_weight: str = settings["total_weight"]
    assert total_weight in df.columns, total_weight

    N_eigen = settings["systematics"][systematic]["N_eigen"]
    syst_weight = settings["systematics"][systematic]["weight"]

    with uproot.update(root_file_path) as newfile:

        logging.info(f"Updating file with uproot: {root_file_path}")
        for reco_channel_name, reco_channel_ids in reco_channel_info.items():

            binning = settings["bins"][reco_channel_name]

            for template_name in template_names:

                q = f"{ctgy_id_column} == '{template_name}' and {reco_channel_id_column} in {reco_channel_ids}"

                tmp_df = df.query(q)
                if len(tmp_df) > 0:
                    logging.info(
                        f"Computing templates in region: {reco_channel_ids} for template: {template_name}"
                    )

                    for variation in range(N_eigen):
                        hist_up = np.histogramdd(
                            np.array(tmp_df[[*binning.keys()]]),
                            bins=[bins for bins in binning.values()],
                            weights=np.array(
                                tmp_df[total_weight]
                                / tmp_df[syst_weight]
                                * tmp_df[f"{syst_weight}_up{variation}"].fillna(1)
                            ),
                        )

                        hist_down = np.histogramdd(
                            np.array(tmp_df[[*binning.keys()]]),
                            bins=[bins for bins in binning.values()],
                            weights=tmp_df[total_weight]
                            / tmp_df[syst_weight]
                            * tmp_df[f"{syst_weight}_down{variation}"].fillna(1),
                        )

                        newfile[
                            f"{reco_channel_name}/{template_name}/{systematic}_up{variation}"
                        ] = hist_up[0].flatten(), np.linspace(
                            0, 1, hist_up[0].flatten().shape[0] + 1
                        )
                        newfile[
                            f"{reco_channel_name}/{template_name}/{systematic}_down{variation}"
                        ] = hist_down[0].flatten(), np.linspace(
                            0, 1, hist_down[0].flatten().shape[0] + 1
                        )

                else:
                    logging.info(
                        f"Skipping template in region: {reco_channel_ids} for template: {template_name}"
                    )
                    continue
