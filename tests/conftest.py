import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sysvar import add_weights_to_dataframe


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(8311311)

    sample_size = 10
    theta = 5.5
    momentum_scaler = MinMaxScaler((0.05, 0.4))
    momentum_scaler.fit(rng.gamma(theta, 1.0, sample_size).reshape(-1, 1))

    df = pd.DataFrame(
        {
            "channel": rng.integers(0, 2, sample_size),
            "template": rng.integers(1, 3, sample_size),
            "slow_pi_p": momentum_scaler.transform(
                rng.gamma(theta, 1.0, sample_size).reshape(-1, 1)
            ).flatten(),
            "slow_pi2_p": momentum_scaler.transform(
                rng.gamma(theta, 1.0, sample_size).reshape(-1, 1)
            ).flatten(),
        }
    )

    df["slow_pi_mcPDG"] = rng.choice([-211, 211], sample_size)
    df["slow_pi2_mcPDG"] = rng.choice([-211, 211], sample_size)
    df["slow_pi_PDG"] = rng.choice([-211, 211], sample_size)
    df["slow_pi2_PDG"] = rng.choice([-211, 211], sample_size)

    df.loc[df.template == 1, "fit_variable1"] = rng.exponential(
        0.2, len(df[df.template == 1])
    )
    df.loc[df.template == 1, "fit_variable2"] = rng.normal(
        2.5, 0.3, len(df[df.template == 1])
    )
    df.loc[df.template == 1, "other_weight"] = rng.normal(
        0.3, 0.04, len(df[df.template == 1])
    )
    df.loc[df.template == 2, "fit_variable1"] = rng.power(
        1.5, len(df[df.template == 2])
    )
    df.loc[df.template == 2, "fit_variable2"] = rng.rayleigh(
        1.5, size=len(df[df.template == 2])
    )
    df.loc[df.template == 2, "other_weight"] = rng.normal(
        0.8, 0.1, len(df[df.template == 2])
    )

    df = df.query("0 < fit_variable1 < 1 and 1 < fit_variable2 < 4")
    df = df.query("0.05 < slow_pi_p < 0.4 and 0.05 < slow_pi2_p < 0.4")

    df["template"].replace(1, "signal", inplace=True)
    df["template"].replace(2, "bkg", inplace=True)

    add_weights_to_dataframe(
        df=df,
        systematic="charged_slow_pi",
        MC_production="sysvar_101",
        csv_path=None,
        prefix="slow_pi",
        weightname="charged_weight",
    )

    df["total_weight"] = df[["other_weight", "slow_pi_charged_weight"]].product(axis=1)
    return df
