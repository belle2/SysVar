import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sysvar import add_weights_to_dataframe


@pytest.fixture
def toy_df():
    sample_size = 10
    theta = 5.5
    momentum_scaler = MinMaxScaler((0.05, 0.4))
    momentum_scaler.fit(np.random.gamma(theta, 1.0, sample_size).reshape(-1, 1))

    df = pd.DataFrame(
        {
            "channel": np.random.randint(0, 2, sample_size),
            "template": np.random.randint(1, 3, sample_size),
            "slow_pi_p": momentum_scaler.transform(
                np.random.gamma(theta, 1.0, sample_size).reshape(-1, 1)
            ).flatten(),
            "slow_pi2_p": momentum_scaler.transform(
                np.random.gamma(theta, 1.0, sample_size).reshape(-1, 1)
            ).flatten(),
        }
    )

    df["slow_pi_mcPDG"] = np.random.choice([-211, 211], sample_size)
    df["slow_pi2_mcPDG"] = np.random.choice([-211, 211], sample_size)
    df["slow_pi_PDG"] = np.random.choice([-211, 211], sample_size)
    df["slow_pi2_PDG"] = np.random.choice([-211, 211], sample_size)

    df.loc[df.template == 1, "fit_variable1"] = np.random.exponential(
        0.2, len(df[df.template == 1])
    )
    df.loc[df.template == 1, "fit_variable2"] = np.random.normal(
        2.5, 0.3, len(df[df.template == 1])
    )
    df.loc[df.template == 1, "other_weight"] = np.random.normal(
        0.3, 0.04, len(df[df.template == 1])
    )
    df.loc[df.template == 2, "fit_variable1"] = np.random.power(
        1.5, len(df[df.template == 2])
    )
    df.loc[df.template == 2, "fit_variable2"] = np.random.rayleigh(
        1.5, size=len(df[df.template == 2])
    )
    df.loc[df.template == 2, "other_weight"] = np.random.normal(
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
        prefix="slow_pi",
        weightname="charged_weight",
    )

    df["total_weight"] = df[["other_weight", "slow_pi_charged_weight"]].product(axis=1)
    return df
