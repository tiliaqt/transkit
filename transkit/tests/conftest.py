import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_meds():
    """Return a representative Bateman-function dose response."""

    med_name = "lin"

    def bateman(T, D, ka, ke):
        return (
            D
            * ka
            / (ka - ke)
            * (np.power(np.e, -ke * T) - np.power(np.e, -ka * T))
        )

    # Bateman function parameters
    D = 400.0
    ka = 0.50
    ke = 0.45

    # Medication domain
    min_t = 0.0
    max_t = 30.0

    def med(T):
        return np.where(
            np.logical_or(T <= min_t, T >= max_t),
            np.zeros_like(T),
            bateman(T, D, ka, ke),
        )

    med.domain = (
        pd.to_timedelta(min_t, unit="D"),
        pd.to_timedelta(max_t, unit="D"),
    )

    medications = {med_name: med}
    return (med_name, medications)
