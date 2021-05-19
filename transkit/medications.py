import numpy as np
import pandas as pd


##############
### Models ###


def cmpt3(t, D, k1, k2, k3):
    k1t = np.clip(-k1 * t, None, 300.0)
    k2t = np.clip(-k2 * t, None, 300.0)
    k3t = np.clip(-k3 * t, None, 300.0)

    C_num = (
        D
        * k1
        * k2
        * (
            (k2 - k3) * np.exp(k1t)
            + (k3 - k1) * np.exp(k2t)
            + (k1 - k2) * np.exp(k3t)
        )
    )
    C_den = (k1 - k2) * (k1 - k3) * (k2 - k3)

    if C_den != 0:
        return C_num / C_den
    else:
        return 1e20


##########################################
### Medication Dose Response Functions ###


def zero_dose(t):
    return 0.0 * t


zero_dose.domain = (
    pd.to_timedelta(0.0, unit="D"),
    pd.to_timedelta(1.0, unit="D"),
)


def ef_ec(t):
    return cmpt3(
        t,
        47.8110240,
        1.75321309,
        1.04533601,
        0.11270428,
    )


ef_ec.domain = (
    pd.to_timedelta(0.0, unit="D"),
    pd.to_timedelta(90.0, unit="D"),
)


def ef_ev(t):
    return cmpt3(
        t,
        91.5928578,
        0.92440910,
        1000.00000,
        0.22498508,
    )


ef_ev.domain = (
    pd.to_timedelta(0.0, unit="D"),
    pd.to_timedelta(90.0, unit="D"),
)


medications = {
    "ec": ef_ec,
    "ev": ef_ev,
    "pill-oral": zero_dose,
    "pill-subl": zero_dose,
    "pill-bucc": zero_dose,
}
