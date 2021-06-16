import pandas as pd

from transkit import models, pharma


##########################################
### Medication Dose Response Functions ###


def zero_dose(t):
    return 0.0 * t


zero_dose.domain = (
    pd.to_timedelta(0.0, unit="D"),
    pd.to_timedelta(1.0, unit="D"),
)


def ef_ec(t):
    return models.Cmpt3VModel().model(
        t,
        47.8110240,
        1.75321309,
        1.04533601,
        0.11270428,
    )


ef_ec.domain = (
    pd.to_timedelta(0.0, unit="D"),
    pd.to_timedelta(pharma.findDecayToZero(ef_ec), unit="D"),
)


def ef_ev(t):
    return models.Cmpt3VModel().model(
        t,
        91.5928578,
        0.92440910,
        1000.00000,
        0.22498508,
    )


ef_ev.domain = (
    pd.to_timedelta(0.0, unit="D"),
    pd.to_timedelta(pharma.findDecayToZero(ef_ev), unit="D"),
)


medications = {
    "ec": ef_ec,
    "ev": ef_ev,
    "pill-oral": zero_dose,
    "pill-subl": zero_dose,
    "pill-bucc": zero_dose,
}
