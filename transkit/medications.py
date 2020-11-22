import numpy as np
import pandas as pd
from scipy import interpolate


#################################################################
### Helper functions for working with Dose Response functions ###


def normalizedDoseResponse(ef, ef_dose):
    def ef_norm(T):
        return ef(T) / ef_dose

    ef_norm.domain = ef.domain
    return ef_norm


def calibratedDoseResponse(ef, X):
    """
    Wraps the dose response function ef and transforms it by the linear
    power series specified by the first-order polynomial coefficients in
    ndarray X, such that:

        calibratedDoseResponse(ef, X)(T) = X[0] + X[1]*ef(T)
    """

    def ef_cali(T):
        return X[0] + X[1] * ef(T)

    ef_cali.domain = ef.domain
    return ef_cali


def rawDoseResponseToEf(raw):
    """
    Converts an ndarray of flaot(Day),float(pg/mL) pairs into an
    interpolated function ef:

        ef(time_since_dose [Days]) = blood concentration [pg/mL]

    representing the instantaneous total blood concentration of the
    medication at the given time after administration.

    Expects two zero values at both the start and end, used to tweak the
    slope of the interpolated curve to end at 0.0. Ef must be positive
    and continuous across the domain of the input data, and equal to 0.0
    outside that domain. This is required for later computations using
    this function to operate correctly.

    The returned function ef is called in the inner loop of all
    subsequent computations, so it is heavily optimized for performance
    by constructing lookup tables of the interpolated function. LUT
    table sizes < 100000 seem to make first derivatives of ef get
    wiggly."""

    interp_ef = interpolate.interp1d(
        raw[:, 0],
        raw[:, 1],
        kind="cubic",
        fill_value=(0.0, 0.0),
        bounds_error=False,
    )
    min_x = raw[1, 0]
    max_x = raw[-2, 0]
    sample_x = np.linspace(min_x, max_x, num=100000)
    sample_y = np.array([interp_ef(x) for x in sample_x])
    slope_lut = (sample_y[1:] - sample_y[:-1]) / (sample_x[1:] - sample_x[:-1])

    def ef(x):
        idx = np.searchsorted(sample_x, x) - 1
        return np.where(
            np.logical_or(x <= min_x, x >= max_x),
            np.zeros_like(x),
            sample_y[idx]
            + (x - sample_x[idx]) * np.take(slope_lut, idx, mode="clip"),
        )

    ef.domain = (
        pd.to_timedelta(min_x, unit="D"),
        pd.to_timedelta(max_x, unit="D"),
    )

    return ef


##########################################
### Medication Dose Response Functions ###


def zero_dose(T):
    return 0.0 * T


zero_dose.domain = (
    pd.to_timedelta(0.0, unit="D"),
    pd.to_timedelta(1.0, unit="D"),
)


# Estradiol Cypionate 1.0mg
# https://en.wikipedia.org/wiki/Template:Hormone_levels_with_estradiol_cypionate_by_intramuscular_injection#/media/File:Estradiol_levels_after_a_single_intramuscular_injection_of_1.0_to_1.5-mg_estradiol_cypionate_in_hypogonadal_girls.png
ec_level_1mg = np.array(
    [
        [-1.0, 0.0],
        [0.0, 0.0],
        [2.0, 50.0],
        [3.0, 62.0],
        [4.0, 65.0],
        [5.0, 65.0],
        [6.0, 63.0],
        [7.0, 59.0],
        [10.0, 45.0],
        [11.0, 39.0],
        [13.0, 28.0],
        [14.0, 25.0],
        [15.0, 23.0],
        [18.0, 19.0],
        [20.0, 17.0],
        [37.0, 0.0],
        [38.0, 0.0],
    ]
)
ef_ec_1mg = rawDoseResponseToEf(ec_level_1mg)


# Estradiol Cypionate 5.0mg
# https://en.wikipedia.org/wiki/Template:Hormone_levels_with_estradiol_valerate_by_intramuscular_injection#/media/File:Estradiol_levels_after_a_single_5_mg_intramuscular_injection_of_estradiol_esters.png
ec_level_5mg = np.array(
    [
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 80.0],
        [2.0, 230.0],
        [2.1, 230.0],  # control point
        [3.0, 210.0],
        [3.8, 220.0],  # control point
        [4.0, 230.0],
        [4.4, 330.0],
        [4.5, 315.0],  # control point
        [5.0, 225.0],
        [6.0, 185.0],
        [7.0, 175.0],
        [8.0, 155.0],
        [9.0, 115.0],
        [10.0, 100.0],
        [11.0, 90.0],
        [12.0, 80.0],
        [13.0, 55.0],
        [14.0, 45.0],
        [24.0, 0.0],
        [25.0, 0.0],
    ]
)
ef_ec_5mg = rawDoseResponseToEf(ec_level_5mg)
ef_ec_5mg_norm = normalizedDoseResponse(ef_ec_5mg, 5.0)


# Estradiol Valerate 5.0mg
# https://en.wikipedia.org/wiki/Template:Hormone_levels_with_estradiol_valerate_by_intramuscular_injection#/media/File:Estradiol_levels_after_a_single_5_mg_intramuscular_injection_of_estradiol_esters.png
ev_level_5mg = np.array(
    [
        [-1.0, 0.0],
        [0.0, 0.0],
        [0.5, 160.0],
        [2.0, 610.0],
        [2.5, 675.0],
        [3.0, 515.0],
        [4.0, 310.0],
        [5.0, 210.0],
        [6.0, 115.0],
        [7.0, 70.0],
        [8.0, 60.0],
        [9.0, 45.0],
        [10.0, 35.0],
        [11.0, 20.0],
        [12.0, 15.0],
        [13.0, 10.0],
        [21.0, 0.0],
        [22.0, 0.0],
    ]
)
ef_ev_5mg = rawDoseResponseToEf(ev_level_5mg)
ef_ev_5mg_norm = normalizedDoseResponse(ef_ev_5mg, 5.0)


medications = {
    "ec": ef_ec_1mg,
    "ev": ef_ev_5mg_norm,
    "pill-oral": zero_dose,
    "pill-subl": zero_dose,
    "pill-bucc": zero_dose,
}
