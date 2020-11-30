import numpy as np
import pandas as pd

from transkit import pharma


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
ef_ec_1mg = pharma.rawDoseResponseToEf(ec_level_1mg)


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
ef_ec_5mg = pharma.rawDoseResponseToEf(ec_level_5mg)
ef_ec_5mg_norm = pharma.normalizedDoseResponse(ef_ec_5mg, 5.0)


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
ef_ev_5mg = pharma.rawDoseResponseToEf(ev_level_5mg)
ef_ev_5mg_norm = pharma.normalizedDoseResponse(ef_ev_5mg, 5.0)


medications = {
    "ec": ef_ec_1mg,
    "ev": ef_ev_5mg_norm,
    "pill-oral": zero_dose,
    "pill-subl": zero_dose,
    "pill-bucc": zero_dose,
}
