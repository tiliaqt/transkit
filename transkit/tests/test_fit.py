import numpy as np

from transkit import pharma, fit


def test_partitioning():
    # damn this is needed!
    pass


def test_X0_and_bounds(simple_meds):
    med = simple_meds[0]

    doses = pharma.createDoses(
        np.array(
            [*pharma.rep_from(["2020.Jan.01.1200", 1.0, med], n=4, freq="4D")]
        )
    )

    # Check with no partitions and no fixed bounds
    X0, X_bounds, X_mapping = fit.createX0AndBounds(doses)
    assert len(X0) == 2 * len(doses)
