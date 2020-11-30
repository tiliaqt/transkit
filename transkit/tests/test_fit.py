import functools
import numpy as np
import pytest

from transkit import pharma, fit


def validateX0AndBounds(
    doses, n, X0, X_bounds, X_mapping, equal_doses=None, nonoverlapping=True
):
    assert len(X0) == n
    if n > 0:
        assert (
            len(X_bounds) == 2
            and len(X_bounds[0]) == n
            and len(X_bounds[1]) == n
        )
    else:
        assert X_bounds == (-np.inf, np.inf)

    unmapped = fit.createDosesFromX(X0, X_mapping)
    assert (unmapped.index == doses.index).all()
    assert (unmapped["medication"] == doses["medication"]).all()

    ud = unmapped["dose"]
    dd = doses["dose"]
    if equal_doses is None:
        assert (ud == dd).all()
    else:
        for eqd in equal_doses:
            # Every dose in the partition must be equal
            assert (ud.loc[eqd[0]] == ud.loc[eqd]).all()

            # The original value of the first dose in the partition is
            # the value of the entire partition. If the partitions are
            # overlapping though, it's more complicated, so manually
            # check that outside this function.
            if nonoverlapping:
                assert (ud.loc[eqd[0]] == dd.loc[eqd[0]]).all()

        # Every unpartitioned dose must be equal to its original values
        all_parted = functools.reduce(lambda a, b: a.union(b), equal_doses)
        ud_unpart = ud[ud.index.isin(all_parted) == False]  # noqa: E712
        dd_unpart = dd[dd.index.isin(all_parted) == False]  # noqa: E712
        assert (ud_unpart == dd_unpart).all()


@pytest.mark.filterwarnings(
    "ignore:Midpoint time bounds are only supported:RuntimeWarning"
)
def test_X0_and_bounds():
    with pytest.raises(ValueError):
        fit.createX0AndBounds(pharma.createDoses([]))

    single_dose = pharma.createDoses([["2020", 1.0, "med"]])
    with pytest.warns(RuntimeWarning):
        validateX0AndBounds(
            single_dose,
            1,
            *fit.createX0AndBounds(single_dose),
        )
    with pytest.raises(ValueError):
        fit.createX0AndBounds(
            single_dose,
            dose_bounds=[(0.0, 10.0), (1.0, 9.0)],
        )
    with pytest.raises(ValueError):
        fit.createX0AndBounds(single_dose, time_bounds=["fixed", "midpoints"])

    doses = pharma.createDoses(
        [
            ["2020.Jan.01.1200", 1.0, "one"],
            ["2020.Jan.05.1200", 2.0, "two"],
            ["2020.Jan.09.1200", 3.0, "three"],
            ["2020.Jan.13.1200", 4.0, "four"],
        ]
    )

    with pytest.raises(ValueError):
        fit.createX0AndBounds(doses, dose_bounds="dexif")
    with pytest.raises(ValueError):
        fit.createX0AndBounds(doses, time_bounds="dexif")

    # No partitions and no fixed bounds
    validateX0AndBounds(
        doses,
        2 * len(doses),
        *fit.createX0AndBounds(doses),
    )

    # One partition and no fixed bounds
    equal_doses = [doses.index[0:2]]
    validateX0AndBounds(
        doses,
        len(doses) + len(doses) - 1,
        *fit.createX0AndBounds(doses, equal_doses=equal_doses),
        equal_doses,
    )

    # Two partitions and no fixed bounds
    equal_doses = [doses.index[0:2], doses.index[2:]]
    validateX0AndBounds(
        doses,
        len(doses) + len(doses) - 2,
        *fit.createX0AndBounds(doses, equal_doses=equal_doses),
        equal_doses,
    )

    # Two overlapping partitions covering all doses, and no fixed bounds
    equal_doses = [doses.index[0:2], doses.index[1:]]
    X0, X_bounds, X_mapping = fit.createX0AndBounds(
        doses,
        equal_doses=equal_doses,
    )
    validateX0AndBounds(
        doses,
        len(doses) + 1,
        X0,
        X_bounds,
        X_mapping,
        equal_doses,
        nonoverlapping=False,
    )
    # The entire merged partition should have the original value of the
    # first dose in the first un-merged partition.
    assert X0[len(doses)] == doses["dose"].iloc[0]

    # One big partition and no fixed bounds
    equal_doses = [doses.index[0:]]
    validateX0AndBounds(
        doses,
        len(doses) + 1,
        *fit.createX0AndBounds(doses, equal_doses=equal_doses),
        equal_doses,
    )

    # No partitions, two fixed bounds
    validateX0AndBounds(
        doses,
        len(doses) - 1 + len(doses) - 1,
        *fit.createX0AndBounds(
            doses,
            time_bounds=["fixed", "midpoints", "midpoints", "midpoints"],
            dose_bounds=[(0.0, 10.0), (1.0, 9.0), (2.0, 7.0), "fixed"],
        ),
    )

    # No partitions, all fixed
    validateX0AndBounds(
        doses,
        0,
        *fit.createX0AndBounds(
            doses,
            time_bounds="fixed",
            dose_bounds="fixed",
        ),
    )

    # One partition, two fixed bounds, one fixed in the partition
    equal_doses = [doses.index[-2:]]
    validateX0AndBounds(
        doses,
        len(doses) - 1 + len(doses) - 2,
        *fit.createX0AndBounds(
            doses,
            time_bounds=["fixed", "midpoints", "midpoints", "midpoints"],
            dose_bounds=[(0.0, 10.0), (1.0, 9.0), (2.0, 7.0), "fixed"],
            equal_doses=equal_doses,
        ),
        equal_doses,
    )
