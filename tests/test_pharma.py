import math
import numpy as np
import pandas as pd
import pytest

from transkit import pharma, medications


@pytest.fixture
def doses_5doses_4days():
    doses = pharma.createDoses(
        np.array(
            pharma.rep_from(["2020.Jan.01.1200", 1.0, "ec"], n=5, freq="4D")
        )
    )

    ec = medications.medications["ec"]
    actual_levels = pharma.createMeasurements(
        np.array(
            pharma.rep_from_dose(
                "2020.Jan.01.1200",
                [
                    ec(0.0) + 0.0 + 0.0 + 0.0 + 0.0,
                    ec(2.0) + 0.0 + 0.0 + 0.0 + 0.0,
                    ec(4.0) + ec(0.0) + 0.0 + 0.0 + 0.0,
                    ec(6.0) + ec(2.0) + 0.0 + 0.0 + 0.0,
                    ec(8.0) + ec(4.0) + ec(0.0) + 0.0 + 0.0,
                    ec(10.0) + ec(6.0) + ec(2.0) + 0.0 + 0.0,
                    ec(12.0) + ec(8.0) + ec(4.0) + ec(0.0) + 0.0,
                    ec(14.0) + ec(10.0) + ec(6.0) + ec(2.0) + 0.0,
                    ec(16.0) + ec(12.0) + ec(8.0) + ec(4.0) + ec(0.0),
                    ec(18.0) + ec(14.0) + ec(10.0) + ec(6.0) + ec(2.0),
                    ec(20.0) + ec(16.0) + ec(12.0) + ec(8.0) + ec(4.0),
                ],
                "",
                n=11,
                freq="2D",
            )
        )
    )["value"]

    return (doses, actual_levels)


@pytest.fixture
def doses_12doses_5days():
    """Doses spread over a greater time than the domain of the medication."""
    doses = pharma.createDoses(
        np.array(
            pharma.rep_from(["2020.Jan.01.1200", 2.0, "ec"], n=12, freq="6D")
        )
    )

    def ec(T):
        return 2.0 * medications.medications["ec"](T)

    actual_levels = pharma.createMeasurements(
        np.array(
            pharma.rep_from_dose(
                "2020.Jan.01.1200",
                # fmt: off
                [
                    ec( 0.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec( 3.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec( 6.0) + ec( 0.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec( 9.0) + ec( 3.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(27.0) + ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(30.0) + ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(33.0) + ec(27.0) + ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                    ec(36.0) + ec(30.0) + ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  + ec(33.0) + ec(27.0) + ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  + ec(36.0) + ec(30.0) + ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  + ec(33.0) + ec(27.0) + ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  + ec(36.0) + ec(30.0) + ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  +     0.0  + ec(33.0) + ec(27.0) + ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  +     0.0  + ec(36.0) + ec(30.0) + ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  +     0.0  +     0.0  + ec(33.0) + ec(27.0) + ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  +     0.0  +     0.0  + ec(36.0) + ec(30.0) + ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  +     0.0  +     0.0  +     0.0  + ec(33.0) + ec(27.0) + ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) +     0.0  ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  +     0.0  +     0.0  +     0.0  + ec(36.0) + ec(30.0) + ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) + ec( 0.0) ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  + ec(33.0) + ec(27.0) + ec(21.0) + ec(15.0) + ec( 9.0) + ec( 3.0) ,  # noqa: E131,E201,E221,E222,E501
                        0.0  +     0.0  +     0.0  +     0.0  +     0.0  +     0.0  + ec(36.0) + ec(30.0) + ec(24.0) + ec(18.0) + ec(12.0) + ec( 6.0) ,  # noqa: E131,E201,E221,E222,E501
                ],
                # fmt: on
                "",
                n=25,
                freq="3D",
            )
        )
    )["value"]

    return (doses, actual_levels)


def test_create_doses():
    arr = np.array(
        [
            ["2020.Jan.01.1200", 1.0, "ec"],
            ["2020.Jan.06.1200", 1.0, "ec"],
            ["2020.Jan.11.1200", 1.0, "ec"],
            ["2020.Jan.16.1200", 1.0, "ec"],
            ["2020.Jan.21.1200", 1.0, "ec"],
        ]
    )
    doses = pharma.createDoses(arr)
    assert len(doses) == 5


def test_levels_exact1(doses_5doses_4days):
    doses = doses_5doses_4days[0]
    actual_levels = doses_5doses_4days[1]

    levels_calcexact = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsAtMoments(actual_levels.index),
        doses,
        medications.medications,
    )
    assert (actual_levels == levels_calcexact).all()


def test_levels_exact2(doses_12doses_5days):
    doses = doses_12doses_5days[0]
    actual_levels = doses_12doses_5days[1]

    levels_calcexact = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsAtMoments(actual_levels.index),
        doses,
        medications.medications,
    )

    # Different calculation method in this test case results in some
    # slight floating point differences (on the order of 1e-14), so just
    # make sure the values are close within a tolerance. I suspect it's
    # a difference from calculating relative time in days.
    close = [
        math.isclose(c, a, rel_tol=1e-15, abs_tol=np.nextafter(0, 1))
        for c, a in zip(levels_calcexact, actual_levels)
    ]
    assert all(close)


def test_levels_conv1(doses_5doses_4days):
    doses = doses_5doses_4days[0]
    actual_levels = doses_5doses_4days[1]

    levels_calcconv = pharma.calcBloodLevelsConv(
        pharma.zeroLevelsAtMoments(actual_levels.index),
        doses,
        medications.medications,
    )
    assert (actual_levels == levels_calcconv).all()


def test_levels_conv2(doses_12doses_5days):
    doses = doses_12doses_5days[0]
    actual_levels = doses_12doses_5days[1]

    levels_calcconv = pharma.calcBloodLevelsConv(
        pharma.zeroLevelsAtMoments(actual_levels.index),
        doses,
        medications.medications,
    )

    # Different calculation method in this test case results in some
    # slight floating point differences (on the order of 1e-14), so just
    # make sure the values are close within a tolerance. I suspect it's
    # a difference from calculating relative time in days.
    close = [
        math.isclose(c, a, rel_tol=1e-15, abs_tol=np.nextafter(0, 1))
        for c, a in zip(levels_calcconv, actual_levels)
    ]
    assert all(close)


def test_conv_aligned(doses_12doses_5days):
    """
    With a 1 minute sampling frequency and dose times aligned to the
    minutes, Exact and Conv should return the same results within
    a very small tolerance."""

    doses = doses_12doses_5days[0]

    levels_calcexact = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsFromDoses(doses, "1min"),
        doses,
        medications.medications,
    )
    levels_calcconv = pharma.calcBloodLevelsConv(
        pharma.zeroLevelsFromDoses(doses, "1min"),
        doses,
        medications.medications,
    )

    close = [
        math.isclose(e, c, rel_tol=1e-15, abs_tol=1e-12)
        for e, c in zip(levels_calcexact, levels_calcconv)
    ]
    assert all(close)


def test_conv_error(doses_12doses_5days):
    """
    With a 31 minute sampling frequency, Exact and Conv should be
    numerically different, but the error of Conv should be within a
    reasonable limit. 31 minutes ensures that doses are not aligned
    with any samples."""

    doses = doses_12doses_5days[0]

    levels_calcexact = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsFromDoses(doses, "31min"),
        doses,
        medications.medications,
    )
    levels_calcconv = pharma.calcBloodLevelsConv(
        pharma.zeroLevelsFromDoses(doses, "31min"),
        doses,
        medications.medications,
    )

    err = (levels_calcexact - levels_calcconv).abs()
    assert (err < 0.003).all()
    assert err.mean() < 0.0006


def test_levels_conv_input(doses_5doses_4days):
    doses = doses_5doses_4days[0]

    # Test that Conv warns when the sample space doesn't appear to have
    # fixed frequency.
    with pytest.warns(RuntimeWarning):
        pharma.calcBloodLevelsConv(
            pharma.zeroLevelsAtMoments(
                pd.to_datetime([0.0, 1.0, 2.0, 4.0], unit="D")
            ),
            doses,
            medications.medications,
        )

    # Test that Conv fails where there are doses before the start of the
    # sample space.
    with pytest.raises(ValueError):
        pharma.calcBloodLevelsConv(
            pharma.zeroLevelsAtMoments(
                pd.date_range(
                    "2020.Jan.02.1200", "2020.Jan.10.1200", freq="1D"
                )
            ),
            doses,
            medications.medications,
        )
