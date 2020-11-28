import math
from matplotlib import collections as mc
from matplotlib import dates as mdates
import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, root_scalar
import warnings


from transkit import pharma, medications


####################################################
### Fitting doses to a desired blood level curve ###


def partitionDoses(doses, equal_doses):
    """
    Partition the doses into disjoint equal-amount equivalence classes
    using the specified equivalence properties.

    Note that any doses with duplicate time indices that are part of an
    equivalence class in equal_doses will be irreversibly collapsed into
    a single dose. This function expects the index of doses to be a set,
    but Pandas actually represents the index as a multiset. If you need
    doses to happen at the same moment in time, just perturb one by an
    infinitesimal amount so that their indices don't compare equal.

    Returns a list of DatetimeIndexes, where each DatetimeIndex represents a
    disjoint equivalence class of doses.

    Parameters:
    ===========
    doses        DataFrame of doses (see pharma.createdoses(...)).
    equal_doses  Sequence of DatetimeIndexes where each DatetimeIndex
                 corresponds to doses that are defined to have equal
                 doses."""

    def areEquivalent(sa, sb, eq_sets):
        for eq in eq_sets:
            if not sa.isdisjoint(eq) and not sb.isdisjoint(eq):
                return True
        return False

    # This implementation is O(n**2)–worse than the optimal O(n*log(n))
    # of a disjoint set forest–but the number of doses n will generally
    # be small, and this is a lot simpler!

    eq_sets = [set(e) for e in equal_doses]
    partitions = [{dose_time} for dose_time in doses.index]
    p = 0
    while p < len(partitions):
        to_del = set()
        for j in range(p + 1, len(partitions)):
            if areEquivalent(partitions[p], partitions[j], eq_sets):
                partitions[p] |= partitions[j]
                to_del.add(j)
        partitions[:] = [
            partitions[d] for d in range(len(partitions)) if d not in to_del
        ]
        p += 1
    return [pd.DatetimeIndex(part).sort_values() for part in partitions]


def createInitialAndBounds(
    doses,
    dose_bounds=(0.0, np.inf),
    time_bounds="midpoints",
    equal_doses=[],
):
    """
    Create the initial X vector of independent variables that will be
    optimized, the partitions of equal-amount doses, and the bounds
    corresponding to X.

    Returns a 3-tuple of: (
        X:          Vector of independent variables that will be
                    optimized by least_squares, represented as a stacked
                    array of len(doses) dose times in days, followed by
                    len(partitions) dose amounts.
        partitions: Equal-amount partitions of doses, represented as a
                    list P of partitions, where each P[j] is a set {i_0,
                    ..., i_k} of integer indices representing the doses
                    contained in the partition, and X[len(doses)-1+j] is
                    the dose amount corresponding to the partition.
        bounds:     2-tuple of ([lower_0, ...],[upper_0, ...]) bounds
                    on the independent variables in X, as expected by
                    least_squares.
    )

    Parameters:
    ===========
    doses        Initial doses to be optimized.
    dose_bounds  Specifies how dose amounts will be bounded in the
                 fitting. A single dose bound can be specified as either
                 a (min_dose, max_dose) 2-tuple or a string specifying
                 specific bound behavior. Dose_bounds can either be an
                 Iterable of those single dose bounds corresponding to
                 the bounds for each dose, or a single dose bound that
                 will be used for all doses. If dose_bounds is used in
                 conjunction with equal_doses, the bound corresponding
                 to the first dose in each partition will be used for
                 the entire parttion. Possible string options are:
                    'fixed': The dose will be fixed to its initial
                             value and won't be optimized.
                 Defaults to (0.0, np.inf).
    time_bounds  Specfies how dose times will be bounded in the
                 fitting. Can either be an Iterable of strings
                 corresponding to the time bound behavior for each
                 dose, or a single string that will be used for all
                 doses. Possible string options are:
                    'fixed':     the dose times will not be optimized.
                    'midpoints': the dose times will be optimized and
                                 bounded to the midpoints between their
                                 nearest neighboring doses (default).
    equal_doses  A list of DatetimeIndexes where each DatetimeIndex
                 corresponds to doses that should be constrained to have
                 the same amount, partitioning the doses. The initial
                 doses are set by the first dose within each
                 partition."""

    if len(doses) == 0:
        raise ValueError("Need at least 1 initial dose to optimize.")
    elif len(doses) == 1 and (
        time_bounds == "midpoints" or time_bounds == ["midpoints"]
    ):
        time_bounds = "fixed"
        warnings.warn(
            "Midpoint time bounds are only supported with more than 1 "
            "dose. Continuing with time_bounds = 'fixed'.",
            RuntimeWarning,
        )

    if isinstance(dose_bounds, str) or (
        isinstance(dose_bounds, tuple)
        and not isinstance(dose_bounds[0], tuple)
    ):
        # It's a single string or a single tuple, and not a
        # tuple-of-tuples, or some other Iterable.
        dose_bounds = len(doses) * [dose_bounds]
    elif len(dose_bounds) != len(doses):
        raise ValueError(
            f"Expecting either a single 2-tuple, a single string, or "
            f"an Iterable of length len(doses)={len(doses)} for "
            f"dose_bounds, but got {type(dose_bounds)} of length "
            f"{len(dose_bounds)}."
        )

    if isinstance(time_bounds, str):
        time_bounds = len(doses) * [time_bounds]
    elif len(time_bounds) != len(doses):
        raise ValueError(
            f"Expecting either a single string or an Iterable of "
            f"length len(doses)={len(doses)} for time_bounds, but got "
            f"{type(time_bounds)} of length {len(time_bounds)}."
        )

    # least_squares works on float parameters, so convert our dose dates
    # into float days.
    times = pharma.dateTimeToDays(doses.index)

    # Pregenerate all the midpoint bounds even if we don't end up using
    # them. It simplifies the loop a lot.
    midpoints = (times[:-1] + times[1:]) / 2.0
    midpoint_bounds = (
        [(2 * times[0] - midpoints[0], midpoints[0])]
        + [(lb, rb) for lb, rb in zip(midpoints[:-1], midpoints[1:])]
        + [(midpoints[-1], 2 * times[-1] - midpoints[-1])]
    )

    t_bounds = []
    for ti in range(len(times)):
        tb = time_bounds[ti]
        if tb == "midpoints":
            # Midpoint time bound handling:
            #   The first dose's minimum time bound is the same distance
            #    away as the first midpoint, but mirrored to the negative
            #    direction.
            #   The last dose's maximum time bound is the
            #    same distance away as the previous midpoint, but
            #    mirrored to the positive direction.
            #   All other bounds are at the midpoints between dose times.
            t_bounds.append(midpoint_bounds[ti])
        elif tb == "fixed":
            # least_squares doesn't have in-built capacity for fixed
            # parameters, and also requires each lower bound to be
            # strictly less than each upper bound. This is basically a
            # hack to avoid requiring different implementations for
            # optimizing on amounts+times vs just amounts, keeping the
            # same array X of independent variables in both cases.
            t_bounds.append(
                (
                    times[ti],
                    np.nextafter(times[ti], times[ti] + 1.0),
                )
            )
        else:
            raise ValueError(f"'{tb}' isn't a valid input in time_bounds.")

    partitions = partitionDoses(doses, equal_doses)

    # partitionDoses gives equivalence classes of Datetimes, but
    # ultimately we need equivalence classes of dose *indices* because
    # the exact times will change through the optimization and no longer
    # be valid indices. What we later return from this function is
    # essentially telling which dose amounts in X correspond to which
    # times in X.
    part_indices = [
        {list(doses.index).index(dose_time) for dose_time in part}
        for part in partitions
    ]

    part_doses = np.zeros(len(partitions))
    d_bounds = []
    for p in range(len(partitions)):
        # Use the amount of the first dose in this partition as the
        # initial value for the entire partition.
        amount = doses["dose"][partitions[p][0]]
        part_doses[p] = amount

        # Use the bound of the first dose in this partition as the bound
        # for the entire partition.
        db = dose_bounds[next(iter(part_indices[p]))]
        if isinstance(db, str):
            if db == "fixed":
                d_bounds.append((amount, np.nextafter(amount, amount + 1.0)))
            else:
                raise ValueError(
                    f"'{db}' isn't a valid string option in dose_bounds."
                )
        elif isinstance(db, tuple) and len(db) == 2:
            d_bounds.append(db)
        else:
            raise ValueError(
                f"'{db}' isn't a valid string or 2-tuple bound in "
                f"dose_bounds"
            )

    return (
        np.concatenate((times, part_doses)),
        part_indices,
        tuple(zip(*t_bounds, *d_bounds)),
    )


def createDosesFromX(X, X_partitions, X_medications):
    """
    Create a doses DataFrame from the stacked X vector of independent
    variables.

    Scipy's optimization functions just pass around the X vector of
    independent variables, but we need to reconstruct the full doses
    DataFrame to pass to pharma.calcBloodLevels(...).

    Returns a DataFrame of doses (see pharma.createDoses).

    Parameters:
    ===========
    X               X vector of independent variables (see
                    createInitialAndBounds).
    X_partitions    Equal-amount partitions of doses (see
                    createInitialAndBounds). It is used to unpartition
                    the dose amounts in X.
    X_medications   Column of medications corresponding to each dose,
                    exactly as it was in the original doses
                    DataFrame."""

    n_doses = X.size - len(X_partitions)

    times = X[0:n_doses]

    # Unpartition the doses
    part_dose_amts = X[n_doses:]
    dose_amts = np.zeros_like(times)
    for p in range(len(X_partitions)):
        for d in X_partitions[p]:
            dose_amts[d] = part_dose_amts[p]

    doses_array = np.concatenate([times, dose_amts, X_medications])
    return pharma.createDoses(
        doses_array.reshape((n_doses, 3), order="F"),
        date_unit="D",
    )


def initializeFit(
    doses_init,
    medications_map,
    target,
    dose_bounds=(0.0, np.inf),
    time_bounds="midpoints",
    equal_doses=[],
    exclude_area=pd.DatetimeIndex([]),
):
    """
    Get everything set up to run least squares, and organize it into a
    tidy dict.

    Returns a dict containing all the relevant information about the
    optimization run, ready to be used with runLeastSquares.

    Parameters:
    ===========
    doses_init        Initial set of doses to be optimized.
    medications_map   Mapping of medication dose-response functions
                      (see medications.medications).
    target            Target function to fit the blood level curve to
                      [Pandas series index by Datetime].
    dose_bounds       See createInitialAndBounds.
    time_bounds       See createInitialAndBounds.
    equal_doses       See createInitialAndBounds.
    exclude_area      DatetimeIndex corresponding to points in target
                      that should not be considered when computing
                      residuals for the fit."""
    run = {}
    run["doses_init"] = doses_init
    run["medications_map"] = medications_map
    run["X0"], run["partitions"], run["bounds"] = createInitialAndBounds(
        doses_init,
        dose_bounds=dose_bounds,
        time_bounds=time_bounds,
        equal_doses=equal_doses,
    )
    run["exclude_area"] = exclude_area
    run["target"] = target
    run["result"] = None
    run["doses_optim"] = None
    return run


def runLeastSquares(run, max_nfev=20, **kwargs):
    X_medications = run["doses_init"]["medication"].values

    # Residuals function
    def fTimesAndDoses(
        X, X_partitions, X_medications, medications_map, target, exclude_area
    ):
        doses = createDosesFromX(X, X_partitions, X_medications)
        included_target = target[
            target.index.isin(exclude_area) == False  # noqa: E712
        ]
        residuals = pharma.zeroLevelsAtMoments(included_target.index)
        pharma.calcBloodLevelsExact(residuals, doses, medications_map)
        residuals -= included_target
        return residuals

    result = least_squares(
        fTimesAndDoses,
        run["X0"],
        args=(
            run["partitions"],
            X_medications,
            run["medications_map"],
            run["target"],
            run["exclude_area"],
        ),
        bounds=run["bounds"],
        max_nfev=max_nfev,
        **kwargs,
    )

    # Put the optimized X vector back into a DataFrame
    doses_optim = createDosesFromX(result.x, run["partitions"], X_medications)

    run["result"] = result
    run["doses_optim"] = doses_optim

    return result


def returnToSteady(
    doses,
    steady_amount,
    steady_interval,
    steady_medication,
    medications,
    soonest_dose_time=None,
    stabilize_time=None,
    ideal_tod=None,
):
    """
    Determine a dosing regimen to get to steady-state.

    Parameters:
    ===========
    doses              Starting doses to continue to steady-state.
    steady_amount      Dose amount at steady-state in mg.
    steady_interval    Dose interval at steady-state, as a Pandas
                       frequency offset or string.
    steady_medication  Medication to dose at steady-state (string
                       corresponding to a medication in medications).
    medications        Dict of medication dose-response functions (see
                       medications.medications).
    soonest_dose_time  Time of the soonest possible dose. Typically
                       you might pass pd.Timestamp.now() as the value.
                       Defaults to None. The first new dose will be at
                       this time if either dosing ASAP is the best
                       approach to return to steady-state, or if the
                       ideal dose time is in the past relative to this
                       time.
    stabilize_time     Determines how new doses will be stabilized to
                       reach steady-state. If None, no stabilization
                       will be done. If 0.0, only the first new dose
                       will be stabilized. If np.inf, all new doses will
                       be stabilized. Any other floating point number is
                       the time range in days after the first new dose
                       within which new doses should be stabilized. When
                       stabilizing, least squares is used to optimize
                       the dose amounts to best fit steady-state
                       concentration.
    ideal_tod          Time of day when doses should ideally occur,
                       as a Pandas frequency offset or string specifying
                       the offset from midnight. Defaults to None, in
                       which case new doses will occurr at
                       steady_interval intervals after the time of the
                       first new dose.
    """

    interval_offset = pd.tseries.frequencies.to_offset(steady_interval)
    interval_days = interval_offset.nanos * 1e-9 / 60.0 / 60.0 / 24.0

    # Estimate steady state concentrations
    med = medications[steady_medication]
    med_dom = pharma.timeDeltaToDays(med.domain[1] - med.domain[0])
    n_doses_ss = math.ceil(med_dom / interval_days)
    med_x = np.linspace(
        pharma.timeDeltaToDays(med.domain[0]),
        pharma.timeDeltaToDays(med.domain[1]),
        1000,
    )
    c_ss = simps(steady_amount * med(med_x), x=med_x) / interval_days
    c_trough = np.sum(
        steady_amount
        * med(
            np.arange(
                interval_days, interval_days * (n_doses_ss + 1), interval_days
            )
        )
    )

    levels_after = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsFromDoses(
            doses.iloc[-1:],
            "15min",
            lower_bound="exact",
            upper_bound=("domain", medications),
        ),
        doses,
        medications,
    )

    # Find where the levels curve instersects with the steady-state trough.
    # We look for the descending-side intersection that occurs after the
    # maximum concentration following the last dose.
    peak_t = levels_after.idxmax()
    root_bracket = (
        pharma.timeDeltaToDays(peak_t - levels_after.index[0]),
        pharma.timeDeltaToDays(levels_after.index[-1] - levels_after.index[0]),
    )

    if levels_after[peak_t] >= c_trough:
        # If levels rise to or over the steady-state trough, time the next dose
        # at the time levels either touch the trough levels or descend back to
        # the trough levels.

        interp_levels = interp1d(
            pharma.timeDeltaToDays(levels_after.index - levels_after.index[0]),
            levels_after.values - c_trough,
        )
        root = root_scalar(interp_levels, bracket=root_bracket)

        if not root.converged:
            # This shouldn't happen
            raise RuntimeError(
                "uh oh, couldn't figure out where to put the next injection"
            )

        next_dose_t = doses.index[-1] + pd.to_timedelta(root.root, unit="D")

    elif peak_t > levels_after.index[0]:
        # If levels don't reach the trough, but do rise after the last dose,
        # time the next dose at either the next steady interval, or the point
        # at which the levels would fall below where they started.

        interp_levels = interp1d(
            pharma.timeDeltaToDays(levels_after.index - levels_after.index[0]),
            levels_after.values - levels_after.iloc[0],
        )
        root = root_scalar(interp_levels, bracket=root_bracket)

        if not root.converged:
            # This shouldn't happen
            raise RuntimeError(
                "uh oh, couldn't figure out where to put the next injection"
            )

        root_dose_t = doses.index[-1] + pd.to_timedelta(root.root, unit="D")
        interval_dose_t = pharma.snapToTime(
            doses.index[-1] + interval_offset,
            snap_to=ideal_tod,
        )
        if root_dose_t <= interval_dose_t:
            next_dose_t = root_dose_t
        else:
            next_dose_t = interval_dose_t

    else:
        # If levels don't reach the trough and immediately fall after the
        # last dose, time the next dose either as soon as possible, or at
        # the next steady interval.

        next_dose_t = pharma.snapToTime(
            doses.index[-1] + interval_offset,
            snap_to=ideal_tod,
        )

        if (
            soonest_dose_time is not None
            and soonest_dose_time > levels_after.index[0]
            and soonest_dose_time < next_dose_t
        ):
            next_dose_t = soonest_dose_time

    # Make sure we're not dosing in the past.
    if soonest_dose_time is not None and soonest_dose_time > next_dose_t:
        next_dose_t = soonest_dose_time

    # Ideally the followup dose would also happen by that same logic,
    # and would happen when crossing the trough, but it's challenging
    # becuase it can only be fully determined after stabilizing the first
    # dose (and stabilizing the first dose requires stabilizing the whole
    # series of doses). We could potentially do this recursively for each
    # subsequent dose--I don't know if it would be faster or slower than
    # doing least_squares.
    followup_dose_t = pharma.snapToTime(
        next_dose_t + interval_offset,
        snap_to=ideal_tod,
    )

    new_doses = pd.concat(
        [
            doses,
            pharma.createDoses(
                np.array([[next_dose_t, steady_amount, steady_medication]])
            ),
            pharma.createDoses(
                np.array(
                    pharma.rep_from(
                        [followup_dose_t, steady_amount, steady_medication],
                        n=n_doses_ss - 1,
                        freq=steady_interval,
                    )
                )
            ),
        ]
    )

    if stabilize_time is None:
        return (new_doses, None)

    else:
        # Now adjust the new doses so they are stable around the average
        # steady-state concentration.
        # TODO: I wonder if i could get close by estimating the number
        # of missed doses?

        if stabilize_time == np.inf:
            # Stabilize with all of the succeeding doses
            stabilizing_doses = new_doses[next_dose_t:]
        else:
            # Stabilize only within the specified window
            stabilizing_max_t = next_dose_t + pd.to_timedelta(
                stabilize_time, unit="D"
            )
            stabilizing_doses = new_doses[next_dose_t:stabilizing_max_t]

        # All doses after the stabilizing doses should be fixed.
        dose_bounds = (
            len(doses) * ["fixed"]
            + len(stabilizing_doses) * [(0.0, np.inf)]
            + (
                (len(new_doses) - len(stabilizing_doses) - len(doses))
                * ["fixed"]
            )
        )

        # All doses after the stabilizing doses should be equal.
        ndi = new_doses.index
        equal_doses = [
            new_doses[
                (ndi.isin(doses.index) == False)  # noqa: E712
                & (ndi.isin(stabilizing_doses.index) == False)  # noqa: E712
            ].index
        ]

        fit_ss = pharma.zeroLevelsFromDoses(new_doses, "12H")
        fit_ss.values[:] += c_ss
        fit_run = initializeFit(
            new_doses,
            medications,
            fit_ss,
            time_bounds="fixed",
            dose_bounds=dose_bounds,
            equal_doses=equal_doses,
            exclude_area=fit_ss[:next_dose_t].index,
        )

        with warnings.catch_warnings():
            # least_squares sometimes divides by zero with the way we use
            # infinitesimal ranges to represent fixed bounds. Just suppress
            # the warning about it.
            warnings.simplefilter("ignore")
            runLeastSquares(fit_run, max_nfev=25, ftol=1e-2)

        return (fit_run["doses_optim"], fit_run)


def plotOptimizationRun(fig, ax, run):
    # Calculate levels at the initial and optimized moments of dose
    init_levels = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsAtMoments(run["doses_init"].index),
        run["doses_init"],
        run["medications_map"],
    )
    optim_levels = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsAtMoments(run["doses_optim"].index),
        run["doses_optim"],
        run["medications_map"],
    )

    ax.plot(run["target"], label="Target Curve")
    ax.plot(
        run["target"][run["exclude_area"]],
        marker="o",
        color=(1.0, 0.1, 0.1, 0.5),
        label="excluded from residuals",
    )
    pharma.plotDoses(
        fig,
        ax,
        run["doses_init"],
        run["medications_map"],
        sample_freq="6H",
        label="Initial condition",
    )
    pharma.plotDoses(
        fig,
        ax,
        run["doses_optim"],
        run["medications_map"],
        sample_freq="6H",
        label="Optimized solution",
    )

    # Draw lines between the initial doses and the optimized doses
    init_points = [
        (mdates.date2num(T), level) for T, level in init_levels.items()
    ]
    optim_points = [
        (mdates.date2num(T), level) for T, level in optim_levels.items()
    ]

    lines = list(zip(init_points, optim_points))
    lc = mc.LineCollection(lines, linestyles="dotted", zorder=10)
    ax.add_collection(lc)
    ax.legend()

    ax.set_xlim(
        (
            mdates.date2num(run["target"].index[0]),
            mdates.date2num(run["target"].index[-1]),
        )
    )


##############################################################
### Fitting a calibration polynomial on ef to measurements ###


def calibrateDoseResponse_lsqpoly(
    doses,
    uncalibrated_medications,
    medication,
    measurements,
):
    """
    Calibrates the dose-response of a specific medication given a series
    of doses and blood-level measurements from the same time period.
    This uses least squares regression to compute polynomial
    coefficients X that calibrate the medication as: X[0] + X[1]*ef(T),
    such that the errors between the simulated blood levels and the
    measurements are minimized.

    Returns a calibrated copy of 'uncalibrated_medications' and the
    calibrated polynomial coefficients for the specified medication.

    Parameters:
    ===========
    doses                     DataFrame of doses (see
                              pharma.createDoses()).
    uncalibrated_medications  Dict of medications (see
                              medications.medications).
    medication                Which medication in
                              'uncalibrated_medications' to calibrate.
    measurements              DataFrame of blood level measurements
                              corresponding to the actual dose response
                              of the to-be-calibrated medication (see
                              pharma.createMeasurements())."""

    calibrated_medications = dict(uncalibrated_medications)

    # Residuals function
    def f(X):
        calibrated_medications[
            medication
        ] = medications.calibratedDoseResponse(
            uncalibrated_medications[medication], X
        )
        levels_at_measurements = pharma.calcBloodLevelsExact(
            pharma.zeroLevelsAtMoments(measurements.index),
            doses,
            calibrated_medications,
        )
        return levels_at_measurements - measurements["value"]

    # Initial condition is un-transformed ef(T)
    X0 = np.array([0.0, 1.0])
    result = least_squares(
        f,
        X0,
        bounds=(np.zeros_like(X0), [np.nextafter(0.0, 1.0), np.inf]),
        max_nfev=10,
        verbose=0,
    )
    calibrated_medications[medication] = medications.calibratedDoseResponse(
        uncalibrated_medications[medication],
        result.x,
    )
    return calibrated_medications, result.x


def calibrateDoseResponse_meanscale(
    doses,
    uncalibrated_medications,
    medication,
    measurements,
):
    """
    Calibrates the dose-response of a specific medication given a series
    of doses and blood-level measurements from the same time period.
    This uses a trivial mean of error ratios to compute a scale factor
    on ef(T) that attempts to minimize the errors between the simulated
    blood levels and the measurements.

    Returns a calibrated copy of 'uncalibrated_medications' and the
    calibrated scale factor (represented as order-1 polynomial
    coefficients) for the specified medication.

    Parameters:
    ===========
    doses                     DataFrame of doses (see
                              pharma.createDoses()).
    uncalibrated_medications  Dict of medications (see
                              medications.medications).
    medication                Which medication in
                              'uncalibated_medications' to calibrate.
    measurements              DataFrame of blood level measurements
                              corresponding to the actual dose response
                              of the to-be-calibrated medication. (see
                              pharma.createMeasurements())."""

    levels_at_measurements = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsAtMoments(measurements.index),
        doses,
        uncalibrated_medications,
    )

    cali_X = np.array(
        [
            0.0,
            (measurements["value"] / levels_at_measurements).mean(),
        ]
    )

    calibrated_medications = dict(uncalibrated_medications)
    calibrated_medications[medication] = medications.calibratedDoseResponse(
        uncalibrated_medications[medication],
        cali_X,
    )
    return calibrated_medications, cali_X
