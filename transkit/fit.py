from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib import dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
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
    partitions = [{dose_idx} for dose_idx in doses.index]
    p = 0
    while p < len(partitions):
        to_del = set()
        for j in range(p+1, len(partitions)):
            if areEquivalent(partitions[p], partitions[j], eq_sets):
                partitions[p] |= partitions[j]
                to_del.add(j)
        partitions[:] = [partitions[d]
                for d in range(len(partitions)) if d not in to_del]
        p += 1
    return [pd.DatetimeIndex(part).sort_values() for part in partitions]

def createInitialAndBounds(doses,
                           max_dose=np.inf,
                           time_bounds='midpoints',
                           equal_doses=[]):
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
    max_dose     Maximum dose amount bound that will be used in the fitting.
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
                 corresponds to doses that should be constrained to
                 have the same amount."""

    if len(doses) == 0:
        raise ValueError("Need at least 1 initial dose to optimize.")
    elif len(doses) == 1 and (time_bounds == 'midpoints' or
                              time_bounds == ['midpoints']):
        time_bounds = 'fixed'
        warnings.warn(
                "Midpoint time bounds are only supported with more than 1 "
                "dose. Continuing with time_bounds = 'fixed'.",
                RuntimeWarning)

    if isinstance(time_bounds, str):
        time_bounds = len(doses) * [time_bounds]
    elif len(time_bounds) != len(doses):
        raise ValueError(
                "Expecting either a single string or an Iterable of length "
                f"len(doses)={len(doses)} for time_bounds, but got "
                f"{type(time_bounds)} of length {len(time_bounds)}.")

    # least_squares works on float parameters, so convert our dose dates
    # into float days.
    times = pharma.dateTimeToDays(doses.index)

    # Pregenerate all the midpoint bounds even if we don't end up using
    # them. It simplifies the loop a lot.
    midpoints = (times[:-1] + times[1:])/2.0
    midpoint_bounds = [(2*times[0] - midpoints[0], midpoints[0])] +\
                      [(l, r) for l,r in zip(midpoints[:-1], midpoints[1:])] +\
                      [(midpoints[-1], 2*times[-1] - midpoints[-1])]

    t_bounds = []
    for ti in range(len(times)):
        tb = time_bounds[ti]
        if tb == 'midpoints':
            # Midpoint time bound handling:
            #   The first dose's minimum time bound is the same distance
            #    away as the first midpoint, but mirrored to the negative
            #    direction.
            #   The last dose's maximum time bound is the
            #    same distance away as the previous midpoint, but
            #    mirrored to the positive direction.
            #   All other bounds are at the midpoints between dose times.
            t_bounds.append(midpoint_bounds[ti])
        elif tb == 'fixed':
            # least_squares doesn't have in-built capacity for fixed
            # parameters, and also requires each lower bound to be
            # strictly less than each upper bound. This is basically a
            # hack to avoid requiring different implementations for
            # optimizing on amounts+times vs just amounts, keeping the
            # same array X of independent variables in both cases.
            t_bounds.append((times[ti], np.nextafter(times[ti], times[ti]+1.0)))
        else:
            raise ValueError(f"'{tb}' isn't a valid input in time_bounds.")

    # This gives equivalence classes of Datetimes, but ultimately we
    # need equivalence classes of dose indices because the exact times
    # will change through the optimization and no longer be valid
    # indices. What we later return from this function is essentially
    # telling which doses amounts in X correspond to which times in X.
    partitions = partitionDoses(doses, equal_doses)
    part_doses = np.zeros(len(partitions))
    for p in range(len(partitions)):
        part_doses[p] = doses["dose"][partitions[p][0]]
    d_bounds = [(0.0, max_dose)]*len(partitions)

    return (
            np.concatenate((times, part_doses)),
            [{list(doses.index).index(dose_idx) for dose_idx in part}
                for part in partitions],
            tuple(zip(*t_bounds, *d_bounds))
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
        doses_array.reshape((n_doses, 3), order='F'),
        date_unit='D')


def emptyResults():
    return defaultdict(lambda: {
        "doses_init": None,
        "X0": None,
        "partitions": None,
        "bounds":None,
        "exclude_area": pd.DatetimeIndex([]),
        "target": None,
        "result": None,
        "doses_optim": None,})

def initializeRun(doses_init,
                  medications_map,
                  target,
                  max_dose=np.inf,
                  time_bounds='midpoints',
                  equal_doses=[],
                  exclude_area=pd.DatetimeIndex([])):
    """
    Get everything set up to run least squares, and organize it into a
    tidy dict.
    
    Returns a dict containing all the relevant information about the
    optimization run (see emptyResults for structure), ready to be used
    with runLeastSquares.
    
    Parameters:
    ===========
    doses_init        Initial set of doses to be optimized.
    medications_map   Mapping of medication dose-response functions
                      (see medications.medications).
    target            Target function to fit the blood level curve to
                      [Pandas series index by Datetime].
    max_dose          See createInitialAndBounds.
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
        max_dose=max_dose,
        time_bounds=time_bounds,
        equal_doses=equal_doses)
    run["exclude_area"] = exclude_area
    run["target"] = target
    run["result"] = None
    run["doses_optim"] = None
    return run

def runLeastSquares(run, max_nfev=20, **kwargs):
    X_medications = run["doses_init"]["medication"].values

    # Residuals function
    def fTimesAndDoses(X,
                       X_partitions,
                       X_medications,
                       medications_map,
                       target,
                       exclude_area):
        doses = createDosesFromX(X, X_partitions, X_medications)
        residuals = pharma.zeroLevelsAtMoments(target.index)
        pharma.calcBloodLevelsConv(residuals, doses, medications_map)
        residuals -= target
        return residuals.drop(exclude_area)

    result = least_squares(
        fTimesAndDoses,
        run["X0"],
        args=(run["partitions"],
              X_medications,
              run["medications_map"],
              run["target"],
              run["exclude_area"]),
        bounds=run["bounds"],
        max_nfev=max_nfev,
        **kwargs)

    # Put the optimized X vector back into a DataFrame
    doses_optim = createDosesFromX(
            result.x,
            run["partitions"],
            X_medications)

    run["result"] = result
    run["doses_optim"] = doses_optim

    return result

def plotOptimizationRun(fig, ax, run):
    # Calculate levels at the initial and optimized moments of dose
    init_levels = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsAtMoments(run["doses_init"].index),
        run["doses_init"],
        run["medications_map"])
    optim_levels = pharma.calcBloodLevelsExact(
        pharma.zeroLevelsAtMoments(run["doses_optim"].index),
        run["doses_optim"],
        run["medications_map"])

    ax.plot(run["target"], label="Target Curve")
    ax.plot(run["target"][run["exclude_area"]],
            marker='o',
            color=(1.0, 0.1, 0.1, 0.5),
            label="excluded from residuals")
    pharma.plotDoses(
        fig, ax,
        run["doses_init"],
        run["medications_map"],
        sample_freq='6H',
        label="Initial condition")
    pharma.plotDoses(
        fig, ax,
        run["doses_optim"],
        run["medications_map"],
        sample_freq='6H',
        label="Optimized solution")

    # Draw lines between the initial doses and the optimized doses
    init_points = [
        (mdates.date2num(T), level) for T,level in init_levels.items()]
    optim_points = [
        (mdates.date2num(T), level) for T,level in optim_levels.items()]

    lines = list(zip(init_points, optim_points))
    lc = mc.LineCollection(lines, linestyles='dotted', zorder=10)
    ax.add_collection(lc);
    ax.legend();

    ax.set_xlim((mdates.date2num(run["target"].index[0]),
                 mdates.date2num(run["target"].index[-1])))


##############################################################
### Fitting a calibration polynomial on ef to measurements ###

def calibrateDoseResponse_lsqpoly(doses,
                                  uncalibrated_medications,
                                  medication,
                                  measurements):
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
        calibrated_medications[medication] = medications.calibratedDoseResponse(
                uncalibrated_medications[medication], X)
        levels_at_measurements = pharma.calcBloodLevelsExact(
            pharma.zeroLevelsAtMoments(measurements.index),
            doses,
            calibrated_medications)
        return levels_at_measurements - measurements["value"]

    # Initial condition is un-transformed ef(T)
    X0 = np.array([0.0, 1.0])
    result = least_squares(f, X0,
                           bounds=(np.zeros_like(X0),
                                   [np.nextafter(0.0, 1.0), np.inf]),
                           max_nfev=10,
                           verbose=0)
    calibrated_medications[medication] = medications.calibratedDoseResponse(
            uncalibrated_medications[medication], result.x)
    return calibrated_medications, result.x

def calibrateDoseResponse_meanscale(doses,
                                    uncalibrated_medications,
                                    medication,
                                    measurements):
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
        uncalibrated_medications)

    cali_X = np.array([0.0, (measurements["value"] / levels_at_measurements).mean()])

    calibrated_medications = dict(uncalibrated_medications)
    calibrated_medications[medication] = medications.calibratedDoseResponse(
        uncalibrated_medications[medication], cali_X)
    return calibrated_medications, cali_X
