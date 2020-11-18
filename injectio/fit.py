from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib import dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import warnings

from injectio import pharma, injectables


#########################################################
### Fitting injections to a desired blood level curve ###

def partitionInjections(injections, equal_injections):
    """
    Partition the injections into disjoint equal-dose equivalence classes
    using the specified equivalence properties.
    
    Note that any injections with duplicate time indices that are part of an
    equivalence class in equal_injections will be irreversibly collapsed into a
    single injection. This function expects the index of injections to be a
    set, but Pandas actually represents the index as a multiset. If you need
    injections to happen at the same moment in time, just perturb one by an
    infinitesimal amount so that their indices don't compare equal.
    
    Returns a list of DatetimeIndexes, where each DatetimeIndex represents a
    disjoint equivalence class of injections.
    
    Parameters:
    ===========
    injections        DataFrame of injections (see pharma.createInjections(...)).
    equal_injections  Sequence of DatetimeIndexes where each DatetimeIndex
                      corresponds to injections that are defined to have equal
                      doses."""
    
    def areEquivalent(sa, sb, eq_sets):
        for eq in eq_sets:
            if not sa.isdisjoint(eq) and not sb.isdisjoint(eq):
                return True
        return False

    # This implementation is O(n**2)–worse than the optimal O(n*log(n)) of a
    # disjoint set forest–but the number of injections n will generally be
    # small, and this is a lot simpler!
    
    eq_sets = [set(e) for e in equal_injections]
    partitions = [{inj} for inj in injections.index]
    p = 0
    while p < len(partitions):
        to_del = set()
        for j in range(p+1, len(partitions)):
            if areEquivalent(partitions[p], partitions[j], eq_sets):
                partitions[p] |= partitions[j]
                to_del.add(j)
        partitions[:] = [partitions[d] for d in range(len(partitions)) if d not in to_del]
        p += 1
    return [pd.DatetimeIndex(p).sort_values() for p in partitions]

def createInitialAndBounds(injections,
                           max_dose=np.inf,
                           time_bounds='midpoints',
                           equal_injections=[]):
    """
    Create the initial X vector of independent variables that will be
    optimized, the partitions of equal-dose injections, and the bounds
    corresponding to X.
    
    Returns a 3-tuple of: (
        X:          Vector of independent variables that will be optimized by
                    least_squares, represented as a stacked array of
                    len(injections) times in days, followed by len(partitions)
                    doses.
        partitions: Equal-dose partitions of injections, represented as a list
                    P of partitions, where each P[j] is a set {i_0, ..., i_k}
                    of integer indices representing the injections contained
                    in the partition, and X[len(injections)-1+j] is the dose
                    variable corresponding to the partition.
        bounds:     2-tuple of ([lower_0, ...],[upper_0, ...]) bounds on the
                    independent variables in X, as expected by least_squares.
    )
    
    Parameters:
    ===========
    injections        Initial injections to be optimized.
    max_dose          Maximum dose bound that will be used in the fitting.
    time_bounds       Specfies how injection times will be bounded in the
                      fitting. Can either be an Iterable of strings
                      corresponding to the time bound behavior for each
                      injection, or a single string that will be used for all
                      injections. Possible string options are:
                         'fixed':     the injection times will not be optimized.
                         'midpoints': the injection times will be optimized and
                                      bounded to the midpoints between their
                                      nearest neighboring injections (default).
    equal_injections  A list of DatetimeIndexes where each DatetimeIndex
                      corresponds to injections that should be constrained to
                      have the same dose."""

    if len(injections) == 0:
        raise ValueError("Need at least 1 initial injection to optimize.")
    elif len(injections) == 1 and (time_bounds == 'midpoints' or
                                   time_bounds == ['midpoints']):
        time_bounds = 'fixed'
        warnings.warn(
                "Midpoint time bounds are only supported with more than 1 "
                "injection. Continuing with time_bounds = 'fixed'.",
                RuntimeWarning)

    if isinstance(time_bounds, str):
        time_bounds = len(injections) * [time_bounds]
    elif len(time_bounds) != len(injections):
        raise ValueError(
                "Expecting either a single string or an Iterable of length "
                f"len(injections)={len(injections)} for time_bounds, but got "
                f"{type(time_bounds)} of length {len(time_bounds)}.")

    # least_squares works on float parameters, so convert our injection dates
    # into float days.
    times = pharma.dateTimeToDays(injections.index)

    # Pregenerate all the midpoint bounds even if we don't end up using them.
    # It simplifies the loop a lot.
    midpoints = (times[:-1] + times[1:])/2.0
    midpoint_bounds = [(2*times[0] - midpoints[0], midpoints[0])] +\
                      [(l, r) for l,r in zip(midpoints[:-1], midpoints[1:])] +\
                      [(midpoints[-1], 2*times[-1] - midpoints[-1])]

    t_bounds = []
    for ti in range(len(times)):
        tb = time_bounds[ti]
        if tb == 'midpoints':
            # Midpoint time bound handling:
            #   The first injection's minimum time bound is the same distance away
            #     as the first midpoint, but mirrored to the negative direction.
            #   The last injection's maximum time bound is the same distance away
            #     as the previous midpoint, but mirrored to the positive direction.
            #   All other bounds are at the midpoints between injection times.
            t_bounds.append(midpoint_bounds[ti])
        elif tb == 'fixed':
            # least_squares doesn't have in-built capacity for fixed parameters,
            # and also requires each lower bound to be strictly less than each
            # upper bound. This is basically a hack to avoid requiring different
            # implementations for optimizing on doses+times vs just doses, keeping
            # the same array X of independent variables in both cases.
            t_bounds.append((times[ti], np.nextafter(times[ti], times[ti]+1.0)))
        else:
            raise ValueError(f"'{tb}' isn't a valid input in time_bounds.")

    # This gives equivalence classes of Datetimes, but ultimately we need
    # equivalence classes of injection indices because the exact times will
    # change through the optimization and no longer be valid indices. What we
    # later return from this function is essentially telling which doses in X
    # correspond to which times in X.
    partitions = partitionInjections(injections, equal_injections)
    part_doses = np.zeros(len(partitions))
    for p in range(len(partitions)):
        part_doses[p] = injections["dose"][partitions[p][0]]
    d_bounds = [(0.0, max_dose)]*len(partitions)

    return np.concatenate((times, part_doses)),\
           [{list(injections.index).index(inj) for inj in part} for part in partitions],\
           tuple(zip(*t_bounds, *d_bounds))


def createInjectionsTimesDoses(X, X_partitions, X_injectables):
    """
    Create an injections DataFrame from the stacked X vector of independent
    variables.
    
    Scipy's optimization functions just pass around the X vector of independent
    variables, but we need to reconstruct the full injections DataFrame to
    pass to pharma.calcInjections(...).
    
    Returns a DataFrame of injections (see pharma.createInjections).
    
    Parameters:
    ===========
    X               X vector of independent variables (see
                    createInitialAndBounds).
    X_partitions    Equal-dose partitions of injections (see
                    createInitialAndBounds). It is used to unpartition
                    the doses in X.
    X_injectables   Column of injectables corresponding to each injection,
                    exactly as it was in the original injections DataFrame
                    (except for the discarded final injection)."""
    
    n_inj = X.size - len(X_partitions)
    
    times = X[0:n_inj]
    
    # Unpartition the doses
    part_doses = X[n_inj:]
    doses = np.zeros_like(times)
    for p in range(len(X_partitions)):
        for d in X_partitions[p]:
            doses[d] = part_doses[p]
    
    return pharma.createInjections(
        np.concatenate([times, doses, X_injectables]).reshape((n_inj, 3), order='F'),
        date_unit='D')


def emptyResults():
    return defaultdict(lambda: {
        "injections_init": None,
        "X0": None,
        "partitions": None,
        "bounds":None,
        "exclude_area": pd.DatetimeIndex([]),
        "target": None,
        "result": None,
        "injections_optim": None,})

def initializeRun(injections_init,
                  injectables_map,
                  target,
                  max_dose=np.inf,
                  time_bounds='midpoints',
                  equal_injections=[],
                  exclude_area=pd.DatetimeIndex([])):
    """
    Get everything set up to run least squares, and organize it into a tidy
    dict.
    
    Returns a dict containing all the relevant information about the
    optimization run (see emptyResults for structure), ready to be used with
    runLeastSquares.
    
    Parameters:
    ===========
    injections_init   Initial set of injections to be optimized.
    injectables_map   Mapping of injectable functions (see
                      injectables.injectables).
    target            Target function to fit the injections curve to [Pandas
                      series index by Datetime].
    max_dose          See createInitialAndBounds.
    time_bounds       See createInitialAndBounds.
    equal_injections  See createInitialAndBounds.
    exclude_area      DatetimeIndex corresponding to points in target that
                      should not be considered when computing residuals for the
                      fit."""
    run = {}
    run["injections_init"] = injections_init
    run["injectables_map"] = injectables_map
    run["X0"], run["partitions"], run["bounds"] = createInitialAndBounds(
        injections_init,
        max_dose=max_dose,
        time_bounds=time_bounds,
        equal_injections=equal_injections)
    run["exclude_area"] = exclude_area
    run["target"] = target
    run["result"] = None
    run["injections_optim"] = None
    return run

def runLeastSquares(run, max_nfev=20, **kwargs):
    X_injectables = run["injections_init"]["injectable"].values
    
    # Residuals function
    def fTimesAndDoses(X,
                       X_partitions,
                       X_injectables,
                       injectables_map,
                       target,
                       exclude_area):
        injections = createInjectionsTimesDoses(X, X_partitions, X_injectables)
        residuals = pharma.zeroLevelsAtMoments(target.index)
        pharma.calcInjectionsConv(residuals, injections, injectables_map)
        residuals -= target
        return residuals.drop(exclude_area)
    
    result = least_squares(
        fTimesAndDoses,
        run["X0"],
        args=(run["partitions"],
              X_injectables,
              run["injectables_map"],
              run["target"],
              run["exclude_area"]),
        bounds=run["bounds"],
        max_nfev=max_nfev,
        **kwargs)
    
    # Put the optimized X vector back into a DataFrame
    injections_optim = createInjectionsTimesDoses(
            result.x,
            run["partitions"],
            X_injectables)
    
    run["result"] = result
    run["injections_optim"] = injections_optim
    
    return result

def plotOptimizationRun(fig, ax, run):
    # Calculate levels at the initial and optimized moments of injection
    init_levels = pharma.calcInjectionsExact(
        pharma.zeroLevelsAtMoments(run["injections_init"].index),
        run["injections_init"],
        run["injectables_map"])
    optim_levels = pharma.calcInjectionsExact(
        pharma.zeroLevelsAtMoments(run["injections_optim"].index),
        run["injections_optim"],
        run["injectables_map"])
    
    ax.plot(run["target"], label="Target Curve")
    ax.plot(run["target"][run["exclude_area"]],
            marker='o',
            color=(1.0, 0.1, 0.1, 0.5),
            label="excluded from fit")
    pharma.plotInjections(
        fig, ax,
        run["injections_init"],
        run["injectables_map"],
        sample_freq='6H',
        label="Initial condition")
    pharma.plotInjections(
        fig, ax,
        run["injections_optim"],
        run["injectables_map"],
        sample_freq='6H',
        label="Optimized solution")
    
    # Draw lines between the initial injections and the optimized injections
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

def calibrateInjections_lsqpoly(injections,
                                uncalibrated_injectables,
                                injectable,
                                measurements):
    """
    Calibrates the dose-response of a specific injectable given a series of
    injections and blood-level measurements from the same time period. This
    uses least squares regression to compute polynomial coefficients X that
    calibrate the injectable as: X[0] + X[1]*ef(T), such the errors between
    the simulated blood levels and the measurements are minimized.
    
    Returns a calibrated copy of 'uncalibrated_injectables' and the calibrated
    polynomial coefficients for the specified injectable.
    
    Parameters:
    ===========
    injections                DataFrame of injections (see injectio.createInjections()).
    uncalibrated_injectables  Dict of injectables (see injectio.injectables).
    injectable                Which injectable in 'injectables' to calibrate.
    measurements              DataFrame of blood level measurements corresponding
                              to the actual dose response of the to-be-calibrated
                              function injectables[injectable] (see
                              injectio.createMeasurements())."""
    
    calibrated_injectables = dict(uncalibrated_injectables)
    
    # Residuals function
    def f(X):
        calibrated_injectables[injectable] =\
            injectables.calibratedInjection(uncalibrated_injectables[injectable], X)
        levels_at_measurements = pharma.calcInjectionsExact(
            pharma.zeroLevelsAtMoments(measurements.index),
            injections,
            calibrated_injectables)
        return levels_at_measurements - measurements["value"]
    
    # Initial condition is un-transformed ef(T)
    X0 = np.array([0.0, 1.0])
    result = least_squares(f, X0,
                           bounds=(np.zeros_like(X0),
                                   [np.nextafter(0.0, 1.0), np.inf]),
                           max_nfev=10,
                           verbose=0)
    calibrated_injectables[injectable] =\
        injectables.calibratedInjection(uncalibrated_injectables[injectable], result.x)
    return calibrated_injectables, result.x


def calibrateInjections_meanscale(injections,
                                  uncalibrated_injectables,
                                  injectable,
                                  measurements):
    """
    Calibrates the dose-response of a specific injectable given a series of
    injections and blood-level measurements from the same time period. This
    uses a trivial mean of error ratios to compute a scale factor on ef(T)
    that attempts to minimize the errors between the simulated blood levels
    and the measurements.
    
    Returns a calibrated copy of 'uncalibrated_injectables' and the calibrated
    scale factor (represented as order-1 polynomial coefficients) for the
    specified injectable.
    
    Parameters:
    ===========
    injections                DataFrame of injections (see injectio.createInjections()).
    uncalibrated_injectables  Dict of injectables (see injectio.injectables).
    injectable                Which injectable in 'injectables' to calibrate.
    measurements              DataFrame of blood level measurements corresponding
                              to the actual dose response of the to-be-calibrated
                              function injectables[injectable] (see
                              injectio.createMeasurements())."""
    
    levels_at_measurements = pharma.calcInjectionsExact(
        pharma.zeroLevelsAtMoments(measurements.index),
        injections,
        uncalibrated_injectables)
    
    cali_X = np.array([0.0, (measurements["value"] / levels_at_measurements).mean()])
    
    calibrated_injectables = dict(uncalibrated_injectables)
    calibrated_injectables[injectable] = injectables.calibratedInjection(
        uncalibrated_injectables[injectable], cali_X)
    return calibrated_injectables, cali_X
