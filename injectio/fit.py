from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib import dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from injectio import pharma, injectables


#########################################################
### Fitting injections to a desired blood level curve ###

def createInitialAndBounds(injections,
                           max_dose=np.inf,
                           time_bounds='midpoints'):
    # least_squares works on float parameters, so convert our injection dates
    # into float days.
    times = np.array([pharma.timeStampToDays(inj_date) for inj_date in injections.index][:-1])
    doses = injections["dose"][:-1].to_numpy()
    
    if time_bounds == 'midpoints':
        # Midpoint time bound handling:
        #   The first injection's minimum time bound is its same time
        #   The last injection's maximum time bound is the dummy injection time
        #   All other bounds are at the midpoints between the initial times
        midpoints = (times[1:] + times[:-1])/2.0
        #time_bounds = [(times[0], midpoints[0])] +\
        #TODO: revert this^ after figuring out zeroLevelsFromInjections flooring inconsistency
        time_bounds = [(times[0], np.nextafter(times[0], times[0]+1.0))] +\
                      [(l, r) for l,r in zip(midpoints[:-1], midpoints[1:])] +\
                      [(midpoints[-1], pharma.timeStampToDays(injections.index[-1]))]
    elif time_bounds == 'fixed':
        # least_squares doesn't have in-built capacity for fixed parameters,
        # and also requires each lower bound to be strictly less than each
        # upper bound. This is basically a hack to avoid requiring different
        # implementations for optimizing on doses+times vs just doses, keeping
        # the same array X of independent variables in both cases.
        time_bounds = [(T, np.nextafter(T, T+1.0)) for T in times]
    else:
        raise ValueError(f"'{time_bounds}' isn't a recognized input for time_bounds")
    
    dose_bounds = [(0.0, max_dose)]*len(doses)
    
    return np.concatenate((times, doses)),\
           list(zip(*time_bounds, *dose_bounds))


def createInjectionsTimesDoses(X_times_doses, X_injectables):
    """
    Create an injections DataFrame from the stacked X vector of doses and times
    used for curve fitting.
    
    X_injectables is the column of injectables corresponding to each injection.
    len(X_injectables) must == len(X_times_doses)/2.
    Scipy's optimization functions just pass around the X vector of minimzation
    parameters, but we need to reconstruct the full injections DataFrame to
    pass to pharma.calcInjections(...).
    
    This doesn't include the final dummy injection! Which is fine for the purpose
    of the optimization function, but the resulting injections Series won't
    behave as expected elsewhere."""
    n_inj = int(X_times_doses.size / 2)
    X_times_doses = np.append(X_times_doses, X_injectables).reshape((n_inj, 3), order='F')
    return pharma.createInjections(X_times_doses, date_unit='D')


# X is a 1D array of injection parameters we optimize on.
# Let i_n == len(injections)-1 (The last injection is always a zero-dose dummy
# point, so we don't optimize on it).
# Then, X is represented as i_n dose values followed by i_n corresponding dates
# in days.
# So, len(X) == 2*i_n.
def fTimesAndDoses(X_times_doses, X_injectables, target_x, target_y):
    injections = createInjectionsTimesDoses(X_times_doses, X_injectables)
    residuals = pharma.zeroLevelsAtMoments(target_x)
    pharma.calcInjections(residuals, injections, injectables.injectables)
    residuals -= target_y
    return residuals


def emptyResults():
    return defaultdict(lambda: {
        "injections_init": None,
        "X0": None,
        "bounds":None,
        "target": None,
        "result": None,
        "injections_optim": None,})

def initializeRun(injections_init,
                  target_x, target_y,
                  max_dose=np.inf, time_bounds='midpoints'):
    run = {}
    run["injections_init"] = injections_init
    run["X0"], run["bounds"] = createInitialAndBounds(
        injections_init,
        max_dose=max_dose,
        time_bounds=time_bounds)
    run["target"] = (target_x, target_y)
    run["result"] = None
    run["injections_optim"] = None
    return run

def runLeastSquares(run, max_nfev=20, **kwargs):
    X_injectables = run["injections_init"]["injectable"][:-1].values
    
    result = least_squares(
        fTimesAndDoses,
        run["X0"],
        args=(X_injectables,
              run["target"][0],
              run["target"][1]),
        bounds=run["bounds"],
        max_nfev=max_nfev,
        **kwargs)
    
    # Put the optimized X vector back into a DataFrame, and add back the final
    # zero injection that we dropped.
    injections_optim = pd.concat([
        createInjectionsTimesDoses(result.x, X_injectables),
        run["injections_init"].iloc[[-1]]])
    
    run["result"] = result
    run["injections_optim"] = injections_optim
    
    return result

def plotOptimizationRun(run):
    # Calculate levels at the initial and optimized moments of injection
    init_levels = pharma.calcInjections(
        pharma.zeroLevelsAtMoments(run["injections_init"].index),
        run["injections_init"],
        injectables.injectables)
    optim_levels = pharma.calcInjections(
        pharma.zeroLevelsAtMoments(run["injections_optim"].index),
        run["injections_optim"],
        injectables.injectables)
    
    ax = plt.gca()
    ax.plot(run["target"][0], run["target"][1], label="Target Curve")
    pharma.plotInjections(
        run["injections_init"],
        injectables.injectables,
        sample_freq='6H',
        label="Initial condition")
    pharma.plotInjections(
        run["injections_optim"],
        injectables.injectables,
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
        levels_at_measurements = pharma.calcInjections(
            pharma.zeroLevelsAtMoments(measurements.index),
            injections,
            calibrated_injectables)
        return levels_at_measurements - measurements["value"]
    
    # Initial condition is un-transformed ef(T)
    X0 = np.array([0.0, 1.0])
    result = least_squares(f, X0,
                           bounds=(np.zeros_like(X0),
                                   np.full_like(X0, np.inf)),
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
    
    injections                DataFrame of injections (see injectio.createInjections()).
    uncalibrated_injectables  Dict of injectables (see injectio.injectables).
    injectable                Which injectable in 'injectables' to calibrate.
    measurements              DataFrame of blood level measurements corresponding
                              to the actual dose response of the to-be-calibrated
                              function injectables[injectable] (see
                              injectio.createMeasurements())."""
    
    levels_at_measurements = pharma.calcInjections(
        pharma.zeroLevelsAtMoments(measurements.index),
        injections,
        uncalibrated_injectables)
    
    cali_X = np.array([0.0, (measurements["value"] / levels_at_measurements).mean()])
    
    calibrated_injectables = dict(uncalibrated_injectables)
    calibrated_injectables[injectable] = injectables.calibratedInjection(
        uncalibrated_injectables[injectable], cali_X)
    return calibrated_injectables, cali_X
