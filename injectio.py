import math
import numpy as np
import pandas as pd

from scipy import interpolate
from numba import jit

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import collections as mc
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize

# Converts an ndarray of flaot(Day),float(pg/mL) pairs into an interpolated function f:
#   val [pg/mL] = f(time_since_injection [Days])
#
# Expects two zero values at both the start and end, used to tweak the
#   slope of the interpolated curve to end at 0.0.
# f is positive and continuous across the domain of the input data,
# and equal to 0.0 outside that domain. This is required for later computations
# using this function to operate correctly.
#
# The function ef returned by rawDVToFunc function is called in the inner loop
# of all subsequent computations, so it is heavily optimized for performance by
# constructing lookup tables of the interpolated function. LUT table
# sizes < 100000 seem to make first derivatives of ef get wiggly.
def rawDVToFunc(raw):
    interp_ef = interpolate.interp1d(raw[:,0], raw[:,1],
                                     kind='cubic',
                                     fill_value=(0.0, 0.0),
                                     bounds_error=False)
    min_x = raw[1,0]
    max_x = raw[-2,0]
    sample_x  = np.linspace(min_x, max_x, num=100000)
    sample_y  = np.array([interp_ef(x) for x in sample_x])
    slope_lut = (sample_y[1:] - sample_y[:-1]) / (sample_x[1:] - sample_x[:-1])

    @jit(nopython=True)
    def ef(x):
        if x <= min_x or x >= max_x:
            return 0.0
        else:
            idx = np.searchsorted(sample_x, x)-1
            # sample_x[-1] == max_x, which is handled by the above conditional,
            # so it's safe to use idx as an index into slope_lut even though
            # it has length of len(sample_x)-1. Could possibly eliminate the
            # conditional entirely by setting the edge values of slope_lut to 0
            # and being tricksy with indexing.
            return sample_y[idx] + (x - sample_x[idx])*slope_lut[idx]
    ef.domain = (min_x, max_x)
    return ef

def dateTimeToDays(dt):
    return dt.astype('double')/1.0e9/60.0/60.0/24.0
def timeDeltaToDays(td):
    return td.total_seconds()/60.0/60.0/24.0


# Reformats a numpy ndarray[object] of the form:
#
#   [
#     [ "date str", dose float, dose func ],
#     [ "date str", dose float, dose func ],
#     ...
#   ]
#
# into a Pandas Series of (dose float, dose func) tuples indexed by DateTime.
#
# This allows the user-typed input array to look nice and be easy to work with.
def createInjections(inj_array):
    return pd.Series([tuple(i) for i in inj_array[:,1:3]], index=pd.to_datetime(inj_array[:,0]))

# Create an injections Series for a particular injection function that cycles
# monotonically for the desired length of time.
#
#   ef        function
#   sim_time  length of simulation [Days]
#   inj_freq  injection every inj_freq [Pandas frequency thing]
def createInjectionsCycle(ef, sim_time, inj_freq):
    end_date = pd.to_datetime(sim_time, unit='D')
    dates = pd.date_range(start=0, end=end_date, freq=inj_freq)
    if dates[-1] != end_date:
        dates = pd.to_datetime(np.append(dates.to_series().values, [end_date.to_datetime64()]))
    inj = [(1.0, ef)] * (len(dates)-1) + [(0.0, ef)]
    return pd.Series(inj, index=dates)


# TODO: this start time handling is funky and not generalizable to different
# contexts this function may be used in, like the more recent optimization
# stuff where it results in a 1 day misalignment.
def zeroLevelsFromInjections(injections, sample_freq):
    start_time = injections.index[0].floor('D')
    end_time   = injections.index[-1].ceil('D')
    dates      = pd.date_range(start_time, end_time, freq=sample_freq)
    return pd.Series(np.zeros(len(dates)), index=dates)
    
def zeroLevelsAtInjections(injections):
    return pd.Series(np.zeros(len(injections)), index=injections.index)


# Conceptually, this constructs a lazy-evaluated continuous function of total
# blood E concentration by additively overlaying the function f (with domain as
# specified in rawDVToFunc) shifted in time for each injection. It then samples
# this at the times specified in zero_levels. Any numeric errors are contained
# with the interpolated function f representing the concentration curve for a
# single injection. It's not numeric integration because the function f already
# returns the total blood level contribution of a single injection depot, and
# each depot contributes linearly to the total systemic blood concentration.
def calcInjections(zero_levels, injections):
    for inj_date, inj in injections.items():
        (inj_ratio, inj_ef) = inj
        
        # If the function has a specified domain, we can use that to avoid
        # doing work we don't need to. If it doesn't have a specified domain,
        # don't make any assumptions about it. This allows for the seamless
        # calculation of derivatives calculated using the scipy wrapper function.
        if hasattr(inj_ef, 'domain') and inj_ef.domain is not None:
            # Only need to compute where inj_ef > 0. This is guaranteed to be true
            # exactly across the domain of inj_ef, so we can save significant
            # computation here.
            max_T = inj_date + pd.to_timedelta(inj_ef.domain[1], unit='D')
            levels_range = zero_levels[inj_date:max_T].index
        else:
            levels_range = zero_levels.index
            
        for T in levels_range:
            zero_levels[T] += inj_ratio * inj_ef(timeDeltaToDays(T - inj_date))

    return zero_levels


def startPlot():
    plt.rcParams['figure.figsize'] = [15, 12]
    plt.figure(dpi=150)
    plt.xlabel('Time (days)')
    plt.ylabel('Estradiol (pg/mL)')
    plt.grid(which='major', linestyle=':', color=(0.8, 0.8, 0.8), alpha=0.7)
    plt.grid(which='minor', linestyle=':', color=(0.8, 0.8, 0.8), alpha=0.2)
    plt.gca().set_axisbelow(True)
    plt.gca().set_zorder(0)

# Plot the continuous, simulated curve of blood levels for a series of
# injections, along with associated blood level measurements if they exist.
#
#   injections    Injections to plot, represented as a Pandas Series of
#                 (dose [mg], ef [function]) tuples indexed by DateTime.
#                 See createInjections(...).
#   sample_freq   Frequency at which the calculated continuous curve is sampled
#                 at. [Pandas frequency thing]
#   measurements  (Optional) Actual blood level measurements to plot alongside
#                 the simulated curve, represented as a Pandas series of pg/mL
#                 values indexed by DateTime.
#   label         Matplotlib label to attach to the curve. [String]
def plotInjections(injections,
                   sample_freq,
                   measurements=pd.Series(dtype=np.float64),
                   label=''):
    e_levels = calcInjections(zeroLevelsFromInjections(injections, sample_freq),
                              injections)
    levels_at_injection = calcInjections(zeroLevelsAtInjections(injections),
                                         injections)[0:-1]
    levels_at_measurements = calcInjections(
        pd.Series(np.zeros(len(measurements)), index=measurements.index),
        injections)
    
    # Plot with relative x-axis times so the labels are more readable
    sim_time = timeDeltaToDays(e_levels.index[-1] - e_levels.index[0])
    plt.xlim((0.0, sim_time))
    plt.xticks(np.arange(0.0, sim_time+1.0, 3.0), rotation=-45)
    plt.gca().xaxis.set_minor_locator(mticker.AutoMinorLocator(n=3))
    
    # Plot simulated curve
    plt.plot([timeDeltaToDays(d - e_levels.index[0]) for d in e_levels.index],
             e_levels.values,
             label=label,
             zorder=1)
    
    # Plot moments of injection as dose-scaled points on top of the simulated curce
    doses         = [dose for dose,ef in injections.values[0:-1]]
    norm_doses    = Normalize(vmin=-1.0*max(doses), vmax=max(doses)+0.2)(doses)
    marker_sizes  = [(9.0*dose+2.0)**2 for dose in norm_doses]
    marker_colors = [(dose, 1.0-dose, 0.7, 1.0) for dose in norm_doses]
    plt.scatter([timeDeltaToDays(d - e_levels.index[0]) for d in levels_at_injection.index],
                levels_at_injection.values,
                s=marker_sizes,
                c=marker_colors,
                marker='2',
                zorder=2)
    
    # Plot measured blood levels
    plt.plot([timeDeltaToDays(d - e_levels.index[0]) for d in measurements.index],
             measurements.values,
             'o')

    # Draw a vertical line from the measured points to the simulated curve
    measurements_points = [
        (timeDeltaToDays(T - e_levels.index[0]), d) for T,d in measurements.items()]
    levels_at_measurements_points = [
        (timeDeltaToDays(T - e_levels.index[0]), d) for T,d in levels_at_measurements.items()]
    
    #errors = measurements - levels_at_measurements
    #plt.errorbar([timeDeltaToDays(d - e_levels.index[0]) for d in levels_at_measurements.index],
    #             levels_at_measurements.values,
    #             yerr=errors.values,
    #             color=(0.5, 0.4, 0.4, 0.5), capsize=4.0, fmt='none')

    lines = list(zip(measurements_points, levels_at_measurements_points))
    lc = mc.LineCollection(lines, linestyles=(0, (2, 3)), colors=(0.7, 0.3, 0.3, 1.0), zorder=3)
    plt.gca().add_collection(lc);

# Plot multiple injection curves for a range of injection frequencies.
#
#   ef         Interpolated function ef(T days) = (E mg) (from rawDVToFunc) for single injection
#   sim_time   Total simulation time [Days]
#   sim_freq   Resolution of simulaion [Pandas frequency thing]
#   inj_freqs  List of period curves to plot [Days]
def plotInjectionFrequencies(ef, sim_time, sim_freq, inj_freqs):
    for freq in inj_freqs:
        plotInjections(createInjectionsCycle(ef, sim_time, pd.offsets.Nano(freq*24.0*60.0*60.0*1e9)),
                       sim_freq,
                       label=freq)

    plt.legend(title='Injection frequency\n(days)')
    plt.show()