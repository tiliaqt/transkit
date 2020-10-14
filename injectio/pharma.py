from collections import defaultdict
from collections.abc import Iterable
import math
from matplotlib import pyplot
from matplotlib import dates as mdates
from matplotlib import collections as mc
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from numba import jit
import numpy as np
import pandas as pd
from scipy import interpolate


injectables_markers = defaultdict(lambda: "2", {
    "ec": "2",
    "ev": "1",
    "pill-oral": ".",
    "pill-subl": ".",
    "pill-bucc": ".",
})


def dateTimeToDays(dt):
    return dt.astype('double')/1.0e9/60.0/60.0/24.0
def timeStampToDays(ts):
    return dateTimeToDays(ts.to_datetime64())
def timeDeltaToDays(td):
    return td.total_seconds()/60.0/60.0/24.0


def rawDVToFunc(raw):
    """
    Converts an ndarray of flaot(Day),float(pg/mL) pairs into an interpolated
    function ef:

        ef(time_since_injection [Days]) = blood concentration [pg/mL]

    representing the instantaneous total blood concentration of the medication
    at the given time after administration.

    Expects two zero values at both the start and end, used to tweak the slope
    of the interpolated curve to end at 0.0. Ef must be positive and continuous
    across the domain of the input data, and equal to 0.0 outside that domain.
    This is required for later computations using this function to operate
    correctly.

    The returned function ef is called in the inner loop of all subsequent
    computations, so it is heavily optimized for performance by constructing
    lookup tables of the interpolated function. LUT table sizes < 100000 seem
    to make first derivatives of ef get wiggly."""

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
    ef.domain = (pd.to_timedelta(min_x, unit='D'), pd.to_timedelta(max_x, unit='D'))
    return ef


def createInjections(inj_array, date_format=None, date_unit='ns'):
    """
    Create a DataFrame of injections for use in computation.

    This reformats a numpy ndarray[object] of the form:

    [
        [ "date str", dose float, "injectable str" ],
        [ "date str", dose float, "injectable str" ],
        ...
    ]

    into a Pandas DataFrame with the date as index, and dose and injectable as
    columns. This allows the user-typed input array to look nice and be easy
    to work with. Date values can be anything parsable by pd.to_datetime(...).

    Parameters:
    ===========
    inj_array    ndarray of injections.
    date_format  Passed through to the format parameter of
                 pandas.to_datetime(...).
    date_unit    Passed through to the unit parameter of
                 pandas.to_datetime(...)."""

    df = pd.DataFrame(inj_array[:,1:3],
                      index=pd.to_datetime(inj_array[:,0], format=date_format, unit=date_unit),
                      columns=["dose", "injectable"])
    df.loc[:, "dose"] = df["dose"].apply(pd.to_numeric)
    return df


def createMeasurements(measurements_array, date_format=None, date_unit='ns'):
    """
    Create a DataFrame of measurements for use in computation.

    This reformats a numpy ndarray[object] of the form:

    [
        [ "date str", measurement float, "method str" ],
        [ "date str", measurement float, "method str" ],
        ...
    ]

    into a Pandas DataFrame with the date as index, and measurement value and
    method as columns. This allows the user-typed input array to look nice and
    be easy to work with. Date values can be anything parsable by
    pd.to_datetime(...).

    Parameters:
    ===========
    measurements_array  ndarray of measurements.
    date_format         Passed through to the format parameter of
                        pandas.to_datetime(...).
    date_unit           Passed through to the unit parameter of
                        pandas.to_datetime(...)."""

    df = pd.DataFrame(measurements_array[:,1:3],
                      index=pd.to_datetime(measurements_array[:,0], format=date_format, unit=date_unit),
                      columns=["value", "method"])
    df.loc[:, "value"] = df["value"].apply(pd.to_numeric)
    return df

def createInjectionsCycle(ef, sim_time, inj_freq, start_date=0):
    """
    Creates a DataFrame of injections for a particular injectable that cycles
    monotonically for the desired length of time.

    Parameters:
    ===========
    ef          Injectable
    sim_time    Length of simulation [Days]
    inj_freq    Injection every inj_freq [Pandas frequency thing]
    start_date  Cycle starting from this date. Can be anything parsable
                by pd.to_datetime(...)."""

    n_inj = math.ceil(sim_time / (pd.tseries.frequencies.to_offset(inj_freq).nanos * 1.0e-9/60.0/60.0/24.0)) + 1
    injs = createInjections(np.array(rep_from([start_date, 1.0, ef], n=n_inj, freq=inj_freq)))
    injs.loc[injs.iloc[[-1]].index, "dose"] = 0
    return injs

def rep_from(inj, n, freq):
    """
    Repeats injections with cycling frequencies from a given first injection.
    Returns a list of lists.

    This is used for composing and generating injections for use in the
    ndarray passed to createInjections().

    Parameters:
    ===========
    inj   Starting injection specified as a length-3 sequence of
          ("date", dose, "injectable").
    n     Number of repetitions.
    freq  Frequency of repetition. Either a single Pandas frequency string, or
          an Iterable of frequency strings that will be cycled through as the
          repetition continues. The latter format allows for a cycle of
          injections where, eg. you inject after 2 days, then after 3 days,
          then after 2 days, then after 3 days, etc."""

    if isinstance(freq, str) or not isinstance(freq, Iterable):
        freq = (freq,)

    offsets = [pd.tseries.frequencies.to_offset(f) for f in (math.floor(n / len(freq)) * freq)[0:n]]
    inj_dates = [pd.to_datetime(inj[0])]
    for o in offsets:
        inj_dates.append(inj_dates[-1] + o)

    return [*zip(inj_dates, n*[inj[1]], n*[inj[2]])]

def rep_from_dose(date, dose, ef, n, freq):
    """
    Repeats injections with cycling doses and cycling frequencies from a
    given first injection. Returns a list of lists.

    This is used for composing and generating injections for use in the
    ndarray passed to createInjections().

    Parameters:
    ===========
    date  Starting injection date, as string parsable by pandas.to_datetime().
    dose  Injection dose. Either a single scalar that will be repeated in each
          injection, or an Iterable of scalar doses that will be cycled through
          as the repetition continues.
    ef    Injectable string corresponding to an injectable dose-reponse
          function.
    n     Number of repetitions.
    freq  Frequency of repetition. Either a single Pandas frequency string, or
          a sequence of frequency strings that will be cycled through as the
          repetition continues. The latter format allows for a cycle of
          injections where, eg. you inject after 2 days, then after 3 days,
          then after 2 days, then after 3 days, etc."""

    if not isinstance(dose, Iterable):
        dose = (dose,)
    if isinstance(freq, str) or not isinstance(freq, Iterable):
        freq = (freq,)

    offsets = [pd.tseries.frequencies.to_offset(f) for f in (math.floor(n / len(freq)) * freq)[0:n]]
    inj_dates = [pd.to_datetime(date)]
    for o in offsets:
        inj_dates.append(inj_dates[-1] + o)

    doses = [d for d in (math.floor(n / len(dose)) * dose)[0:n]]
    return [*zip(inj_dates, doses, n*[ef])]


def zeroLevelsFromInjections(injections, sample_freq):
    """
    Returns a Pandas series of zeros indexed by time across the entire range of
    injections, sampled at a give frequency.

    The range of the returned series is big enough to include all the input
    injections and is expanded to the nearest day in either direction such that
    the range is large enough.

    Parameters:
    ===========
    injections  DataFrame of injections (see createInjections(...)).
    sample_freq  Sample frequency [Pandas frequency string]."""

    # TODO: this start time handling is funky and not generalizable to different
    # contexts this function may be used in, like the more recent optimization
    # stuff where it results in a 1 day misalignment.
    start_time = injections.index[0].floor('D')
    end_time   = injections.index[-1].ceil('D')
    dates      = pd.date_range(start_time, end_time, freq=sample_freq)
    return pd.Series(np.zeros(len(dates)), index=dates)

def zeroLevelsAtMoments(moments):
    """
    Returns a Pandas Series of zeros indexed by the input.
    
    This is used to create the output buffer for calcInjections when you are
    interested in the calculated blood level at a particular moment in time,
    such as at the moment of injection, or a moment of measurement. It works
    with any DatetimeIndex as input."""

    return pd.Series(np.zeros(len(moments)), index=moments)


def calcInjections(zero_levels, injections, injectables):
    """
    Compute blood levels for the given injections using a noncompartmental
    pharmacokinetic model.

    Given a DataFrame of injections (injection time, injection dose, and
    medication), this computes the simulated total blood concentration at every
    point in time in the index of zero_levels using the medication dose-reponse
    functions given in injectables.

    Conceptually, this additively superimposes the dose-response functions
    shifted in time for each dose. Each dose-response function returns the total
    blood levels contribution for a single dose of the medication, and each dose
    contributes linearly to the total systemic blood concentraion.

    This mutates the zero_levels input series and also returns it.

    Parameters:
    ===========
    zero_levels   Series of zeros indexed by Datetime of times at which the
                  total blood concentration curve curve should be sampled at.
    injections    DataFrame of injections that specify the blood concentration
                  curve (see createInjections(...)).
    injectables   Dict of injectable dose-reponse functions where keys are
                  strings corresponding with the values of the injectable
                  column of injections (see injectables.injectables)."""

    for inj_date, inj_dose, inj_injectable in injections[["dose", "injectable"]].itertuples():
        inj_ef = injectables[inj_injectable]
        
        # If the function has a specified domain, we can use that to avoid
        # doing work we don't need to. If it doesn't have a specified domain,
        # don't make any assumptions about it. This allows for the seamless
        # calculation of derivatives calculated using the scipy wrapper function.
        if hasattr(inj_ef, 'domain') and inj_ef.domain is not None:
            # Only need to compute where inj_ef > 0. This is guaranteed to be true
            # exactly across the domain of inj_ef, so we can save significant
            # computation here.
            max_T = inj_date + inj_ef.domain[1]
            levels_range = zero_levels[inj_date:max_T].index
        else:
            levels_range = zero_levels.index
        
        for T in levels_range:
            zero_levels[T] += inj_dose * inj_ef(timeDeltaToDays(T - inj_date))
    
    return zero_levels


def startPlot():
    fig, ax_pri = pyplot.subplots(figsize=[15, 12], dpi=150)
    
    ax_pri.set_axisbelow(True)
    ax_pri.set_zorder(0)
    
    ax_pri.set_xlabel('Date')
    ax_pri.set_ylabel('Estradiol (pg/mL)')
    ax_pri.xaxis_date()
    
    ax_pri.xaxis.set_major_locator(mdates.MonthLocator())
    ax_pri.xaxis.set_major_formatter(mdates.DateFormatter("1%Y.%b.%d.%H%M"))
    ax_pri.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=range(1, 32, 3)))
    ax_pri.xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
    ax_pri.tick_params(which='major', axis='x', labelrotation=-45, pad=12)
    pyplot.setp(ax_pri.xaxis.get_majorticklabels(), ha="left")
    ax_pri.tick_params(which='minor', axis='x', labelrotation=-90, labelsize=6)
    
    ax_pri.grid(which='major', axis='both', linestyle=':', color=(0.8, 0.8, 0.8), alpha=0.7)
    ax_pri.grid(which='minor', axis='both', linestyle=':', color=(0.8, 0.8, 0.8), alpha=0.2)
    
    return fig, ax_pri

def plotInjections(fig, ax,
                   injections,
                   injectables,
                   estradiol_measurements=pd.DataFrame(),
                   sample_freq='1H',
                   label=''):
    """
    Plot the continuous, simulated curve of blood levels for a series of
    injections, along with associated blood level measurements if they exist.

    Parameters:
    ===========
    injections              Injections to plot, represented as a Pandas
                            DataFrame with "dose" and "injectable" columns
                            indexed by DateTime. See createInjections(...). By
                            convention, the last injection is expected to be a
                            zero-dose marker of the simulation window and is
                            discarded.
    injectables             Mapping of injectable name to ef function. [dict]
    estradiol_measurements  (Optional) Actual blood level measurements to plot
                            alongside the simulated curve, represented as a
                            Pandas DataFrame with "value" and "method" columns
                            indexed by DateTime. See createMeasurements(...).
    sample_freq             (Optional) Frequency at which the calculated
                            continuous curve is sampled at. [Pandas frequency
                            thing]
    label                   (Optional) Matplotlib label to attach to the curve.
                            [String]"""

    # Easy way to display a vertical line at the current time
    estradiol_measurements = pd.concat([
        estradiol_measurements,
        createMeasurements(np.array([[pd.Timestamp.now(), -10.0, np.nan]]))])
    
    # Calculate everything we'll need to plot
    e_levels = calcInjections(
        zeroLevelsFromInjections(injections, sample_freq),
        injections,
        injectables)
    levels_at_injections = calcInjections(
        zeroLevelsAtMoments(injections.index),
        injections,
        injectables)[0:-1]
    levels_at_measurements = calcInjections(
        zeroLevelsAtMoments(estradiol_measurements.index),
        injections,
        injectables)
    
    # The primary axis displays absolute date.
    ax_pri = ax
    
    # matplotlib uses *FLOAT DAYS SINCE EPOCH* to represent dates.
    # set_xlim can take a pd DateTime, but converts it to that ^
    ax_pri.set_xlim((mdates.date2num(e_levels.index[0]),
                     mdates.date2num(e_levels.index[-1])))
    
    # The secondary axis displays relative date in days.
    def mdate2reldays(X):
        return np.array([d - mdates.date2num(e_levels.index[0]) for d in X])
    def reldays2mdate(X):
        return np.array([mdates.date2num(e_levels.index[0]) + d for d in X])
    
    ax_sec = ax_pri.secondary_xaxis('top', functions=(mdate2reldays, reldays2mdate))
    ax_sec.set_xlabel("Time (days)")
    ax_sec.set_xticks(np.arange(0.0,
                                timeDeltaToDays(e_levels.index[-1] - e_levels.index[0]) + 1.0,
                                9.0))
    ax_sec.xaxis.set_minor_locator(mticker.AutoMinorLocator(n=3))
    ax_sec.tick_params(axis='x', labelrotation=45)
    
    # Plot simulated curve
    ax_pri.plot(e_levels.index,
                e_levels.values,
                label=label,
                zorder=1)
    
    # Plot moments of injection as dose-scaled points on top of the simulated
    # curve, independently for each kind of injectable.
    for injectable, group in injections[0:-1].groupby(by="injectable"):
        doses         = group["dose"].values
        norm_doses    = Normalize(vmin=-1.0*max(doses), vmax=max(doses)+0.2)(doses)
        marker_sizes  = [(9.0*dose+2.0)**2 for dose in norm_doses]
        marker_colors = [(dose, 1.0-dose, 0.7, 1.0) for dose in norm_doses]
        levels_at_group = levels_at_injections[group.index]
        ax_pri.scatter(levels_at_group.index,
                       levels_at_group.values,
                       s=marker_sizes,
                       c=marker_colors,
                       marker=injectables_markers[injectable],
                       zorder=2,
                       label=f"{injectable} inj")
    ax_pri.legend()
    
    # Plot measured blood levels
    ax_pri.plot(estradiol_measurements.index,
                estradiol_measurements["value"].values,
                'o')

    # Draw vertical lines from the measured points to the simulated curve
    measurements_points = [
        (mdates.date2num(d), m) for d,m in estradiol_measurements["value"].items()]
    levels_at_measurements_points = [
        (mdates.date2num(d), m) for d,m in levels_at_measurements.items()]
    
    lines = list(zip(measurements_points, levels_at_measurements_points))
    lc = mc.LineCollection(lines, linestyles=(0, (2, 3)), colors=(0.7, 0.3, 0.3, 1.0), zorder=3)
    ax_pri.add_collection(lc);
    
    for line in lines:
        # if length of line in figure coords is > length of text in figure
        # coords, then draw the text at the midpoint of the line. We assume
        # this is true to begin with, but then alter the text if it turns out
        # to not be true.
        mp = (0.5*(line[0][0] + line[1][0]), 0.5*(line[0][1] + line[1][1]))
        txt = ax_pri.text(
            mp[0], mp[1],
            mdates.DateFormatter("1%Y.%b.%d.%H%M").format_data(line[0][0]),
            rotation=-90,
            ha="left", va="center",
            fontsize="x-small",
            color=(0.7, 0.3, 0.3, 1.0),
            zorder=3,
            clip_on=True)
        
        # elif length of line in display coords is <= length of text in display
        # coords, then draw the text above or below the line segment, with
        # ha="center".
        txt_bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        line_dc = ax_pri.transData.transform(line)
        if abs(line_dc[0][1] - line_dc[1][1]) <=\
           abs(txt_bbox.p0[1] - txt_bbox.p1[1]):
            txt.set_y(line[0][1])
            if line[0][1] >= line[1][1]:
                txt.set_va("bottom")
            else:
                txt.set_va("top")
            txt.set_ha("center")

    
def plotInjectionFrequencies(fig, ax, ef, sim_time, sim_freq, inj_freqs):
    """
    Plot multiple injection curves for a range of injection frequencies.

    Parameters:
    ===========
    ef         Interpolated function ef(T days) = (E mg) (from rawDVToFunc) for
               a single injection.
    sim_time   Total simulation time. [Days]
    sim_freq   Resolution of simulaion. [Pandas frequency thing]
    inj_freqs  List of period curves to plot. [Pandas frequency things]
    """

    injectables = {"ef": ef}
    for freq in inj_freqs:
        plotInjections(
            fig, ax,
            createInjectionsCycle("ef", sim_time, freq),
            injectables,
            sample_freq=sim_freq,
            label=f"{freq} freq")

    ax.set_xlim(
        ax.get_xlim()[0],
        pd.to_datetime(sim_time, unit='D'))
    ax.legend()
