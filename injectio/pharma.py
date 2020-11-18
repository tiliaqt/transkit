from collections import defaultdict
from collections.abc import Iterable
import math
from matplotlib import pyplot
from matplotlib import dates as mdates
from matplotlib import collections as mc
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy import interpolate, signal
import warnings


injectables_markers = defaultdict(lambda: "2", {
    "ec": "2",
    "ev": "1",
    "pill-oral": ".",
    "pill-subl": ".",
    "pill-bucc": ".",
})


def dateTimeToDays(dt):
    if isinstance(dt, pd.DatetimeIndex) or isinstance(dt, pd.Timestamp):
        dt = dt.to_numpy(dtype='datetime64[ns]')

    return (dt.astype('datetime64[ns]').astype(np.float64)
            * 1e-9 # ns to s
            / 60.0 # s to min
            / 60.0 # min to hr
            / 24.0) # hr to days
def timeDeltaToDays(td):
    if isinstance(td, pd.TimedeltaIndex) or isinstance(td, pd.Timedelta):
        td = td.to_numpy(dtype='timedelta64[ns]')

    return (td.astype('timedelta64[ns]').astype(np.float64)
            * 1e-9 # ns to s
            / 60.0 # s to min
            / 60.0 # min to hr
            / 24.0) # hr to days



###################################
### Injection Creation Routines ###

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

    n_inj = math.ceil(sim_time / (pd.tseries.frequencies.to_offset(inj_freq).nanos * 1.0e-9/60.0/60.0/24.0))
    injs = createInjections(np.array(rep_from([start_date, 1.0, ef], n=n_inj, freq=inj_freq)))
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


def zeroLevelsFromInjections(injections, sample_freq, upper_bound='midnight'):
    """
    Intelligently create a simulation window for the injections.

    Returns a Pandas series of zeros indexed by time across the entire range of
    injections, sampled at a given frequency. The range of the returned series
    is big enough to include all the input injections. The lower bound of the
    window is midnight (00h:00m) of the day of the first injection, and the
    upper bound is controlled by the upper_bound parameter.

    Parameters:
    ===========
    injections   DataFrame of injections (see createInjections(...)).
    sample_freq  Sample frequency [Pandas frequency string].
    upper_bound  1- or 2-tuple specifying how to handle the upper bound of the
                 window. The first value of the tuple is a string chooding the
                 the method of handling, and if present the second value is the
                 paramter to that method. Possible methods are:
                     ('midnight', n=1): Upper bound will be the first sample
                                        greater than midnight (00h:00m) of the
                                        nth day following the last injection.
                     ('continue', n=1): Upper bound will be the first sample
                                        greater than the period as if the last
                                        injection was repeated n times, using
                                        the time between the last two
                                        injections as the period. Requires
                                        at least two injections.
                    ('timedelta', t):   Upper bound will be the first sample
                                        greater than timedelta t after the
                                        last injection. t is required and may
                                        be either a pandas.TimeDelta or a
                                        number of days.
                    ('datetime', t):    Upper bound will be the first sample
                                        greater than the time t. t is required
                                        and can be any date parsable by
                                        pandas.to_datetime.
                 ('midnight') is the default method."""

    if isinstance(upper_bound, str):
        upper_bound = (upper_bound,)

    ub_method = upper_bound[0] if len(upper_bound) > 0 else None
    ub_param  = upper_bound[1] if len(upper_bound) > 1 else None

    if ub_method == 'midnight':
        n = 1 if ub_param is None else ub_param
        end_time = (injections.index[-1] + pd.to_timedelta(n, unit='D')).floor('D')
    elif ub_method == 'continue':
        if len(injections) < 2:
            raise ValueError("upper_bound method 'continue' requires at least "
                             "two injections.")

        n = 1 if ub_param is None else ub_param
        freq = injections.index[-1] - injections.index[-2]
        end_time = injections.index[-1] + n*freq
    elif ub_method == 'timedelta':
        if ub_param is None:
            raise ValueError("upper_bound method 'timedelta' requires an "
                             "argument.")

        t = pd.to_timedelta(ub_param, unit='D')
        end_time = injections.index[-1] + t
    elif ub_method == 'datetime':
        if ub_param is None:
            raise ValueError("upper_bound method 'datetime' requires an "
                             "argument.")

        end_time = pd.to_datetime(ub_param)
    else:
        raise ValueError("upper_bound method '{ub_method}' isn't valid.")

    start_time = injections.index[0].floor('D')

    # If our end_time isn't properly aligned, increment it by one sample
    # period and let date_range truncate to the correctly aligned end.
    # This ensures date_range doesn't pick an end_date that doesn't
    # include our entire window.
    freq_offset = pd.tseries.frequencies.to_offset(sample_freq)
    freq_sec = (freq_offset.nanos * 1e-9)
    if (end_time - start_time).total_seconds() % freq_sec != 0:
        end_time = freq_offset.apply(end_time)

    dates = pd.date_range(start_time, end_time, freq=sample_freq)
    return pd.Series(np.zeros(len(dates)), index=dates)


def zeroLevelsAtMoments(moments):
    """
    Returns a Pandas Series of zeros indexed by the input.
    
    This is used to create the output buffer for calcInjections when you are
    interested in the calculated blood level at a particular moment in time,
    such as at the moment of injection, or a moment of measurement. It works
    with any DatetimeIndex as input."""

    return pd.Series(np.zeros(len(moments)), index=moments)



###################################
### Pharmacokinetic Computation ###

def calcInjectionsExact(zero_levels, injections, injectables):
    """
    Compute blood levels for the given injections using a
    noncompartmental pharmacokinetic model.

    Given a DataFrame of injections (injection time, injection dose, and
    medication), this computes the simulated total blood concentration
    at every point in time in the index of zero_levels using the
    medication dose-reponse functions given in injectables.

    This additively superimposes the dose-response functions shifted in
    time for each dose. Each dose-response function returns the total
    blood levels contribution for a single dose of the medication, and
    each dose contributes linearly to the total systemic blood
    concentraion. This implementation is O(N^2) in the number of
    samples. Unlike the convolution implementation, it uses a continous
    dose-response calculation and doesn't suffer from sample-rate
    aliasing. This makes it well suited for computing blood levels at
    sparse or irregular times, or in cases where the sample rate is slow
    enough that errors in the convolution approach become untenable.

    This mutates the zero_levels input series and also returns it.

    Parameters:
    ===========
    zero_levels   Series of zeros indexed by Datetime of times at which
                  the total blood concentration curve curve should be
                  sampled at.
    injections    DataFrame of injections that specify the blood
                  concentration curve (see createInjections(...)).
    injectables   Dict of injectable dose-reponse functions where keys
                  are strings corresponding with the values of the
                  injectable column of injections (see
                  injectables.injectables)."""

    for inj_date, inj_dose, inj_injectable in injections[["dose", "injectable"]].itertuples():
        inj_ef = injectables[inj_injectable]

        # If the function has a specified domain, we only need to calculate
        # levels across that. If it doesn't, then we need to calculate levels
        # across the entire sample space.
        if hasattr(inj_ef, 'domain') and inj_ef.domain is not None:
            np_date = inj_date.to_numpy()
            max_T = np_date + inj_ef.domain[1].to_numpy()

            # Find where in the sample space we need to modify. Doing it this
            # way is a lot faster than Panda's time-based slices.
            idxs = np.searchsorted(zero_levels.index.values, [np_date, max_T])
            level_idxs = np.arange(
                    idxs[0],
                    idxs[1] + (idxs[1] < len(zero_levels))) # Inclusive

            zero_levels.values[level_idxs] += (inj_dose * inj_ef(timeDeltaToDays(
                zero_levels.index.values[level_idxs] - np_date)))
        else:
            zero_levels += (inj_dose * inj_ef(timeDeltaToDays(
                zero_levels.index - inj_date)))

    return zero_levels


def calcInjectionsConv(zero_levels, injections, injectables):
    """
    Compute blood levels for the given injections using a
    noncompartmental pharmacokinetic model.

    Given a DataFrame of injections (injection time, injection dose, and
    medication), this computes the simulated total blood concentration
    at every point in time in the index of zero_levels using the
    medication dose-reponse functions given in injectables.

    This mutates the zero_levels input series and also returns it.

    The noncompartmental pharmacokinetic model is a Linear
    Shift-Invariant system, which allows us to perform the computation
    using the discrete convolution, which can be computed in O(NlogN) in
    the number of samples. This makes it faster than calcInjectionsExact
    for large numbers of samples.

    The trade off of this approach is that it introduces numeric errors
    into the results, dependent on the sample rate. Injection times must
    be projected into the sample space, and they will not typically be
    aligned with any samples. This creates temporal aliasing where the
    dose-response impulses will be shifted in time to the nearest
    sample. Similarly, the dose-response functions need to be resampled,
    which can introduce further amplitude aliasing, especially where
    there are sharp peaks (looking at you, Estradiol Valerate).

    The error can be nearly zero if injections are aligned with samples.
    This happens when, e.g., injection times are specified with 1 minute
    precision, there is 1 minute between each sample, and samples are
    aligned with the start of each minute. In pathological cases, the
    error stays well below 1 pg/mL up to around 50 minutes between each
    sample. With faster sample rates (below 15 minutes between each
    sample), the error becomes negligible, below 0.1 pg/mL.

    Parameters:
    ===========
    zero_levels   Series of zeros indexed by Datetime of times at which
                  the total blood concentration curve curve should be
                  sampled at. The index must be fixed-frequency and
                  monotonically increasing.
    injections    DataFrame of injections that specify the blood
                  concentration curve (see createInjections(...)).
    injectables   Dict of injectable dose-reponse functions where keys
                  are strings corresponding with the values of the
                  injectable column of injections (see
                  injectables.injectables)."""

    if zero_levels.index.freq is None:
        warnings.warn(
                "It doesn't look like zero_levels has a fixed frequency. "
                "calcInjectionsConv requires fixed frequency samples or "
                "the results will be incorrect!",
                RuntimeWarning)

    def project_injs(injs, samples):
        # Find where each injection belongs in the sample space.
        it0 = np.searchsorted(samples.index, injs.index, side='right')

        # With side='right', zeros in it0 represent times that are
        # before the beginning of the sample window.
        if not np.all(it0):
            raise ValueError(
                f"Injections before the lower bound of zero_levels "
                f" at {zero_levels.index[0]} are unsupported with "
                f" calcInjectionsConv -- use calcInjectionsExact if "
                f"you need this functionality, or use a bigger window.")

        it0 -= 1
        t0 = samples.index[it0]
        it1 = it0 + (t0 < injs.index)

        # Drop any injections that are after the sample window.
        in_window = it1 < len(samples)
        it0 = it0[in_window]
        it1 = it1[in_window]
        t0 = t0[in_window]
        t1 = samples.index[it1]

        # Improve temporal aliasing by proportionally splitting the
        # injection dose between the two adjacent samples.
        dt = (t1 - t0).to_numpy()
        p = np.divide((injs.index[in_window] - t0).to_numpy(), dt,
                      out=np.zeros_like(dt, dtype=np.float64),
                      where=(dt != pd.to_timedelta(0))).astype(np.float64)

        inj_doses = injs[["dose"]].values[:,0][in_window]
        doses = np.zeros(len(samples), dtype=np.float64)
        np.add.at(doses, it0, inj_doses * (1 - p))
        np.add.at(doses, it1, inj_doses * p)

        return doses

    def resample_ef(inj_ef, samples):
        if hasattr(inj_ef, 'domain') and inj_ef.domain is not None:
            min_T = samples.index[0] + inj_ef.domain[0]
            max_T = samples.index[0] + inj_ef.domain[1]
            ef_range = samples[min_T:max_T].index
        else:
            ef_range = samples.index

        return inj_ef(timeDeltaToDays(ef_range - ef_range[0]))

    # We run one convolution for each injectable, and combine the results.
    grp_injectables = injections.groupby(by="injectable")
    for injectable_idx, (injectable, group) in zip(
            range(len(grp_injectables)), grp_injectables):
        # First, project the injections into the sample space.
        doses = project_injs(group, zero_levels)

        # Next, resample the dose response function into our sample space
        # to create the kernel for the convolution.
        inj_ef = injectables[injectable]
        dose_response = resample_ef(inj_ef, zero_levels)

        # Now do the convolution!
        zero_levels += pd.Series(
                signal.convolve(
                    doses,
                    dose_response,
                    mode='full')[0:len(zero_levels)],
                index=zero_levels.index)

    return zero_levels


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

    def ef(x):
        idx = np.searchsorted(sample_x, x)-1
        return np.where(
                np.logical_or(x <= min_x, x >= max_x),
                np.zeros_like(x),
                sample_y[idx] + ((x - sample_x[idx])
                                 * np.take(slope_lut, idx, mode='clip')))
    ef.domain = (pd.to_timedelta(min_x, unit='D'), pd.to_timedelta(max_x, unit='D'))

    return ef



################
### Plotting ###

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
                   sample_freq='15min',
                   label='',
                   upper_bound=('continue', 1)):
    """
    Plot the continuous, simulated curve of blood levels for a series of
    injections, along with associated blood level measurements if they exist.

    Parameters:
    ===========
    injections              Injections to plot, represented as a Pandas
                            DataFrame with "dose" and "injectable" columns
                            indexed by DateTime. See createInjections(...).
    injectables             Mapping of injectable name to ef function. [dict]
    estradiol_measurements  (Optional) Actual blood level measurements to plot
                            alongside the simulated curve, represented as a
                            Pandas DataFrame with "value" and "method" columns
                            indexed by DateTime. See createMeasurements(...).
    sample_freq             (Optional) Frequency at which the calculated
                            continuous curve is sampled at. [Pandas frequency
                            thing]
    label                   (Optional) Matplotlib label to attach to the curve.
                            [String]
    upper_bound             Upper bound of the simulation window (see
                            pharma.zeroLevelsFromInjections(...))."""

    # Easy way to display a vertical line at the current time
    estradiol_measurements = pd.concat([
        estradiol_measurements,
        createMeasurements(np.array([[pd.Timestamp.now(), -10.0, np.nan]]))])

    # Calculate everything we'll need to plot
    e_levels = calcInjectionsConv(
        zeroLevelsFromInjections(injections, sample_freq, upper_bound=upper_bound),
        injections,
        injectables)
    levels_at_injections = calcInjectionsExact(
        zeroLevelsAtMoments(injections.index),
        injections,
        injectables)
    levels_at_measurements = calcInjectionsExact(
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
    for injectable, group in injections.groupby(by="injectable"):
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
