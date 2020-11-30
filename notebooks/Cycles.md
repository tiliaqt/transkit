---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Sacred Cycles
*Simulating Menstrual-esque Cycles Using Injected Estrogen*

```python
%load_ext autoreload
%autoreload 2
%matplotlib inline

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "../")
from transkit import pharma, fit, medications
```

```python
t3_ec_cali = np.array([0.000000000, 0.246635358])
t2_ev_cali = np.array([0.000000000, 1.428689307])

calibrated_medications = dict(medications.medications)
calibrated_medications["ec"] = pharma.calibratedDoseResponse(
    calibrated_medications["ec"],
    t3_ec_cali)
calibrated_medications["ev"] = pharma.calibratedDoseResponse(
    calibrated_medications["ev"],
    t2_ev_cali)
```

# Optimization

This constructs an optimization problem solving for optimal injection doses and times to fit a desired blood level profile.


## Menstrual

```python
def molarConcToMassConc(molar_conc, molar_weight):
    """Converts pmol/L to pg/mL for a substance of a particular molar weight.
    
    molar         Molar concentration in pmol/L [array like].
    molar_weight  Molar weight of substance in g/mol."""
    return molar_conc * molar_weight * 10**-3


e2_mol = 272.388 # g/mol

# Stricker2006, referenced from https://en.wikiversity.org/wiki/WikiJournal_of_Medicine/Reference_ranges_for_estradiol,_progesterone,_luteinizing_hormone_and_follicle-stimulating_hormone_during_the_menstrual_cycle
menstrual_e2 = molarConcToMassConc(np.array([118.42, 133.01, 133.17, 125.95, 134.65, 151.33, 159.48, 170.34, 195.9, 228.2, 269.07, 343.68, 477.27, 661.19, 914.84, 780.76, 320.59, 261.32, 338.94, 454.07, 499.49, 497.07, 531.14, 504.39, 499.18, 526.68, 350.65, 322.24, 229.7, 249.28, 150.0, 118.42, 118.42, 118.42, 118.42]),
                                   e2_mol)

menstrual_fit_start_t = pd.to_datetime("2020-01-01 12:00:00")
menstrual_fit_range_t = 6*7 - 1 + len(menstrual_e2) # 6 weeks to steady state, then the menstrual curve.
menstrual_fit_t  = pd.date_range(menstrual_fit_start_t,
                                 menstrual_fit_start_t + pd.to_timedelta(menstrual_fit_range_t, unit='D'),
                                 freq='1D')
menstrual_fit_e2 = np.append(np.full(6*7, menstrual_e2[0]), menstrual_e2)

menstrual_fit = pd.Series(menstrual_fit_e2, index=menstrual_fit_t)
menstrual_fit_results = fit.emptyResults()
```

### Estradiol Cypionate

Least squares figures out some real gymnastics to get this close, but EC doesn't decay quickly enough to mimic the big peak very well. I wonder if this result is still in range for some realistic cycle, even if it doesn't quite match the mean?

```python
doses = pharma.createDosesCycle(
    "ec",
    menstrual_fit_range_t,
    '3D',
    start_date=menstrual_fit_start_t)
menstrual_fit_results["ec"] = fit.initializeFit(
    doses,
    calibrated_medications,
    menstrual_fit,
    dose_bounds=(0.0, 10.0),
    time_bounds=14*['fixed'] + (len(doses)-14)*['midpoints'],
    equal_doses=[doses.index[:13]], # All injections in the ramp-up to steady-state should be equal
    exclude_area=menstrual_fit[:doses.index[12]].index) # Only optimize starting from the first injection after steady-state
fit.runLeastSquares(
    menstrual_fit_results["ec"],
    max_nfev=35,
    ftol=1e-2,
    verbose=2);
```

```python
fig,ax = pharma.startPlot()
ax.set_title('Optimizing Estradiol Cypionate doses and times')
ax.set_ylim(bottom=0, top=260)
ax.set_yticks(range(0, 261, 50))
fit.plotOptimizationRun(fig, ax, menstrual_fit_results["ec"])

with pd.option_context("display.float_format", "{:10,.3f}".format):
    print("Optimized Doses:\n", menstrual_fit_results["ec"]["doses_optim"])
```

### Estradiol Valerate

After all this work, it looks like EV can actually fit a realistic menstrual curve better than EC can. Wow! That's incredibly unexpected for me but I guess I was doing all of the curve fitting stuff at unrealistic time scales. The interesting bit here is that, because of the big intense spike in estradiol prior to ovulation, the decay rate of EC isn't able to keep up. EV, with it's big quick peak, and fast decay, is perfectly suited to mimicing that peak in estradiol.

Somatically it's almost like taking EV is like ovulating with every injection–and that actually resonates with my subjective experience pretty well. EV has a kind of dynamicism and responsiveness and punchiness to it. EC is much softer, pillowy, flowy, floaty. EV is unsurprisingly well suited to fitting to quicker, more dynamic curves; EC is better suited to gentler, more stable curves. They each have strengths. I have no idea how you'd get it prescribed, but I wonder if mixing medicines could work. Like using EC for most injections, but using EV for just around the big peak. Or maybe spiking with pills there. Once you've figured out the calibration for your body, a whole lot of magic is possible!

At realistic time scales there's a lot more to "stable levels" than just optimizing for constant e2 levels, that's for sure! I feel a bit sad and disappointed, and I wish I had done this piece with a real menstrual cycle curve much sooner. I am feeling a big expansion in my understanding of all this though, and how it fits together somatically. It's not what I was expecting, and that's really exciting! Laying it all out on a timeline is really important. I feel some inklings and sparking connections happening...

```python
doses = pharma.createDosesCycle(
    "ev",
    menstrual_fit_range_t,
    '3D',
    start_date=menstrual_fit_start_t)
menstrual_fit_results["ev"] = fit.initializeFit(
    doses,
    calibrated_medications,
    menstrual_fit,
    dose_bounds=(0.0, 10.0),
    time_bounds=14*['fixed'] + (len(doses)-14)*['midpoints'],
    equal_doses=[doses.index[:14]], # All injections in the ramp-up to steady-state should be equal
    exclude_area=menstrual_fit[:doses.index[13]].index) # Only optimize starting from the first injection after steady-state
fit.runLeastSquares(
    menstrual_fit_results["ev"],
    max_nfev=35,
    ftol=1e-2,
    verbose=2);
```

```python
fig,ax = pharma.startPlot()
ax.set_title('Optimizing Estradiol Valerate doses and times')
ax.set_ylim(bottom=0, top=260)
ax.set_yticks(range(0, 261, 50))
fit.plotOptimizationRun(fig, ax, menstrual_fit_results["ev"])

with pd.option_context("display.float_format", "{:10,.3f}".format):
    print("Optimized Doses:\n", menstrual_fit_results["ev"]["doses_optim"])
```

## Sine Wave

*try to fit blood levels to a sine wave curve*

```python
cycling_start_t = 42.0 # days
sine_target_func = lambda T: int(T>=cycling_start_t) * (250.0 + 50.0*np.sin((2*math.pi/30)*(T-cycling_start_t))) +\
                             int(T<cycling_start_t)  * (250.0)
sine_start_t = pd.to_datetime("2020")
sine_target_x = pd.date_range(sine_start_t, sine_start_t + pd.to_timedelta(90.0, unit='D'), freq='18H')
sine_target_y = np.array([sine_target_func(pharma.timeDeltaToDays(T - sine_target_x[0])) for T in sine_target_x])

sine_target = pd.Series(sine_target_y, index=sine_target_x)
sine_results = fit.emptyResults()
```

### Least Squares (Estradiol Cypionate, optimizing doses)

Works ok, doesn't find a great solution right now.

```python
sine_results["ec_doses"] = fit.initializeFit(
    pharma.createDosesCycle("ec", 90.0, '3D', start_date="2020"),
    calibrated_medications,
    sine_target,
    dose_bounds=(0.0, 5.0),
    time_bounds='fixed')
fit.runLeastSquares(
    sine_results["ec_doses"],
    max_nfev=35,
    ftol=1e-3,
    verbose=2);
```

```python
fig,ax = pharma.startPlot()
ax.set_title('Optimizing Estradiol Cypionate doses with fixed times')
ax.set_ylim(bottom=0, top=350)
ax.set_yticks(range(0, 351, 50))
fit.plotOptimizationRun(fig, ax, sine_results["ec_doses"])

with pd.option_context("display.float_format", "{:10,.3f}".format):
    print("Optimized Doses:\n", sine_results["ec_doses"]["doses_optim"])
```

### Least Squares (Estradiol Cypionate, Doses & Times)

This works awesome!

```python
doses = pharma.createDosesCycle("ec", 90.0, '3D', start_date="2020")
sine_results["ec_doses_and_times"] = fit.initializeFit(
    doses,
    calibrated_medications,
    sine_target,
    dose_bounds=(0.0, 5.0),
    time_bounds=['fixed'] + (len(doses)-1)*['midpoints'])
fit.runLeastSquares(
    sine_results["ec_doses_and_times"],
    max_nfev=30,
    ftol=1e-3,
    verbose=2);
```

```python
fig,ax = pharma.startPlot()
ax.set_title('Optimizing Estradiol Cypionate doses and times')
ax.set_ylim(bottom=0, top=350)
ax.set_yticks(range(0, 351, 50))
fit.plotOptimizationRun(fig, ax, sine_results["ec_doses_and_times"])

with pd.option_context("display.float_format", "{:10,.3f}".format):
    print("Optimized Doses:\n", sine_results["ec_doses_and_times"]["doses_optim"])
```

### Least Squares (Estradiol Valerate, Doses & Times)

At reasonable injection schedules, EV just really doesn't work for doing any kind of cycling. You can do it alright if you inject more frequently than the single-dose peak ($\tau \lt t_{max}$), but the smoother curve of Estradiol Cypionate makes it more suitable for this use.

```python
doses = pharma.createDosesCycle("ev", 90.0, '3D', start_date="2020")
sine_results["ev_doses_and_times"] = fit.initializeFit(
    doses,
    calibrated_medications,
    sine_target,
    dose_bounds=(0.0, 5.0),
    time_bounds=['fixed'] + (len(doses)-1)*['midpoints'])
fit.runLeastSquares(
    sine_results["ev_doses_and_times"],
    max_nfev=25,
    ftol=1e-3,
    verbose=2);
```

```python
fig,ax = pharma.startPlot()
ax.set_title('Optimizing Estradiol Valerate doses and times')
ax.set_ylim(bottom=0, top=350)
ax.set_yticks(range(0, 351, 50))
fit.plotOptimizationRun(fig, ax, sine_results["ev_doses_and_times"])

with pd.option_context("display.float_format", "{:10,.3f}".format,
                       "display.max_rows", sys.maxsize):
    print("Optimized Doses:\n", sine_results["ev_doses_and_times"]["doses_optim"])
```

### Step func (Estradiol Cypionate, optimize doses & times)

It's a loading dose! This demonstrates the control theory intuition that, if desired, you can reach a steady state level more quickly by using several large injections to reach the target level and then reducing to a regular consistent dose. That doesn't mean it would be good for your body! But, looking purely at blood concentration levels, it's possible. The lsq solution here achieves a 150pg/mL change in level, steady-to-steady, in just 9 days (in comparison to 6 weeks for a linear, consistent increase in dose.

```python
step_func = lambda T: int(T>=pd.to_datetime('1970-02-01')) * (250.0) +\
                       int(T<pd.to_datetime('1970-02-01'))  * (100.0)
step_target_x = pd.date_range(0, pd.to_datetime(62.0, unit='D'), freq='12H')
step_target_y = np.array([step_func(T) for T in step_target_x])

step_target = pd.Series(step_target_y, index=step_target_x)
step_results = fit.emptyResults()
```

```python
step_results["ec_doses_and_times"] = fit.initializeFit(
    pharma.createDosesCycle("ec", 62.0, '3D'),
    calibrated_medications,
    step_target,
    dose_bounds=(0.0, 5.0),
    time_bounds=7*['fixed'] + 8*['midpoints'] + 6*['fixed'])
fit.runLeastSquares(
    step_results["ec_doses_and_times"],
    max_nfev=35,
    ftol=1e-3,
    verbose=2);
```

```python
fig,ax = pharma.startPlot()
ax.set_title('Optimizing Estradiol Cypionate doses and times')
ax.set_ylim(bottom=0, top=350)
ax.set_yticks(range(0, 351, 50))
fit.plotOptimizationRun(fig, ax, step_results["ec_doses_and_times"])

with pd.option_context("display.float_format", "{:10,.3f}".format):
    print("Optimized Doses:\n", step_results["ec_doses_and_times"]["doses_optim"])
```

```python

```
