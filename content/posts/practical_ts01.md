---
title: "Explainable insights from sequence regression"
date: 2023-01-08T15:18:18-05:00
draft: false
author: "Aaron Slowey"
math: true
tags: ["modeling", "statsmodels"]
categories: ["technical"]
---


# Business considerations

Linear models are among the most explainable, and yet producing insights 
salient to business problems is not trivial.  Adopt a simple formulation, 
and a direct interpretation of parameters will require multiple footnotes to 
bridge the gap between what is meaningful on the terms of learned 
associations and what can confirm or alter a manager's point of view and 
strategy.  Adopt a more complex formulation and, well, you must have amazing 
infrastructure.

One reality that makes the gap so wide is that to discern any pattern, 
data sets need to be subdivided into coherent segments.  A linear model 
can do so through indicator variables, but practically speaking it can 
be necessary or even legally required to separate data into tens of 
thousands of segments.  A model will be learned for each segment, and 
each will require some level of validation.

Once validated, an interesting problem is how to build downstream analytics 
that consume the model output (predictions, parameters, performance metrics, 
etc.) and inform managers.  The granularity of their decisions is key to 
designing those analytics, which should refine the models' design.  

In this context, I address a detail that can appreciably erode the 
veracity and salience of model-derived insights.  This erosion is a risk 
when applying canned routines in general, and time series/forecasting 
packages in particular.  They abstract away important details, most notably  
how data are prepared for modeling.  As a result, such packages 
lead users to implicitly, rather than explicitly, choose parameters that are 
incongruent with, for example, the temporal structure of
the time series.  I have not researched it closely, but I fail to see how 
recent meta-learners that deploy dozens of forecasting algorithms 
solve this congruence issue, unless they adequately prepare data as well.  

Packages like `statsmodels.tsa` are workhorses, but do not check, for 
instance, whether a daily time 
series contains only business days, while the user has specified a 
'seasonality' period of 7.  When I started using it, I was unsure if 
it did or not, and so I created artificial data with known 
infidelities and effects, observed how `statsmodels` responded, and what 
distortions ensued.  This post is a recounting of some of those experiments.

The following covers _in-sample_ deconstruction of temporally
sensitive effects that can be applied to a variety of problems, including 
forecasting.  The objectives include predicting the future _and_ understanding 
temporal patterns, for a variety of reasons including explaining models to 
non-technical managers and assessing the tradeoff between model capacity, 
maintenance costs, failure risk, and other things that ultimately determine 
adoption and the delivery of value to an enterprise.

# Setting

We denote a sequence (e.g., time series) model by adding additional 
subscripts to a multivariate linear regression:
$$y_t = \alpha + \sum_{i=1}^m \beta_i x_{i,t} + \epsilon_t$$
The properties of variables that constitute $x$ determine what kind of time
series regression we perform.

- An auto-regression model includes up to $p$ lags: $x_{t-p}, \ldots, x_{t-1}$
- A linear trend is included by $x_{1, t} = t$; assuming equally spaced
  observations, $t$ would be equivalent to `numpy.linspace(1, T)`
- Day of week is achieved by having one binary variable for all but one day;
  i.e., $x_{i, t}=1$ if the observation occurs on a particular day and zero
  otherwise. For any categorical variable having $k$ unique values, include
  $k-1$ binary variables into the model. So if we leave out Sunday, $x_{Monday,
  t}$ measures the effect of Monday on $y$ _compared to the effect of Sunday_.
- Spike: A dummy variable that is 1 in a specific period, zero before and after
- Step: A dummy variable that is zero up to a point, 1 from that point on. 
  Similar for a change in slope.

As with its `OLS` class, you can apply statsmodels' `.summary()` method to the 
fitted `tsa` model object, as well as `.plot_predict(start=t0, end=t1)`.

The 'errors' should

- Be normally distributed with mean zero and a constant variance 
- Not be auto-, or serially, correlated; checks include the Breusch-Godfrey or 
  Lagrange Multiplier test, in which a small `p-value` indicates significant 
  autocorrelation remains.
- Unrelated to the predictors

Apply `.plot_diagnostics()` to the fitted model object to obtain  
some of these diagnostics.

`statsmodels.tsa` has multiple methods for time series
analysis. We do not have to use `.tsa` methods to model
sequence data; multivariate linear regression with univariate lags and time
characteristic variables could achieve roughly the same model. A variety of
considerations may determine the choice of model class, such as being
able to say you used a certain package.

In any case, we are tackling the challenge of building a linear model with
familiar performance criteria. The most profound difference is that the
observations are possibly auto-correlated, not I.I.D., but this may 'normalize'
out by including lags and time characteristics (seasonal components).  
Alternatively, we may model the (fractional) differences between observations.  
Some problems may call for harnessing the 'memory' of a time series, rather 
than erasing it for the sake of stationarity. 

A good starting point is the `AutoReg` class of `statsmodels.tsa.ar_model`.

[A worked example](https://www.statsmodels.
org/dev/examples/notebooks/generated/autoregressions.html?highlight=ar_select_order)
in the `statsmodels` documentation does not show how each component
manifests and how parameters affect the fit. That ambiguity ends here.

To what extent do `statsmodels` implementations
include utilities that recognize and utilize `datetimes`? The work presented 
here suggests not at all, despite warnings when `seasonal=True`
but `period` is unspecified. To obtain expected behavior from statistical
learning algorithms, it is crucial to know and potentially modify the sequential
structure (spacing) of data, because there do not appear to be intelligent
checks and automated cleaning processes. Other key questions include

- Consider a period $m=52$ weeks per year; unless `tsa` interprets datetime 
  values intelligently, setting `period=52` should only imply a weekly 
  periodicity (seasonality) if there are $7\cdot52$ rows of data.
- If we have $T=1.5$ years of data, would we not have to pre-specify
  the period as $m=p\cdot\frac{52}{q} \cdot T$, where p and q define the
  period of interest?
- What if there are a few days missing, seemingly randomly?
- ...or even not randomly?  For example, the data lack Saturdays, 
  Sundays, and/or holidays.
- Does the data need to be processed to ensure there are 7 days per week, 52
  weeks per year, imputing zeroes where needed?

To begin to shed light on these questions, create artificial data with known 
patterns. Create one year of daily
timestamps and initialize the observations with random numbers $\in (0, 1)$. For
other possibilities, see also `sklego.datasets.make_simpleseries`.

var spec = {{ stage0 }}

We'll work with several functions that are available in the project repo.

Encode a time characteristic, such as day of the week (dow) and boost the signal
on certain days (or weeks, etc.). Here, we spike the signal on Fridays and
experiment with the `seasonal` and `period` parameters.  The charts are
produced by Altair.

```python
ts = artifice()
ts = seq_ordinal(ts)
ts = signal_boost(ts,every_other_week=False)
s1model, s1params, _, stage1 = run_autoreg(ts.y,
                                           lags_=2,
                                           seasonal_=False,
                                           )
```

Here's a sample of our input data.

{{ s1data }}

With `seasonal=False`, we obtain an upwardly trending oscillation:

var spec = {{ stage1 }}

Enable `seasonal`, and we get the expected level baseline with weekly peaks:

```python
_, s2params, _, stage2 = run_autoreg(ts.y,
                     lags_=4,
                     seasonal_=True,
                     period_=7
                    )
```

var spec = {{ stage2 }}

If we increase `lags=7` to include the weekly effect, we get
almost as good a model as with seasonal terms:

```python
_, s3params, _, stage3 = run_autoreg(ts.y,
                     lags_=7,
                     seasonal_=False,
                    )
```

var spec = {{ stage3 }}

Any number of lags above seven, and we see no improvement. Although not shown,
the model with `lags=7` and `seasonal=True` with `period=7` looks identical to
the  `seasonal=True`, `period=7` model with `lags=4` above, suggesting that the
$x_{t-p}$ and $s_d$ terms are collinear.

`statsmodels.tsa.ar_model.AutoReg` interprets `period=7` as the _longest step_
from one data point to the next in the DataFrame; shorter steps from 1 to $p-1$
are also included. We think that something happens every 7th observation in the
sequence, and  `AutoReg` checks whether anything
happens on shorter cadences. As discussed in detail below, whether those _steps_
correspond to a meaningful time interval or period depends on the structure of
the sequence.

Look at the coefficients by applying the `.params` method to the fitted model
object; e.g., with `lags=2`, `seasonal=True`, and `period=7`:

{{ s4params }}

Note that with `period` and `lags` set to an integer, multiple terms are
included up to that value: e.g., $[1, \text{lags}]$. Unlike `period`, you may
provide a list of integers for `lags`, in which case _only_ those lags are
included. As expected, we find the `seasonal.6` coefficient to be higher 
than
those of narrower periodicity. It's also no accident that seasonal components
0-5's coefficients are _similar_ to each other. We will investigate how this
model responds to data with multiple periods.

`statsmodels.tsa` does not require the time or sequence index to be of
a `datetime` dtype. Replacing datetimes by integers, we obtain the same
result (not shown). But note that `AutoReg` is not being explicitly given a
sequence or time variable; it is implicit in the `pandas.Series` index of `ts.y`
, so the algorithm is unaware of the change in the time column `ts.t` . If we
remove a small number of points at random such that there are gaps in the index,
the model falls apart (not shown). Can we make the seasonal regression algorithm
aware that observations are made on calendar days?

```python
ts, idx_mask = abscond(ts, points=5)
```

Setting the index with the datetime-formatted values (maintaining the five
randomly placed gaps) leads
to `ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.`
The predictions from `.predict()` are all null.

What matters is a logical correspondence between `period` and the frequency of
the data as presented by their sequence in the array or DataFrame. Let's
say `period=7`; if the frequency is unspecified, the algorithm considers if
a data point six steps from the current point tends to be higher, lower, or
about the same as the current point. If the data happen to be observations
recorded every nanosecond with a perturbation every 13 ns, `period=13` should
fit that sequence nicely. This naive behavior is helpful for modeling
observations that occur with a regularity that is meaningful, if not in a
temporal way. For example, every fourth trip to buy groceries, the family goes
to Costco, not Trader Joe's. But it poses a problem for incomplete and irregular
sequences when effects pertain to certain fixed time qualities.

When `.set_index('dt_col')` involves a datetime column, we obtain
a `DateTimeIndex` with `freq=None`, which `statsmodels.tsa.ar_model`
complains about. We can specify the frequency

Having reproduced the result shown in Stage 2 with a complete data set indexed
in this way, we return to the case where points are randomly missing; remove
them _prior to_ setting the datetime index and frequency. Here, we introduce a
potential problem. As an aside, `.asfreq('B')` sets an index to daily business
day. `.asfreq` can be applied to a DataFrame; it will return the DataFrame 
"reindexed to the specified frequency."
Meaning the "original data conformed to a new index with the specified
frequency."  In this case, to conform to daily frequency, rows of `nan` are
placed where time points were missing. Before and after applying `.asfreq('d')`:

__Before__

{{ s5_before_asfreq }}

__After__

```python
ts = ts.set_index('t').asfreq('d')
```

{{ s5_after_asfreq }}

`AutoReg` will raise a `MissingDataError: exog contains inf or nans`, resolvable
by including `missing='drop'`. And yet even though we have a `DatetimeIndex`
with `freq='D'`, we still get
a `ValueWarning: A date index has been provided, but it has no associated frequency information`
. Instead of a `DatetimeIndex` with `.asfreq('d')`, we can try a `PeriodIndex`,
a subclass of `Index` that is regularly spaced:

```python
ts.set_index('t', inplace=True)
ts.index = pd.DatetimeIndex(ts.index).to_period('D')
```

```
PeriodIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
             '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08',
             '2022-01-09', '2022-01-10',
             ...
             '2022-12-22', '2022-12-23', '2022-12-24', '2022-12-25',
             '2022-12-26', '2022-12-27', '2022-12-28', '2022-12-29',
             '2022-12-30', '2022-12-31'],
            dtype='period[D]', name='t', length=360, freq='D')
```

We still have gaps in the time series, but no `nan` have been inserted:

{{ s5_periodIndex }}

It is unnecessary to include `missing='drop'` in `AutoReg()`, since the data has
no missing values (it is good practice to include `missing='raise'` as
a check).  `.fit()` runs and `.predict()` returns values without altering the
input data structure: it still has a `PeriodIndex`:

```python
_, s5params, ts5, stage5 = run_autoreg(ts.y,
                                       lags_=7,
                                       seasonal_=False,
                                      )
```

{{ ts5md }}

Did the model fit well? While the `PeriodIndex` facilitated the fit, it
complicates plotting with Altair. Assuming the index needs to be reset into a
column, a `PeriodIndex`  resets to a column of `'period[D]'` dtype, which
causes a `TypeError`. After fitting the model with `PeriodIndex`-ed data,
reformat the index:

```python
ts.index = ts.index.to_timestamp()
```

You will lose the `freq` and have a `DatetimeIndex` once again. Then reset 
_that_ index and plot as usual.

var spec = {{ stage5 }}

{{ s5params }}

Our model did not capture the Friday signal as before, presumably because it
naively assumed every $7\cdot n^{th}$ point was Friday, ignoring the days
skipped in the index. Note how the parameters' seasonal components are written
as `s(1, 7)`, whereas before it was `seasonal.0`. The connotation is that we're
capturing effects that occur every 1st of 7 days.

We've tried two ways to structure our data such that `statsmodels.tsa` is aware
of timing. But this data transformation did not apparently raise such awareness,
as the model failed to fit the elevated values on Fridays. It could not even use
the `DateTimeIndex` with a daily frequency. All of these observations suggest
that what ultimately matters to `statsmodels.tsa` is the numerical index of the
table; row 6 means Friday, whether the Tuesday before is missing or not (in
which case Friday is actually in row 5). To the machine, they are all just row
indices, not days.

Notice how the model suggests a stronger signal around day 7, adjoined by
expectation of signal quite a bit before; the behavior is smeared or muddled.
Absence of even a handful of time points throws off the periodicity of this time
series.

This is a practical problem, as there are bound to be missing time points,
either random or systematic (e.g., transactions on business days only). A path
forward would be to have all dates present and put zeroes where no transactions
occurred (does a package like `tbats` do this?). For some
systematic effects like weekend dormancy, the model should fit coefficients on
those days accordingly; i.e., $\beta_{s(5)}$ and $\beta_{s(6)}$ should be close
to zero. Holidays are another systematic effect we would need an exogenous
binary variable to capture to avoid errors in seasonal effect estimates.
Lastly, randomly occurring times without transactions will rationally be
factored in by $\beta_{s(t)}$ to the extent they occur.

Using `.asfreq` to conform a daily time series of 365 points minus 10 removed at
random to a `DatetimeIndex`-ed dataframe with `freq='D'`, we get one of our 
better fits.

```python
ts = ts.asfreq('d')
ts.fillna(0, inplace=True)
_, s6params, _, stage6 = run_autoreg(ts.y,
                                     lags_=2,
                                     seasonal_=True,
                                     period=7
                                     )
```

{{ s6params }}

The coefficient 2.37 is lower than 2.55 for the fit of the complete dataset,
which suggests that some Fridays may have been zeroed out by the data
preparation. But as long as such a random effect is not too prevalent, the
autoregression should provide reasonable results. Here the incidence of 
missing days is about 3%.

Another method may allow discrete seasonality, rather than the
inclusive/cumulative sort employed by `AutoReg`. The choice involves whether 
to use autoregression plus seasonality versus engineered features in a 
multivariate regression.

For example, no matter how many time points are absent, if we encode Friday
correctly, that feature will light up on this artificial data set, and so is a
more reliable approach than an autoregressive seasonality model. Since
it cannot include custom features, the `AutoReg` class is more suitable for EDA
than deployment as an estimator or forecasting service.

> No matter what method we choose, we need to verify, using synthetic data, that
> the choice and its parameters is congruent with the _structure_ of our sequence
> data.

What if the signal boost occurred _every other_ Friday? s(7,7) = 1.5, while
other seasonal variables are similar to before, although a bit elevated (0.51 to
0.56); what seems to have happened is a halving of the Friday effect. The model
is unaware of what week each transaction occurred.

var spec = {{ stage7 }}

Increasing `period` to 15 leads to worse results, probably due to how larger
interval effects do not consistently align to even or odd Fridays across months
of the year. In sum, if we believe such specific periodic effects are there, or
conversely want to probe for them, we should simply encode it as a binary
variable consistent with how the data are prepared. The coefficients on those
variables from the regression provide the evidence. From there, we can decide
whether to remove extraneous variables, to improve our estimates of the
remaining effects. To probe for surprises in production, we may deploy the more
inclusive, though, problematic (collinearity, etc.) alongside the selected
model.

Binary variables and periodic components in the `AutoReg` model class 
produce spiky or jagged forecasts. The variability common to
real world data--something recurring _around_, rather than precisely _on_, a 
particular day, attenuates the weight (coefficient) of the effect(s). 
However, the attenuation may be unpredictably unstable. In the next post 
in this series, we ask the model to learn how to relax periodic effects.