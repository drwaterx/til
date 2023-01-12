---
title: "Linear regression of time series"
date: 2023-01-08T15:18:18-05:00
draft: false
author: "Aaron Slowey"
tags: ["modeling", "statsmodels"]
categories: ["technical"]
---

This post is under construction, with missing graphs and unbaked LaTax math.

# Time series regression

The following text covers mostly _in-sample_ deconstruction of temporally
sensitive effects that can be applied to other problems, including forecasting.

We denote a time series context with some additional subscripts:
$$y_t = \alpha + \sum_{i=1}^m \beta_i x_{i,t} + \epsilon_t$$
The properties of variables that constitute $x$ determine what kind of time
series regression we perform.

- An auto-regression model includes up to $p$ lags: $x_{t-p}, \ldots, x_{t-1}$
- A linear trend is included by $x_{1, t} = t$; assuming equally spaced
  observations, $t$ would be `np.linspace(1, T)`
- Day of week is achieved by having one binary variable for all but one day;
  i.e., $x_{i, t}=1$ if the observation occurs on a particular day and zero
  otherwise. For any categorical variable having $k$ unique values, include
  $k-1$ binary variables into the model. So if we leave out Sunday, $x_{Monday,
  t}$ measures the effect of Monday on $y$ _compared to the effect of Sunday_.
- Spike: A dummy variable that is 1 in a specific period, zero before and after
- Step: A dummy variable zero up to a point, 1 from that point on. Related is a
  change in slope.

As with `OLS`, you can apply statsmodels' `.summary()` method to the fitted
model object, as well as `.plot_predict(start=720, end=840)`.

The 'errors' should

- Have mean zero
- Not be auto-, or serially, correlated; Breusch-Godfrey or Lagrange Multiplier
  test, in which a small `p-value` is bad (significant autocorrelation remains).
- Unrelated to the predictors
- Be normally distributed with a constant variance

`.plot_diagnostics()` provides some of these diagnostics.

`statsmodels.tsa` has multiple methods for time series
analysis/regression/forecasting. Each term has a different connotation,
depending on how you deploy them. We do not have to use `.tsa` methods to model
sequence data; multivariate linear regression with univariate lags and time
characteristic variables could achieve roughly the same model. A variety of
considerations may determine the choice of model class, such as simply being
able to report that you used a certain time series analysis package.

In any case, we are tackling the challenge of building a linear model with
familiar performance criteria. The most profound difference is that the
observations are possibly auto-correlated, not I.I.D., but this may 'normalize'
out by simply including lags and time characteristics (seasonal components).

A good starting point is the `AutoReg` class of  `statsmodels.tsa.ar_model`.

```python
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

auto_reg = AutoReg(data,
                   lags=3,  # AR(3)
                   seasonal=True,
                   period=7,
                   )
auto_reg0 = auto_reg.fit()
```

[Worked example](https://www.statsmodels.org/dev/examples/notebooks/generated/autoregressions.html?highlight=ar_select_order)
in the `statsmodels` documentation that fails to clearly show how each component
manifests, and exactly what parameters affect the fit. That ambiguity ends here.

An overarching question is -- to what extent do `statsmodels` implementations
include utilities that recognize and utilize `datetimes`? The work below
suggests not at all, despite some warning or errors when `seasonal=True`,
but `period` is unspecified. To obtain expected behavior from statistical
learning algorithms, it is crucial to know and potentially modify the sequential
structure (spacing) of data, because there do not appear to be intelligent
checks and automated cleaning processes. Other key questions include

- For example, $m=52$, unless `tsa` interprets datetime values intelligently,
  only implies a weekly periodicity if there are $7\cdot52$ rows of data.
- What if we have $T=1.5$ years of data; haven't seen any caution to carefully
  compute the period as $m=p\cdot\frac{52}{q} \cdot T$, where p and q define the
  period of interest
- There are a few days missing?
- What if the data lack Saturdays and Sundays, as with `BOLT` data?
- Does the data need to be processed to ensure there are 7 days per week, 52
  weeks per year, imputing zeroes where needed?

To begin, create artificial data with known patterns. Create one year of daily
timestamps and initialize the observations with random numbers $\in (0, 1)$. For
other possibilities, see also `sklego.datasets.make_simpleseries`.

```python
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

days = np.arange('2022-01-01', '2023-01-01', dtype='datetime64[D]')
print(
    f'There are {len(days)} days in the year 2022 (start {days.min()}; end {days.max()}).')

ts = pd.DataFrame({'t': days,  # np.linspace(1, 100, 100)
                   'y': np.random.random(len(days))}
                  )
```

![[professional/know_how/stat_modeling/attachments/visualization.png]]

Encode a time characteristic, such as day of the week (dow) and boost the signal
on certain days (or weeks, etc.). Here, we spike the signal on Fridays. Then we
play with the `seasonal` and `period` parameters.

```python
ts.loc[:, 'dow'] = ts.t.dt.weekday
ts.y = ts.y + np.where(ts.dow == 4, 2, 0)

auto_reg = AutoReg(ts.y,
                   lags=4,
                   trend='t',
                   seasonal=False,
                   period=7,
                   )
auto_reg0 = auto_reg.fit()
ts.loc[:, 'y_hat'] = auto_reg0.predict()
c2 = altair_ts_line(ts, 't', 'y_hat', 't')
(c0 + c2).add_selection(pan_zoom)
```

With `seasonal=False`, we obtain an upwardly trending oscillation:
![[professional/know_how/stat_modeling/attachments/visualization (1).png]]

Enable `seasonal`, and we get the expected level baseline with weekly peaks:
__Exhibit A__
![[professional/know_how/stat_modeling/attachments/visualization (2).png]]

If we set `period=365`, we get
a `ValueError: The model specification cannot be estimated. The model contains 370 regressors (1 trend, 365 seasonal, 4 lags) but after adjustment for hold_back and creation of the lags, there are only 361 data points available to estimate parameters.`

And if we set to a feasible, but wrong value -- `period=30`  -- we will get a
somewhat better result, than without any periodicity, but clearly missing the
effect:
![[professional/know_how/stat_modeling/attachments/visualization (4).png]]

Interestingly, if we increase `lags=7` to include the weekly effect, we get
almost as good a model as with seasonal terms:

```
auto_reg = AutoReg(ts.y,
                   lags=7,
                   trend='t',
                   seasonal=False,
                   period=7)
```

![[professional/know_how/stat_modeling/attachments/visualization (5).png]]

Any number of lags above seven, and we see no improvement. Although not shown,
the model with `lags=7` and `seasonal=True` with `period=7` looks identical to
the  `seasonal=True`, `period=7` model with `lags=4` above, suggesting that the
$x_{t-p}$ and $s_d$ terms are collinear.

`statsmodels.tsa.ar_model.AutoReg` interprets `period=7` as the _longest step_
from one data point to the next in the dataframe; shorter steps from 1 to $p-1$
are also included. We think that something happens every 7th observation in the
sequence, and whether we like it or not,  `AutoReg` checks whether anything
happens on shorter cadences. As discussed in detail below, whether those _steps_
correspond to a meaningful time interval or period depends on the structure of
the sequence.

Look at the coefficients by applying the `.params` method to the fitted model
object; e.g., with `lags=2`, `seasonal=True`, and `period=7`:
![[Pasted image 20221003171824.png]]

Note that with `period` and `lags` set to an integer, multiple terms are
included up to that value: e.g., $[1, \text{lags}]$. Unlike `period`, you may
provide a list of integers for `lags`, in which case _only_ those lags are
included. As expected, we find `seasonal.6` coefficient to be much higher than
those of narrower periodicities. It's also no accident that seasonal components
0-5's coefficients are _similar_ to each other. We will investigate how this
model responds to data with multiple periodicities.

Continuing on with sanity checks, no matter the length of the sequence, as long
as there are more data points than `lags` (and other parameters), the model will
fit properly. Here, we have data from Jan through March 2022.
![[visualization (6).png]]

`statsmodels.tsa` does not require the time or sequence index to be of
a `datetime` dtype. Replacing datetime values by integers, we obtain the same
result (not shown). But note that `AutoReg` is not being explicitly given a
sequence or time variable; it is implicit in the `pandas.Series` index of `ts.y`
, so the algorithm is unaware of the change in the time column `ts.t` . If we
remove a small number of points at random such that there are gaps in the index,
the model falls apart (not shown). Can we make the seasonal regression algorithm
aware that observations are made on calendar days?

```python
points = 5
idx_mask = np.random.randint(0, len(ts), points)
ts = ts[~ts.index.isin(idx_mask)]
```

Incidentally, avoid `df.sample(frac=0.9)`, as it shuffles the rows.

Setting the index with the datetime-formatted values (maintaining the five
randomly placed gaps) leads
to `ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.`
The predictions from `.predict()` are all null.

What matters is a logical correspondence between `period` and the frequency of
the data as presented by their sequence in the array or DataFrame. Let's
say `period=7`; if the frequency is unspecified, the algorithm just considers if
a data point six steps from the current point tends to be higher, lower, or
about the same as the current point. If the data happen to be observations
recorded every nanosecond with a perturbation every 13 ns, `period=13` should
fit that sequence nicely. This naive behavior is helpful for modeling
observations that occur with a regularity that is meaningful, if not in a
temporal way. For example, every fourth trip to buy groceries, the family goes
to Costco, not Trader Joe's. But it poses a problem for incomplete and irregular
sequences when effects pertain to certain fixed time qualities.

When `.set_index('dt_col')` involves a datetime column, we obtain
a `DateTimeIndex` with `freq=None`, which `statsmodels.tsa.ar_model` is
complaining about. We can specify the frequency

```python
ts = ts.set_index('t').asfreq('d')
```

Having reproduced the result shown in Exhibit A with a complete data set indexed
in this way, we return to the case where points are randomly missing; remove
them _prior to_ setting the datetime index and frequency. Here, we introduce a
potential problem. As an aside, `.asfreq('B')` sets an index to daily business
day. [More](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asfreq.html?highlight=asfreq#pandas.DataFrame.asfreq) `.asfreq`
can be applied to a DataFrame; it will return the DataFrame "reindexed to the
specified frequency."
Meaning the "original data conformed to a new index with the specified
frequency."  In this case, to conform to daily frequency, rows of `nan` are
placed where time points were missing. Before and after applying `.asfreq('d')`:
![[Pasted image 20221004101943.png]]
![[Pasted image 20221004102013.png]]

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
![[Pasted image 20221004104608.png]]

It is unnecessary to include `missing='drop'` in `AutoReg()`, since the data has
no missing values (it is probably good practice to include `missing='raise'` as
a check).  `.fit()` runs and `.predict()` returns values without alterring the
input data structure: it still has a `PeriodIndex`:
![[Pasted image 20221004105317.png]]

Did the model fit well? While the `PeriodIndex` facilitated the fit, it
complicates plotting with Altair. Assuming the index needs to be reset into a
column, a  `PeriodIndex`  resets to a column of `'period[D]' `dtype, and this
leads to a `TypeError`. After fitting the model with `PeriodIndex`-ed data,
reformat the index:

```python
ts.index = ts.index.to_timestamp()
```

You will lose the `freq` and have a `DatetimeIndex` once again. Then reset _
that_ index and plot as usual.
![[professional/know_how/stat_modeling/attachments/visualization (2) 1.png]]
![[Pasted image 20221004121816.png]]
Our model did not capture the Friday signal as before, presumably because it
naively assumed every $7\cdot n^{th}$ point was Friday, ignoring the days
skipped in the index. Note how the parameters' seasonal components are written
as `s(1, 7)`, whereas before it was `seasonal.0`. The connotation is that we're
capturing effects that occur every 1st of 7 days?

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

As for the datetime, period, and all that functionality in pandas...it serves
other uses, such as resampling, aggregations, synthesizing data, and more.

This is a practical problem, as there are bound to be missing time points,
either random or systematic (e.g., transactions on business days only). A path
forward would be to have all dates present and put zeroes where no transactions
occurred (perhaps packages like `tbats` do this under the hood?). For some
systematic effects like weekend dormancy, the model should fit coefficients on
those days accordingly; i.e., $\beta_{s(5)}$ and $\beta_{s(6)}$ should be close
to zero. Holidays are another systematic effect we would need an exogenous
binary variable to capture to avoid errors in the seasonal component estimates.
Lastly, randomly occurring times without transactions will just have to be
factored in by $\beta_{s(t)}$ to the extent they occur, which is rational.

Using `.asfreq` to conform a daily time series of 365 points minus 10 removed at
random to a `DatetimeIndex`-ed dataframe with `freq='D'`, we get a good fit.

```python
ts = ts.asfreq('d')
ts.fillna(0, inplace=True)

auto_reg = AutoReg(ts.y,
                   missing='raise',
                   lags=2,
                   trend='t',
                   seasonal=True,
                   period=7,
                   old_names=False,
                   )
auto_reg0 = auto_reg.fit()
ts.loc[:, 'y_hat'] = auto_reg0.predict()
```

![[Pasted image 20221004163931.png]]

The coefficient 2.37 is lower than 2.55 for the fit of the complete dataset,
which suggests that some Fridays may have been zeroed out by the data
preparation. But as long as such a random effect is not too prevalent, the
autoregression should provide reasonable results. Here the incidence of '
missing' days is about 3%.

Another method may allow discrete seasonality, rather than the
inclusive/cumulative sort employed by `AutoReg`. The choice calls for a design
principle:

- Autoregression plus seasonality
- Hand-derived features
- Both: what are each capturing?

For example, no matter how many time points are absent, if we encode Friday
correctly, that feature will light up on this artificial data set, and so is a
more reliable approach than an autoregressive seasonality model (alone). Since
it cannot include custom features, the `AutoReg` class is more suitable for EDA
than deployment as an estimator or forecasting service.

> No matter what method we choose, we need to verify, using synthetic data, that
> the choice and its parameters is congruent with the _structure_ of our sequence
> data.

What if the signal boost occurred _every other_ Friday? s(7,7) = 1.5, while
other seasonal variables are similar to before, although a bit elevated (0.51 to
0.56); what seems to have happened is a halving of the Friday effect. The model
is unaware of what week each transaction occurred.
![[visualization (8).png]]

Increasing `period` to 15 leads to worse results, probably due to how larger
interval effects do not consistently align to even or odd Fridays across months
of the year. In sum, if we believe such specific periodic effects are there, or
conversely want to probe for them, we should simply encode it as a binary
variable consistent with how the data are prepared. The coefficients on those
variables from the regression provide the evidence. From there, we can decide
whether to remove extraneous variables, to improve our estimates of the
remaining effects. To probe for suprises in production, we may deploy the more
inclusive, though, problematic (collinearity, etc.) alongside the 'selected'
model.

Up until this point, binary variables and periodic components in the `AutoReg`
model class produce spiky or jagged looking forecasts. The variability common to
real world data--a recurring transaction happening _around_, rather than
consistently precisely _on_, a particular day, attenuates the weight (
coefficient) of the effect(s), as it should. However, the attenuation may be
unpredictably unstable, because we're asking the model to size a discrete effect
that may never exist. Instead, we can relax the effect to have some
flexibility--a day or two before or after Friday, for instance. We just need to
change the Dirac deltas to humps, or radial basis functions.

In a 2018 Pydata London [talk](https://www.youtube.com/watch?v=68ABAU_V8qI),
Vincent Warmerdam spoke about linear models, including using RBFs. See also his
talk [How to Constrain Artificial Stupidity](https://www.youtube.com/watch?v=Z8MEFI7ZJlA)
. He has co-led initiatives including `evol` and `scikit-lego`.

Most web searches go to RBF kernels for SVMs or RBF neural networks, neither of
which are tactically relevant here.  `SciPy` has an RBF fitting routine; closer
perhaps, but still not in-line with our objective to produce artificial peaks as
regression predictors; that is, to _generate_ artificial peaks and let the
regression weigh its importance and tune their widths, _given the presence_ of
other behaviorally (temporally) specific effects on the RHS. The math that
fulfills this purpose is $$\phi(x_i) = \text{exp}
\Big[-\frac{1}{2\alpha} (x-m_i)^2 \Big]~ \forall ~\text{week, month, or} \dots$$
where $$
m_i=\begin{cases}
1, & \text{if}\ x ~ \text{mod}~ im_1=0 \\
0, & \text{otherwise}
\end{cases}
$$
and $i$ ranges from 1 to however many instances of period $m_1$ exist in the
data; for instance, if $m_1=7$, and we have 56 days of data,
$i\in[1, 2, 3, 4,5 ,6, 7, 8]$ and $m \in [7, 14, 21, 28, 35, 42, 49, 56]$.
Recall the $\text{mod}$, or modulo, is the _remainder_ that, alongside the _
quotient_, is an answer to a division of a _dividend_ by a _divisor_. For
example, `14 mod 3 = 2`; `9 mod 3` and `121 mod 11` are zero.

Note $m$ is the _period_ as defined in relation to the sequence structure;
e.g., `7` implies weekly periodicity if data are daily; `6` would imply a
semiannual periodicity of monthly data and annual periodicity for bimonthly
data.

The RBF should

- Peak at `1` at the characteristic points in the sequence and
- Smoothly decay to zero on either side.

The exponential will only evaluate to `1` when the quantity in the brackets is
zero; i.e., when $x = m_i$; $i$ denotes the $i^{\text{th}}$ week, month, etc.
within the data's boundaries. For weekly periodicity, the RBF evaluates to 1
when $x \in [7, 14, 21, ...]$. While it is easy to perform this calculation for
one value of $m$ (see below), to get the other humps we need, we need to also
subtract `14` and so on.

```python
x1, x2 = 1, 56
points_per_interval = 10
steps = (x2 - x1) * points_per_interval + 1
rbf = pd.DataFrame({'x': np.linspace(x1, x2, steps)})

alpha = 1.2  # higher values broaden the hump
m_ = 14
rbf = rbf.assign(y=np.exp(-1 * (rbf.x - m_) ** 2 / alpha))
chrt = altair_ts_scatter(rbf, 'x', 'y', 'x')
```

![[single_hump 1.png]]

Proposed procedure:

1. Determine the highest multiple of $m$ from the range of the data; e.g., if we
   have two years of data and we want a monthly RBF, the multiple is 24.
2. Create an array for each multiple of $m$, starting one time step after the
   end of the preceding array
3. Concatentate the arrays into one RBF.

```python
def rbf_builder(positions: np.ndarray,
                period: int = 7,
                alpha: float = 1.2,
                points_per_interval: int = 1,
                ):
    """For the range of the sequence, produce a radial basis function (RBF) comprising smooth peaks around all multiples within a chosen period.
    
    The multiples are located where the modulo of the position (time) values and the chosen period are zero.  The value of the multiple is equal to the input position 
    array at those locations.  For each of those locations and associated values,
    an RBF is formed by subtracting the value from the position array, as the 
    exponent will evaluate to `1` when its argument is zero.
    
    Each RBF (one per period multiple present in the data) is stored in a separate
    array, which are collected in a list and stacked into a 2D array. The final step, summing vertically over the 2D aray, provides a single RBF with multiple humps--the form we need to include in a linear regression.
    
    Parameters
    ----------
    positions
        Position (time) values associated with a sequence of observations.
    period
        Number of sequence or time points (or rows) in the series that define a putative characteristic
        of the sequence. `7` implies weekly periodicity if data were recorded daily. The period by itself 
        has no inherent meaning; it is always in relation to the data structure (spacing of observations).
    alpha
        Width of the humps; higher is wider.
    points_per_interval
        Always one, except when unit testing with synthetic data using `numpy.linspace`.
    """
    rbf_segments = []

    # 0 index pulls array out of tuple
    for m in np.where(np.fmod(positions, period) == 0)[0]:
        segment = np.exp(-(positions - positions[m]) ** 2 / alpha)
        rbf_segments.append(
            segment)  # ['rbf_' + str(int(m/points_per_interval))] = segment
    return pd.DataFrame({'x': positions,
                         'rbf_' + str(period): np.sum(
                             np.vstack(rbf_segments), axis=0
                         )})
```

![[multi_humps.png]]

Note the last hump is half cutoff, which is probably appropriate to avoid
extending beyond the scope of the data. It's conceivable we might want RBF with
half humps, if something happens mostly on Fridays, sometimes on Thursday,
rarely on Wednesdays, but never on Saturdays.

The width of the humps, controlled by hyperparameter $\alpha$, should be tuned
or at least dialed into a sensible expectation. In the example above,
$\alpha=1.2$ allows for about $\pm 2$ days. Warmerdam
mentioned [Evol](https://evol.readthedocs.io/en/latest/), a library to define a
complex algorithm in a composable way (sounds like a kindred spirit to `patsy`).
Compare the above function to `sklego.preprocessing.RepeatingBasisFunction`.

We have a function that creates an RBF that peaks every $m^\text{th}$ instance
of a sequence, sometimes including the zeroth position. To become part of the
design matrix $X$, the RBF needs to align congruently to the desired DoW, WoM,
etc. that $y$ and other RHS predictors align to.

There are probably mutliple ways to do this. Let's say we have arrays $t$, $y$,
and some $X$, including the ordinal encoding of $t$ (`.dt.weekday`) and the
binary variables `dow_0`, ... `dow_6`. We could

1. Tell NumPy or pandas to concatenate the RBF starting where the ordinal
   variable matches the day we want.
2. Or align where RBF=1 to where the right binary variable is 1, which requires
   an extra step: the binary encoding of the ordinal. We'll take the first
   approach, though the second seems to facilitate error checking, since we
   could test that _all_ rows where 1 should be 1 are the same position in the
   sequence.

Ensure complete coverage of the sequence by the RBF. If the RBF was originally
tailored to the length of the sequence, there will be empty leading cells to
backfill after merging the RBF to the sequence, unless the DoW happened to be in
position `0`. This dependency alone motivates the use of `pandas.merge` in
concert with indexed collections (ndarray-> series and DataFrame). When
using `pandas.merge(how='left')`, where the left df is the sequence, there will
be no trailing RBF values.

For the backfill, count leading nulls, compute RBF at $x=t-i$ for $i=1$ to $n_
{nulls}$, or if the arrays are in a DataFrame and pandas has nulls in the
leading rows of the RBF, use the (integer) index for $x$.

```python
def rbf_builder(positions: np.ndarray,
                period: int = 7,
                alpha: float = 1.2,
                points_per_interval: int = 1,
                ) -> np.ndarray:
    """For the range of the sequence, produce a radial basis function (RBF) comprising
    smooth peaks around all existing multiples of a chosen period.
    
    The multiples are located where the modulo of the position (time) values and the 
    chosen period are zero.  The value of the multiple is equal to the input position 
    array at those locations.  For each of those locations and associated values,
    an RBF is formed by subtracting the value from the position array, as the 
    exponent will evaluate to `1` when its argument is zero.
    
    Each RBF (one per period multiple present in the data) is stored in a separate
    array, which are collected in a list and stacked into a 2D array.
    
    The final step, summing vertically over the 2D aray, provides a single RBF with 
    multiple humps--the form we need to include in a linear regression.  We do not need to
    return the results as a dataframe; a NumPy array may be preferable.
    
    Parameters
    ----------
    positions
        Position (time) values associated with a sequence of observations.
    period
        Number of sequence or time points (or rows) in the series that define a putative characteristic
        of the sequence. `7` implies weekly periodicity if data were recorded daily. The period by itself 
        has no inherent meaning; it is always in relation to the data structure (spacing of observations).
    alpha
        Width of the humps; higher is wider.
    points_per_interval
        Always one, except when unit testing with synthetic data using `numpy.linspace`.
    """
    rbf_segments = []
    # rbf_segments = {'x': sequence}

    for m in np.where(np.fmod(positions, period) == 0)[
        0]:  # zero index pulls array out of tuple
        segment = np.exp(-(positions - positions[
            m]) ** 2 / alpha)  # todo: trim sequence to around m
        rbf_segments.append(
            segment)  # ['rbf_' + str(int(m/points_per_interval))] = segment
    return np.sum(np.vstack(rbf_segments), axis=0)


def rbf_stitcher(seq: pd.DataFrame,
                 rbf: np.ndarray,
                 m_: int = 7,
                 characteristic: str = 'dow',
                 characteristic_value: int = 4,
                 ) -> pd.DataFrame:
    """Extends regression design matrix X with a radial basis function (RBF),
    ensuring the RBF peak aligns to the desired positions within the sequence.
    
    Parameters
    ----------
    seq
        Sequence of observations.
    rbf
        Radial basis, a function of recurring humps.
    m_
        Period of the RBF.
    characteristic
        Name of the sequence or time characteristic.
    characteristic_value
        Ordinal value of the characteristic.  For example, if the characteristic is
        the day of the week (DoW) and we want the RBF to peak on Fridays,
        we have prepared a sequence that includes an ordinal encoding via `pandas.dt.weekday`
        where DoW = 4 corresponds to Fridays.
    """
    # Convert RBF into a named Series
    rbf_name = 'rbf_' + str(m_) + '_' + str(characteristic_value)
    rbf = pd.Series(rbf, name=rbf_name)

    # locate first instance where specified time characteristic appears in the sequence
    delay = seq[seq[characteristic] == characteristic_value].index[0]

    # Adjust index of RBF
    rbf.index = rbf.index + delay

    # Merge the RBF to the sequence
    seq = seq.merge(rbf, left_index=True, right_index=True, how='left')

    # Verify alignment -- possible?

    # Backfill any nulls in the RBF resulting from the index alignment
    idx_null_rbf = seq[np.isnan(seq[rbf_name])].index
    seq.loc[idx_null_rbf, rbf_name] = seq.loc[
                                          idx_null_rbf + max(
                                              idx_null_rbf) + 2, rbf_name][
                                      ::-1].to_numpy()
    # verify that the +2 scalar is universal and not dependent on the period m_

    return seq
```

Along with `alpha`, does sequence spacing matter? No. We find that, with
$\alpha=1.2$, $\pm 1$ day from the characteristic date, the RBF is sizable at
just over 0.4, dropping precipitously $\pm 2$ days to about 0.04. Keeping the
spacing fixed, changes to alpha adjust those proportions:

| $\alpha$ | $\pm 1$ day | $\pm 2$ days |
| :--- | ---: | ---: |
| 1.5 | 0.51 | 0.069 |
| 1.2 | 0.43 | 0.036 |
| 0.8 | 0.29 | 0.007 |
| 0.4 | 0.082 | $4.5 \cdot 10^{-5}$ |

Choosing alpha is a two step process. First, minimize forecast error. Second,
look at the value of the tuned RBF around peak time points. Consider the unit
test case where the signal is elevated on or around Fridays. If alpha of 1.2 is,
to a first cut, optimal, it implies that for every $\beta$ dollars repeatedly
paid on Friday, there tends to be $0.40 $\cdot \beta$ paid on either Thursday or
Saturday, as the coefficient for the RBF will weigh its values. Unless there is
a clear reason to reject those proportions, driven by alpha and the minimization
of the chosen error metric, keep alpha as is.

When the RBF's shape (alpha) is chosen this way, we apply an objective basis to
attenuate the main effect (payments recurring on Fridays). We can assume that
coefficient for `dow_4` would have otherwise been underestimated, while that
of `dow_3` and/or `dow_5` (if present) would be overestimated. To verify this
intuition, observe beta as alpha shrinks; it should converge to the beta for the
corresponding binary variable.

Given Friday is the last business day of the week, _that_ RBF maybe should be
asymmetrical (to the left), while that for Monday should be asymmetrical (on the
right). We are simply trying to obtain both a good fit (low error) and insight
through meaningful betas.

It is not necessary for the RBF to be super smooth over a fine grid, because the
values at the key time points will be the same. Substeps just enhance how the
RBF looks by itself, in method documentation (it is unlikely to ever be exposed
to the client).

### Deploy the RBF to the artificial time series with signal boosted every Friday

Using the `statsmodels.tsa.ar_model.AutoReg`, disabling `seasonal`, and adding
the RBF as an `exog` variable:

```python
auto_reg = AutoReg(tsrbf.y,
                   missing='raise',
                   lags=2,
                   trend='t',
                   seasonal=False,
                   # period=7,
                   exog=tsrbf.rbf_7_4,
                   old_names=False,
                   )
auto_reg1 = auto_reg.fit()
tsrbf.loc[:, 'y_hat'] = auto_reg1.predict()
pprint(auto_reg1.params)

trend
0.001264
y.L1 - 0.119650
y.L2
0.141311
rbf_7_4
2.158667
```

![RBF test](/rbf_test1.png)

The periodic effect is clearly captured, with less amplitude compared to the AR
seasonal components due to attenuation. Expect higher error as a cost of
accommodating variance in the periodicity. We might prove these phenomena by
computing the MAPE of each fit, with and without such variance.