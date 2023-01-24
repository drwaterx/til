---
title: "Practical_ts02"
date: 2023-01-16T10:17:45-05:00
draft: true
author: "Aaron Slowey"
---

In the [previous post](https://drwaterx.github.io/til/posts/practical_ts01/) 
of this series, binary variables and periodic components in the `AutoReg`
model class produce spiky or jagged-looking forecasts. The variability common to
real world data--a recurring transaction happening _around_, rather than
consistently precisely _on_, a particular day, attenuates the weight (
coefficient) of the effect(s), as it should. However, the attenuation may be
unpredictably unstable, because we're asking the model to size a discrete
effect that may never exist. In this post, I demonstrate how we can 
include in the machine learning task the precision with which periodic 
effects should be specified--a day or two before or after Friday, for 
instance. To do that, we change the discrete features to humps,
or radial basis functions.

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
rbf = pd.DataFrame({'x': numpy.linspace(x1, x2, steps)})

alpha = 1.2  # higher values broaden the hump
m_ = 14
rbf = rbf.assign(y=numpy.exp(-1 * (rbf.x - m_) ** 2 / alpha))
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
def rbf_builder(positions: numpy.ndarray,
                period: int = 7,
                alpha: float = 1.2,
                points_per_interval: int = 1,
                ):
    """For the range of the sequence, produce a radial basis function (RBF) comprising smooth peaks around all multiples within a chosen period.
    
    The multiples are located where the modulo of the position (time) values 
    and the chosen period are zero.  The value of the multiple is equal to 
    the inumpyut position 
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
    for m in numpy.where(numpy.fmod(positions, period) == 0)[0]:
        segment = numpy.exp(-(positions - positions[m]) ** 2 / alpha)
        rbf_segments.append(
            segment)  # ['rbf_' + str(int(m/points_per_interval))] = segment
    return pd.DataFrame({'x': positions,
                         'rbf_' + str(period): numpy.sum(
                             numpy.vstack(rbf_segments), axis=0
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
def rbf_builder(positions: numpy.ndarray,
                period: int = 7,
                alpha: float = 1.2,
                points_per_interval: int = 1,
                ) -> numpy.ndarray:
    """For the range of the sequence, produce a radial basis function (RBF) comprising
    smooth peaks around all existing multiples of a chosen period.
    
    The multiples are located where the modulo of the position (time) values and the 
    chosen period are zero.  The value of the multiple is equal to the 
    inumpyut position 
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

    for m in numpy.where(numpy.fmod(positions, period) == 0)[
        0]:  # zero index pulls array out of tuple
        segment = numpy.exp(-(positions - positions[
            m]) ** 2 / alpha)  # todo: trim sequence to around m
        rbf_segments.append(
            segment)  # ['rbf_' + str(int(m/points_per_interval))] = segment
    return numpy.sum(numpy.vstack(rbf_segments), axis=0)


def rbf_stitcher(seq: pd.DataFrame,
                 rbf: numpy.ndarray,
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
    idx_null_rbf = seq[numpy.isnan(seq[rbf_name])].index
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