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

var spec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
  "config": {
    "view": {
      "continuousHeight": 300,
      "continuousWidth": 400
    }
  },
  "data": {
    "name": "data-460851078ef436e285d50a6cff3c9a96"
  },
  "datasets": {
    "data-460851078ef436e285d50a6cff3c9a96": [
      {
        "dow": 5,
        "t": "2022-01-01T00:00:00",
        "wom": 1,
        "y": 0.9955981828907805,
        "y_hat": null
      },
      {
        "dow": 6,
        "t": "2022-01-02T00:00:00",
        "wom": 1,
        "y": 0.6234109104925726,
        "y_hat": null
      },
      {
        "dow": 0,
        "t": "2022-01-03T00:00:00",
        "wom": 2,
        "y": 0.8760936874425284,
        "y_hat": 0.4967907586604228
      },
      {
        "dow": 1,
        "t": "2022-01-04T00:00:00",
        "wom": 2,
        "y": 0.8893778611747504,
        "y_hat": 0.5184239775556261
      },
      {
        "dow": 2,
        "t": "2022-01-05T00:00:00",
        "wom": 2,
        "y": 0.3867624782099456,
        "y_hat": 0.4972472553191131
      },
      {
        "dow": 3,
        "t": "2022-01-06T00:00:00",
        "wom": 2,
        "y": 0.6899198960223786,
        "y_hat": 0.5262597140421478
      },
      {
        "dow": 4,
        "t": "2022-01-07T00:00:00",
        "wom": 2,
        "y": 2.992852550566659,
        "y_hat": 2.4299590582515473
      },
      {
        "dow": 5,
        "t": "2022-01-08T00:00:00",
        "wom": 2,
        "y": 0.46804208563249405,
        "y_hat": 0.5525014522166138
      },
      {
        "dow": 6,
        "t": "2022-01-09T00:00:00",
        "wom": 2,
        "y": 0.9455841386245034,
        "y_hat": 0.5668882974435845
      },
      {
        "dow": 0,
        "t": "2022-01-10T00:00:00",
        "wom": 3,
        "y": 0.6381233538848571,
        "y_hat": 0.4776279996404177
      },
      {
        "dow": 1,
        "t": "2022-01-11T00:00:00",
        "wom": 3,
        "y": 0.7595024746539746,
        "y_hat": 0.5291292866690476
      },
      {
        "dow": 2,
        "t": "2022-01-12T00:00:00",
        "wom": 3,
        "y": 0.23738191602671632,
        "y_hat": 0.4808008912256273
      },
      {
        "dow": 3,
        "t": "2022-01-13T00:00:00",
        "wom": 3,
        "y": 0.09680924548766545,
        "y_hat": 0.515091715171366
      },
      {
        "dow": 4,
        "t": "2022-01-14T00:00:00",
        "wom": 3,
        "y": 2.588571863863769,
        "y_hat": 2.405058230192902
      },
      {
        "dow": 5,
        "t": "2022-01-15T00:00:00",
        "wom": 3,
        "y": 0.16573502501513393,
        "y_hat": 0.5090408212208333
      },
      {
        "dow": 6,
        "t": "2022-01-16T00:00:00",
        "wom": 3,
        "y": 0.38942124330865624,
        "y_hat": 0.5365359980531161
      },
      {
        "dow": 0,
        "t": "2022-01-17T00:00:00",
        "wom": 4,
        "y": 0.688787521402102,
        "y_hat": 0.44552691006817247
      },
      {
        "dow": 1,
        "t": "2022-01-18T00:00:00",
        "wom": 4,
        "y": 0.6697600544042848,
        "y_hat": 0.5006636268518205
      },
      {
        "dow": 2,
        "t": "2022-01-19T00:00:00",
        "wom": 4,
        "y": 0.36885391553882474,
        "y_hat": 0.4810838917599954
      },
      {
        "dow": 3,
        "t": "2022-01-20T00:00:00",
        "wom": 4,
        "y": 0.9325008856726638,
        "y_hat": 0.5141158767799752
      },
      {
        "dow": 4,
        "t": "2022-01-21T00:00:00",
        "wom": 4,
        "y": 2.141135576285614,
        "y_hat": 2.436148536768076
      },
      {
        "dow": 5,
        "t": "2022-01-22T00:00:00",
        "wom": 4,
        "y": 0.9772509899400809,
        "y_hat": 0.5414839206074327
      },
      {
        "dow": 6,
        "t": "2022-01-23T00:00:00",
        "wom": 4,
        "y": 0.5683493004574759,
        "y_hat": 0.5356818812038745
      },
      {
        "dow": 0,
        "t": "2022-01-24T00:00:00",
        "wom": 5,
        "y": 0.9649672826390342,
        "y_hat": 0.4945634970531389
      },
      {
        "dow": 1,
        "t": "2022-01-25T00:00:00",
        "wom": 5,
        "y": 0.22830327875114365,
        "y_hat": 0.5183276280490623
      },
      {
        "dow": 2,
        "t": "2022-01-26T00:00:00",
        "wom": 5,
        "y": 0.5236014315750456,
        "y_hat": 0.4834914229513924
      },
      {
        "dow": 3,
        "t": "2022-01-27T00:00:00",
        "wom": 5,
        "y": 0.5771602890446812,
        "y_hat": 0.49481710194259054
      },
      {
        "dow": 4,
        "t": "2022-01-28T00:00:00",
        "wom": 5,
        "y": 2.3198686688954586,
        "y_hat": 2.4344611018946396
      },
      {
        "dow": 5,
        "t": "2022-01-29T00:00:00",
        "wom": 5,
        "y": 0.4375103099359736,
        "y_hat": 0.5275197097634949
      },
      {
        "dow": 6,
        "t": "2022-01-30T00:00:00",
        "wom": 5,
        "y": 0.7071449184091295,
        "y_hat": 0.5300200373997083
      },
      {
        "dow": 0,
        "t": "2022-01-31T00:00:00",
        "wom": 6,
        "y": 0.1524637438937454,
        "y_hat": 0.46950282399087084
      },
      {
        "dow": 1,
        "t": "2022-02-01T00:00:00",
        "wom": 1,
        "y": 0.011748418393797033,
        "y_hat": 0.5027153449515473
      },
      {
        "dow": 2,
        "t": "2022-02-02T00:00:00",
        "wom": 1,
        "y": 0.25517769811183577,
        "y_hat": 0.4335507573741657
      },
      {
        "dow": 3,
        "t": "2022-02-03T00:00:00",
        "wom": 1,
        "y": 0.5156949359371945,
        "y_hat": 0.47556781187960434
      },
      {
        "dow": 4,
        "t": "2022-02-04T00:00:00",
        "wom": 1,
        "y": 2.6710173501944316,
        "y_hat": 2.418325496049474
      },
      {
        "dow": 5,
        "t": "2022-02-05T00:00:00",
        "wom": 1,
        "y": 0.8240477378755111,
        "y_hat": 0.5343478357671186
      },
      {
        "dow": 6,
        "t": "2022-02-06T00:00:00",
        "wom": 1,
        "y": 0.9508838016129778,
        "y_hat": 0.5601352577647144
      },
      {
        "dow": 0,
        "t": "2022-02-07T00:00:00",
        "wom": 2,
        "y": 0.05520043850117129,
        "y_hat": 0.4974480437451285
      },
      {
        "dow": 1,
        "t": "2022-02-08T00:00:00",
        "wom": 2,
        "y": 0.36173925787444183,
        "y_hat": 0.513206980650459
      },
      {
        "dow": 2,
        "t": "2022-02-09T00:00:00",
        "wom": 2,
        "y": 0.3287722160717247,
        "y_hat": 0.4384131691013086
      },
      {
        "dow": 3,
        "t": "2022-02-10T00:00:00",
        "wom": 2,
        "y": 0.26344459815995447,
        "y_hat": 0.49667802938134803
      },
      {
        "dow": 4,
        "t": "2022-02-11T00:00:00",
        "wom": 2,
        "y": 2.105522116894078,
        "y_hat": 2.415202698211017
      },
      {
        "dow": 5,
        "t": "2022-02-12T00:00:00",
        "wom": 2,
        "y": 0.9004373252539722,
        "y_hat": 0.5046824885073314
      },
      {
        "dow": 6,
        "t": "2022-02-13T00:00:00",
        "wom": 2,
        "y": 0.3083956070489431,
        "y_hat": 0.5319008881433882
      },
      {
        "dow": 0,
        "t": "2022-02-14T00:00:00",
        "wom": 3,
        "y": 0.8443045110438598,
        "y_hat": 0.483324909023404
      },
      {
        "dow": 1,
        "t": "2022-02-15T00:00:00",
        "wom": 3,
        "y": 0.9448650092640142,
        "y_hat": 0.5011821422416533
      },
      {
        "dow": 2,
        "t": "2022-02-16T00:00:00",
        "wom": 3,
        "y": 0.5570587072891441,
        "y_hat": 0.4977899462465503
      },
      {
        "dow": 3,
        "t": "2022-02-17T00:00:00",
        "wom": 3,
        "y": 0.5374558873035822,
        "y_hat": 0.5347949272417398
      },
      {
        "dow": 4,
        "t": "2022-02-18T00:00:00",
        "wom": 3,
        "y": 2.8223117437123824,
        "y_hat": 2.4354694540369994
      },
      {
        "dow": 5,
        "t": "2022-02-19T00:00:00",
        "wom": 3,
        "y": 0.02226637799800346,
        "y_hat": 0.5400703987126106
      },
      {
        "dow": 6,
        "t": "2022-02-20T00:00:00",
        "wom": 3,
        "y": 0.15725472652557881,
        "y_hat": 0.5456163598479873
      },
      {
        "dow": 0,
        "t": "2022-02-21T00:00:00",
        "wom": 4,
        "y": 0.9269915515225546,
        "y_hat": 0.43170828507018294
      },
      {
        "dow": 1,
        "t": "2022-02-22T00:00:00",
        "wom": 4,
        "y": 0.4501913471576654,
        "y_hat": 0.4954975240418244
      },
      {
        "dow": 2,
        "t": "2022-02-23T00:00:00",
        "wom": 4,
        "y": 0.9531441262485346,
        "y_hat": 0.48823066431783424
      },
      {
        "dow": 3,
        "t": "2022-02-24T00:00:00",
        "wom": 4,
        "y": 0.5844501255113494,
        "y_hat": 0.519519472671327
      },
      {
        "dow": 4,
        "t": "2022-02-25T00:00:00",
        "wom": 4,
        "y": 2.80397776091471,
        "y_hat": 2.458308069387769
      },
      {
        "dow": 5,
        "t": "2022-02-26T00:00:00",
        "wom": 4,
        "y": 0.26383392179603904,
        "y_hat": 0.542195800807761
      },
      {
        "dow": 6,
        "t": "2022-02-27T00:00:00",
        "wom": 4,
        "y": 0.7531464817944895,
        "y_hat": 0.5516416906073818
      },
      {
        "dow": 0,
        "t": "2022-02-28T00:00:00",
        "wom": 5,
        "y": 0.15885513834885823,
        "y_hat": 0.4618899359780873
      },
      {
        "dow": 1,
        "t": "2022-03-01T00:00:00",
        "wom": 1,
        "y": 0.14535499394438622,
        "y_hat": 0.5058303574877435
      },
      {
        "dow": 2,
        "t": "2022-03-02T00:00:00",
        "wom": 1,
        "y": 0.4410675050392091,
        "y_hat": 0.43816256596576625
      },
      {
        "dow": 3,
        "t": "2022-03-03T00:00:00",
        "wom": 1,
        "y": 0.7517726853328405,
        "y_hat": 0.48854162038072846
      },
      {
        "dow": 4,
        "t": "2022-03-04T00:00:00",
        "wom": 1,
        "y": 2.7945001756665624,
        "y_hat": 2.435556070318829
      },
      {
        "dow": 5,
        "t": "2022-03-05T00:00:00",
        "wom": 1,
        "y": 0.04870630413527299,
        "y_hat": 0.5510704584107008
      },
      {
        "dow": 6,
        "t": "2022-03-06T00:00:00",
        "wom": 1,
        "y": 0.2161041338955525,
        "y_hat": 0.5450948686636378
      },
      {
        "dow": 0,
        "t": "2022-03-07T00:00:00",
        "wom": 2,
        "y": 0.37010898460177155,
        "y_hat": 0.435041792407649
      },
      {
        "dow": 1,
        "t": "2022-03-08T00:00:00",
        "wom": 2,
        "y": 0.09520299889511219,
        "y_hat": 0.48298588836607886
      },
      {
        "dow": 2,
        "t": "2022-03-09T00:00:00",
        "wom": 2,
        "y": 0.11977796440076782,
        "y_hat": 0.44824665641173894
      },
      {
        "dow": 3,
        "t": "2022-03-10T00:00:00",
        "wom": 2,
        "y": 0.5000495405840498,
        "y_hat": 0.47676527608495756
      },
      {
        "dow": 4,
        "t": "2022-03-11T00:00:00",
        "wom": 2,
        "y": 2.453037332781335,
        "y_hat": 2.411129683739495
      },
      {
        "dow": 5,
        "t": "2022-03-12T00:00:00",
        "wom": 2,
        "y": 0.1454828110192531,
        "y_hat": 0.5278354061996601
      },
      {
        "dow": 6,
        "t": "2022-03-13T00:00:00",
        "wom": 2,
        "y": 0.37079700609072885,
        "y_hat": 0.5295379288986366
      },
      {
        "dow": 0,
        "t": "2022-03-14T00:00:00",
        "wom": 3,
        "y": 0.6368565984388843,
        "y_hat": 0.4447991215598311
      },
      {
        "dow": 1,
        "t": "2022-03-15T00:00:00",
        "wom": 3,
        "y": 0.5235082648923333,
        "y_hat": 0.4990719735426933
      },
      {
        "dow": 2,
        "t": "2022-03-16T00:00:00",
        "wom": 3,
        "y": 0.7192823846677754,
        "y_hat": 0.4749988421987305
      },
      {
        "dow": 3,
        "t": "2022-03-17T00:00:00",
        "wom": 3,
        "y": 0.6045077428498761,
        "y_hat": 0.5171315926773352
      },
      {
        "dow": 4,
        "t": "2022-03-18T00:00:00",
        "wom": 3,
        "y": 2.5404792929526683,
        "y_hat": 2.446592365808845
      },
      {
        "dow": 5,
        "t": "2022-03-19T00:00:00",
        "wom": 3,
        "y": 0.4830778428824394,
        "y_hat": 0.5360857207829948
      },
      {
        "dow": 6,
        "t": "2022-03-20T00:00:00",
        "wom": 3,
        "y": 0.9022692634866183,
        "y_hat": 0.5440178300153817
      },
      {
        "dow": 0,
        "t": "2022-03-21T00:00:00",
        "wom": 4,
        "y": 0.06386988650342573,
        "y_hat": 0.4783242019394053
      },
      {
        "dow": 1,
        "t": "2022-03-22T00:00:00",
        "wom": 4,
        "y": 0.2631672977407843,
        "y_hat": 0.5115034804148847
      },
      {
        "dow": 2,
        "t": "2022-03-23T00:00:00",
        "wom": 4,
        "y": 0.6967761906616411,
        "y_hat": 0.43673779150020503
      },
      {
        "dow": 3,
        "t": "2022-03-24T00:00:00",
        "wom": 4,
        "y": 0.6938964314225743,
        "y_hat": 0.5025456288446649
      },
      {
        "dow": 4,
        "t": "2022-03-25T00:00:00",
        "wom": 4,
        "y": 2.3402313329841156,
        "y_hat": 2.448043866842587
      },
      {
        "dow": 5,
        "t": "2022-03-26T00:00:00",
        "wom": 4,
        "y": 0.060442940833932424,
        "y_hat": 0.5353015977298601
      },
      {
        "dow": 6,
        "t": "2022-03-27T00:00:00",
        "wom": 4,
        "y": 0.8673761163808922,
        "y_hat": 0.5212422441741162
      },
      {
        "dow": 0,
        "t": "2022-03-28T00:00:00",
        "wom": 5,
        "y": 0.5544851701436769,
        "y_hat": 0.4546224926233655
      },
      {
        "dow": 1,
        "t": "2022-03-29T00:00:00",
        "wom": 5,
        "y": 0.8547133638306371,
        "y_hat": 0.5237514886317916
      },
      {
        "dow": 2,
        "t": "2022-03-30T00:00:00",
        "wom": 5,
        "y": 0.36375693048623126,
        "y_hat": 0.4802406327655775
      },
      {
        "dow": 3,
        "t": "2022-03-31T00:00:00",
        "wom": 5,
        "y": 0.7322858415787377,
        "y_hat": 0.5250775331729335
      },
      {
        "dow": 4,
        "t": "2022-04-01T00:00:00",
        "wom": 1,
        "y": 2.1005642322427573,
        "y_hat": 2.4312743324017627
      },
      {
        "dow": 5,
        "t": "2022-04-02T00:00:00",
        "wom": 1,
        "y": 0.6446246806801427,
        "y_hat": 0.530637749525594
      },
      {
        "dow": 6,
        "t": "2022-04-03T00:00:00",
        "wom": 1,
        "y": 0.523704664544059,
        "y_hat": 0.5251088026683788
      },
      {
        "dow": 0,
        "t": "2022-04-04T00:00:00",
        "wom": 2,
        "y": 0.6478284421164979,
        "y_hat": 0.47645242667294535
      },
      {
        "dow": 1,
        "t": "2022-04-05T00:00:00",
        "wom": 2,
        "y": 0.8035144383488594,
        "y_hat": 0.5079772060871229
      },
      {
        "dow": 2,
        "t": "2022-04-06T00:00:00",
        "wom": 2,
        "y": 0.06175772835110849,
        "y_hat": 0.4839291542047948
      },
      {
        "dow": 3,
        "t": "2022-04-07T00:00:00",
        "wom": 2,
        "y": 0.10101798118758876,
        "y_hat": 0.5137958988269249
      },
      {
        "dow": 4,
        "t": "2022-04-08T00:00:00",
        "wom": 2,
        "y": 2.328695425168449,
        "y_hat": 2.3970437026869673
      },
      {
        "dow": 5,
        "t": "2022-04-09T00:00:00",
        "wom": 2,
        "y": 0.7405205134959488,
        "y_hat": 0.503188594546576
      },
      {
        "dow": 6,
        "t": "2022-04-10T00:00:00",
        "wom": 2,
        "y": 0.08236194536441421,
        "y_hat": 0.5402774394697502
      },
      {
        "dow": 0,
        "t": "2022-04-11T00:00:00",
        "wom": 3,
        "y": 0.48567949453078907,
        "y_hat": 0.4691302067870829
      },
      {
        "dow": 1,
        "t": "2022-04-12T00:00:00",
        "wom": 3,
        "y": 0.5552603671504405,
        "y_hat": 0.4796291198149867
      },
      {
        "dow": 2,
        "t": "2022-04-13T00:00:00",
        "wom": 3,
        "y": 0.2511650942997945,
        "y_hat": 0.4681934405589045
      },
      {
        "dow": 3,
        "t": "2022-04-14T00:00:00",
        "wom": 3,
        "y": 0.4804870698202284,
        "y_hat": 0.5059180037110255
      },
      {
        "dow": 4,
        "t": "2022-04-15T00:00:00",
        "wom": 3,
        "y": 2.0257783736909825,
        "y_hat": 2.4182249938567857
      },
      {
        "dow": 5,
        "t": "2022-04-16T00:00:00",
        "wom": 3,
        "y": 0.9687966319838246,
        "y_hat": 0.5151312584930192
      },
      {
        "dow": 6,
        "t": "2022-04-17T00:00:00",
        "wom": 3,
        "y": 0.7151278400228451,
        "y_hat": 0.5305591407286968
      },
      {
        "dow": 0,
        "t": "2022-04-18T00:00:00",
        "wom": 4,
        "y": 0.4718442940718869,
        "y_hat": 0.49964798819154504
      },
      {
        "dow": 1,
        "t": "2022-04-19T00:00:00",
        "wom": 4,
        "y": 0.9026979968878186,
        "y_hat": 0.5135072021996053
      },
      {
        "dow": 2,
        "t": "2022-04-20T00:00:00",
        "wom": 4,
        "y": 0.39245084658370566,
        "y_hat": 0.47748693630179173
      },
      {
        "dow": 3,
        "t": "2022-04-21T00:00:00",
        "wom": 4,
        "y": 0.4827795920741036,
        "y_hat": 0.5288246903791284
      },
      {
        "dow": 4,
        "t": "2022-04-22T00:00:00",
        "wom": 4,
        "y": 2.551414450198103,
        "y_hat": 2.4260303356637705
      },
      {
        "dow": 5,
        "t": "2022-04-23T00:00:00",
        "wom": 4,
        "y": 0.650135286544272,
        "y_hat": 0.5303875499655886
      },
      {
        "dow": 6,
        "t": "2022-04-24T00:00:00",
        "wom": 4,
        "y": 0.467587186604593,
        "y_hat": 0.549943038786494
      },
      {
        "dow": 0,
        "t": "2022-04-25T00:00:00",
        "wom": 5,
        "y": 0.8420434940525285,
        "y_hat": 0.4754830070083764
      },
      {
        "dow": 1,
        "t": "2022-04-26T00:00:00",
        "wom": 5,
        "y": 0.44245377300469013,
        "y_hat": 0.5108340361542065
      },
      {
        "dow": 2,
        "t": "2022-04-27T00:00:00",
        "wom": 5,
        "y": 0.10472408289231805,
        "y_hat": 0.4844334459505021
      },
      {
        "dow": 3,
        "t": "2022-04-28T00:00:00",
        "wom": 5,
        "y": 0.07965539705381441,
        "y_hat": 0.4958677197277991
      },
      {
        "dow": 4,
        "t": "2022-04-29T00:00:00",
        "wom": 5,
        "y": 2.1204279487951343,
        "y_hat": 2.3990895488892217
      },
      {
        "dow": 5,
        "t": "2022-04-30T00:00:00",
        "wom": 5,
        "y": 0.09911422212585952,
        "y_hat": 0.4964206092595712
      },
      {
        "dow": 6,
        "t": "2022-05-01T00:00:00",
        "wom": 1,
        "y": 0.7653139992562019,
        "y_hat": 0.5110418469407004
      },
      {
        "dow": 0,
        "t": "2022-05-02T00:00:00",
        "wom": 2,
        "y": 0.6928765519708857,
        "y_hat": 0.45435487387021417
      },
      {
        "dow": 1,
        "t": "2022-05-03T00:00:00",
        "wom": 2,
        "y": 0.3293225432083243,
        "y_hat": 0.5227571533370909
      },
      {
        "dow": 2,
        "t": "2022-05-04T00:00:00",
        "wom": 2,
        "y": 0.6649514990871527,
        "y_hat": 0.47325979365624726
      },
      {
        "dow": 3,
        "t": "2022-05-05T00:00:00",
        "wom": 2,
        "y": 0.15101171079274767,
        "y_hat": 0.5058810792932257
      },
      {
        "dow": 4,
        "t": "2022-05-06T00:00:00",
        "wom": 2,
        "y": 2.1798428980656723,
        "y_hat": 2.431485876177647
      },
      {
        "dow": 5,
        "t": "2022-05-07T00:00:00",
        "wom": 2,
        "y": 0.2354835900991824,
        "y_hat": 0.5020829628448384
      },
      {
        "dow": 6,
        "t": "2022-05-08T00:00:00",
        "wom": 2,
        "y": 0.05741790510662548,
        "y_hat": 0.5182585301985556
      },
      {
        "dow": 0,
        "t": "2022-05-09T00:00:00",
        "wom": 3,
        "y": 0.23077326722872582,
        "y_hat": 0.4416008126128215
      },
      {
        "dow": 1,
        "t": "2022-05-10T00:00:00",
        "wom": 3,
        "y": 0.1944400629234596,
        "y_hat": 0.4714472781064954
      },
      {
        "dow": 2,
        "t": "2022-05-11T00:00:00",
        "wom": 3,
        "y": 0.7807354431480255,
        "y_hat": 0.44457005664505445
      },
      {
        "dow": 3,
        "t": "2022-05-12T00:00:00",
        "wom": 3,
        "y": 0.31021821068384303,
        "y_hat": 0.5020199652718704
      },
      {
        "dow": 4,
        "t": "2022-05-13T00:00:00",
        "wom": 3,
        "y": 2.1275853244649534,
        "y_hat": 2.4423983412290537
      },
      {
        "dow": 5,
        "t": "2022-05-14T00:00:00",
        "wom": 3,
        "y": 0.56813301154376,
        "y_hat": 0.5092969989732901
      },
      {
        "dow": 6,
        "t": "2022-05-15T00:00:00",
        "wom": 3,
        "y": 0.10167470187610927,
        "y_hat": 0.5250551324538184
      },
      {
        "dow": 0,
        "t": "2022-05-16T00:00:00",
        "wom": 4,
        "y": 0.37826726087578866,
        "y_hat": 0.46093647736965676
      },
      {
        "dow": 1,
        "t": "2022-05-17T00:00:00",
        "wom": 4,
        "y": 0.7138146434013722,
        "y_hat": 0.4781635078824496
      },
      {
        "dow": 2,
        "t": "2022-05-18T00:00:00",
        "wom": 4,
        "y": 0.12852670113262654,
        "y_hat": 0.46748643229002546
      },
      {
        "dow": 3,
        "t": "2022-05-19T00:00:00",
        "wom": 4,
        "y": 0.3585977195944233,
        "y_hat": 0.5115345457435468
      },
      {
        "dow": 4,
        "t": "2022-05-20T00:00:00",
        "wom": 4,
        "y": 2.22258022407796,
        "y_hat": 2.408682157424716
      },
      {
        "dow": 5,
        "t": "2022-05-21T00:00:00",
        "wom": 4,
        "y": 0.932770577180065,
        "y_hat": 0.5147356152220035
      },
      {
        "dow": 6,
        "t": "2022-05-22T00:00:00",
        "wom": 4,
        "y": 0.9336405466022045,
        "y_hat": 0.5407155484496845
      },
      {
        "dow": 0,
        "t": "2022-05-23T00:00:00",
        "wom": 5,
        "y": 0.7765773218651726,
        "y_hat": 0.5045082552262019
      },
      {
        "dow": 1,
        "t": "2022-05-24T00:00:00",
        "wom": 5,
        "y": 0.9092507321087694,
        "y_hat": 0.5345730713865291
      },
      {
        "dow": 2,
        "t": "2022-05-25T00:00:00",
        "wom": 5,
        "y": 0.9912972693958269,
        "y_hat": 0.4946869470039545
      },
      {
        "dow": 3,
        "t": "2022-05-26T00:00:00",
        "wom": 5,
        "y": 0.17273741694999156,
        "y_hat": 0.5468518827619808
      },
      {
        "dow": 4,
        "t": "2022-05-27T00:00:00",
        "wom": 5,
        "y": 2.7305657647776744,
        "y_hat": 2.4500618298869807
      },
      {
        "dow": 5,
        "t": "2022-05-28T00:00:00",
        "wom": 5,
        "y": 0.9331495324820412,
        "y_hat": 0.5193297047723543
      },
      {
        "dow": 6,
        "t": "2022-05-29T00:00:00",
        "wom": 5,
        "y": 0.9747799262042263,
        "y_hat": 0.5682632750846622
      },
      {
        "dow": 0,
        "t": "2022-05-30T00:00:00",
        "wom": 6,
        "y": 0.6159827699504155,
        "y_hat": 0.5058165123564484
      },
      {
        "dow": 1,
        "t": "2022-05-31T00:00:00",
        "wom": 6,
        "y": 0.18957160929478611,
        "y_hat": 0.5323172222676634
      },
      {
        "dow": 2,
        "t": "2022-06-01T00:00:00",
        "wom": 1,
        "y": 0.08329858559961756,
        "y_hat": 0.4655639528916069
      },
      {
        "dow": 3,
        "t": "2022-06-02T00:00:00",
        "wom": 1,
        "y": 0.7811845717468601,
        "y_hat": 0.48216420003370863
      },
      {
        "dow": 4,
        "t": "2022-06-03T00:00:00",
        "wom": 1,
        "y": 2.5856317189042306,
        "y_hat": 2.41854048611295
      },
      {
        "dow": 5,
        "t": "2022-06-04T00:00:00",
        "wom": 1,
        "y": 0.9915634439508605,
        "y_hat": 0.5481486757761187
      },
      {
        "dow": 6,
        "t": "2022-06-05T00:00:00",
        "wom": 1,
        "y": 0.9909525874581544,
        "y_hat": 0.5622201296718374
      },
      {
        "dow": 0,
        "t": "2022-06-06T00:00:00",
        "wom": 2,
        "y": 0.546656872738867,
        "y_hat": 0.509544473702861
      },
      {
        "dow": 1,
        "t": "2022-06-07T00:00:00",
        "wom": 2,
        "y": 0.10472218993110805,
        "y_hat": 0.5313215376478715
      },
      {
        "dow": 2,
        "t": "2022-06-08T00:00:00",
        "wom": 2,
        "y": 0.04250860920798383,
        "y_hat": 0.45950885511347206
      },
      {
        "dow": 3,
        "t": "2022-06-09T00:00:00",
        "wom": 2,
        "y": 0.3032781068949767,
        "y_hat": 0.4765300517785524
      },
      {
        "dow": 4,
        "t": "2022-06-10T00:00:00",
        "wom": 2,
        "y": 2.5924710028233178,
        "y_hat": 2.4027941662417835
      },
      {
        "dow": 5,
        "t": "2022-06-11T00:00:00",
        "wom": 2,
        "y": 0.1823450459040853,
        "y_hat": 0.5226555382078407
      },
      {
        "dow": 6,
        "t": "2022-06-12T00:00:00",
        "wom": 2,
        "y": 0.7714328053698969,
        "y_hat": 0.5395777784515343
      },
      {
        "dow": 0,
        "t": "2022-06-13T00:00:00",
        "wom": 3,
        "y": 0.7117102710960721,
        "y_hat": 0.4596964393488171
      },
      {
        "dow": 1,
        "t": "2022-06-14T00:00:00",
        "wom": 3,
        "y": 0.5078428447537274,
        "y_hat": 0.5242989977320872
      },
      {
        "dow": 2,
        "t": "2022-06-15T00:00:00",
        "wom": 3,
        "y": 0.4559806484621759,
        "y_hat": 0.48005120057372996
      },
      {
        "dow": 3,
        "t": "2022-06-16T00:00:00",
        "wom": 3,
        "y": 0.3663298759696507,
        "y_hat": 0.5102207588634888
      },
      {
        "dow": 4,
        "t": "2022-06-17T00:00:00",
        "wom": 3,
        "y": 2.036192987755432,
        "y_hat": 2.427030286845141
      },
      {
        "dow": 5,
        "t": "2022-06-18T00:00:00",
        "wom": 3,
        "y": 0.5857499825271564,
        "y_hat": 0.5102758304570155
      },
      {
        "dow": 6,
        "t": "2022-06-19T00:00:00",
        "wom": 3,
        "y": 0.050005294335790484,
        "y_hat": 0.5211856409518846
      },
      {
        "dow": 0,
        "t": "2022-06-20T00:00:00",
        "wom": 4,
        "y": 0.5061272547669947,
        "y_hat": 0.46097219295144776
      },
      {
        "dow": 1,
        "t": "2022-06-21T00:00:00",
        "wom": 4,
        "y": 0.7904390381333787,
        "y_hat": 0.4795887904276564
      },
      {
        "dow": 2,
        "t": "2022-06-22T00:00:00",
        "wom": 4,
        "y": 0.09323256170059835,
        "y_hat": 0.4771399084781224
      },
      {
        "dow": 3,
        "t": "2022-06-23T00:00:00",
        "wom": 4,
        "y": 0.40800965871918415,
        "y_hat": 0.5152238334334927
      },
      {
        "dow": 4,
        "t": "2022-06-24T00:00:00",
        "wom": 4,
        "y": 2.570342151877391,
        "y_hat": 2.4087498026209566
      },
      {
        "dow": 5,
        "t": "2022-06-25T00:00:00",
        "wom": 4,
        "y": 0.07936746817059592,
        "y_hat": 0.5279017923132528
      },
      {
        "dow": 6,
        "t": "2022-06-26T00:00:00",
        "wom": 4,
        "y": 0.017413769472798823,
        "y_hat": 0.5356649133238703
      },
      {
        "dow": 0,
        "t": "2022-06-27T00:00:00",
        "wom": 5,
        "y": 0.4856034209379412,
        "y_hat": 0.4328149611023629
      },
      {
        "dow": 1,
        "t": "2022-06-28T00:00:00",
        "wom": 5,
        "y": 0.10626439009089561,
        "y_hat": 0.47735501023779975
      },
      {
        "dow": 2,
        "t": "2022-06-29T00:00:00",
        "wom": 5,
        "y": 0.7531865011385493,
        "y_hat": 0.4565934866610085
      },
      {
        "dow": 3,
        "t": "2022-06-30T00:00:00",
        "wom": 5,
        "y": 0.2215890993417199,
        "y_hat": 0.4972579361761011
      },
      {
        "dow": 4,
        "t": "2022-07-01T00:00:00",
        "wom": 1,
        "y": 2.2444907708382646,
        "y_hat": 2.4391639795535345
      },
      {
        "dow": 5,
        "t": "2022-07-02T00:00:00",
        "wom": 1,
        "y": 0.41066189647640805,
        "y_hat": 0.5086383437322588
      },
      {
        "dow": 6,
        "t": "2022-07-03T00:00:00",
        "wom": 1,
        "y": 0.8875209104102515,
        "y_hat": 0.5276522539533361
      },
      {
        "dow": 0,
        "t": "2022-07-04T00:00:00",
        "wom": 2,
        "y": 0.29879768250458105,
        "y_hat": 0.4756765448644015
      },
      {
        "dow": 1,
        "t": "2022-07-05T00:00:00",
        "wom": 2,
        "y": 0.8642111944128015,
        "y_hat": 0.519103746855454
      },
      {
        "dow": 2,
        "t": "2022-07-06T00:00:00",
        "wom": 2,
        "y": 0.5035308154420571,
        "y_hat": 0.468279298304797
      },
      {
        "dow": 3,
        "t": "2022-07-07T00:00:00",
        "wom": 2,
        "y": 0.4608669940295842,
        "y_hat": 0.5311554916042507
      },
      {
        "dow": 4,
        "t": "2022-07-08T00:00:00",
        "wom": 2,
        "y": 2.1336407534117914,
        "y_hat": 2.43263548967435
      },
      {
        "dow": 5,
        "t": "2022-07-09T00:00:00",
        "wom": 2,
        "y": 0.07559350276009646,
        "y_hat": 0.5185008966658524
      },
      {
        "dow": 6,
        "t": "2022-07-10T00:00:00",
        "wom": 2,
        "y": 0.21152592170248208,
        "y_hat": 0.5122052608058499
      },
      {
        "dow": 0,
        "t": "2022-07-11T00:00:00",
        "wom": 3,
        "y": 0.7992198246512687,
        "y_hat": 0.4383825054702067
      },
      {
        "dow": 1,
        "t": "2022-07-12T00:00:00",
        "wom": 3,
        "y": 0.08134234250438865,
        "y_hat": 0.49702074794504336
      },
      {
        "dow": 2,
        "t": "2022-07-13T00:00:00",
        "wom": 3,
        "y": 0.8545474044325597,
        "y_hat": 0.4730370156455408
      },
      {
        "dow": 3,
        "t": "2022-07-14T00:00:00",
        "wom": 3,
        "y": 0.7192996694052958,
        "y_hat": 0.49903334487542134
      },
      {
        "dow": 4,
        "t": "2022-07-15T00:00:00",
        "wom": 3,
        "y": 2.839731861280379,
        "y_hat": 2.4590829237763088
      },
      {
        "dow": 5,
        "t": "2022-07-16T00:00:00",
        "wom": 3,
        "y": 0.6866567952654717,
        "y_hat": 0.5527420470316485
      },
      {
        "dow": 6,
        "t": "2022-07-17T00:00:00",
        "wom": 3,
        "y": 0.3025925308532783,
        "y_hat": 0.5678987356529254
      },
      {
        "dow": 0,
        "t": "2022-07-18T00:00:00",
        "wom": 4,
        "y": 0.35527032358475563,
        "y_hat": 0.4740865465555972
      },
      {
        "dow": 1,
        "t": "2022-07-19T00:00:00",
        "wom": 4,
        "y": 0.2998867051787012,
        "y_hat": 0.4893633221283946
      },
      {
        "dow": 2,
        "t": "2022-07-20T00:00:00",
        "wom": 4,
        "y": 0.32169056052383393,
        "y_hat": 0.4554266913463155
      },
      {
        "dow": 3,
        "t": "2022-07-21T00:00:00",
        "wom": 4,
        "y": 0.8817188595233318,
        "y_hat": 0.49571750586434926
      },
      {
        "dow": 4,
        "t": "2022-07-22T00:00:00",
        "wom": 4,
        "y": 2.6482046952131784,
        "y_hat": 2.4350689441557574
      },
      {
        "dow": 5,
        "t": "2022-07-23T00:00:00",
        "wom": 4,
        "y": 0.5392997474632014,
        "y_hat": 0.5561498314017669
      },
      {
        "dow": 6,
        "t": "2022-07-24T00:00:00",
        "wom": 4,
        "y": 0.28113347961902946,
        "y_hat": 0.5534601575845951
      },
      {
        "dow": 0,
        "t": "2022-07-25T00:00:00",
        "wom": 5,
        "y": 0.4047240553693535,
        "y_hat": 0.4656301818804722
      },
      {
        "dow": 1,
        "t": "2022-07-26T00:00:00",
        "wom": 5,
        "y": 0.09656379233844459,
        "y_hat": 0.4897301947692827
      },
      {
        "dow": 2,
        "t": "2022-07-27T00:00:00",
        "wom": 5,
        "y": 0.9545109848618508,
        "y_hat": 0.45239872668182046
      },
      {
        "dow": 3,
        "t": "2022-07-28T00:00:00",
        "wom": 5,
        "y": 0.2844751590965856,
        "y_hat": 0.5029360562621922
      },
      {
        "dow": 4,
        "t": "2022-07-29T00:00:00",
        "wom": 5,
        "y": 2.19828814692932,
        "y_hat": 2.4522787916765902
      },
      {
        "dow": 5,
        "t": "2022-07-30T00:00:00",
        "wom": 5,
        "y": 0.7872041161790215,
        "y_hat": 0.5111620022247799
      },
      {
        "dow": 6,
        "t": "2022-07-31T00:00:00",
        "wom": 5,
        "y": 0.8320061228006757,
        "y_hat": 0.5363666658917704
      },
      {
        "dow": 0,
        "t": "2022-08-01T00:00:00",
        "wom": 1,
        "y": 0.48050923278633095,
        "y_hat": 0.49486749692129434
      },
      {
        "dow": 1,
        "t": "2022-08-02T00:00:00",
        "wom": 1,
        "y": 0.9186312133845865,
        "y_hat": 0.5217480417049952
      },
      {
        "dow": 2,
        "t": "2022-08-03T00:00:00",
        "wom": 1,
        "y": 0.28383987157951585,
        "y_hat": 0.4800933434215706
      },
      {
        "dow": 3,
        "t": "2022-08-04T00:00:00",
        "wom": 1,
        "y": 0.6963665748778668,
        "y_hat": 0.5282645806595277
      },
      {
        "dow": 4,
        "t": "2022-08-05T00:00:00",
        "wom": 1,
        "y": 2.09245143951022,
        "y_hat": 2.427953405345737
      },
      {
        "dow": 5,
        "t": "2022-08-06T00:00:00",
        "wom": 1,
        "y": 0.7172188720261602,
        "y_hat": 0.5304867199912241
      },
      {
        "dow": 6,
        "t": "2022-08-07T00:00:00",
        "wom": 1,
        "y": 0.6862602995181284,
        "y_hat": 0.5287652055403764
      },
      {
        "dow": 0,
        "t": "2022-08-08T00:00:00",
        "wom": 2,
        "y": 0.12495616249592223,
        "y_hat": 0.4870366588398229
      },
      {
        "dow": 1,
        "t": "2022-08-09T00:00:00",
        "wom": 2,
        "y": 0.7803068072936611,
        "y_hat": 0.5038317649617358
      },
      {
        "dow": 2,
        "t": "2022-08-10T00:00:00",
        "wom": 2,
        "y": 0.1711236164496699,
        "y_hat": 0.45705759106594746
      },
      {
        "dow": 3,
        "t": "2022-08-11T00:00:00",
        "wom": 2,
        "y": 0.6418292052213659,
        "y_hat": 0.5176881437709107
      },
      {
        "dow": 4,
        "t": "2022-08-12T00:00:00",
        "wom": 2,
        "y": 2.398223140821866,
        "y_hat": 2.420421968469209
      },
      {
        "dow": 5,
        "t": "2022-08-13T00:00:00",
        "wom": 2,
        "y": 0.7667020108493909,
        "y_hat": 0.5363921988276147
      },
      {
        "dow": 6,
        "t": "2022-08-14T00:00:00",
        "wom": 2,
        "y": 0.41815335304653145,
        "y_hat": 0.5467991648029131
      },
      {
        "dow": 0,
        "t": "2022-08-15T00:00:00",
        "wom": 3,
        "y": 0.6891797613670014,
        "y_hat": 0.48215904682588656
      },
      {
        "dow": 1,
        "t": "2022-08-16T00:00:00",
        "wom": 3,
        "y": 0.5101490274593742,
        "y_hat": 0.5055926245703029
      },
      {
        "dow": 2,
        "t": "2022-08-17T00:00:00",
        "wom": 3,
        "y": 0.7796622447746596,
        "y_hat": 0.47991073110965105
      },
      {
        "dow": 3,
        "t": "2022-08-18T00:00:00",
        "wom": 3,
        "y": 0.5627361856480227,
        "y_hat": 0.5206046080919271
      },
      {
        "dow": 4,
        "t": "2022-08-19T00:00:00",
        "wom": 3,
        "y": 2.149269114111338,
        "y_hat": 2.4511273168906147
      },
      {
        "dow": 5,
        "t": "2022-08-20T00:00:00",
        "wom": 3,
        "y": 0.4028888263767394,
        "y_hat": 0.5251204340687972
      },
      {
        "dow": 6,
        "t": "2022-08-21T00:00:00",
        "wom": 3,
        "y": 0.29581445643538484,
        "y_hat": 0.5230749375317849
      },
      {
        "dow": 0,
        "t": "2022-08-22T00:00:00",
        "wom": 4,
        "y": 0.22454795399899974,
        "y_hat": 0.4591341483141852
      },
      {
        "dow": 1,
        "t": "2022-08-23T00:00:00",
        "wom": 4,
        "y": 0.5368502058660074,
        "y_hat": 0.48582304298727946
      },
      {
        "dow": 2,
        "t": "2022-08-24T00:00:00",
        "wom": 4,
        "y": 0.8352731932733497,
        "y_hat": 0.455701817281319
      },
      {
        "dow": 3,
        "t": "2022-08-25T00:00:00",
        "wom": 4,
        "y": 0.7156495255143432,
        "y_hat": 0.5237474583198937
      },
      {
        "dow": 4,
        "t": "2022-08-26T00:00:00",
        "wom": 4,
        "y": 2.081170707633651,
        "y_hat": 2.4586113850595304
      },
      {
        "dow": 5,
        "t": "2022-08-27T00:00:00",
        "wom": 4,
        "y": 0.9846573243966786,
        "y_hat": 0.5315420611782877
      },
      {
        "dow": 6,
        "t": "2022-08-28T00:00:00",
        "wom": 4,
        "y": 0.11556538882109113,
        "y_hat": 0.5361350363720931
      },
      {
        "dow": 0,
        "t": "2022-08-29T00:00:00",
        "wom": 5,
        "y": 0.26133230623941184,
        "y_hat": 0.4855036716924154
      },
      {
        "dow": 1,
        "t": "2022-08-30T00:00:00",
        "wom": 5,
        "y": 0.9006219041976188,
        "y_hat": 0.47725526614685454
      },
      {
        "dow": 2,
        "t": "2022-08-31T00:00:00",
        "wom": 5,
        "y": 0.2515146218161567,
        "y_hat": 0.4681948708253787
      },
      {
        "dow": 3,
        "t": "2022-09-01T00:00:00",
        "wom": 1,
        "y": 0.13098484463014348,
        "y_hat": 0.5268174837110191
      },
      {
        "dow": 4,
        "t": "2022-09-02T00:00:00",
        "wom": 1,
        "y": 2.9173298508007353,
        "y_hat": 2.410501086412229
      },
      {
        "dow": 5,
        "t": "2022-09-03T00:00:00",
        "wom": 1,
        "y": 0.902033118492389,
        "y_hat": 0.5239835691087095
      },
      {
        "dow": 6,
        "t": "2022-09-04T00:00:00",
        "wom": 1,
        "y": 0.6094812461745855,
        "y_hat": 0.579028055768711
      },
      {
        "dow": 0,
        "t": "2022-09-05T00:00:00",
        "wom": 2,
        "y": 0.21237629900528165,
        "y_hat": 0.4952691329353708
      },
      {
        "dow": 1,
        "t": "2022-09-06T00:00:00",
        "wom": 2,
        "y": 0.8674613314753005,
        "y_hat": 0.5026336407752172
      },
      {
        "dow": 2,
        "t": "2022-09-07T00:00:00",
        "wom": 2,
        "y": 0.7318690107796962,
        "y_hat": 0.46471651844447337
      },
      {
        "dow": 3,
        "t": "2022-09-08T00:00:00",
        "wom": 2,
        "y": 0.9273400869945371,
        "y_hat": 0.538865818625751
      },
      {
        "dow": 4,
        "t": "2022-09-09T00:00:00",
        "wom": 2,
        "y": 2.491296824471812,
        "y_hat": 2.4593024923288445
      },
      {
        "dow": 5,
        "t": "2022-09-10T00:00:00",
        "wom": 2,
        "y": 0.3736267511181386,
        "y_hat": 0.5549146181128354
      },
      {
        "dow": 6,
        "t": "2022-09-11T00:00:00",
        "wom": 2,
        "y": 0.6741226397964637,
        "y_hat": 0.5410405136378633
      },
      {
        "dow": 0,
        "t": "2022-09-12T00:00:00",
        "wom": 3,
        "y": 0.5690598209381145,
        "y_hat": 0.468701374121668
      },
      {
        "dow": 1,
        "t": "2022-09-13T00:00:00",
        "wom": 3,
        "y": 0.5845921058651229,
        "y_hat": 0.5164280677357482
      },
      {
        "dow": 2,
        "t": "2022-09-14T00:00:00",
        "wom": 3,
        "y": 0.06190141469816057,
        "y_hat": 0.47600193151568615
      },
      {
        "dow": 3,
        "t": "2022-09-15T00:00:00",
        "wom": 3,
        "y": 0.7726353296869438,
        "y_hat": 0.504562097693305
      },
      {
        "dow": 4,
        "t": "2022-09-16T00:00:00",
        "wom": 3,
        "y": 2.3982766116278027,
        "y_hat": 2.418824328436912
      },
      {
        "dow": 5,
        "t": "2022-09-17T00:00:00",
        "wom": 3,
        "y": 0.02485301435952192,
        "y_hat": 0.5440166771084493
      },
      {
        "dow": 6,
        "t": "2022-09-18T00:00:00",
        "wom": 3,
        "y": 0.3885550460991051,
        "y_hat": 0.5261644635357546
      },
      {
        "dow": 0,
        "t": "2022-09-19T00:00:00",
        "wom": 4,
        "y": 0.5124897271631259,
        "y_hat": 0.44182407349217223
      },
      {
        "dow": 1,
        "t": "2022-09-20T00:00:00",
        "wom": 4,
        "y": 0.6717284818449774,
        "y_hat": 0.499506798622325
      },
      {
        "dow": 2,
        "t": "2022-09-21T00:00:00",
        "wom": 4,
        "y": 0.8010250603087118,
        "y_hat": 0.47555005974701947
      },
      {
        "dow": 3,
        "t": "2022-09-22T00:00:00",
        "wom": 4,
        "y": 0.7215828184438076,
        "y_hat": 0.5304993746500212
      },
      {
        "dow": 4,
        "t": "2022-09-23T00:00:00",
        "wom": 4,
        "y": 2.467671973989098,
        "y_hat": 2.457380857040162
      },
      {
        "dow": 5,
        "t": "2022-09-24T00:00:00",
        "wom": 4,
        "y": 0.7177366636762655,
        "y_hat": 0.5433557257312887
      },
      {
        "dow": 6,
        "t": "2022-09-25T00:00:00",
        "wom": 4,
        "y": 0.48856096479123934,
        "y_hat": 0.5498226240838627
      },
      {
        "dow": 0,
        "t": "2022-09-26T00:00:00",
        "wom": 5,
        "y": 0.5960952391022855,
        "y_hat": 0.4822008051720972
      },
      {
        "dow": 1,
        "t": "2022-09-27T00:00:00",
        "wom": 5,
        "y": 0.8681440058433928,
        "y_hat": 0.5074071188774416
      },
      {
        "dow": 2,
        "t": "2022-09-28T00:00:00",
        "wom": 5,
        "y": 0.8669839542345549,
        "y_hat": 0.48578856966099204
      },
      {
        "dow": 3,
        "t": "2022-09-29T00:00:00",
        "wom": 5,
        "y": 0.8884123252558981,
        "y_hat": 0.5431003113470626
      },
      {
        "dow": 4,
        "t": "2022-09-30T00:00:00",
        "wom": 5,
        "y": 2.3463008214785965,
        "y_hat": 2.465821242163437
      },
      {
        "dow": 5,
        "t": "2022-10-01T00:00:00",
        "wom": 1,
        "y": 0.2954617156792265,
        "y_hat": 0.5490063522668106
      },
      {
        "dow": 6,
        "t": "2022-10-02T00:00:00",
        "wom": 1,
        "y": 0.633109100839611,
        "y_hat": 0.5313156558678173
      },
      {
        "dow": 0,
        "t": "2022-10-03T00:00:00",
        "wom": 2,
        "y": 0.3669341564363222,
        "y_hat": 0.46364614979584606
      },
      {
        "dow": 1,
        "t": "2022-10-04T00:00:00",
        "wom": 2,
        "y": 0.24876598531702476,
        "y_hat": 0.5087746891999738
      },
      {
        "dow": 2,
        "t": "2022-10-05T00:00:00",
        "wom": 2,
        "y": 0.16587282213408583,
        "y_hat": 0.4558300182291027
      },
      {
        "dow": 3,
        "t": "2022-10-06T00:00:00",
        "wom": 2,
        "y": 0.9919913685200391,
        "y_hat": 0.48973950777965536
      },
      {
        "dow": 4,
        "t": "2022-10-07T00:00:00",
        "wom": 2,
        "y": 2.3533296873989085,
        "y_hat": 2.4310423123676665
      },
      {
        "dow": 5,
        "t": "2022-10-08T00:00:00",
        "wom": 2,
        "y": 0.7213297581947791,
        "y_hat": 0.554911360704057
      },
      {
        "dow": 6,
        "t": "2022-10-09T00:00:00",
        "wom": 2,
        "y": 0.7224275916390493,
        "y_hat": 0.5439767292961218
      },
      {
        "dow": 0,
        "t": "2022-10-10T00:00:00",
        "wom": 3,
        "y": 0.29155319532221713,
        "y_hat": 0.4893020791489958
      },
      {
        "dow": 1,
        "t": "2022-10-11T00:00:00",
        "wom": 3,
        "y": 0.36991001906594234,
        "y_hat": 0.5115549109796871
      },
      {
        "dow": 2,
        "t": "2022-10-12T00:00:00",
        "wom": 3,
        "y": 0.28417739594797153,
        "y_hat": 0.4553343870788139
      },
      {
        "dow": 3,
        "t": "2022-10-13T00:00:00",
        "wom": 3,
        "y": 0.19840529083955494,
        "y_hat": 0.4997725564131675
      },
      {
        "dow": 4,
        "t": "2022-10-14T00:00:00",
        "wom": 3,
        "y": 2.4606866711518958,
        "y_hat": 2.4148643523441606
      },
      {
        "dow": 5,
        "t": "2022-10-15T00:00:00",
        "wom": 3,
        "y": 0.6335089949594493,
        "y_hat": 0.5152479312806922
      },
      {
        "dow": 6,
        "t": "2022-10-16T00:00:00",
        "wom": 3,
        "y": 0.5680543096824044,
        "y_hat": 0.5473753240467025
      },
      {
        "dow": 0,
        "t": "2022-10-17T00:00:00",
        "wom": 4,
        "y": 0.016790840072704594,
        "y_hat": 0.48026181895173703
      },
      {
        "dow": 1,
        "t": "2022-10-18T00:00:00",
        "wom": 4,
        "y": 0.15482355482083832,
        "y_hat": 0.4954814961576285
      },
      {
        "dow": 2,
        "t": "2022-10-19T00:00:00",
        "wom": 4,
        "y": 0.32294713218572024,
        "y_hat": 0.43446678008048273
      },
      {
        "dow": 3,
        "t": "2022-10-20T00:00:00",
        "wom": 4,
        "y": 0.09723518582862622,
        "y_hat": 0.48938074212176336
      },
      {
        "dow": 4,
        "t": "2022-10-21T00:00:00",
        "wom": 4,
        "y": 2.161951112634909,
        "y_hat": 2.4141786551814923
      },
      {
        "dow": 5,
        "t": "2022-10-22T00:00:00",
        "wom": 4,
        "y": 0.9308281099209696,
        "y_hat": 0.5013617569960774
      },
      {
        "dow": 6,
        "t": "2022-10-23T00:00:00",
        "wom": 4,
        "y": 0.3700513293335844,
        "y_hat": 0.5398557089374105
      },
      {
        "dow": 0,
        "t": "2022-10-24T00:00:00",
        "wom": 5,
        "y": 0.33730301215287395,
        "y_hat": 0.4907674124047824
      },
      {
        "dow": 1,
        "t": "2022-10-25T00:00:00",
        "wom": 5,
        "y": 0.2979813390418188,
        "y_hat": 0.49406289572899337
      },
      {
        "dow": 2,
        "t": "2022-10-26T00:00:00",
        "wom": 5,
        "y": 0.8989252622362512,
        "y_hat": 0.45597333301628634
      },
      {
        "dow": 3,
        "t": "2022-10-27T00:00:00",
        "wom": 5,
        "y": 0.9678609978612648,
        "y_hat": 0.5136804919832462
      },
      {
        "dow": 4,
        "t": "2022-10-28T00:00:00",
        "wom": 5,
        "y": 2.852867990371385,
        "y_hat": 2.47026482888401
      },
      {
        "dow": 5,
        "t": "2022-10-29T00:00:00",
        "wom": 5,
        "y": 0.09421902072623589,
        "y_hat": 0.5682198415030082
      },
      {
        "dow": 6,
        "t": "2022-10-30T00:00:00",
        "wom": 5,
        "y": 0.8581628861015397,
        "y_hat": 0.553362035505034
      },
      {
        "dow": 0,
        "t": "2022-10-31T00:00:00",
        "wom": 6,
        "y": 0.20114187846747755,
        "y_hat": 0.4596615396792977
      },
      {
        "dow": 1,
        "t": "2022-11-01T00:00:00",
        "wom": 1,
        "y": 0.6181376931339703,
        "y_hat": 0.516635984405509
      },
      {
        "dow": 2,
        "t": "2022-11-02T00:00:00",
        "wom": 1,
        "y": 0.19978354259674502,
        "y_hat": 0.4578832281890113
      },
      {
        "dow": 3,
        "t": "2022-11-03T00:00:00",
        "wom": 1,
        "y": 0.1768860986420725,
        "y_hat": 0.5110987214886396
      },
      {
        "dow": 4,
        "t": "2022-11-04T00:00:00",
        "wom": 1,
        "y": 2.4752377813951942,
        "y_hat": 2.4100299124147666
      },
      {
        "dow": 5,
        "t": "2022-11-05T00:00:00",
        "wom": 1,
        "y": 0.9055115735426204,
        "y_hat": 0.5148386395620688
      },
      {
        "dow": 6,
        "t": "2022-11-06T00:00:00",
        "wom": 1,
        "y": 0.6649858316616734,
        "y_hat": 0.5562701637739151
      },
      {
        "dow": 0,
        "t": "2022-11-07T00:00:00",
        "wom": 2,
        "y": 0.5304475702709496,
        "y_hat": 0.4980529825768307
      },
      {
        "dow": 1,
        "t": "2022-11-08T00:00:00",
        "wom": 2,
        "y": 0.1761904542362559,
        "y_hat": 0.5157292008835385
      },
      {
        "dow": 2,
        "t": "2022-11-09T00:00:00",
        "wom": 2,
        "y": 0.9624710144321974,
        "y_hat": 0.46314486020995904
      },
      {
        "dow": 3,
        "t": "2022-11-10T00:00:00",
        "wom": 2,
        "y": 0.7059542876896551,
        "y_hat": 0.5091456448776946
      },
      {
        "dow": 4,
        "t": "2022-11-11T00:00:00",
        "wom": 2,
        "y": 2.506728336965063,
        "y_hat": 2.46643581246614
      },
      {
        "dow": 5,
        "t": "2022-11-12T00:00:00",
        "wom": 2,
        "y": 0.8433091095294604,
        "y_hat": 0.5444135866449329
      },
      {
        "dow": 6,
        "t": "2022-11-13T00:00:00",
        "wom": 2,
        "y": 0.7560702301487369,
        "y_hat": 0.5563050037044083
      },
      {
        "dow": 0,
        "t": "2022-11-14T00:00:00",
        "wom": 3,
        "y": 0.06592714264449118,
        "y_hat": 0.4974098519973051
      },
      {
        "dow": 1,
        "t": "2022-11-15T00:00:00",
        "wom": 3,
        "y": 0.2504072631614749,
        "y_hat": 0.5074849113317751
      },
      {
        "dow": 2,
        "t": "2022-11-16T00:00:00",
        "wom": 3,
        "y": 0.5805948341045363,
        "y_hat": 0.44029974147134476
      },
      {
        "dow": 3,
        "t": "2022-11-17T00:00:00",
        "wom": 3,
        "y": 0.5803663521377741,
        "y_hat": 0.5023523218787471
      },
      {
        "dow": 4,
        "t": "2022-11-18T00:00:00",
        "wom": 3,
        "y": 2.044237322520029,
        "y_hat": 2.44234290195584
      },
      {
        "dow": 5,
        "t": "2022-11-19T00:00:00",
        "wom": 3,
        "y": 0.5560838530176727,
        "y_hat": 0.5245297729912423
      },
      {
        "dow": 6,
        "t": "2022-11-20T00:00:00",
        "wom": 3,
        "y": 0.28756910769465927,
        "y_hat": 0.5232410661077838
      },
      {
        "dow": 0,
        "t": "2022-11-21T00:00:00",
        "wom": 4,
        "y": 0.32658389395269083,
        "y_hat": 0.46862796297499915
      },
      {
        "dow": 1,
        "t": "2022-11-22T00:00:00",
        "wom": 4,
        "y": 0.9255075128439307,
        "y_hat": 0.4897524974527008
      },
      {
        "dow": 2,
        "t": "2022-11-23T00:00:00",
        "wom": 4,
        "y": 0.2351401173462816,
        "y_hat": 0.4737753846011047
      },
      {
        "dow": 3,
        "t": "2022-11-24T00:00:00",
        "wom": 4,
        "y": 0.6755009569121625,
        "y_hat": 0.5290397268854274
      },
      {
        "dow": 4,
        "t": "2022-11-25T00:00:00",
        "wom": 4,
        "y": 2.6532627672211397,
        "y_hat": 2.4265235342689375
      },
      {
        "dow": 5,
        "t": "2022-11-26T00:00:00",
        "wom": 4,
        "y": 0.13334860717528818,
        "y_hat": 0.5471812325695693
      },
      {
        "dow": 6,
        "t": "2022-11-27T00:00:00",
        "wom": 4,
        "y": 0.21224946729617233,
        "y_hat": 0.5441529542842961
      },
      {
        "dow": 0,
        "t": "2022-11-28T00:00:00",
        "wom": 5,
        "y": 0.10485005588135898,
        "y_hat": 0.4437656298775255
      },
      {
        "dow": 1,
        "t": "2022-11-29T00:00:00",
        "wom": 5,
        "y": 0.571653845644994,
        "y_hat": 0.4794622765680875
      },
      {
        "dow": 2,
        "t": "2022-11-30T00:00:00",
        "wom": 5,
        "y": 0.6455551020000664,
        "y_hat": 0.451805294954433
      },
      {
        "dow": 3,
        "t": "2022-12-01T00:00:00",
        "wom": 1,
        "y": 0.07205418156187893,
        "y_hat": 0.5217762126874631
      },
      {
        "dow": 4,
        "t": "2022-12-02T00:00:00",
        "wom": 1,
        "y": 2.8390812934919314,
        "y_hat": 2.4315490965073923
      },
      {
        "dow": 5,
        "t": "2022-12-03T00:00:00",
        "wom": 1,
        "y": 0.3495218764848188,
        "y_hat": 0.5200249530600702
      },
      {
        "dow": 6,
        "t": "2022-12-04T00:00:00",
        "wom": 1,
        "y": 0.9605832472674337,
        "y_hat": 0.5604742376767212
      },
      {
        "dow": 0,
        "t": "2022-12-05T00:00:00",
        "wom": 2,
        "y": 0.47976044149700026,
        "y_hat": 0.4769324212585106
      },
      {
        "dow": 1,
        "t": "2022-12-06T00:00:00",
        "wom": 2,
        "y": 0.6055161236619802,
        "y_hat": 0.5306881286657961
      },
      {
        "dow": 2,
        "t": "2022-12-07T00:00:00",
        "wom": 2,
        "y": 0.41871964951333474,
        "y_hat": 0.47312547912871106
      },
      {
        "dow": 3,
        "t": "2022-12-08T00:00:00",
        "wom": 2,
        "y": 0.966612891389878,
        "y_hat": 0.5172346281782415
      },
      {
        "dow": 4,
        "t": "2022-12-09T00:00:00",
        "wom": 2,
        "y": 2.5654467983822786,
        "y_hat": 2.4449775700971856
      },
      {
        "dow": 5,
        "t": "2022-12-10T00:00:00",
        "wom": 2,
        "y": 0.34560834727361145,
        "y_hat": 0.5606125859661195
      },
      {
        "dow": 6,
        "t": "2022-12-11T00:00:00",
        "wom": 2,
        "y": 0.7570085832854714,
        "y_hat": 0.5457018888866806
      },
      {
        "dow": 0,
        "t": "2022-12-12T00:00:00",
        "wom": 3,
        "y": 0.9652744314834171,
        "y_hat": 0.4710161178959069
      },
      {
        "dow": 1,
        "t": "2022-12-13T00:00:00",
        "wom": 3,
        "y": 0.5648309593161226,
        "y_hat": 0.5336837354630672
      },
      {
        "dow": 2,
        "t": "2022-12-14T00:00:00",
        "wom": 3,
        "y": 0.7595290379916108,
        "y_hat": 0.49828660444260364
      },
      {
        "dow": 3,
        "t": "2022-12-15T00:00:00",
        "wom": 3,
        "y": 0.4539146559253099,
        "y_hat": 0.5248891664811817
      },
      {
        "dow": 4,
        "t": "2022-12-16T00:00:00",
        "wom": 3,
        "y": 2.6626571950380864,
        "y_hat": 2.448838509795448
      },
      {
        "dow": 5,
        "t": "2022-12-17T00:00:00",
        "wom": 3,
        "y": 0.901352733645678,
        "y_hat": 0.5358235389156042
      },
      {
        "dow": 6,
        "t": "2022-12-18T00:00:00",
        "wom": 3,
        "y": 0.1389757185937477,
        "y_hat": 0.5669428825994003
      },
      {
        "dow": 0,
        "t": "2022-12-19T00:00:00",
        "wom": 4,
        "y": 0.523376913529589,
        "y_hat": 0.4834707916350209
      },
      {
        "dow": 1,
        "t": "2022-12-20T00:00:00",
        "wom": 4,
        "y": 0.22793475704817479,
        "y_hat": 0.48780270680743293
      },
      {
        "dow": 2,
        "t": "2022-12-21T00:00:00",
        "wom": 4,
        "y": 0.3918232223846172,
        "y_hat": 0.46491507924850445
      },
      {
        "dow": 3,
        "t": "2022-12-22T00:00:00",
        "wom": 4,
        "y": 0.47576866552495234,
        "y_hat": 0.4963059605100148
      },
      {
        "dow": 4,
        "t": "2022-12-23T00:00:00",
        "wom": 4,
        "y": 2.3131731287518735,
        "y_hat": 2.4297238410504325
      },
      {
        "dow": 5,
        "t": "2022-12-24T00:00:00",
        "wom": 4,
        "y": 0.11547575723384762,
        "y_hat": 0.5271289193762412
      },
      {
        "dow": 6,
        "t": "2022-12-25T00:00:00",
        "wom": 4,
        "y": 0.880622203620008,
        "y_hat": 0.5257306518423508
      },
      {
        "dow": 0,
        "t": "2022-12-26T00:00:00",
        "wom": 5,
        "y": 0.4962845210256144,
        "y_hat": 0.4623486822690522
      },
      {
        "dow": 1,
        "t": "2022-12-27T00:00:00",
        "wom": 5,
        "y": 0.5406198774645495,
        "y_hat": 0.5271801081764019
      },
      {
        "dow": 2,
        "t": "2022-12-28T00:00:00",
        "wom": 5,
        "y": 0.26315431946537116,
        "y_hat": 0.4724997876723675
      },
      {
        "dow": 3,
        "t": "2022-12-29T00:00:00",
        "wom": 5,
        "y": 0.0927341711398395,
        "y_hat": 0.5096223750049854
      },
      {
        "dow": 4,
        "t": "2022-12-30T00:00:00",
        "wom": 5,
        "y": 2.42594516701449,
        "y_hat": 2.4119442031524905
      },
      {
        "dow": 5,
        "t": "2022-12-31T00:00:00",
        "wom": 5,
        "y": 0.9191390581073097,
        "y_hat": 0.5097847258328849
      }
    ]
  },
  "encoding": {
    "tooltip": {
      "field": "t",
      "type": "temporal"
    },
    "x": {
      "axis": {
        "grid": false,
        "ticks": true
      },
      "field": "t",
      "title": "",
      "type": "temporal"
    },
    "y": {
      "axis": {
        "domain": false,
        "grid": true,
        "ticks": true
      },
      "field": "y",
      "scale": {
        "zero": false
      },
      "title": "",
      "type": "quantitative"
    }
  },
  "height": 200,
  "mark": {
    "color": "darkblue",
    "opacity": 0.6,
    "type": "circle"
  },
  "title": "",
  "width": 400
}

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

| t                   |         y |   dow |   wom |
|:--------------------|----------:|------:|------:|
| 2022-01-01 00:00:00 | 0.995598  |     5 |     1 |
| 2022-01-02 00:00:00 | 0.623411  |     6 |     1 |
| 2022-01-03 00:00:00 | 0.876094  |     0 |     2 |
| 2022-01-04 00:00:00 | 0.889378  |     1 |     2 |
| 2022-01-05 00:00:00 | 0.386762  |     2 |     2 |
| 2022-01-06 00:00:00 | 0.68992   |     3 |     2 |
| 2022-01-07 00:00:00 | 2.99285   |     4 |     2 |
| 2022-01-08 00:00:00 | 0.468042  |     5 |     2 |
| 2022-01-09 00:00:00 | 0.945584  |     6 |     2 |
| 2022-01-10 00:00:00 | 0.638123  |     0 |     3 |
| 2022-01-11 00:00:00 | 0.759502  |     1 |     3 |
| 2022-01-12 00:00:00 | 0.237382  |     2 |     3 |
| 2022-01-13 00:00:00 | 0.0968092 |     3 |     3 |
| 2022-01-14 00:00:00 | 2.58857   |     4 |     3 |
| 2022-01-15 00:00:00 | 0.165735  |     5 |     3 |
| 2022-01-16 00:00:00 | 0.389421  |     6 |     3 |
| 2022-01-17 00:00:00 | 0.688788  |     0 |     4 |
| 2022-01-18 00:00:00 | 0.66976   |     1 |     4 |
| 2022-01-19 00:00:00 | 0.368854  |     2 |     4 |

With `seasonal=False`, we obtain an upwardly trending oscillation:

var spec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
  "config": {
    "view": {
      "continuousHeight": 300,
      "continuousWidth": 400
    }
  },
  "data": {
    "name": "data-460851078ef436e285d50a6cff3c9a96"
  },
  "datasets": {
    "data-460851078ef436e285d50a6cff3c9a96": [
      {
        "dow": 5,
        "t": "2022-01-01T00:00:00",
        "wom": 1,
        "y": 0.9955981828907805,
        "y_hat": null
      },
      {
        "dow": 6,
        "t": "2022-01-02T00:00:00",
        "wom": 1,
        "y": 0.6234109104925726,
        "y_hat": null
      },
      {
        "dow": 0,
        "t": "2022-01-03T00:00:00",
        "wom": 2,
        "y": 0.8760936874425284,
        "y_hat": 0.4967907586604228
      },
      {
        "dow": 1,
        "t": "2022-01-04T00:00:00",
        "wom": 2,
        "y": 0.8893778611747504,
        "y_hat": 0.5184239775556261
      },
      {
        "dow": 2,
        "t": "2022-01-05T00:00:00",
        "wom": 2,
        "y": 0.3867624782099456,
        "y_hat": 0.4972472553191131
      },
      {
        "dow": 3,
        "t": "2022-01-06T00:00:00",
        "wom": 2,
        "y": 0.6899198960223786,
        "y_hat": 0.5262597140421478
      },
      {
        "dow": 4,
        "t": "2022-01-07T00:00:00",
        "wom": 2,
        "y": 2.992852550566659,
        "y_hat": 2.4299590582515473
      },
      {
        "dow": 5,
        "t": "2022-01-08T00:00:00",
        "wom": 2,
        "y": 0.46804208563249405,
        "y_hat": 0.5525014522166138
      },
      {
        "dow": 6,
        "t": "2022-01-09T00:00:00",
        "wom": 2,
        "y": 0.9455841386245034,
        "y_hat": 0.5668882974435845
      },
      {
        "dow": 0,
        "t": "2022-01-10T00:00:00",
        "wom": 3,
        "y": 0.6381233538848571,
        "y_hat": 0.4776279996404177
      },
      {
        "dow": 1,
        "t": "2022-01-11T00:00:00",
        "wom": 3,
        "y": 0.7595024746539746,
        "y_hat": 0.5291292866690476
      },
      {
        "dow": 2,
        "t": "2022-01-12T00:00:00",
        "wom": 3,
        "y": 0.23738191602671632,
        "y_hat": 0.4808008912256273
      },
      {
        "dow": 3,
        "t": "2022-01-13T00:00:00",
        "wom": 3,
        "y": 0.09680924548766545,
        "y_hat": 0.515091715171366
      },
      {
        "dow": 4,
        "t": "2022-01-14T00:00:00",
        "wom": 3,
        "y": 2.588571863863769,
        "y_hat": 2.405058230192902
      },
      {
        "dow": 5,
        "t": "2022-01-15T00:00:00",
        "wom": 3,
        "y": 0.16573502501513393,
        "y_hat": 0.5090408212208333
      },
      {
        "dow": 6,
        "t": "2022-01-16T00:00:00",
        "wom": 3,
        "y": 0.38942124330865624,
        "y_hat": 0.5365359980531161
      },
      {
        "dow": 0,
        "t": "2022-01-17T00:00:00",
        "wom": 4,
        "y": 0.688787521402102,
        "y_hat": 0.44552691006817247
      },
      {
        "dow": 1,
        "t": "2022-01-18T00:00:00",
        "wom": 4,
        "y": 0.6697600544042848,
        "y_hat": 0.5006636268518205
      },
      {
        "dow": 2,
        "t": "2022-01-19T00:00:00",
        "wom": 4,
        "y": 0.36885391553882474,
        "y_hat": 0.4810838917599954
      },
      {
        "dow": 3,
        "t": "2022-01-20T00:00:00",
        "wom": 4,
        "y": 0.9325008856726638,
        "y_hat": 0.5141158767799752
      },
      {
        "dow": 4,
        "t": "2022-01-21T00:00:00",
        "wom": 4,
        "y": 2.141135576285614,
        "y_hat": 2.436148536768076
      },
      {
        "dow": 5,
        "t": "2022-01-22T00:00:00",
        "wom": 4,
        "y": 0.9772509899400809,
        "y_hat": 0.5414839206074327
      },
      {
        "dow": 6,
        "t": "2022-01-23T00:00:00",
        "wom": 4,
        "y": 0.5683493004574759,
        "y_hat": 0.5356818812038745
      },
      {
        "dow": 0,
        "t": "2022-01-24T00:00:00",
        "wom": 5,
        "y": 0.9649672826390342,
        "y_hat": 0.4945634970531389
      },
      {
        "dow": 1,
        "t": "2022-01-25T00:00:00",
        "wom": 5,
        "y": 0.22830327875114365,
        "y_hat": 0.5183276280490623
      },
      {
        "dow": 2,
        "t": "2022-01-26T00:00:00",
        "wom": 5,
        "y": 0.5236014315750456,
        "y_hat": 0.4834914229513924
      },
      {
        "dow": 3,
        "t": "2022-01-27T00:00:00",
        "wom": 5,
        "y": 0.5771602890446812,
        "y_hat": 0.49481710194259054
      },
      {
        "dow": 4,
        "t": "2022-01-28T00:00:00",
        "wom": 5,
        "y": 2.3198686688954586,
        "y_hat": 2.4344611018946396
      },
      {
        "dow": 5,
        "t": "2022-01-29T00:00:00",
        "wom": 5,
        "y": 0.4375103099359736,
        "y_hat": 0.5275197097634949
      },
      {
        "dow": 6,
        "t": "2022-01-30T00:00:00",
        "wom": 5,
        "y": 0.7071449184091295,
        "y_hat": 0.5300200373997083
      },
      {
        "dow": 0,
        "t": "2022-01-31T00:00:00",
        "wom": 6,
        "y": 0.1524637438937454,
        "y_hat": 0.46950282399087084
      },
      {
        "dow": 1,
        "t": "2022-02-01T00:00:00",
        "wom": 1,
        "y": 0.011748418393797033,
        "y_hat": 0.5027153449515473
      },
      {
        "dow": 2,
        "t": "2022-02-02T00:00:00",
        "wom": 1,
        "y": 0.25517769811183577,
        "y_hat": 0.4335507573741657
      },
      {
        "dow": 3,
        "t": "2022-02-03T00:00:00",
        "wom": 1,
        "y": 0.5156949359371945,
        "y_hat": 0.47556781187960434
      },
      {
        "dow": 4,
        "t": "2022-02-04T00:00:00",
        "wom": 1,
        "y": 2.6710173501944316,
        "y_hat": 2.418325496049474
      },
      {
        "dow": 5,
        "t": "2022-02-05T00:00:00",
        "wom": 1,
        "y": 0.8240477378755111,
        "y_hat": 0.5343478357671186
      },
      {
        "dow": 6,
        "t": "2022-02-06T00:00:00",
        "wom": 1,
        "y": 0.9508838016129778,
        "y_hat": 0.5601352577647144
      },
      {
        "dow": 0,
        "t": "2022-02-07T00:00:00",
        "wom": 2,
        "y": 0.05520043850117129,
        "y_hat": 0.4974480437451285
      },
      {
        "dow": 1,
        "t": "2022-02-08T00:00:00",
        "wom": 2,
        "y": 0.36173925787444183,
        "y_hat": 0.513206980650459
      },
      {
        "dow": 2,
        "t": "2022-02-09T00:00:00",
        "wom": 2,
        "y": 0.3287722160717247,
        "y_hat": 0.4384131691013086
      },
      {
        "dow": 3,
        "t": "2022-02-10T00:00:00",
        "wom": 2,
        "y": 0.26344459815995447,
        "y_hat": 0.49667802938134803
      },
      {
        "dow": 4,
        "t": "2022-02-11T00:00:00",
        "wom": 2,
        "y": 2.105522116894078,
        "y_hat": 2.415202698211017
      },
      {
        "dow": 5,
        "t": "2022-02-12T00:00:00",
        "wom": 2,
        "y": 0.9004373252539722,
        "y_hat": 0.5046824885073314
      },
      {
        "dow": 6,
        "t": "2022-02-13T00:00:00",
        "wom": 2,
        "y": 0.3083956070489431,
        "y_hat": 0.5319008881433882
      },
      {
        "dow": 0,
        "t": "2022-02-14T00:00:00",
        "wom": 3,
        "y": 0.8443045110438598,
        "y_hat": 0.483324909023404
      },
      {
        "dow": 1,
        "t": "2022-02-15T00:00:00",
        "wom": 3,
        "y": 0.9448650092640142,
        "y_hat": 0.5011821422416533
      },
      {
        "dow": 2,
        "t": "2022-02-16T00:00:00",
        "wom": 3,
        "y": 0.5570587072891441,
        "y_hat": 0.4977899462465503
      },
      {
        "dow": 3,
        "t": "2022-02-17T00:00:00",
        "wom": 3,
        "y": 0.5374558873035822,
        "y_hat": 0.5347949272417398
      },
      {
        "dow": 4,
        "t": "2022-02-18T00:00:00",
        "wom": 3,
        "y": 2.8223117437123824,
        "y_hat": 2.4354694540369994
      },
      {
        "dow": 5,
        "t": "2022-02-19T00:00:00",
        "wom": 3,
        "y": 0.02226637799800346,
        "y_hat": 0.5400703987126106
      },
      {
        "dow": 6,
        "t": "2022-02-20T00:00:00",
        "wom": 3,
        "y": 0.15725472652557881,
        "y_hat": 0.5456163598479873
      },
      {
        "dow": 0,
        "t": "2022-02-21T00:00:00",
        "wom": 4,
        "y": 0.9269915515225546,
        "y_hat": 0.43170828507018294
      },
      {
        "dow": 1,
        "t": "2022-02-22T00:00:00",
        "wom": 4,
        "y": 0.4501913471576654,
        "y_hat": 0.4954975240418244
      },
      {
        "dow": 2,
        "t": "2022-02-23T00:00:00",
        "wom": 4,
        "y": 0.9531441262485346,
        "y_hat": 0.48823066431783424
      },
      {
        "dow": 3,
        "t": "2022-02-24T00:00:00",
        "wom": 4,
        "y": 0.5844501255113494,
        "y_hat": 0.519519472671327
      },
      {
        "dow": 4,
        "t": "2022-02-25T00:00:00",
        "wom": 4,
        "y": 2.80397776091471,
        "y_hat": 2.458308069387769
      },
      {
        "dow": 5,
        "t": "2022-02-26T00:00:00",
        "wom": 4,
        "y": 0.26383392179603904,
        "y_hat": 0.542195800807761
      },
      {
        "dow": 6,
        "t": "2022-02-27T00:00:00",
        "wom": 4,
        "y": 0.7531464817944895,
        "y_hat": 0.5516416906073818
      },
      {
        "dow": 0,
        "t": "2022-02-28T00:00:00",
        "wom": 5,
        "y": 0.15885513834885823,
        "y_hat": 0.4618899359780873
      },
      {
        "dow": 1,
        "t": "2022-03-01T00:00:00",
        "wom": 1,
        "y": 0.14535499394438622,
        "y_hat": 0.5058303574877435
      },
      {
        "dow": 2,
        "t": "2022-03-02T00:00:00",
        "wom": 1,
        "y": 0.4410675050392091,
        "y_hat": 0.43816256596576625
      },
      {
        "dow": 3,
        "t": "2022-03-03T00:00:00",
        "wom": 1,
        "y": 0.7517726853328405,
        "y_hat": 0.48854162038072846
      },
      {
        "dow": 4,
        "t": "2022-03-04T00:00:00",
        "wom": 1,
        "y": 2.7945001756665624,
        "y_hat": 2.435556070318829
      },
      {
        "dow": 5,
        "t": "2022-03-05T00:00:00",
        "wom": 1,
        "y": 0.04870630413527299,
        "y_hat": 0.5510704584107008
      },
      {
        "dow": 6,
        "t": "2022-03-06T00:00:00",
        "wom": 1,
        "y": 0.2161041338955525,
        "y_hat": 0.5450948686636378
      },
      {
        "dow": 0,
        "t": "2022-03-07T00:00:00",
        "wom": 2,
        "y": 0.37010898460177155,
        "y_hat": 0.435041792407649
      },
      {
        "dow": 1,
        "t": "2022-03-08T00:00:00",
        "wom": 2,
        "y": 0.09520299889511219,
        "y_hat": 0.48298588836607886
      },
      {
        "dow": 2,
        "t": "2022-03-09T00:00:00",
        "wom": 2,
        "y": 0.11977796440076782,
        "y_hat": 0.44824665641173894
      },
      {
        "dow": 3,
        "t": "2022-03-10T00:00:00",
        "wom": 2,
        "y": 0.5000495405840498,
        "y_hat": 0.47676527608495756
      },
      {
        "dow": 4,
        "t": "2022-03-11T00:00:00",
        "wom": 2,
        "y": 2.453037332781335,
        "y_hat": 2.411129683739495
      },
      {
        "dow": 5,
        "t": "2022-03-12T00:00:00",
        "wom": 2,
        "y": 0.1454828110192531,
        "y_hat": 0.5278354061996601
      },
      {
        "dow": 6,
        "t": "2022-03-13T00:00:00",
        "wom": 2,
        "y": 0.37079700609072885,
        "y_hat": 0.5295379288986366
      },
      {
        "dow": 0,
        "t": "2022-03-14T00:00:00",
        "wom": 3,
        "y": 0.6368565984388843,
        "y_hat": 0.4447991215598311
      },
      {
        "dow": 1,
        "t": "2022-03-15T00:00:00",
        "wom": 3,
        "y": 0.5235082648923333,
        "y_hat": 0.4990719735426933
      },
      {
        "dow": 2,
        "t": "2022-03-16T00:00:00",
        "wom": 3,
        "y": 0.7192823846677754,
        "y_hat": 0.4749988421987305
      },
      {
        "dow": 3,
        "t": "2022-03-17T00:00:00",
        "wom": 3,
        "y": 0.6045077428498761,
        "y_hat": 0.5171315926773352
      },
      {
        "dow": 4,
        "t": "2022-03-18T00:00:00",
        "wom": 3,
        "y": 2.5404792929526683,
        "y_hat": 2.446592365808845
      },
      {
        "dow": 5,
        "t": "2022-03-19T00:00:00",
        "wom": 3,
        "y": 0.4830778428824394,
        "y_hat": 0.5360857207829948
      },
      {
        "dow": 6,
        "t": "2022-03-20T00:00:00",
        "wom": 3,
        "y": 0.9022692634866183,
        "y_hat": 0.5440178300153817
      },
      {
        "dow": 0,
        "t": "2022-03-21T00:00:00",
        "wom": 4,
        "y": 0.06386988650342573,
        "y_hat": 0.4783242019394053
      },
      {
        "dow": 1,
        "t": "2022-03-22T00:00:00",
        "wom": 4,
        "y": 0.2631672977407843,
        "y_hat": 0.5115034804148847
      },
      {
        "dow": 2,
        "t": "2022-03-23T00:00:00",
        "wom": 4,
        "y": 0.6967761906616411,
        "y_hat": 0.43673779150020503
      },
      {
        "dow": 3,
        "t": "2022-03-24T00:00:00",
        "wom": 4,
        "y": 0.6938964314225743,
        "y_hat": 0.5025456288446649
      },
      {
        "dow": 4,
        "t": "2022-03-25T00:00:00",
        "wom": 4,
        "y": 2.3402313329841156,
        "y_hat": 2.448043866842587
      },
      {
        "dow": 5,
        "t": "2022-03-26T00:00:00",
        "wom": 4,
        "y": 0.060442940833932424,
        "y_hat": 0.5353015977298601
      },
      {
        "dow": 6,
        "t": "2022-03-27T00:00:00",
        "wom": 4,
        "y": 0.8673761163808922,
        "y_hat": 0.5212422441741162
      },
      {
        "dow": 0,
        "t": "2022-03-28T00:00:00",
        "wom": 5,
        "y": 0.5544851701436769,
        "y_hat": 0.4546224926233655
      },
      {
        "dow": 1,
        "t": "2022-03-29T00:00:00",
        "wom": 5,
        "y": 0.8547133638306371,
        "y_hat": 0.5237514886317916
      },
      {
        "dow": 2,
        "t": "2022-03-30T00:00:00",
        "wom": 5,
        "y": 0.36375693048623126,
        "y_hat": 0.4802406327655775
      },
      {
        "dow": 3,
        "t": "2022-03-31T00:00:00",
        "wom": 5,
        "y": 0.7322858415787377,
        "y_hat": 0.5250775331729335
      },
      {
        "dow": 4,
        "t": "2022-04-01T00:00:00",
        "wom": 1,
        "y": 2.1005642322427573,
        "y_hat": 2.4312743324017627
      },
      {
        "dow": 5,
        "t": "2022-04-02T00:00:00",
        "wom": 1,
        "y": 0.6446246806801427,
        "y_hat": 0.530637749525594
      },
      {
        "dow": 6,
        "t": "2022-04-03T00:00:00",
        "wom": 1,
        "y": 0.523704664544059,
        "y_hat": 0.5251088026683788
      },
      {
        "dow": 0,
        "t": "2022-04-04T00:00:00",
        "wom": 2,
        "y": 0.6478284421164979,
        "y_hat": 0.47645242667294535
      },
      {
        "dow": 1,
        "t": "2022-04-05T00:00:00",
        "wom": 2,
        "y": 0.8035144383488594,
        "y_hat": 0.5079772060871229
      },
      {
        "dow": 2,
        "t": "2022-04-06T00:00:00",
        "wom": 2,
        "y": 0.06175772835110849,
        "y_hat": 0.4839291542047948
      },
      {
        "dow": 3,
        "t": "2022-04-07T00:00:00",
        "wom": 2,
        "y": 0.10101798118758876,
        "y_hat": 0.5137958988269249
      },
      {
        "dow": 4,
        "t": "2022-04-08T00:00:00",
        "wom": 2,
        "y": 2.328695425168449,
        "y_hat": 2.3970437026869673
      },
      {
        "dow": 5,
        "t": "2022-04-09T00:00:00",
        "wom": 2,
        "y": 0.7405205134959488,
        "y_hat": 0.503188594546576
      },
      {
        "dow": 6,
        "t": "2022-04-10T00:00:00",
        "wom": 2,
        "y": 0.08236194536441421,
        "y_hat": 0.5402774394697502
      },
      {
        "dow": 0,
        "t": "2022-04-11T00:00:00",
        "wom": 3,
        "y": 0.48567949453078907,
        "y_hat": 0.4691302067870829
      },
      {
        "dow": 1,
        "t": "2022-04-12T00:00:00",
        "wom": 3,
        "y": 0.5552603671504405,
        "y_hat": 0.4796291198149867
      },
      {
        "dow": 2,
        "t": "2022-04-13T00:00:00",
        "wom": 3,
        "y": 0.2511650942997945,
        "y_hat": 0.4681934405589045
      },
      {
        "dow": 3,
        "t": "2022-04-14T00:00:00",
        "wom": 3,
        "y": 0.4804870698202284,
        "y_hat": 0.5059180037110255
      },
      {
        "dow": 4,
        "t": "2022-04-15T00:00:00",
        "wom": 3,
        "y": 2.0257783736909825,
        "y_hat": 2.4182249938567857
      },
      {
        "dow": 5,
        "t": "2022-04-16T00:00:00",
        "wom": 3,
        "y": 0.9687966319838246,
        "y_hat": 0.5151312584930192
      },
      {
        "dow": 6,
        "t": "2022-04-17T00:00:00",
        "wom": 3,
        "y": 0.7151278400228451,
        "y_hat": 0.5305591407286968
      },
      {
        "dow": 0,
        "t": "2022-04-18T00:00:00",
        "wom": 4,
        "y": 0.4718442940718869,
        "y_hat": 0.49964798819154504
      },
      {
        "dow": 1,
        "t": "2022-04-19T00:00:00",
        "wom": 4,
        "y": 0.9026979968878186,
        "y_hat": 0.5135072021996053
      },
      {
        "dow": 2,
        "t": "2022-04-20T00:00:00",
        "wom": 4,
        "y": 0.39245084658370566,
        "y_hat": 0.47748693630179173
      },
      {
        "dow": 3,
        "t": "2022-04-21T00:00:00",
        "wom": 4,
        "y": 0.4827795920741036,
        "y_hat": 0.5288246903791284
      },
      {
        "dow": 4,
        "t": "2022-04-22T00:00:00",
        "wom": 4,
        "y": 2.551414450198103,
        "y_hat": 2.4260303356637705
      },
      {
        "dow": 5,
        "t": "2022-04-23T00:00:00",
        "wom": 4,
        "y": 0.650135286544272,
        "y_hat": 0.5303875499655886
      },
      {
        "dow": 6,
        "t": "2022-04-24T00:00:00",
        "wom": 4,
        "y": 0.467587186604593,
        "y_hat": 0.549943038786494
      },
      {
        "dow": 0,
        "t": "2022-04-25T00:00:00",
        "wom": 5,
        "y": 0.8420434940525285,
        "y_hat": 0.4754830070083764
      },
      {
        "dow": 1,
        "t": "2022-04-26T00:00:00",
        "wom": 5,
        "y": 0.44245377300469013,
        "y_hat": 0.5108340361542065
      },
      {
        "dow": 2,
        "t": "2022-04-27T00:00:00",
        "wom": 5,
        "y": 0.10472408289231805,
        "y_hat": 0.4844334459505021
      },
      {
        "dow": 3,
        "t": "2022-04-28T00:00:00",
        "wom": 5,
        "y": 0.07965539705381441,
        "y_hat": 0.4958677197277991
      },
      {
        "dow": 4,
        "t": "2022-04-29T00:00:00",
        "wom": 5,
        "y": 2.1204279487951343,
        "y_hat": 2.3990895488892217
      },
      {
        "dow": 5,
        "t": "2022-04-30T00:00:00",
        "wom": 5,
        "y": 0.09911422212585952,
        "y_hat": 0.4964206092595712
      },
      {
        "dow": 6,
        "t": "2022-05-01T00:00:00",
        "wom": 1,
        "y": 0.7653139992562019,
        "y_hat": 0.5110418469407004
      },
      {
        "dow": 0,
        "t": "2022-05-02T00:00:00",
        "wom": 2,
        "y": 0.6928765519708857,
        "y_hat": 0.45435487387021417
      },
      {
        "dow": 1,
        "t": "2022-05-03T00:00:00",
        "wom": 2,
        "y": 0.3293225432083243,
        "y_hat": 0.5227571533370909
      },
      {
        "dow": 2,
        "t": "2022-05-04T00:00:00",
        "wom": 2,
        "y": 0.6649514990871527,
        "y_hat": 0.47325979365624726
      },
      {
        "dow": 3,
        "t": "2022-05-05T00:00:00",
        "wom": 2,
        "y": 0.15101171079274767,
        "y_hat": 0.5058810792932257
      },
      {
        "dow": 4,
        "t": "2022-05-06T00:00:00",
        "wom": 2,
        "y": 2.1798428980656723,
        "y_hat": 2.431485876177647
      },
      {
        "dow": 5,
        "t": "2022-05-07T00:00:00",
        "wom": 2,
        "y": 0.2354835900991824,
        "y_hat": 0.5020829628448384
      },
      {
        "dow": 6,
        "t": "2022-05-08T00:00:00",
        "wom": 2,
        "y": 0.05741790510662548,
        "y_hat": 0.5182585301985556
      },
      {
        "dow": 0,
        "t": "2022-05-09T00:00:00",
        "wom": 3,
        "y": 0.23077326722872582,
        "y_hat": 0.4416008126128215
      },
      {
        "dow": 1,
        "t": "2022-05-10T00:00:00",
        "wom": 3,
        "y": 0.1944400629234596,
        "y_hat": 0.4714472781064954
      },
      {
        "dow": 2,
        "t": "2022-05-11T00:00:00",
        "wom": 3,
        "y": 0.7807354431480255,
        "y_hat": 0.44457005664505445
      },
      {
        "dow": 3,
        "t": "2022-05-12T00:00:00",
        "wom": 3,
        "y": 0.31021821068384303,
        "y_hat": 0.5020199652718704
      },
      {
        "dow": 4,
        "t": "2022-05-13T00:00:00",
        "wom": 3,
        "y": 2.1275853244649534,
        "y_hat": 2.4423983412290537
      },
      {
        "dow": 5,
        "t": "2022-05-14T00:00:00",
        "wom": 3,
        "y": 0.56813301154376,
        "y_hat": 0.5092969989732901
      },
      {
        "dow": 6,
        "t": "2022-05-15T00:00:00",
        "wom": 3,
        "y": 0.10167470187610927,
        "y_hat": 0.5250551324538184
      },
      {
        "dow": 0,
        "t": "2022-05-16T00:00:00",
        "wom": 4,
        "y": 0.37826726087578866,
        "y_hat": 0.46093647736965676
      },
      {
        "dow": 1,
        "t": "2022-05-17T00:00:00",
        "wom": 4,
        "y": 0.7138146434013722,
        "y_hat": 0.4781635078824496
      },
      {
        "dow": 2,
        "t": "2022-05-18T00:00:00",
        "wom": 4,
        "y": 0.12852670113262654,
        "y_hat": 0.46748643229002546
      },
      {
        "dow": 3,
        "t": "2022-05-19T00:00:00",
        "wom": 4,
        "y": 0.3585977195944233,
        "y_hat": 0.5115345457435468
      },
      {
        "dow": 4,
        "t": "2022-05-20T00:00:00",
        "wom": 4,
        "y": 2.22258022407796,
        "y_hat": 2.408682157424716
      },
      {
        "dow": 5,
        "t": "2022-05-21T00:00:00",
        "wom": 4,
        "y": 0.932770577180065,
        "y_hat": 0.5147356152220035
      },
      {
        "dow": 6,
        "t": "2022-05-22T00:00:00",
        "wom": 4,
        "y": 0.9336405466022045,
        "y_hat": 0.5407155484496845
      },
      {
        "dow": 0,
        "t": "2022-05-23T00:00:00",
        "wom": 5,
        "y": 0.7765773218651726,
        "y_hat": 0.5045082552262019
      },
      {
        "dow": 1,
        "t": "2022-05-24T00:00:00",
        "wom": 5,
        "y": 0.9092507321087694,
        "y_hat": 0.5345730713865291
      },
      {
        "dow": 2,
        "t": "2022-05-25T00:00:00",
        "wom": 5,
        "y": 0.9912972693958269,
        "y_hat": 0.4946869470039545
      },
      {
        "dow": 3,
        "t": "2022-05-26T00:00:00",
        "wom": 5,
        "y": 0.17273741694999156,
        "y_hat": 0.5468518827619808
      },
      {
        "dow": 4,
        "t": "2022-05-27T00:00:00",
        "wom": 5,
        "y": 2.7305657647776744,
        "y_hat": 2.4500618298869807
      },
      {
        "dow": 5,
        "t": "2022-05-28T00:00:00",
        "wom": 5,
        "y": 0.9331495324820412,
        "y_hat": 0.5193297047723543
      },
      {
        "dow": 6,
        "t": "2022-05-29T00:00:00",
        "wom": 5,
        "y": 0.9747799262042263,
        "y_hat": 0.5682632750846622
      },
      {
        "dow": 0,
        "t": "2022-05-30T00:00:00",
        "wom": 6,
        "y": 0.6159827699504155,
        "y_hat": 0.5058165123564484
      },
      {
        "dow": 1,
        "t": "2022-05-31T00:00:00",
        "wom": 6,
        "y": 0.18957160929478611,
        "y_hat": 0.5323172222676634
      },
      {
        "dow": 2,
        "t": "2022-06-01T00:00:00",
        "wom": 1,
        "y": 0.08329858559961756,
        "y_hat": 0.4655639528916069
      },
      {
        "dow": 3,
        "t": "2022-06-02T00:00:00",
        "wom": 1,
        "y": 0.7811845717468601,
        "y_hat": 0.48216420003370863
      },
      {
        "dow": 4,
        "t": "2022-06-03T00:00:00",
        "wom": 1,
        "y": 2.5856317189042306,
        "y_hat": 2.41854048611295
      },
      {
        "dow": 5,
        "t": "2022-06-04T00:00:00",
        "wom": 1,
        "y": 0.9915634439508605,
        "y_hat": 0.5481486757761187
      },
      {
        "dow": 6,
        "t": "2022-06-05T00:00:00",
        "wom": 1,
        "y": 0.9909525874581544,
        "y_hat": 0.5622201296718374
      },
      {
        "dow": 0,
        "t": "2022-06-06T00:00:00",
        "wom": 2,
        "y": 0.546656872738867,
        "y_hat": 0.509544473702861
      },
      {
        "dow": 1,
        "t": "2022-06-07T00:00:00",
        "wom": 2,
        "y": 0.10472218993110805,
        "y_hat": 0.5313215376478715
      },
      {
        "dow": 2,
        "t": "2022-06-08T00:00:00",
        "wom": 2,
        "y": 0.04250860920798383,
        "y_hat": 0.45950885511347206
      },
      {
        "dow": 3,
        "t": "2022-06-09T00:00:00",
        "wom": 2,
        "y": 0.3032781068949767,
        "y_hat": 0.4765300517785524
      },
      {
        "dow": 4,
        "t": "2022-06-10T00:00:00",
        "wom": 2,
        "y": 2.5924710028233178,
        "y_hat": 2.4027941662417835
      },
      {
        "dow": 5,
        "t": "2022-06-11T00:00:00",
        "wom": 2,
        "y": 0.1823450459040853,
        "y_hat": 0.5226555382078407
      },
      {
        "dow": 6,
        "t": "2022-06-12T00:00:00",
        "wom": 2,
        "y": 0.7714328053698969,
        "y_hat": 0.5395777784515343
      },
      {
        "dow": 0,
        "t": "2022-06-13T00:00:00",
        "wom": 3,
        "y": 0.7117102710960721,
        "y_hat": 0.4596964393488171
      },
      {
        "dow": 1,
        "t": "2022-06-14T00:00:00",
        "wom": 3,
        "y": 0.5078428447537274,
        "y_hat": 0.5242989977320872
      },
      {
        "dow": 2,
        "t": "2022-06-15T00:00:00",
        "wom": 3,
        "y": 0.4559806484621759,
        "y_hat": 0.48005120057372996
      },
      {
        "dow": 3,
        "t": "2022-06-16T00:00:00",
        "wom": 3,
        "y": 0.3663298759696507,
        "y_hat": 0.5102207588634888
      },
      {
        "dow": 4,
        "t": "2022-06-17T00:00:00",
        "wom": 3,
        "y": 2.036192987755432,
        "y_hat": 2.427030286845141
      },
      {
        "dow": 5,
        "t": "2022-06-18T00:00:00",
        "wom": 3,
        "y": 0.5857499825271564,
        "y_hat": 0.5102758304570155
      },
      {
        "dow": 6,
        "t": "2022-06-19T00:00:00",
        "wom": 3,
        "y": 0.050005294335790484,
        "y_hat": 0.5211856409518846
      },
      {
        "dow": 0,
        "t": "2022-06-20T00:00:00",
        "wom": 4,
        "y": 0.5061272547669947,
        "y_hat": 0.46097219295144776
      },
      {
        "dow": 1,
        "t": "2022-06-21T00:00:00",
        "wom": 4,
        "y": 0.7904390381333787,
        "y_hat": 0.4795887904276564
      },
      {
        "dow": 2,
        "t": "2022-06-22T00:00:00",
        "wom": 4,
        "y": 0.09323256170059835,
        "y_hat": 0.4771399084781224
      },
      {
        "dow": 3,
        "t": "2022-06-23T00:00:00",
        "wom": 4,
        "y": 0.40800965871918415,
        "y_hat": 0.5152238334334927
      },
      {
        "dow": 4,
        "t": "2022-06-24T00:00:00",
        "wom": 4,
        "y": 2.570342151877391,
        "y_hat": 2.4087498026209566
      },
      {
        "dow": 5,
        "t": "2022-06-25T00:00:00",
        "wom": 4,
        "y": 0.07936746817059592,
        "y_hat": 0.5279017923132528
      },
      {
        "dow": 6,
        "t": "2022-06-26T00:00:00",
        "wom": 4,
        "y": 0.017413769472798823,
        "y_hat": 0.5356649133238703
      },
      {
        "dow": 0,
        "t": "2022-06-27T00:00:00",
        "wom": 5,
        "y": 0.4856034209379412,
        "y_hat": 0.4328149611023629
      },
      {
        "dow": 1,
        "t": "2022-06-28T00:00:00",
        "wom": 5,
        "y": 0.10626439009089561,
        "y_hat": 0.47735501023779975
      },
      {
        "dow": 2,
        "t": "2022-06-29T00:00:00",
        "wom": 5,
        "y": 0.7531865011385493,
        "y_hat": 0.4565934866610085
      },
      {
        "dow": 3,
        "t": "2022-06-30T00:00:00",
        "wom": 5,
        "y": 0.2215890993417199,
        "y_hat": 0.4972579361761011
      },
      {
        "dow": 4,
        "t": "2022-07-01T00:00:00",
        "wom": 1,
        "y": 2.2444907708382646,
        "y_hat": 2.4391639795535345
      },
      {
        "dow": 5,
        "t": "2022-07-02T00:00:00",
        "wom": 1,
        "y": 0.41066189647640805,
        "y_hat": 0.5086383437322588
      },
      {
        "dow": 6,
        "t": "2022-07-03T00:00:00",
        "wom": 1,
        "y": 0.8875209104102515,
        "y_hat": 0.5276522539533361
      },
      {
        "dow": 0,
        "t": "2022-07-04T00:00:00",
        "wom": 2,
        "y": 0.29879768250458105,
        "y_hat": 0.4756765448644015
      },
      {
        "dow": 1,
        "t": "2022-07-05T00:00:00",
        "wom": 2,
        "y": 0.8642111944128015,
        "y_hat": 0.519103746855454
      },
      {
        "dow": 2,
        "t": "2022-07-06T00:00:00",
        "wom": 2,
        "y": 0.5035308154420571,
        "y_hat": 0.468279298304797
      },
      {
        "dow": 3,
        "t": "2022-07-07T00:00:00",
        "wom": 2,
        "y": 0.4608669940295842,
        "y_hat": 0.5311554916042507
      },
      {
        "dow": 4,
        "t": "2022-07-08T00:00:00",
        "wom": 2,
        "y": 2.1336407534117914,
        "y_hat": 2.43263548967435
      },
      {
        "dow": 5,
        "t": "2022-07-09T00:00:00",
        "wom": 2,
        "y": 0.07559350276009646,
        "y_hat": 0.5185008966658524
      },
      {
        "dow": 6,
        "t": "2022-07-10T00:00:00",
        "wom": 2,
        "y": 0.21152592170248208,
        "y_hat": 0.5122052608058499
      },
      {
        "dow": 0,
        "t": "2022-07-11T00:00:00",
        "wom": 3,
        "y": 0.7992198246512687,
        "y_hat": 0.4383825054702067
      },
      {
        "dow": 1,
        "t": "2022-07-12T00:00:00",
        "wom": 3,
        "y": 0.08134234250438865,
        "y_hat": 0.49702074794504336
      },
      {
        "dow": 2,
        "t": "2022-07-13T00:00:00",
        "wom": 3,
        "y": 0.8545474044325597,
        "y_hat": 0.4730370156455408
      },
      {
        "dow": 3,
        "t": "2022-07-14T00:00:00",
        "wom": 3,
        "y": 0.7192996694052958,
        "y_hat": 0.49903334487542134
      },
      {
        "dow": 4,
        "t": "2022-07-15T00:00:00",
        "wom": 3,
        "y": 2.839731861280379,
        "y_hat": 2.4590829237763088
      },
      {
        "dow": 5,
        "t": "2022-07-16T00:00:00",
        "wom": 3,
        "y": 0.6866567952654717,
        "y_hat": 0.5527420470316485
      },
      {
        "dow": 6,
        "t": "2022-07-17T00:00:00",
        "wom": 3,
        "y": 0.3025925308532783,
        "y_hat": 0.5678987356529254
      },
      {
        "dow": 0,
        "t": "2022-07-18T00:00:00",
        "wom": 4,
        "y": 0.35527032358475563,
        "y_hat": 0.4740865465555972
      },
      {
        "dow": 1,
        "t": "2022-07-19T00:00:00",
        "wom": 4,
        "y": 0.2998867051787012,
        "y_hat": 0.4893633221283946
      },
      {
        "dow": 2,
        "t": "2022-07-20T00:00:00",
        "wom": 4,
        "y": 0.32169056052383393,
        "y_hat": 0.4554266913463155
      },
      {
        "dow": 3,
        "t": "2022-07-21T00:00:00",
        "wom": 4,
        "y": 0.8817188595233318,
        "y_hat": 0.49571750586434926
      },
      {
        "dow": 4,
        "t": "2022-07-22T00:00:00",
        "wom": 4,
        "y": 2.6482046952131784,
        "y_hat": 2.4350689441557574
      },
      {
        "dow": 5,
        "t": "2022-07-23T00:00:00",
        "wom": 4,
        "y": 0.5392997474632014,
        "y_hat": 0.5561498314017669
      },
      {
        "dow": 6,
        "t": "2022-07-24T00:00:00",
        "wom": 4,
        "y": 0.28113347961902946,
        "y_hat": 0.5534601575845951
      },
      {
        "dow": 0,
        "t": "2022-07-25T00:00:00",
        "wom": 5,
        "y": 0.4047240553693535,
        "y_hat": 0.4656301818804722
      },
      {
        "dow": 1,
        "t": "2022-07-26T00:00:00",
        "wom": 5,
        "y": 0.09656379233844459,
        "y_hat": 0.4897301947692827
      },
      {
        "dow": 2,
        "t": "2022-07-27T00:00:00",
        "wom": 5,
        "y": 0.9545109848618508,
        "y_hat": 0.45239872668182046
      },
      {
        "dow": 3,
        "t": "2022-07-28T00:00:00",
        "wom": 5,
        "y": 0.2844751590965856,
        "y_hat": 0.5029360562621922
      },
      {
        "dow": 4,
        "t": "2022-07-29T00:00:00",
        "wom": 5,
        "y": 2.19828814692932,
        "y_hat": 2.4522787916765902
      },
      {
        "dow": 5,
        "t": "2022-07-30T00:00:00",
        "wom": 5,
        "y": 0.7872041161790215,
        "y_hat": 0.5111620022247799
      },
      {
        "dow": 6,
        "t": "2022-07-31T00:00:00",
        "wom": 5,
        "y": 0.8320061228006757,
        "y_hat": 0.5363666658917704
      },
      {
        "dow": 0,
        "t": "2022-08-01T00:00:00",
        "wom": 1,
        "y": 0.48050923278633095,
        "y_hat": 0.49486749692129434
      },
      {
        "dow": 1,
        "t": "2022-08-02T00:00:00",
        "wom": 1,
        "y": 0.9186312133845865,
        "y_hat": 0.5217480417049952
      },
      {
        "dow": 2,
        "t": "2022-08-03T00:00:00",
        "wom": 1,
        "y": 0.28383987157951585,
        "y_hat": 0.4800933434215706
      },
      {
        "dow": 3,
        "t": "2022-08-04T00:00:00",
        "wom": 1,
        "y": 0.6963665748778668,
        "y_hat": 0.5282645806595277
      },
      {
        "dow": 4,
        "t": "2022-08-05T00:00:00",
        "wom": 1,
        "y": 2.09245143951022,
        "y_hat": 2.427953405345737
      },
      {
        "dow": 5,
        "t": "2022-08-06T00:00:00",
        "wom": 1,
        "y": 0.7172188720261602,
        "y_hat": 0.5304867199912241
      },
      {
        "dow": 6,
        "t": "2022-08-07T00:00:00",
        "wom": 1,
        "y": 0.6862602995181284,
        "y_hat": 0.5287652055403764
      },
      {
        "dow": 0,
        "t": "2022-08-08T00:00:00",
        "wom": 2,
        "y": 0.12495616249592223,
        "y_hat": 0.4870366588398229
      },
      {
        "dow": 1,
        "t": "2022-08-09T00:00:00",
        "wom": 2,
        "y": 0.7803068072936611,
        "y_hat": 0.5038317649617358
      },
      {
        "dow": 2,
        "t": "2022-08-10T00:00:00",
        "wom": 2,
        "y": 0.1711236164496699,
        "y_hat": 0.45705759106594746
      },
      {
        "dow": 3,
        "t": "2022-08-11T00:00:00",
        "wom": 2,
        "y": 0.6418292052213659,
        "y_hat": 0.5176881437709107
      },
      {
        "dow": 4,
        "t": "2022-08-12T00:00:00",
        "wom": 2,
        "y": 2.398223140821866,
        "y_hat": 2.420421968469209
      },
      {
        "dow": 5,
        "t": "2022-08-13T00:00:00",
        "wom": 2,
        "y": 0.7667020108493909,
        "y_hat": 0.5363921988276147
      },
      {
        "dow": 6,
        "t": "2022-08-14T00:00:00",
        "wom": 2,
        "y": 0.41815335304653145,
        "y_hat": 0.5467991648029131
      },
      {
        "dow": 0,
        "t": "2022-08-15T00:00:00",
        "wom": 3,
        "y": 0.6891797613670014,
        "y_hat": 0.48215904682588656
      },
      {
        "dow": 1,
        "t": "2022-08-16T00:00:00",
        "wom": 3,
        "y": 0.5101490274593742,
        "y_hat": 0.5055926245703029
      },
      {
        "dow": 2,
        "t": "2022-08-17T00:00:00",
        "wom": 3,
        "y": 0.7796622447746596,
        "y_hat": 0.47991073110965105
      },
      {
        "dow": 3,
        "t": "2022-08-18T00:00:00",
        "wom": 3,
        "y": 0.5627361856480227,
        "y_hat": 0.5206046080919271
      },
      {
        "dow": 4,
        "t": "2022-08-19T00:00:00",
        "wom": 3,
        "y": 2.149269114111338,
        "y_hat": 2.4511273168906147
      },
      {
        "dow": 5,
        "t": "2022-08-20T00:00:00",
        "wom": 3,
        "y": 0.4028888263767394,
        "y_hat": 0.5251204340687972
      },
      {
        "dow": 6,
        "t": "2022-08-21T00:00:00",
        "wom": 3,
        "y": 0.29581445643538484,
        "y_hat": 0.5230749375317849
      },
      {
        "dow": 0,
        "t": "2022-08-22T00:00:00",
        "wom": 4,
        "y": 0.22454795399899974,
        "y_hat": 0.4591341483141852
      },
      {
        "dow": 1,
        "t": "2022-08-23T00:00:00",
        "wom": 4,
        "y": 0.5368502058660074,
        "y_hat": 0.48582304298727946
      },
      {
        "dow": 2,
        "t": "2022-08-24T00:00:00",
        "wom": 4,
        "y": 0.8352731932733497,
        "y_hat": 0.455701817281319
      },
      {
        "dow": 3,
        "t": "2022-08-25T00:00:00",
        "wom": 4,
        "y": 0.7156495255143432,
        "y_hat": 0.5237474583198937
      },
      {
        "dow": 4,
        "t": "2022-08-26T00:00:00",
        "wom": 4,
        "y": 2.081170707633651,
        "y_hat": 2.4586113850595304
      },
      {
        "dow": 5,
        "t": "2022-08-27T00:00:00",
        "wom": 4,
        "y": 0.9846573243966786,
        "y_hat": 0.5315420611782877
      },
      {
        "dow": 6,
        "t": "2022-08-28T00:00:00",
        "wom": 4,
        "y": 0.11556538882109113,
        "y_hat": 0.5361350363720931
      },
      {
        "dow": 0,
        "t": "2022-08-29T00:00:00",
        "wom": 5,
        "y": 0.26133230623941184,
        "y_hat": 0.4855036716924154
      },
      {
        "dow": 1,
        "t": "2022-08-30T00:00:00",
        "wom": 5,
        "y": 0.9006219041976188,
        "y_hat": 0.47725526614685454
      },
      {
        "dow": 2,
        "t": "2022-08-31T00:00:00",
        "wom": 5,
        "y": 0.2515146218161567,
        "y_hat": 0.4681948708253787
      },
      {
        "dow": 3,
        "t": "2022-09-01T00:00:00",
        "wom": 1,
        "y": 0.13098484463014348,
        "y_hat": 0.5268174837110191
      },
      {
        "dow": 4,
        "t": "2022-09-02T00:00:00",
        "wom": 1,
        "y": 2.9173298508007353,
        "y_hat": 2.410501086412229
      },
      {
        "dow": 5,
        "t": "2022-09-03T00:00:00",
        "wom": 1,
        "y": 0.902033118492389,
        "y_hat": 0.5239835691087095
      },
      {
        "dow": 6,
        "t": "2022-09-04T00:00:00",
        "wom": 1,
        "y": 0.6094812461745855,
        "y_hat": 0.579028055768711
      },
      {
        "dow": 0,
        "t": "2022-09-05T00:00:00",
        "wom": 2,
        "y": 0.21237629900528165,
        "y_hat": 0.4952691329353708
      },
      {
        "dow": 1,
        "t": "2022-09-06T00:00:00",
        "wom": 2,
        "y": 0.8674613314753005,
        "y_hat": 0.5026336407752172
      },
      {
        "dow": 2,
        "t": "2022-09-07T00:00:00",
        "wom": 2,
        "y": 0.7318690107796962,
        "y_hat": 0.46471651844447337
      },
      {
        "dow": 3,
        "t": "2022-09-08T00:00:00",
        "wom": 2,
        "y": 0.9273400869945371,
        "y_hat": 0.538865818625751
      },
      {
        "dow": 4,
        "t": "2022-09-09T00:00:00",
        "wom": 2,
        "y": 2.491296824471812,
        "y_hat": 2.4593024923288445
      },
      {
        "dow": 5,
        "t": "2022-09-10T00:00:00",
        "wom": 2,
        "y": 0.3736267511181386,
        "y_hat": 0.5549146181128354
      },
      {
        "dow": 6,
        "t": "2022-09-11T00:00:00",
        "wom": 2,
        "y": 0.6741226397964637,
        "y_hat": 0.5410405136378633
      },
      {
        "dow": 0,
        "t": "2022-09-12T00:00:00",
        "wom": 3,
        "y": 0.5690598209381145,
        "y_hat": 0.468701374121668
      },
      {
        "dow": 1,
        "t": "2022-09-13T00:00:00",
        "wom": 3,
        "y": 0.5845921058651229,
        "y_hat": 0.5164280677357482
      },
      {
        "dow": 2,
        "t": "2022-09-14T00:00:00",
        "wom": 3,
        "y": 0.06190141469816057,
        "y_hat": 0.47600193151568615
      },
      {
        "dow": 3,
        "t": "2022-09-15T00:00:00",
        "wom": 3,
        "y": 0.7726353296869438,
        "y_hat": 0.504562097693305
      },
      {
        "dow": 4,
        "t": "2022-09-16T00:00:00",
        "wom": 3,
        "y": 2.3982766116278027,
        "y_hat": 2.418824328436912
      },
      {
        "dow": 5,
        "t": "2022-09-17T00:00:00",
        "wom": 3,
        "y": 0.02485301435952192,
        "y_hat": 0.5440166771084493
      },
      {
        "dow": 6,
        "t": "2022-09-18T00:00:00",
        "wom": 3,
        "y": 0.3885550460991051,
        "y_hat": 0.5261644635357546
      },
      {
        "dow": 0,
        "t": "2022-09-19T00:00:00",
        "wom": 4,
        "y": 0.5124897271631259,
        "y_hat": 0.44182407349217223
      },
      {
        "dow": 1,
        "t": "2022-09-20T00:00:00",
        "wom": 4,
        "y": 0.6717284818449774,
        "y_hat": 0.499506798622325
      },
      {
        "dow": 2,
        "t": "2022-09-21T00:00:00",
        "wom": 4,
        "y": 0.8010250603087118,
        "y_hat": 0.47555005974701947
      },
      {
        "dow": 3,
        "t": "2022-09-22T00:00:00",
        "wom": 4,
        "y": 0.7215828184438076,
        "y_hat": 0.5304993746500212
      },
      {
        "dow": 4,
        "t": "2022-09-23T00:00:00",
        "wom": 4,
        "y": 2.467671973989098,
        "y_hat": 2.457380857040162
      },
      {
        "dow": 5,
        "t": "2022-09-24T00:00:00",
        "wom": 4,
        "y": 0.7177366636762655,
        "y_hat": 0.5433557257312887
      },
      {
        "dow": 6,
        "t": "2022-09-25T00:00:00",
        "wom": 4,
        "y": 0.48856096479123934,
        "y_hat": 0.5498226240838627
      },
      {
        "dow": 0,
        "t": "2022-09-26T00:00:00",
        "wom": 5,
        "y": 0.5960952391022855,
        "y_hat": 0.4822008051720972
      },
      {
        "dow": 1,
        "t": "2022-09-27T00:00:00",
        "wom": 5,
        "y": 0.8681440058433928,
        "y_hat": 0.5074071188774416
      },
      {
        "dow": 2,
        "t": "2022-09-28T00:00:00",
        "wom": 5,
        "y": 0.8669839542345549,
        "y_hat": 0.48578856966099204
      },
      {
        "dow": 3,
        "t": "2022-09-29T00:00:00",
        "wom": 5,
        "y": 0.8884123252558981,
        "y_hat": 0.5431003113470626
      },
      {
        "dow": 4,
        "t": "2022-09-30T00:00:00",
        "wom": 5,
        "y": 2.3463008214785965,
        "y_hat": 2.465821242163437
      },
      {
        "dow": 5,
        "t": "2022-10-01T00:00:00",
        "wom": 1,
        "y": 0.2954617156792265,
        "y_hat": 0.5490063522668106
      },
      {
        "dow": 6,
        "t": "2022-10-02T00:00:00",
        "wom": 1,
        "y": 0.633109100839611,
        "y_hat": 0.5313156558678173
      },
      {
        "dow": 0,
        "t": "2022-10-03T00:00:00",
        "wom": 2,
        "y": 0.3669341564363222,
        "y_hat": 0.46364614979584606
      },
      {
        "dow": 1,
        "t": "2022-10-04T00:00:00",
        "wom": 2,
        "y": 0.24876598531702476,
        "y_hat": 0.5087746891999738
      },
      {
        "dow": 2,
        "t": "2022-10-05T00:00:00",
        "wom": 2,
        "y": 0.16587282213408583,
        "y_hat": 0.4558300182291027
      },
      {
        "dow": 3,
        "t": "2022-10-06T00:00:00",
        "wom": 2,
        "y": 0.9919913685200391,
        "y_hat": 0.48973950777965536
      },
      {
        "dow": 4,
        "t": "2022-10-07T00:00:00",
        "wom": 2,
        "y": 2.3533296873989085,
        "y_hat": 2.4310423123676665
      },
      {
        "dow": 5,
        "t": "2022-10-08T00:00:00",
        "wom": 2,
        "y": 0.7213297581947791,
        "y_hat": 0.554911360704057
      },
      {
        "dow": 6,
        "t": "2022-10-09T00:00:00",
        "wom": 2,
        "y": 0.7224275916390493,
        "y_hat": 0.5439767292961218
      },
      {
        "dow": 0,
        "t": "2022-10-10T00:00:00",
        "wom": 3,
        "y": 0.29155319532221713,
        "y_hat": 0.4893020791489958
      },
      {
        "dow": 1,
        "t": "2022-10-11T00:00:00",
        "wom": 3,
        "y": 0.36991001906594234,
        "y_hat": 0.5115549109796871
      },
      {
        "dow": 2,
        "t": "2022-10-12T00:00:00",
        "wom": 3,
        "y": 0.28417739594797153,
        "y_hat": 0.4553343870788139
      },
      {
        "dow": 3,
        "t": "2022-10-13T00:00:00",
        "wom": 3,
        "y": 0.19840529083955494,
        "y_hat": 0.4997725564131675
      },
      {
        "dow": 4,
        "t": "2022-10-14T00:00:00",
        "wom": 3,
        "y": 2.4606866711518958,
        "y_hat": 2.4148643523441606
      },
      {
        "dow": 5,
        "t": "2022-10-15T00:00:00",
        "wom": 3,
        "y": 0.6335089949594493,
        "y_hat": 0.5152479312806922
      },
      {
        "dow": 6,
        "t": "2022-10-16T00:00:00",
        "wom": 3,
        "y": 0.5680543096824044,
        "y_hat": 0.5473753240467025
      },
      {
        "dow": 0,
        "t": "2022-10-17T00:00:00",
        "wom": 4,
        "y": 0.016790840072704594,
        "y_hat": 0.48026181895173703
      },
      {
        "dow": 1,
        "t": "2022-10-18T00:00:00",
        "wom": 4,
        "y": 0.15482355482083832,
        "y_hat": 0.4954814961576285
      },
      {
        "dow": 2,
        "t": "2022-10-19T00:00:00",
        "wom": 4,
        "y": 0.32294713218572024,
        "y_hat": 0.43446678008048273
      },
      {
        "dow": 3,
        "t": "2022-10-20T00:00:00",
        "wom": 4,
        "y": 0.09723518582862622,
        "y_hat": 0.48938074212176336
      },
      {
        "dow": 4,
        "t": "2022-10-21T00:00:00",
        "wom": 4,
        "y": 2.161951112634909,
        "y_hat": 2.4141786551814923
      },
      {
        "dow": 5,
        "t": "2022-10-22T00:00:00",
        "wom": 4,
        "y": 0.9308281099209696,
        "y_hat": 0.5013617569960774
      },
      {
        "dow": 6,
        "t": "2022-10-23T00:00:00",
        "wom": 4,
        "y": 0.3700513293335844,
        "y_hat": 0.5398557089374105
      },
      {
        "dow": 0,
        "t": "2022-10-24T00:00:00",
        "wom": 5,
        "y": 0.33730301215287395,
        "y_hat": 0.4907674124047824
      },
      {
        "dow": 1,
        "t": "2022-10-25T00:00:00",
        "wom": 5,
        "y": 0.2979813390418188,
        "y_hat": 0.49406289572899337
      },
      {
        "dow": 2,
        "t": "2022-10-26T00:00:00",
        "wom": 5,
        "y": 0.8989252622362512,
        "y_hat": 0.45597333301628634
      },
      {
        "dow": 3,
        "t": "2022-10-27T00:00:00",
        "wom": 5,
        "y": 0.9678609978612648,
        "y_hat": 0.5136804919832462
      },
      {
        "dow": 4,
        "t": "2022-10-28T00:00:00",
        "wom": 5,
        "y": 2.852867990371385,
        "y_hat": 2.47026482888401
      },
      {
        "dow": 5,
        "t": "2022-10-29T00:00:00",
        "wom": 5,
        "y": 0.09421902072623589,
        "y_hat": 0.5682198415030082
      },
      {
        "dow": 6,
        "t": "2022-10-30T00:00:00",
        "wom": 5,
        "y": 0.8581628861015397,
        "y_hat": 0.553362035505034
      },
      {
        "dow": 0,
        "t": "2022-10-31T00:00:00",
        "wom": 6,
        "y": 0.20114187846747755,
        "y_hat": 0.4596615396792977
      },
      {
        "dow": 1,
        "t": "2022-11-01T00:00:00",
        "wom": 1,
        "y": 0.6181376931339703,
        "y_hat": 0.516635984405509
      },
      {
        "dow": 2,
        "t": "2022-11-02T00:00:00",
        "wom": 1,
        "y": 0.19978354259674502,
        "y_hat": 0.4578832281890113
      },
      {
        "dow": 3,
        "t": "2022-11-03T00:00:00",
        "wom": 1,
        "y": 0.1768860986420725,
        "y_hat": 0.5110987214886396
      },
      {
        "dow": 4,
        "t": "2022-11-04T00:00:00",
        "wom": 1,
        "y": 2.4752377813951942,
        "y_hat": 2.4100299124147666
      },
      {
        "dow": 5,
        "t": "2022-11-05T00:00:00",
        "wom": 1,
        "y": 0.9055115735426204,
        "y_hat": 0.5148386395620688
      },
      {
        "dow": 6,
        "t": "2022-11-06T00:00:00",
        "wom": 1,
        "y": 0.6649858316616734,
        "y_hat": 0.5562701637739151
      },
      {
        "dow": 0,
        "t": "2022-11-07T00:00:00",
        "wom": 2,
        "y": 0.5304475702709496,
        "y_hat": 0.4980529825768307
      },
      {
        "dow": 1,
        "t": "2022-11-08T00:00:00",
        "wom": 2,
        "y": 0.1761904542362559,
        "y_hat": 0.5157292008835385
      },
      {
        "dow": 2,
        "t": "2022-11-09T00:00:00",
        "wom": 2,
        "y": 0.9624710144321974,
        "y_hat": 0.46314486020995904
      },
      {
        "dow": 3,
        "t": "2022-11-10T00:00:00",
        "wom": 2,
        "y": 0.7059542876896551,
        "y_hat": 0.5091456448776946
      },
      {
        "dow": 4,
        "t": "2022-11-11T00:00:00",
        "wom": 2,
        "y": 2.506728336965063,
        "y_hat": 2.46643581246614
      },
      {
        "dow": 5,
        "t": "2022-11-12T00:00:00",
        "wom": 2,
        "y": 0.8433091095294604,
        "y_hat": 0.5444135866449329
      },
      {
        "dow": 6,
        "t": "2022-11-13T00:00:00",
        "wom": 2,
        "y": 0.7560702301487369,
        "y_hat": 0.5563050037044083
      },
      {
        "dow": 0,
        "t": "2022-11-14T00:00:00",
        "wom": 3,
        "y": 0.06592714264449118,
        "y_hat": 0.4974098519973051
      },
      {
        "dow": 1,
        "t": "2022-11-15T00:00:00",
        "wom": 3,
        "y": 0.2504072631614749,
        "y_hat": 0.5074849113317751
      },
      {
        "dow": 2,
        "t": "2022-11-16T00:00:00",
        "wom": 3,
        "y": 0.5805948341045363,
        "y_hat": 0.44029974147134476
      },
      {
        "dow": 3,
        "t": "2022-11-17T00:00:00",
        "wom": 3,
        "y": 0.5803663521377741,
        "y_hat": 0.5023523218787471
      },
      {
        "dow": 4,
        "t": "2022-11-18T00:00:00",
        "wom": 3,
        "y": 2.044237322520029,
        "y_hat": 2.44234290195584
      },
      {
        "dow": 5,
        "t": "2022-11-19T00:00:00",
        "wom": 3,
        "y": 0.5560838530176727,
        "y_hat": 0.5245297729912423
      },
      {
        "dow": 6,
        "t": "2022-11-20T00:00:00",
        "wom": 3,
        "y": 0.28756910769465927,
        "y_hat": 0.5232410661077838
      },
      {
        "dow": 0,
        "t": "2022-11-21T00:00:00",
        "wom": 4,
        "y": 0.32658389395269083,
        "y_hat": 0.46862796297499915
      },
      {
        "dow": 1,
        "t": "2022-11-22T00:00:00",
        "wom": 4,
        "y": 0.9255075128439307,
        "y_hat": 0.4897524974527008
      },
      {
        "dow": 2,
        "t": "2022-11-23T00:00:00",
        "wom": 4,
        "y": 0.2351401173462816,
        "y_hat": 0.4737753846011047
      },
      {
        "dow": 3,
        "t": "2022-11-24T00:00:00",
        "wom": 4,
        "y": 0.6755009569121625,
        "y_hat": 0.5290397268854274
      },
      {
        "dow": 4,
        "t": "2022-11-25T00:00:00",
        "wom": 4,
        "y": 2.6532627672211397,
        "y_hat": 2.4265235342689375
      },
      {
        "dow": 5,
        "t": "2022-11-26T00:00:00",
        "wom": 4,
        "y": 0.13334860717528818,
        "y_hat": 0.5471812325695693
      },
      {
        "dow": 6,
        "t": "2022-11-27T00:00:00",
        "wom": 4,
        "y": 0.21224946729617233,
        "y_hat": 0.5441529542842961
      },
      {
        "dow": 0,
        "t": "2022-11-28T00:00:00",
        "wom": 5,
        "y": 0.10485005588135898,
        "y_hat": 0.4437656298775255
      },
      {
        "dow": 1,
        "t": "2022-11-29T00:00:00",
        "wom": 5,
        "y": 0.571653845644994,
        "y_hat": 0.4794622765680875
      },
      {
        "dow": 2,
        "t": "2022-11-30T00:00:00",
        "wom": 5,
        "y": 0.6455551020000664,
        "y_hat": 0.451805294954433
      },
      {
        "dow": 3,
        "t": "2022-12-01T00:00:00",
        "wom": 1,
        "y": 0.07205418156187893,
        "y_hat": 0.5217762126874631
      },
      {
        "dow": 4,
        "t": "2022-12-02T00:00:00",
        "wom": 1,
        "y": 2.8390812934919314,
        "y_hat": 2.4315490965073923
      },
      {
        "dow": 5,
        "t": "2022-12-03T00:00:00",
        "wom": 1,
        "y": 0.3495218764848188,
        "y_hat": 0.5200249530600702
      },
      {
        "dow": 6,
        "t": "2022-12-04T00:00:00",
        "wom": 1,
        "y": 0.9605832472674337,
        "y_hat": 0.5604742376767212
      },
      {
        "dow": 0,
        "t": "2022-12-05T00:00:00",
        "wom": 2,
        "y": 0.47976044149700026,
        "y_hat": 0.4769324212585106
      },
      {
        "dow": 1,
        "t": "2022-12-06T00:00:00",
        "wom": 2,
        "y": 0.6055161236619802,
        "y_hat": 0.5306881286657961
      },
      {
        "dow": 2,
        "t": "2022-12-07T00:00:00",
        "wom": 2,
        "y": 0.41871964951333474,
        "y_hat": 0.47312547912871106
      },
      {
        "dow": 3,
        "t": "2022-12-08T00:00:00",
        "wom": 2,
        "y": 0.966612891389878,
        "y_hat": 0.5172346281782415
      },
      {
        "dow": 4,
        "t": "2022-12-09T00:00:00",
        "wom": 2,
        "y": 2.5654467983822786,
        "y_hat": 2.4449775700971856
      },
      {
        "dow": 5,
        "t": "2022-12-10T00:00:00",
        "wom": 2,
        "y": 0.34560834727361145,
        "y_hat": 0.5606125859661195
      },
      {
        "dow": 6,
        "t": "2022-12-11T00:00:00",
        "wom": 2,
        "y": 0.7570085832854714,
        "y_hat": 0.5457018888866806
      },
      {
        "dow": 0,
        "t": "2022-12-12T00:00:00",
        "wom": 3,
        "y": 0.9652744314834171,
        "y_hat": 0.4710161178959069
      },
      {
        "dow": 1,
        "t": "2022-12-13T00:00:00",
        "wom": 3,
        "y": 0.5648309593161226,
        "y_hat": 0.5336837354630672
      },
      {
        "dow": 2,
        "t": "2022-12-14T00:00:00",
        "wom": 3,
        "y": 0.7595290379916108,
        "y_hat": 0.49828660444260364
      },
      {
        "dow": 3,
        "t": "2022-12-15T00:00:00",
        "wom": 3,
        "y": 0.4539146559253099,
        "y_hat": 0.5248891664811817
      },
      {
        "dow": 4,
        "t": "2022-12-16T00:00:00",
        "wom": 3,
        "y": 2.6626571950380864,
        "y_hat": 2.448838509795448
      },
      {
        "dow": 5,
        "t": "2022-12-17T00:00:00",
        "wom": 3,
        "y": 0.901352733645678,
        "y_hat": 0.5358235389156042
      },
      {
        "dow": 6,
        "t": "2022-12-18T00:00:00",
        "wom": 3,
        "y": 0.1389757185937477,
        "y_hat": 0.5669428825994003
      },
      {
        "dow": 0,
        "t": "2022-12-19T00:00:00",
        "wom": 4,
        "y": 0.523376913529589,
        "y_hat": 0.4834707916350209
      },
      {
        "dow": 1,
        "t": "2022-12-20T00:00:00",
        "wom": 4,
        "y": 0.22793475704817479,
        "y_hat": 0.48780270680743293
      },
      {
        "dow": 2,
        "t": "2022-12-21T00:00:00",
        "wom": 4,
        "y": 0.3918232223846172,
        "y_hat": 0.46491507924850445
      },
      {
        "dow": 3,
        "t": "2022-12-22T00:00:00",
        "wom": 4,
        "y": 0.47576866552495234,
        "y_hat": 0.4963059605100148
      },
      {
        "dow": 4,
        "t": "2022-12-23T00:00:00",
        "wom": 4,
        "y": 2.3131731287518735,
        "y_hat": 2.4297238410504325
      },
      {
        "dow": 5,
        "t": "2022-12-24T00:00:00",
        "wom": 4,
        "y": 0.11547575723384762,
        "y_hat": 0.5271289193762412
      },
      {
        "dow": 6,
        "t": "2022-12-25T00:00:00",
        "wom": 4,
        "y": 0.880622203620008,
        "y_hat": 0.5257306518423508
      },
      {
        "dow": 0,
        "t": "2022-12-26T00:00:00",
        "wom": 5,
        "y": 0.4962845210256144,
        "y_hat": 0.4623486822690522
      },
      {
        "dow": 1,
        "t": "2022-12-27T00:00:00",
        "wom": 5,
        "y": 0.5406198774645495,
        "y_hat": 0.5271801081764019
      },
      {
        "dow": 2,
        "t": "2022-12-28T00:00:00",
        "wom": 5,
        "y": 0.26315431946537116,
        "y_hat": 0.4724997876723675
      },
      {
        "dow": 3,
        "t": "2022-12-29T00:00:00",
        "wom": 5,
        "y": 0.0927341711398395,
        "y_hat": 0.5096223750049854
      },
      {
        "dow": 4,
        "t": "2022-12-30T00:00:00",
        "wom": 5,
        "y": 2.42594516701449,
        "y_hat": 2.4119442031524905
      },
      {
        "dow": 5,
        "t": "2022-12-31T00:00:00",
        "wom": 5,
        "y": 0.9191390581073097,
        "y_hat": 0.5097847258328849
      }
    ]
  },
  "layer": [
    {
      "encoding": {
        "tooltip": {
          "field": "t",
          "type": "temporal"
        },
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "darkblue",
        "opacity": 0.6,
        "type": "circle"
      },
      "selection": {
        "selector001": {
          "bind": "scales",
          "type": "interval"
        }
      },
      "title": "",
      "width": 400
    },
    {
      "encoding": {
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y_hat",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "t",
        "type": "line"
      },
      "title": "2 lags, period: 0, MAE: 0.562",
      "width": 400
    }
  ]
}

Enable `seasonal`, and we get the expected level baseline with weekly peaks:

```python
_, s2params, _, stage2 = run_autoreg(ts.y,
                     lags_=4,
                     seasonal_=True,
                     period_=7
                    )
```

var spec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
  "config": {
    "view": {
      "continuousHeight": 300,
      "continuousWidth": 400
    }
  },
  "data": {
    "name": "data-460851078ef436e285d50a6cff3c9a96"
  },
  "datasets": {
    "data-460851078ef436e285d50a6cff3c9a96": [
      {
        "dow": 5,
        "t": "2022-01-01T00:00:00",
        "wom": 1,
        "y": 0.9955981828907805,
        "y_hat": null
      },
      {
        "dow": 6,
        "t": "2022-01-02T00:00:00",
        "wom": 1,
        "y": 0.6234109104925726,
        "y_hat": null
      },
      {
        "dow": 0,
        "t": "2022-01-03T00:00:00",
        "wom": 2,
        "y": 0.8760936874425284,
        "y_hat": 0.4967907586604228
      },
      {
        "dow": 1,
        "t": "2022-01-04T00:00:00",
        "wom": 2,
        "y": 0.8893778611747504,
        "y_hat": 0.5184239775556261
      },
      {
        "dow": 2,
        "t": "2022-01-05T00:00:00",
        "wom": 2,
        "y": 0.3867624782099456,
        "y_hat": 0.4972472553191131
      },
      {
        "dow": 3,
        "t": "2022-01-06T00:00:00",
        "wom": 2,
        "y": 0.6899198960223786,
        "y_hat": 0.5262597140421478
      },
      {
        "dow": 4,
        "t": "2022-01-07T00:00:00",
        "wom": 2,
        "y": 2.992852550566659,
        "y_hat": 2.4299590582515473
      },
      {
        "dow": 5,
        "t": "2022-01-08T00:00:00",
        "wom": 2,
        "y": 0.46804208563249405,
        "y_hat": 0.5525014522166138
      },
      {
        "dow": 6,
        "t": "2022-01-09T00:00:00",
        "wom": 2,
        "y": 0.9455841386245034,
        "y_hat": 0.5668882974435845
      },
      {
        "dow": 0,
        "t": "2022-01-10T00:00:00",
        "wom": 3,
        "y": 0.6381233538848571,
        "y_hat": 0.4776279996404177
      },
      {
        "dow": 1,
        "t": "2022-01-11T00:00:00",
        "wom": 3,
        "y": 0.7595024746539746,
        "y_hat": 0.5291292866690476
      },
      {
        "dow": 2,
        "t": "2022-01-12T00:00:00",
        "wom": 3,
        "y": 0.23738191602671632,
        "y_hat": 0.4808008912256273
      },
      {
        "dow": 3,
        "t": "2022-01-13T00:00:00",
        "wom": 3,
        "y": 0.09680924548766545,
        "y_hat": 0.515091715171366
      },
      {
        "dow": 4,
        "t": "2022-01-14T00:00:00",
        "wom": 3,
        "y": 2.588571863863769,
        "y_hat": 2.405058230192902
      },
      {
        "dow": 5,
        "t": "2022-01-15T00:00:00",
        "wom": 3,
        "y": 0.16573502501513393,
        "y_hat": 0.5090408212208333
      },
      {
        "dow": 6,
        "t": "2022-01-16T00:00:00",
        "wom": 3,
        "y": 0.38942124330865624,
        "y_hat": 0.5365359980531161
      },
      {
        "dow": 0,
        "t": "2022-01-17T00:00:00",
        "wom": 4,
        "y": 0.688787521402102,
        "y_hat": 0.44552691006817247
      },
      {
        "dow": 1,
        "t": "2022-01-18T00:00:00",
        "wom": 4,
        "y": 0.6697600544042848,
        "y_hat": 0.5006636268518205
      },
      {
        "dow": 2,
        "t": "2022-01-19T00:00:00",
        "wom": 4,
        "y": 0.36885391553882474,
        "y_hat": 0.4810838917599954
      },
      {
        "dow": 3,
        "t": "2022-01-20T00:00:00",
        "wom": 4,
        "y": 0.9325008856726638,
        "y_hat": 0.5141158767799752
      },
      {
        "dow": 4,
        "t": "2022-01-21T00:00:00",
        "wom": 4,
        "y": 2.141135576285614,
        "y_hat": 2.436148536768076
      },
      {
        "dow": 5,
        "t": "2022-01-22T00:00:00",
        "wom": 4,
        "y": 0.9772509899400809,
        "y_hat": 0.5414839206074327
      },
      {
        "dow": 6,
        "t": "2022-01-23T00:00:00",
        "wom": 4,
        "y": 0.5683493004574759,
        "y_hat": 0.5356818812038745
      },
      {
        "dow": 0,
        "t": "2022-01-24T00:00:00",
        "wom": 5,
        "y": 0.9649672826390342,
        "y_hat": 0.4945634970531389
      },
      {
        "dow": 1,
        "t": "2022-01-25T00:00:00",
        "wom": 5,
        "y": 0.22830327875114365,
        "y_hat": 0.5183276280490623
      },
      {
        "dow": 2,
        "t": "2022-01-26T00:00:00",
        "wom": 5,
        "y": 0.5236014315750456,
        "y_hat": 0.4834914229513924
      },
      {
        "dow": 3,
        "t": "2022-01-27T00:00:00",
        "wom": 5,
        "y": 0.5771602890446812,
        "y_hat": 0.49481710194259054
      },
      {
        "dow": 4,
        "t": "2022-01-28T00:00:00",
        "wom": 5,
        "y": 2.3198686688954586,
        "y_hat": 2.4344611018946396
      },
      {
        "dow": 5,
        "t": "2022-01-29T00:00:00",
        "wom": 5,
        "y": 0.4375103099359736,
        "y_hat": 0.5275197097634949
      },
      {
        "dow": 6,
        "t": "2022-01-30T00:00:00",
        "wom": 5,
        "y": 0.7071449184091295,
        "y_hat": 0.5300200373997083
      },
      {
        "dow": 0,
        "t": "2022-01-31T00:00:00",
        "wom": 6,
        "y": 0.1524637438937454,
        "y_hat": 0.46950282399087084
      },
      {
        "dow": 1,
        "t": "2022-02-01T00:00:00",
        "wom": 1,
        "y": 0.011748418393797033,
        "y_hat": 0.5027153449515473
      },
      {
        "dow": 2,
        "t": "2022-02-02T00:00:00",
        "wom": 1,
        "y": 0.25517769811183577,
        "y_hat": 0.4335507573741657
      },
      {
        "dow": 3,
        "t": "2022-02-03T00:00:00",
        "wom": 1,
        "y": 0.5156949359371945,
        "y_hat": 0.47556781187960434
      },
      {
        "dow": 4,
        "t": "2022-02-04T00:00:00",
        "wom": 1,
        "y": 2.6710173501944316,
        "y_hat": 2.418325496049474
      },
      {
        "dow": 5,
        "t": "2022-02-05T00:00:00",
        "wom": 1,
        "y": 0.8240477378755111,
        "y_hat": 0.5343478357671186
      },
      {
        "dow": 6,
        "t": "2022-02-06T00:00:00",
        "wom": 1,
        "y": 0.9508838016129778,
        "y_hat": 0.5601352577647144
      },
      {
        "dow": 0,
        "t": "2022-02-07T00:00:00",
        "wom": 2,
        "y": 0.05520043850117129,
        "y_hat": 0.4974480437451285
      },
      {
        "dow": 1,
        "t": "2022-02-08T00:00:00",
        "wom": 2,
        "y": 0.36173925787444183,
        "y_hat": 0.513206980650459
      },
      {
        "dow": 2,
        "t": "2022-02-09T00:00:00",
        "wom": 2,
        "y": 0.3287722160717247,
        "y_hat": 0.4384131691013086
      },
      {
        "dow": 3,
        "t": "2022-02-10T00:00:00",
        "wom": 2,
        "y": 0.26344459815995447,
        "y_hat": 0.49667802938134803
      },
      {
        "dow": 4,
        "t": "2022-02-11T00:00:00",
        "wom": 2,
        "y": 2.105522116894078,
        "y_hat": 2.415202698211017
      },
      {
        "dow": 5,
        "t": "2022-02-12T00:00:00",
        "wom": 2,
        "y": 0.9004373252539722,
        "y_hat": 0.5046824885073314
      },
      {
        "dow": 6,
        "t": "2022-02-13T00:00:00",
        "wom": 2,
        "y": 0.3083956070489431,
        "y_hat": 0.5319008881433882
      },
      {
        "dow": 0,
        "t": "2022-02-14T00:00:00",
        "wom": 3,
        "y": 0.8443045110438598,
        "y_hat": 0.483324909023404
      },
      {
        "dow": 1,
        "t": "2022-02-15T00:00:00",
        "wom": 3,
        "y": 0.9448650092640142,
        "y_hat": 0.5011821422416533
      },
      {
        "dow": 2,
        "t": "2022-02-16T00:00:00",
        "wom": 3,
        "y": 0.5570587072891441,
        "y_hat": 0.4977899462465503
      },
      {
        "dow": 3,
        "t": "2022-02-17T00:00:00",
        "wom": 3,
        "y": 0.5374558873035822,
        "y_hat": 0.5347949272417398
      },
      {
        "dow": 4,
        "t": "2022-02-18T00:00:00",
        "wom": 3,
        "y": 2.8223117437123824,
        "y_hat": 2.4354694540369994
      },
      {
        "dow": 5,
        "t": "2022-02-19T00:00:00",
        "wom": 3,
        "y": 0.02226637799800346,
        "y_hat": 0.5400703987126106
      },
      {
        "dow": 6,
        "t": "2022-02-20T00:00:00",
        "wom": 3,
        "y": 0.15725472652557881,
        "y_hat": 0.5456163598479873
      },
      {
        "dow": 0,
        "t": "2022-02-21T00:00:00",
        "wom": 4,
        "y": 0.9269915515225546,
        "y_hat": 0.43170828507018294
      },
      {
        "dow": 1,
        "t": "2022-02-22T00:00:00",
        "wom": 4,
        "y": 0.4501913471576654,
        "y_hat": 0.4954975240418244
      },
      {
        "dow": 2,
        "t": "2022-02-23T00:00:00",
        "wom": 4,
        "y": 0.9531441262485346,
        "y_hat": 0.48823066431783424
      },
      {
        "dow": 3,
        "t": "2022-02-24T00:00:00",
        "wom": 4,
        "y": 0.5844501255113494,
        "y_hat": 0.519519472671327
      },
      {
        "dow": 4,
        "t": "2022-02-25T00:00:00",
        "wom": 4,
        "y": 2.80397776091471,
        "y_hat": 2.458308069387769
      },
      {
        "dow": 5,
        "t": "2022-02-26T00:00:00",
        "wom": 4,
        "y": 0.26383392179603904,
        "y_hat": 0.542195800807761
      },
      {
        "dow": 6,
        "t": "2022-02-27T00:00:00",
        "wom": 4,
        "y": 0.7531464817944895,
        "y_hat": 0.5516416906073818
      },
      {
        "dow": 0,
        "t": "2022-02-28T00:00:00",
        "wom": 5,
        "y": 0.15885513834885823,
        "y_hat": 0.4618899359780873
      },
      {
        "dow": 1,
        "t": "2022-03-01T00:00:00",
        "wom": 1,
        "y": 0.14535499394438622,
        "y_hat": 0.5058303574877435
      },
      {
        "dow": 2,
        "t": "2022-03-02T00:00:00",
        "wom": 1,
        "y": 0.4410675050392091,
        "y_hat": 0.43816256596576625
      },
      {
        "dow": 3,
        "t": "2022-03-03T00:00:00",
        "wom": 1,
        "y": 0.7517726853328405,
        "y_hat": 0.48854162038072846
      },
      {
        "dow": 4,
        "t": "2022-03-04T00:00:00",
        "wom": 1,
        "y": 2.7945001756665624,
        "y_hat": 2.435556070318829
      },
      {
        "dow": 5,
        "t": "2022-03-05T00:00:00",
        "wom": 1,
        "y": 0.04870630413527299,
        "y_hat": 0.5510704584107008
      },
      {
        "dow": 6,
        "t": "2022-03-06T00:00:00",
        "wom": 1,
        "y": 0.2161041338955525,
        "y_hat": 0.5450948686636378
      },
      {
        "dow": 0,
        "t": "2022-03-07T00:00:00",
        "wom": 2,
        "y": 0.37010898460177155,
        "y_hat": 0.435041792407649
      },
      {
        "dow": 1,
        "t": "2022-03-08T00:00:00",
        "wom": 2,
        "y": 0.09520299889511219,
        "y_hat": 0.48298588836607886
      },
      {
        "dow": 2,
        "t": "2022-03-09T00:00:00",
        "wom": 2,
        "y": 0.11977796440076782,
        "y_hat": 0.44824665641173894
      },
      {
        "dow": 3,
        "t": "2022-03-10T00:00:00",
        "wom": 2,
        "y": 0.5000495405840498,
        "y_hat": 0.47676527608495756
      },
      {
        "dow": 4,
        "t": "2022-03-11T00:00:00",
        "wom": 2,
        "y": 2.453037332781335,
        "y_hat": 2.411129683739495
      },
      {
        "dow": 5,
        "t": "2022-03-12T00:00:00",
        "wom": 2,
        "y": 0.1454828110192531,
        "y_hat": 0.5278354061996601
      },
      {
        "dow": 6,
        "t": "2022-03-13T00:00:00",
        "wom": 2,
        "y": 0.37079700609072885,
        "y_hat": 0.5295379288986366
      },
      {
        "dow": 0,
        "t": "2022-03-14T00:00:00",
        "wom": 3,
        "y": 0.6368565984388843,
        "y_hat": 0.4447991215598311
      },
      {
        "dow": 1,
        "t": "2022-03-15T00:00:00",
        "wom": 3,
        "y": 0.5235082648923333,
        "y_hat": 0.4990719735426933
      },
      {
        "dow": 2,
        "t": "2022-03-16T00:00:00",
        "wom": 3,
        "y": 0.7192823846677754,
        "y_hat": 0.4749988421987305
      },
      {
        "dow": 3,
        "t": "2022-03-17T00:00:00",
        "wom": 3,
        "y": 0.6045077428498761,
        "y_hat": 0.5171315926773352
      },
      {
        "dow": 4,
        "t": "2022-03-18T00:00:00",
        "wom": 3,
        "y": 2.5404792929526683,
        "y_hat": 2.446592365808845
      },
      {
        "dow": 5,
        "t": "2022-03-19T00:00:00",
        "wom": 3,
        "y": 0.4830778428824394,
        "y_hat": 0.5360857207829948
      },
      {
        "dow": 6,
        "t": "2022-03-20T00:00:00",
        "wom": 3,
        "y": 0.9022692634866183,
        "y_hat": 0.5440178300153817
      },
      {
        "dow": 0,
        "t": "2022-03-21T00:00:00",
        "wom": 4,
        "y": 0.06386988650342573,
        "y_hat": 0.4783242019394053
      },
      {
        "dow": 1,
        "t": "2022-03-22T00:00:00",
        "wom": 4,
        "y": 0.2631672977407843,
        "y_hat": 0.5115034804148847
      },
      {
        "dow": 2,
        "t": "2022-03-23T00:00:00",
        "wom": 4,
        "y": 0.6967761906616411,
        "y_hat": 0.43673779150020503
      },
      {
        "dow": 3,
        "t": "2022-03-24T00:00:00",
        "wom": 4,
        "y": 0.6938964314225743,
        "y_hat": 0.5025456288446649
      },
      {
        "dow": 4,
        "t": "2022-03-25T00:00:00",
        "wom": 4,
        "y": 2.3402313329841156,
        "y_hat": 2.448043866842587
      },
      {
        "dow": 5,
        "t": "2022-03-26T00:00:00",
        "wom": 4,
        "y": 0.060442940833932424,
        "y_hat": 0.5353015977298601
      },
      {
        "dow": 6,
        "t": "2022-03-27T00:00:00",
        "wom": 4,
        "y": 0.8673761163808922,
        "y_hat": 0.5212422441741162
      },
      {
        "dow": 0,
        "t": "2022-03-28T00:00:00",
        "wom": 5,
        "y": 0.5544851701436769,
        "y_hat": 0.4546224926233655
      },
      {
        "dow": 1,
        "t": "2022-03-29T00:00:00",
        "wom": 5,
        "y": 0.8547133638306371,
        "y_hat": 0.5237514886317916
      },
      {
        "dow": 2,
        "t": "2022-03-30T00:00:00",
        "wom": 5,
        "y": 0.36375693048623126,
        "y_hat": 0.4802406327655775
      },
      {
        "dow": 3,
        "t": "2022-03-31T00:00:00",
        "wom": 5,
        "y": 0.7322858415787377,
        "y_hat": 0.5250775331729335
      },
      {
        "dow": 4,
        "t": "2022-04-01T00:00:00",
        "wom": 1,
        "y": 2.1005642322427573,
        "y_hat": 2.4312743324017627
      },
      {
        "dow": 5,
        "t": "2022-04-02T00:00:00",
        "wom": 1,
        "y": 0.6446246806801427,
        "y_hat": 0.530637749525594
      },
      {
        "dow": 6,
        "t": "2022-04-03T00:00:00",
        "wom": 1,
        "y": 0.523704664544059,
        "y_hat": 0.5251088026683788
      },
      {
        "dow": 0,
        "t": "2022-04-04T00:00:00",
        "wom": 2,
        "y": 0.6478284421164979,
        "y_hat": 0.47645242667294535
      },
      {
        "dow": 1,
        "t": "2022-04-05T00:00:00",
        "wom": 2,
        "y": 0.8035144383488594,
        "y_hat": 0.5079772060871229
      },
      {
        "dow": 2,
        "t": "2022-04-06T00:00:00",
        "wom": 2,
        "y": 0.06175772835110849,
        "y_hat": 0.4839291542047948
      },
      {
        "dow": 3,
        "t": "2022-04-07T00:00:00",
        "wom": 2,
        "y": 0.10101798118758876,
        "y_hat": 0.5137958988269249
      },
      {
        "dow": 4,
        "t": "2022-04-08T00:00:00",
        "wom": 2,
        "y": 2.328695425168449,
        "y_hat": 2.3970437026869673
      },
      {
        "dow": 5,
        "t": "2022-04-09T00:00:00",
        "wom": 2,
        "y": 0.7405205134959488,
        "y_hat": 0.503188594546576
      },
      {
        "dow": 6,
        "t": "2022-04-10T00:00:00",
        "wom": 2,
        "y": 0.08236194536441421,
        "y_hat": 0.5402774394697502
      },
      {
        "dow": 0,
        "t": "2022-04-11T00:00:00",
        "wom": 3,
        "y": 0.48567949453078907,
        "y_hat": 0.4691302067870829
      },
      {
        "dow": 1,
        "t": "2022-04-12T00:00:00",
        "wom": 3,
        "y": 0.5552603671504405,
        "y_hat": 0.4796291198149867
      },
      {
        "dow": 2,
        "t": "2022-04-13T00:00:00",
        "wom": 3,
        "y": 0.2511650942997945,
        "y_hat": 0.4681934405589045
      },
      {
        "dow": 3,
        "t": "2022-04-14T00:00:00",
        "wom": 3,
        "y": 0.4804870698202284,
        "y_hat": 0.5059180037110255
      },
      {
        "dow": 4,
        "t": "2022-04-15T00:00:00",
        "wom": 3,
        "y": 2.0257783736909825,
        "y_hat": 2.4182249938567857
      },
      {
        "dow": 5,
        "t": "2022-04-16T00:00:00",
        "wom": 3,
        "y": 0.9687966319838246,
        "y_hat": 0.5151312584930192
      },
      {
        "dow": 6,
        "t": "2022-04-17T00:00:00",
        "wom": 3,
        "y": 0.7151278400228451,
        "y_hat": 0.5305591407286968
      },
      {
        "dow": 0,
        "t": "2022-04-18T00:00:00",
        "wom": 4,
        "y": 0.4718442940718869,
        "y_hat": 0.49964798819154504
      },
      {
        "dow": 1,
        "t": "2022-04-19T00:00:00",
        "wom": 4,
        "y": 0.9026979968878186,
        "y_hat": 0.5135072021996053
      },
      {
        "dow": 2,
        "t": "2022-04-20T00:00:00",
        "wom": 4,
        "y": 0.39245084658370566,
        "y_hat": 0.47748693630179173
      },
      {
        "dow": 3,
        "t": "2022-04-21T00:00:00",
        "wom": 4,
        "y": 0.4827795920741036,
        "y_hat": 0.5288246903791284
      },
      {
        "dow": 4,
        "t": "2022-04-22T00:00:00",
        "wom": 4,
        "y": 2.551414450198103,
        "y_hat": 2.4260303356637705
      },
      {
        "dow": 5,
        "t": "2022-04-23T00:00:00",
        "wom": 4,
        "y": 0.650135286544272,
        "y_hat": 0.5303875499655886
      },
      {
        "dow": 6,
        "t": "2022-04-24T00:00:00",
        "wom": 4,
        "y": 0.467587186604593,
        "y_hat": 0.549943038786494
      },
      {
        "dow": 0,
        "t": "2022-04-25T00:00:00",
        "wom": 5,
        "y": 0.8420434940525285,
        "y_hat": 0.4754830070083764
      },
      {
        "dow": 1,
        "t": "2022-04-26T00:00:00",
        "wom": 5,
        "y": 0.44245377300469013,
        "y_hat": 0.5108340361542065
      },
      {
        "dow": 2,
        "t": "2022-04-27T00:00:00",
        "wom": 5,
        "y": 0.10472408289231805,
        "y_hat": 0.4844334459505021
      },
      {
        "dow": 3,
        "t": "2022-04-28T00:00:00",
        "wom": 5,
        "y": 0.07965539705381441,
        "y_hat": 0.4958677197277991
      },
      {
        "dow": 4,
        "t": "2022-04-29T00:00:00",
        "wom": 5,
        "y": 2.1204279487951343,
        "y_hat": 2.3990895488892217
      },
      {
        "dow": 5,
        "t": "2022-04-30T00:00:00",
        "wom": 5,
        "y": 0.09911422212585952,
        "y_hat": 0.4964206092595712
      },
      {
        "dow": 6,
        "t": "2022-05-01T00:00:00",
        "wom": 1,
        "y": 0.7653139992562019,
        "y_hat": 0.5110418469407004
      },
      {
        "dow": 0,
        "t": "2022-05-02T00:00:00",
        "wom": 2,
        "y": 0.6928765519708857,
        "y_hat": 0.45435487387021417
      },
      {
        "dow": 1,
        "t": "2022-05-03T00:00:00",
        "wom": 2,
        "y": 0.3293225432083243,
        "y_hat": 0.5227571533370909
      },
      {
        "dow": 2,
        "t": "2022-05-04T00:00:00",
        "wom": 2,
        "y": 0.6649514990871527,
        "y_hat": 0.47325979365624726
      },
      {
        "dow": 3,
        "t": "2022-05-05T00:00:00",
        "wom": 2,
        "y": 0.15101171079274767,
        "y_hat": 0.5058810792932257
      },
      {
        "dow": 4,
        "t": "2022-05-06T00:00:00",
        "wom": 2,
        "y": 2.1798428980656723,
        "y_hat": 2.431485876177647
      },
      {
        "dow": 5,
        "t": "2022-05-07T00:00:00",
        "wom": 2,
        "y": 0.2354835900991824,
        "y_hat": 0.5020829628448384
      },
      {
        "dow": 6,
        "t": "2022-05-08T00:00:00",
        "wom": 2,
        "y": 0.05741790510662548,
        "y_hat": 0.5182585301985556
      },
      {
        "dow": 0,
        "t": "2022-05-09T00:00:00",
        "wom": 3,
        "y": 0.23077326722872582,
        "y_hat": 0.4416008126128215
      },
      {
        "dow": 1,
        "t": "2022-05-10T00:00:00",
        "wom": 3,
        "y": 0.1944400629234596,
        "y_hat": 0.4714472781064954
      },
      {
        "dow": 2,
        "t": "2022-05-11T00:00:00",
        "wom": 3,
        "y": 0.7807354431480255,
        "y_hat": 0.44457005664505445
      },
      {
        "dow": 3,
        "t": "2022-05-12T00:00:00",
        "wom": 3,
        "y": 0.31021821068384303,
        "y_hat": 0.5020199652718704
      },
      {
        "dow": 4,
        "t": "2022-05-13T00:00:00",
        "wom": 3,
        "y": 2.1275853244649534,
        "y_hat": 2.4423983412290537
      },
      {
        "dow": 5,
        "t": "2022-05-14T00:00:00",
        "wom": 3,
        "y": 0.56813301154376,
        "y_hat": 0.5092969989732901
      },
      {
        "dow": 6,
        "t": "2022-05-15T00:00:00",
        "wom": 3,
        "y": 0.10167470187610927,
        "y_hat": 0.5250551324538184
      },
      {
        "dow": 0,
        "t": "2022-05-16T00:00:00",
        "wom": 4,
        "y": 0.37826726087578866,
        "y_hat": 0.46093647736965676
      },
      {
        "dow": 1,
        "t": "2022-05-17T00:00:00",
        "wom": 4,
        "y": 0.7138146434013722,
        "y_hat": 0.4781635078824496
      },
      {
        "dow": 2,
        "t": "2022-05-18T00:00:00",
        "wom": 4,
        "y": 0.12852670113262654,
        "y_hat": 0.46748643229002546
      },
      {
        "dow": 3,
        "t": "2022-05-19T00:00:00",
        "wom": 4,
        "y": 0.3585977195944233,
        "y_hat": 0.5115345457435468
      },
      {
        "dow": 4,
        "t": "2022-05-20T00:00:00",
        "wom": 4,
        "y": 2.22258022407796,
        "y_hat": 2.408682157424716
      },
      {
        "dow": 5,
        "t": "2022-05-21T00:00:00",
        "wom": 4,
        "y": 0.932770577180065,
        "y_hat": 0.5147356152220035
      },
      {
        "dow": 6,
        "t": "2022-05-22T00:00:00",
        "wom": 4,
        "y": 0.9336405466022045,
        "y_hat": 0.5407155484496845
      },
      {
        "dow": 0,
        "t": "2022-05-23T00:00:00",
        "wom": 5,
        "y": 0.7765773218651726,
        "y_hat": 0.5045082552262019
      },
      {
        "dow": 1,
        "t": "2022-05-24T00:00:00",
        "wom": 5,
        "y": 0.9092507321087694,
        "y_hat": 0.5345730713865291
      },
      {
        "dow": 2,
        "t": "2022-05-25T00:00:00",
        "wom": 5,
        "y": 0.9912972693958269,
        "y_hat": 0.4946869470039545
      },
      {
        "dow": 3,
        "t": "2022-05-26T00:00:00",
        "wom": 5,
        "y": 0.17273741694999156,
        "y_hat": 0.5468518827619808
      },
      {
        "dow": 4,
        "t": "2022-05-27T00:00:00",
        "wom": 5,
        "y": 2.7305657647776744,
        "y_hat": 2.4500618298869807
      },
      {
        "dow": 5,
        "t": "2022-05-28T00:00:00",
        "wom": 5,
        "y": 0.9331495324820412,
        "y_hat": 0.5193297047723543
      },
      {
        "dow": 6,
        "t": "2022-05-29T00:00:00",
        "wom": 5,
        "y": 0.9747799262042263,
        "y_hat": 0.5682632750846622
      },
      {
        "dow": 0,
        "t": "2022-05-30T00:00:00",
        "wom": 6,
        "y": 0.6159827699504155,
        "y_hat": 0.5058165123564484
      },
      {
        "dow": 1,
        "t": "2022-05-31T00:00:00",
        "wom": 6,
        "y": 0.18957160929478611,
        "y_hat": 0.5323172222676634
      },
      {
        "dow": 2,
        "t": "2022-06-01T00:00:00",
        "wom": 1,
        "y": 0.08329858559961756,
        "y_hat": 0.4655639528916069
      },
      {
        "dow": 3,
        "t": "2022-06-02T00:00:00",
        "wom": 1,
        "y": 0.7811845717468601,
        "y_hat": 0.48216420003370863
      },
      {
        "dow": 4,
        "t": "2022-06-03T00:00:00",
        "wom": 1,
        "y": 2.5856317189042306,
        "y_hat": 2.41854048611295
      },
      {
        "dow": 5,
        "t": "2022-06-04T00:00:00",
        "wom": 1,
        "y": 0.9915634439508605,
        "y_hat": 0.5481486757761187
      },
      {
        "dow": 6,
        "t": "2022-06-05T00:00:00",
        "wom": 1,
        "y": 0.9909525874581544,
        "y_hat": 0.5622201296718374
      },
      {
        "dow": 0,
        "t": "2022-06-06T00:00:00",
        "wom": 2,
        "y": 0.546656872738867,
        "y_hat": 0.509544473702861
      },
      {
        "dow": 1,
        "t": "2022-06-07T00:00:00",
        "wom": 2,
        "y": 0.10472218993110805,
        "y_hat": 0.5313215376478715
      },
      {
        "dow": 2,
        "t": "2022-06-08T00:00:00",
        "wom": 2,
        "y": 0.04250860920798383,
        "y_hat": 0.45950885511347206
      },
      {
        "dow": 3,
        "t": "2022-06-09T00:00:00",
        "wom": 2,
        "y": 0.3032781068949767,
        "y_hat": 0.4765300517785524
      },
      {
        "dow": 4,
        "t": "2022-06-10T00:00:00",
        "wom": 2,
        "y": 2.5924710028233178,
        "y_hat": 2.4027941662417835
      },
      {
        "dow": 5,
        "t": "2022-06-11T00:00:00",
        "wom": 2,
        "y": 0.1823450459040853,
        "y_hat": 0.5226555382078407
      },
      {
        "dow": 6,
        "t": "2022-06-12T00:00:00",
        "wom": 2,
        "y": 0.7714328053698969,
        "y_hat": 0.5395777784515343
      },
      {
        "dow": 0,
        "t": "2022-06-13T00:00:00",
        "wom": 3,
        "y": 0.7117102710960721,
        "y_hat": 0.4596964393488171
      },
      {
        "dow": 1,
        "t": "2022-06-14T00:00:00",
        "wom": 3,
        "y": 0.5078428447537274,
        "y_hat": 0.5242989977320872
      },
      {
        "dow": 2,
        "t": "2022-06-15T00:00:00",
        "wom": 3,
        "y": 0.4559806484621759,
        "y_hat": 0.48005120057372996
      },
      {
        "dow": 3,
        "t": "2022-06-16T00:00:00",
        "wom": 3,
        "y": 0.3663298759696507,
        "y_hat": 0.5102207588634888
      },
      {
        "dow": 4,
        "t": "2022-06-17T00:00:00",
        "wom": 3,
        "y": 2.036192987755432,
        "y_hat": 2.427030286845141
      },
      {
        "dow": 5,
        "t": "2022-06-18T00:00:00",
        "wom": 3,
        "y": 0.5857499825271564,
        "y_hat": 0.5102758304570155
      },
      {
        "dow": 6,
        "t": "2022-06-19T00:00:00",
        "wom": 3,
        "y": 0.050005294335790484,
        "y_hat": 0.5211856409518846
      },
      {
        "dow": 0,
        "t": "2022-06-20T00:00:00",
        "wom": 4,
        "y": 0.5061272547669947,
        "y_hat": 0.46097219295144776
      },
      {
        "dow": 1,
        "t": "2022-06-21T00:00:00",
        "wom": 4,
        "y": 0.7904390381333787,
        "y_hat": 0.4795887904276564
      },
      {
        "dow": 2,
        "t": "2022-06-22T00:00:00",
        "wom": 4,
        "y": 0.09323256170059835,
        "y_hat": 0.4771399084781224
      },
      {
        "dow": 3,
        "t": "2022-06-23T00:00:00",
        "wom": 4,
        "y": 0.40800965871918415,
        "y_hat": 0.5152238334334927
      },
      {
        "dow": 4,
        "t": "2022-06-24T00:00:00",
        "wom": 4,
        "y": 2.570342151877391,
        "y_hat": 2.4087498026209566
      },
      {
        "dow": 5,
        "t": "2022-06-25T00:00:00",
        "wom": 4,
        "y": 0.07936746817059592,
        "y_hat": 0.5279017923132528
      },
      {
        "dow": 6,
        "t": "2022-06-26T00:00:00",
        "wom": 4,
        "y": 0.017413769472798823,
        "y_hat": 0.5356649133238703
      },
      {
        "dow": 0,
        "t": "2022-06-27T00:00:00",
        "wom": 5,
        "y": 0.4856034209379412,
        "y_hat": 0.4328149611023629
      },
      {
        "dow": 1,
        "t": "2022-06-28T00:00:00",
        "wom": 5,
        "y": 0.10626439009089561,
        "y_hat": 0.47735501023779975
      },
      {
        "dow": 2,
        "t": "2022-06-29T00:00:00",
        "wom": 5,
        "y": 0.7531865011385493,
        "y_hat": 0.4565934866610085
      },
      {
        "dow": 3,
        "t": "2022-06-30T00:00:00",
        "wom": 5,
        "y": 0.2215890993417199,
        "y_hat": 0.4972579361761011
      },
      {
        "dow": 4,
        "t": "2022-07-01T00:00:00",
        "wom": 1,
        "y": 2.2444907708382646,
        "y_hat": 2.4391639795535345
      },
      {
        "dow": 5,
        "t": "2022-07-02T00:00:00",
        "wom": 1,
        "y": 0.41066189647640805,
        "y_hat": 0.5086383437322588
      },
      {
        "dow": 6,
        "t": "2022-07-03T00:00:00",
        "wom": 1,
        "y": 0.8875209104102515,
        "y_hat": 0.5276522539533361
      },
      {
        "dow": 0,
        "t": "2022-07-04T00:00:00",
        "wom": 2,
        "y": 0.29879768250458105,
        "y_hat": 0.4756765448644015
      },
      {
        "dow": 1,
        "t": "2022-07-05T00:00:00",
        "wom": 2,
        "y": 0.8642111944128015,
        "y_hat": 0.519103746855454
      },
      {
        "dow": 2,
        "t": "2022-07-06T00:00:00",
        "wom": 2,
        "y": 0.5035308154420571,
        "y_hat": 0.468279298304797
      },
      {
        "dow": 3,
        "t": "2022-07-07T00:00:00",
        "wom": 2,
        "y": 0.4608669940295842,
        "y_hat": 0.5311554916042507
      },
      {
        "dow": 4,
        "t": "2022-07-08T00:00:00",
        "wom": 2,
        "y": 2.1336407534117914,
        "y_hat": 2.43263548967435
      },
      {
        "dow": 5,
        "t": "2022-07-09T00:00:00",
        "wom": 2,
        "y": 0.07559350276009646,
        "y_hat": 0.5185008966658524
      },
      {
        "dow": 6,
        "t": "2022-07-10T00:00:00",
        "wom": 2,
        "y": 0.21152592170248208,
        "y_hat": 0.5122052608058499
      },
      {
        "dow": 0,
        "t": "2022-07-11T00:00:00",
        "wom": 3,
        "y": 0.7992198246512687,
        "y_hat": 0.4383825054702067
      },
      {
        "dow": 1,
        "t": "2022-07-12T00:00:00",
        "wom": 3,
        "y": 0.08134234250438865,
        "y_hat": 0.49702074794504336
      },
      {
        "dow": 2,
        "t": "2022-07-13T00:00:00",
        "wom": 3,
        "y": 0.8545474044325597,
        "y_hat": 0.4730370156455408
      },
      {
        "dow": 3,
        "t": "2022-07-14T00:00:00",
        "wom": 3,
        "y": 0.7192996694052958,
        "y_hat": 0.49903334487542134
      },
      {
        "dow": 4,
        "t": "2022-07-15T00:00:00",
        "wom": 3,
        "y": 2.839731861280379,
        "y_hat": 2.4590829237763088
      },
      {
        "dow": 5,
        "t": "2022-07-16T00:00:00",
        "wom": 3,
        "y": 0.6866567952654717,
        "y_hat": 0.5527420470316485
      },
      {
        "dow": 6,
        "t": "2022-07-17T00:00:00",
        "wom": 3,
        "y": 0.3025925308532783,
        "y_hat": 0.5678987356529254
      },
      {
        "dow": 0,
        "t": "2022-07-18T00:00:00",
        "wom": 4,
        "y": 0.35527032358475563,
        "y_hat": 0.4740865465555972
      },
      {
        "dow": 1,
        "t": "2022-07-19T00:00:00",
        "wom": 4,
        "y": 0.2998867051787012,
        "y_hat": 0.4893633221283946
      },
      {
        "dow": 2,
        "t": "2022-07-20T00:00:00",
        "wom": 4,
        "y": 0.32169056052383393,
        "y_hat": 0.4554266913463155
      },
      {
        "dow": 3,
        "t": "2022-07-21T00:00:00",
        "wom": 4,
        "y": 0.8817188595233318,
        "y_hat": 0.49571750586434926
      },
      {
        "dow": 4,
        "t": "2022-07-22T00:00:00",
        "wom": 4,
        "y": 2.6482046952131784,
        "y_hat": 2.4350689441557574
      },
      {
        "dow": 5,
        "t": "2022-07-23T00:00:00",
        "wom": 4,
        "y": 0.5392997474632014,
        "y_hat": 0.5561498314017669
      },
      {
        "dow": 6,
        "t": "2022-07-24T00:00:00",
        "wom": 4,
        "y": 0.28113347961902946,
        "y_hat": 0.5534601575845951
      },
      {
        "dow": 0,
        "t": "2022-07-25T00:00:00",
        "wom": 5,
        "y": 0.4047240553693535,
        "y_hat": 0.4656301818804722
      },
      {
        "dow": 1,
        "t": "2022-07-26T00:00:00",
        "wom": 5,
        "y": 0.09656379233844459,
        "y_hat": 0.4897301947692827
      },
      {
        "dow": 2,
        "t": "2022-07-27T00:00:00",
        "wom": 5,
        "y": 0.9545109848618508,
        "y_hat": 0.45239872668182046
      },
      {
        "dow": 3,
        "t": "2022-07-28T00:00:00",
        "wom": 5,
        "y": 0.2844751590965856,
        "y_hat": 0.5029360562621922
      },
      {
        "dow": 4,
        "t": "2022-07-29T00:00:00",
        "wom": 5,
        "y": 2.19828814692932,
        "y_hat": 2.4522787916765902
      },
      {
        "dow": 5,
        "t": "2022-07-30T00:00:00",
        "wom": 5,
        "y": 0.7872041161790215,
        "y_hat": 0.5111620022247799
      },
      {
        "dow": 6,
        "t": "2022-07-31T00:00:00",
        "wom": 5,
        "y": 0.8320061228006757,
        "y_hat": 0.5363666658917704
      },
      {
        "dow": 0,
        "t": "2022-08-01T00:00:00",
        "wom": 1,
        "y": 0.48050923278633095,
        "y_hat": 0.49486749692129434
      },
      {
        "dow": 1,
        "t": "2022-08-02T00:00:00",
        "wom": 1,
        "y": 0.9186312133845865,
        "y_hat": 0.5217480417049952
      },
      {
        "dow": 2,
        "t": "2022-08-03T00:00:00",
        "wom": 1,
        "y": 0.28383987157951585,
        "y_hat": 0.4800933434215706
      },
      {
        "dow": 3,
        "t": "2022-08-04T00:00:00",
        "wom": 1,
        "y": 0.6963665748778668,
        "y_hat": 0.5282645806595277
      },
      {
        "dow": 4,
        "t": "2022-08-05T00:00:00",
        "wom": 1,
        "y": 2.09245143951022,
        "y_hat": 2.427953405345737
      },
      {
        "dow": 5,
        "t": "2022-08-06T00:00:00",
        "wom": 1,
        "y": 0.7172188720261602,
        "y_hat": 0.5304867199912241
      },
      {
        "dow": 6,
        "t": "2022-08-07T00:00:00",
        "wom": 1,
        "y": 0.6862602995181284,
        "y_hat": 0.5287652055403764
      },
      {
        "dow": 0,
        "t": "2022-08-08T00:00:00",
        "wom": 2,
        "y": 0.12495616249592223,
        "y_hat": 0.4870366588398229
      },
      {
        "dow": 1,
        "t": "2022-08-09T00:00:00",
        "wom": 2,
        "y": 0.7803068072936611,
        "y_hat": 0.5038317649617358
      },
      {
        "dow": 2,
        "t": "2022-08-10T00:00:00",
        "wom": 2,
        "y": 0.1711236164496699,
        "y_hat": 0.45705759106594746
      },
      {
        "dow": 3,
        "t": "2022-08-11T00:00:00",
        "wom": 2,
        "y": 0.6418292052213659,
        "y_hat": 0.5176881437709107
      },
      {
        "dow": 4,
        "t": "2022-08-12T00:00:00",
        "wom": 2,
        "y": 2.398223140821866,
        "y_hat": 2.420421968469209
      },
      {
        "dow": 5,
        "t": "2022-08-13T00:00:00",
        "wom": 2,
        "y": 0.7667020108493909,
        "y_hat": 0.5363921988276147
      },
      {
        "dow": 6,
        "t": "2022-08-14T00:00:00",
        "wom": 2,
        "y": 0.41815335304653145,
        "y_hat": 0.5467991648029131
      },
      {
        "dow": 0,
        "t": "2022-08-15T00:00:00",
        "wom": 3,
        "y": 0.6891797613670014,
        "y_hat": 0.48215904682588656
      },
      {
        "dow": 1,
        "t": "2022-08-16T00:00:00",
        "wom": 3,
        "y": 0.5101490274593742,
        "y_hat": 0.5055926245703029
      },
      {
        "dow": 2,
        "t": "2022-08-17T00:00:00",
        "wom": 3,
        "y": 0.7796622447746596,
        "y_hat": 0.47991073110965105
      },
      {
        "dow": 3,
        "t": "2022-08-18T00:00:00",
        "wom": 3,
        "y": 0.5627361856480227,
        "y_hat": 0.5206046080919271
      },
      {
        "dow": 4,
        "t": "2022-08-19T00:00:00",
        "wom": 3,
        "y": 2.149269114111338,
        "y_hat": 2.4511273168906147
      },
      {
        "dow": 5,
        "t": "2022-08-20T00:00:00",
        "wom": 3,
        "y": 0.4028888263767394,
        "y_hat": 0.5251204340687972
      },
      {
        "dow": 6,
        "t": "2022-08-21T00:00:00",
        "wom": 3,
        "y": 0.29581445643538484,
        "y_hat": 0.5230749375317849
      },
      {
        "dow": 0,
        "t": "2022-08-22T00:00:00",
        "wom": 4,
        "y": 0.22454795399899974,
        "y_hat": 0.4591341483141852
      },
      {
        "dow": 1,
        "t": "2022-08-23T00:00:00",
        "wom": 4,
        "y": 0.5368502058660074,
        "y_hat": 0.48582304298727946
      },
      {
        "dow": 2,
        "t": "2022-08-24T00:00:00",
        "wom": 4,
        "y": 0.8352731932733497,
        "y_hat": 0.455701817281319
      },
      {
        "dow": 3,
        "t": "2022-08-25T00:00:00",
        "wom": 4,
        "y": 0.7156495255143432,
        "y_hat": 0.5237474583198937
      },
      {
        "dow": 4,
        "t": "2022-08-26T00:00:00",
        "wom": 4,
        "y": 2.081170707633651,
        "y_hat": 2.4586113850595304
      },
      {
        "dow": 5,
        "t": "2022-08-27T00:00:00",
        "wom": 4,
        "y": 0.9846573243966786,
        "y_hat": 0.5315420611782877
      },
      {
        "dow": 6,
        "t": "2022-08-28T00:00:00",
        "wom": 4,
        "y": 0.11556538882109113,
        "y_hat": 0.5361350363720931
      },
      {
        "dow": 0,
        "t": "2022-08-29T00:00:00",
        "wom": 5,
        "y": 0.26133230623941184,
        "y_hat": 0.4855036716924154
      },
      {
        "dow": 1,
        "t": "2022-08-30T00:00:00",
        "wom": 5,
        "y": 0.9006219041976188,
        "y_hat": 0.47725526614685454
      },
      {
        "dow": 2,
        "t": "2022-08-31T00:00:00",
        "wom": 5,
        "y": 0.2515146218161567,
        "y_hat": 0.4681948708253787
      },
      {
        "dow": 3,
        "t": "2022-09-01T00:00:00",
        "wom": 1,
        "y": 0.13098484463014348,
        "y_hat": 0.5268174837110191
      },
      {
        "dow": 4,
        "t": "2022-09-02T00:00:00",
        "wom": 1,
        "y": 2.9173298508007353,
        "y_hat": 2.410501086412229
      },
      {
        "dow": 5,
        "t": "2022-09-03T00:00:00",
        "wom": 1,
        "y": 0.902033118492389,
        "y_hat": 0.5239835691087095
      },
      {
        "dow": 6,
        "t": "2022-09-04T00:00:00",
        "wom": 1,
        "y": 0.6094812461745855,
        "y_hat": 0.579028055768711
      },
      {
        "dow": 0,
        "t": "2022-09-05T00:00:00",
        "wom": 2,
        "y": 0.21237629900528165,
        "y_hat": 0.4952691329353708
      },
      {
        "dow": 1,
        "t": "2022-09-06T00:00:00",
        "wom": 2,
        "y": 0.8674613314753005,
        "y_hat": 0.5026336407752172
      },
      {
        "dow": 2,
        "t": "2022-09-07T00:00:00",
        "wom": 2,
        "y": 0.7318690107796962,
        "y_hat": 0.46471651844447337
      },
      {
        "dow": 3,
        "t": "2022-09-08T00:00:00",
        "wom": 2,
        "y": 0.9273400869945371,
        "y_hat": 0.538865818625751
      },
      {
        "dow": 4,
        "t": "2022-09-09T00:00:00",
        "wom": 2,
        "y": 2.491296824471812,
        "y_hat": 2.4593024923288445
      },
      {
        "dow": 5,
        "t": "2022-09-10T00:00:00",
        "wom": 2,
        "y": 0.3736267511181386,
        "y_hat": 0.5549146181128354
      },
      {
        "dow": 6,
        "t": "2022-09-11T00:00:00",
        "wom": 2,
        "y": 0.6741226397964637,
        "y_hat": 0.5410405136378633
      },
      {
        "dow": 0,
        "t": "2022-09-12T00:00:00",
        "wom": 3,
        "y": 0.5690598209381145,
        "y_hat": 0.468701374121668
      },
      {
        "dow": 1,
        "t": "2022-09-13T00:00:00",
        "wom": 3,
        "y": 0.5845921058651229,
        "y_hat": 0.5164280677357482
      },
      {
        "dow": 2,
        "t": "2022-09-14T00:00:00",
        "wom": 3,
        "y": 0.06190141469816057,
        "y_hat": 0.47600193151568615
      },
      {
        "dow": 3,
        "t": "2022-09-15T00:00:00",
        "wom": 3,
        "y": 0.7726353296869438,
        "y_hat": 0.504562097693305
      },
      {
        "dow": 4,
        "t": "2022-09-16T00:00:00",
        "wom": 3,
        "y": 2.3982766116278027,
        "y_hat": 2.418824328436912
      },
      {
        "dow": 5,
        "t": "2022-09-17T00:00:00",
        "wom": 3,
        "y": 0.02485301435952192,
        "y_hat": 0.5440166771084493
      },
      {
        "dow": 6,
        "t": "2022-09-18T00:00:00",
        "wom": 3,
        "y": 0.3885550460991051,
        "y_hat": 0.5261644635357546
      },
      {
        "dow": 0,
        "t": "2022-09-19T00:00:00",
        "wom": 4,
        "y": 0.5124897271631259,
        "y_hat": 0.44182407349217223
      },
      {
        "dow": 1,
        "t": "2022-09-20T00:00:00",
        "wom": 4,
        "y": 0.6717284818449774,
        "y_hat": 0.499506798622325
      },
      {
        "dow": 2,
        "t": "2022-09-21T00:00:00",
        "wom": 4,
        "y": 0.8010250603087118,
        "y_hat": 0.47555005974701947
      },
      {
        "dow": 3,
        "t": "2022-09-22T00:00:00",
        "wom": 4,
        "y": 0.7215828184438076,
        "y_hat": 0.5304993746500212
      },
      {
        "dow": 4,
        "t": "2022-09-23T00:00:00",
        "wom": 4,
        "y": 2.467671973989098,
        "y_hat": 2.457380857040162
      },
      {
        "dow": 5,
        "t": "2022-09-24T00:00:00",
        "wom": 4,
        "y": 0.7177366636762655,
        "y_hat": 0.5433557257312887
      },
      {
        "dow": 6,
        "t": "2022-09-25T00:00:00",
        "wom": 4,
        "y": 0.48856096479123934,
        "y_hat": 0.5498226240838627
      },
      {
        "dow": 0,
        "t": "2022-09-26T00:00:00",
        "wom": 5,
        "y": 0.5960952391022855,
        "y_hat": 0.4822008051720972
      },
      {
        "dow": 1,
        "t": "2022-09-27T00:00:00",
        "wom": 5,
        "y": 0.8681440058433928,
        "y_hat": 0.5074071188774416
      },
      {
        "dow": 2,
        "t": "2022-09-28T00:00:00",
        "wom": 5,
        "y": 0.8669839542345549,
        "y_hat": 0.48578856966099204
      },
      {
        "dow": 3,
        "t": "2022-09-29T00:00:00",
        "wom": 5,
        "y": 0.8884123252558981,
        "y_hat": 0.5431003113470626
      },
      {
        "dow": 4,
        "t": "2022-09-30T00:00:00",
        "wom": 5,
        "y": 2.3463008214785965,
        "y_hat": 2.465821242163437
      },
      {
        "dow": 5,
        "t": "2022-10-01T00:00:00",
        "wom": 1,
        "y": 0.2954617156792265,
        "y_hat": 0.5490063522668106
      },
      {
        "dow": 6,
        "t": "2022-10-02T00:00:00",
        "wom": 1,
        "y": 0.633109100839611,
        "y_hat": 0.5313156558678173
      },
      {
        "dow": 0,
        "t": "2022-10-03T00:00:00",
        "wom": 2,
        "y": 0.3669341564363222,
        "y_hat": 0.46364614979584606
      },
      {
        "dow": 1,
        "t": "2022-10-04T00:00:00",
        "wom": 2,
        "y": 0.24876598531702476,
        "y_hat": 0.5087746891999738
      },
      {
        "dow": 2,
        "t": "2022-10-05T00:00:00",
        "wom": 2,
        "y": 0.16587282213408583,
        "y_hat": 0.4558300182291027
      },
      {
        "dow": 3,
        "t": "2022-10-06T00:00:00",
        "wom": 2,
        "y": 0.9919913685200391,
        "y_hat": 0.48973950777965536
      },
      {
        "dow": 4,
        "t": "2022-10-07T00:00:00",
        "wom": 2,
        "y": 2.3533296873989085,
        "y_hat": 2.4310423123676665
      },
      {
        "dow": 5,
        "t": "2022-10-08T00:00:00",
        "wom": 2,
        "y": 0.7213297581947791,
        "y_hat": 0.554911360704057
      },
      {
        "dow": 6,
        "t": "2022-10-09T00:00:00",
        "wom": 2,
        "y": 0.7224275916390493,
        "y_hat": 0.5439767292961218
      },
      {
        "dow": 0,
        "t": "2022-10-10T00:00:00",
        "wom": 3,
        "y": 0.29155319532221713,
        "y_hat": 0.4893020791489958
      },
      {
        "dow": 1,
        "t": "2022-10-11T00:00:00",
        "wom": 3,
        "y": 0.36991001906594234,
        "y_hat": 0.5115549109796871
      },
      {
        "dow": 2,
        "t": "2022-10-12T00:00:00",
        "wom": 3,
        "y": 0.28417739594797153,
        "y_hat": 0.4553343870788139
      },
      {
        "dow": 3,
        "t": "2022-10-13T00:00:00",
        "wom": 3,
        "y": 0.19840529083955494,
        "y_hat": 0.4997725564131675
      },
      {
        "dow": 4,
        "t": "2022-10-14T00:00:00",
        "wom": 3,
        "y": 2.4606866711518958,
        "y_hat": 2.4148643523441606
      },
      {
        "dow": 5,
        "t": "2022-10-15T00:00:00",
        "wom": 3,
        "y": 0.6335089949594493,
        "y_hat": 0.5152479312806922
      },
      {
        "dow": 6,
        "t": "2022-10-16T00:00:00",
        "wom": 3,
        "y": 0.5680543096824044,
        "y_hat": 0.5473753240467025
      },
      {
        "dow": 0,
        "t": "2022-10-17T00:00:00",
        "wom": 4,
        "y": 0.016790840072704594,
        "y_hat": 0.48026181895173703
      },
      {
        "dow": 1,
        "t": "2022-10-18T00:00:00",
        "wom": 4,
        "y": 0.15482355482083832,
        "y_hat": 0.4954814961576285
      },
      {
        "dow": 2,
        "t": "2022-10-19T00:00:00",
        "wom": 4,
        "y": 0.32294713218572024,
        "y_hat": 0.43446678008048273
      },
      {
        "dow": 3,
        "t": "2022-10-20T00:00:00",
        "wom": 4,
        "y": 0.09723518582862622,
        "y_hat": 0.48938074212176336
      },
      {
        "dow": 4,
        "t": "2022-10-21T00:00:00",
        "wom": 4,
        "y": 2.161951112634909,
        "y_hat": 2.4141786551814923
      },
      {
        "dow": 5,
        "t": "2022-10-22T00:00:00",
        "wom": 4,
        "y": 0.9308281099209696,
        "y_hat": 0.5013617569960774
      },
      {
        "dow": 6,
        "t": "2022-10-23T00:00:00",
        "wom": 4,
        "y": 0.3700513293335844,
        "y_hat": 0.5398557089374105
      },
      {
        "dow": 0,
        "t": "2022-10-24T00:00:00",
        "wom": 5,
        "y": 0.33730301215287395,
        "y_hat": 0.4907674124047824
      },
      {
        "dow": 1,
        "t": "2022-10-25T00:00:00",
        "wom": 5,
        "y": 0.2979813390418188,
        "y_hat": 0.49406289572899337
      },
      {
        "dow": 2,
        "t": "2022-10-26T00:00:00",
        "wom": 5,
        "y": 0.8989252622362512,
        "y_hat": 0.45597333301628634
      },
      {
        "dow": 3,
        "t": "2022-10-27T00:00:00",
        "wom": 5,
        "y": 0.9678609978612648,
        "y_hat": 0.5136804919832462
      },
      {
        "dow": 4,
        "t": "2022-10-28T00:00:00",
        "wom": 5,
        "y": 2.852867990371385,
        "y_hat": 2.47026482888401
      },
      {
        "dow": 5,
        "t": "2022-10-29T00:00:00",
        "wom": 5,
        "y": 0.09421902072623589,
        "y_hat": 0.5682198415030082
      },
      {
        "dow": 6,
        "t": "2022-10-30T00:00:00",
        "wom": 5,
        "y": 0.8581628861015397,
        "y_hat": 0.553362035505034
      },
      {
        "dow": 0,
        "t": "2022-10-31T00:00:00",
        "wom": 6,
        "y": 0.20114187846747755,
        "y_hat": 0.4596615396792977
      },
      {
        "dow": 1,
        "t": "2022-11-01T00:00:00",
        "wom": 1,
        "y": 0.6181376931339703,
        "y_hat": 0.516635984405509
      },
      {
        "dow": 2,
        "t": "2022-11-02T00:00:00",
        "wom": 1,
        "y": 0.19978354259674502,
        "y_hat": 0.4578832281890113
      },
      {
        "dow": 3,
        "t": "2022-11-03T00:00:00",
        "wom": 1,
        "y": 0.1768860986420725,
        "y_hat": 0.5110987214886396
      },
      {
        "dow": 4,
        "t": "2022-11-04T00:00:00",
        "wom": 1,
        "y": 2.4752377813951942,
        "y_hat": 2.4100299124147666
      },
      {
        "dow": 5,
        "t": "2022-11-05T00:00:00",
        "wom": 1,
        "y": 0.9055115735426204,
        "y_hat": 0.5148386395620688
      },
      {
        "dow": 6,
        "t": "2022-11-06T00:00:00",
        "wom": 1,
        "y": 0.6649858316616734,
        "y_hat": 0.5562701637739151
      },
      {
        "dow": 0,
        "t": "2022-11-07T00:00:00",
        "wom": 2,
        "y": 0.5304475702709496,
        "y_hat": 0.4980529825768307
      },
      {
        "dow": 1,
        "t": "2022-11-08T00:00:00",
        "wom": 2,
        "y": 0.1761904542362559,
        "y_hat": 0.5157292008835385
      },
      {
        "dow": 2,
        "t": "2022-11-09T00:00:00",
        "wom": 2,
        "y": 0.9624710144321974,
        "y_hat": 0.46314486020995904
      },
      {
        "dow": 3,
        "t": "2022-11-10T00:00:00",
        "wom": 2,
        "y": 0.7059542876896551,
        "y_hat": 0.5091456448776946
      },
      {
        "dow": 4,
        "t": "2022-11-11T00:00:00",
        "wom": 2,
        "y": 2.506728336965063,
        "y_hat": 2.46643581246614
      },
      {
        "dow": 5,
        "t": "2022-11-12T00:00:00",
        "wom": 2,
        "y": 0.8433091095294604,
        "y_hat": 0.5444135866449329
      },
      {
        "dow": 6,
        "t": "2022-11-13T00:00:00",
        "wom": 2,
        "y": 0.7560702301487369,
        "y_hat": 0.5563050037044083
      },
      {
        "dow": 0,
        "t": "2022-11-14T00:00:00",
        "wom": 3,
        "y": 0.06592714264449118,
        "y_hat": 0.4974098519973051
      },
      {
        "dow": 1,
        "t": "2022-11-15T00:00:00",
        "wom": 3,
        "y": 0.2504072631614749,
        "y_hat": 0.5074849113317751
      },
      {
        "dow": 2,
        "t": "2022-11-16T00:00:00",
        "wom": 3,
        "y": 0.5805948341045363,
        "y_hat": 0.44029974147134476
      },
      {
        "dow": 3,
        "t": "2022-11-17T00:00:00",
        "wom": 3,
        "y": 0.5803663521377741,
        "y_hat": 0.5023523218787471
      },
      {
        "dow": 4,
        "t": "2022-11-18T00:00:00",
        "wom": 3,
        "y": 2.044237322520029,
        "y_hat": 2.44234290195584
      },
      {
        "dow": 5,
        "t": "2022-11-19T00:00:00",
        "wom": 3,
        "y": 0.5560838530176727,
        "y_hat": 0.5245297729912423
      },
      {
        "dow": 6,
        "t": "2022-11-20T00:00:00",
        "wom": 3,
        "y": 0.28756910769465927,
        "y_hat": 0.5232410661077838
      },
      {
        "dow": 0,
        "t": "2022-11-21T00:00:00",
        "wom": 4,
        "y": 0.32658389395269083,
        "y_hat": 0.46862796297499915
      },
      {
        "dow": 1,
        "t": "2022-11-22T00:00:00",
        "wom": 4,
        "y": 0.9255075128439307,
        "y_hat": 0.4897524974527008
      },
      {
        "dow": 2,
        "t": "2022-11-23T00:00:00",
        "wom": 4,
        "y": 0.2351401173462816,
        "y_hat": 0.4737753846011047
      },
      {
        "dow": 3,
        "t": "2022-11-24T00:00:00",
        "wom": 4,
        "y": 0.6755009569121625,
        "y_hat": 0.5290397268854274
      },
      {
        "dow": 4,
        "t": "2022-11-25T00:00:00",
        "wom": 4,
        "y": 2.6532627672211397,
        "y_hat": 2.4265235342689375
      },
      {
        "dow": 5,
        "t": "2022-11-26T00:00:00",
        "wom": 4,
        "y": 0.13334860717528818,
        "y_hat": 0.5471812325695693
      },
      {
        "dow": 6,
        "t": "2022-11-27T00:00:00",
        "wom": 4,
        "y": 0.21224946729617233,
        "y_hat": 0.5441529542842961
      },
      {
        "dow": 0,
        "t": "2022-11-28T00:00:00",
        "wom": 5,
        "y": 0.10485005588135898,
        "y_hat": 0.4437656298775255
      },
      {
        "dow": 1,
        "t": "2022-11-29T00:00:00",
        "wom": 5,
        "y": 0.571653845644994,
        "y_hat": 0.4794622765680875
      },
      {
        "dow": 2,
        "t": "2022-11-30T00:00:00",
        "wom": 5,
        "y": 0.6455551020000664,
        "y_hat": 0.451805294954433
      },
      {
        "dow": 3,
        "t": "2022-12-01T00:00:00",
        "wom": 1,
        "y": 0.07205418156187893,
        "y_hat": 0.5217762126874631
      },
      {
        "dow": 4,
        "t": "2022-12-02T00:00:00",
        "wom": 1,
        "y": 2.8390812934919314,
        "y_hat": 2.4315490965073923
      },
      {
        "dow": 5,
        "t": "2022-12-03T00:00:00",
        "wom": 1,
        "y": 0.3495218764848188,
        "y_hat": 0.5200249530600702
      },
      {
        "dow": 6,
        "t": "2022-12-04T00:00:00",
        "wom": 1,
        "y": 0.9605832472674337,
        "y_hat": 0.5604742376767212
      },
      {
        "dow": 0,
        "t": "2022-12-05T00:00:00",
        "wom": 2,
        "y": 0.47976044149700026,
        "y_hat": 0.4769324212585106
      },
      {
        "dow": 1,
        "t": "2022-12-06T00:00:00",
        "wom": 2,
        "y": 0.6055161236619802,
        "y_hat": 0.5306881286657961
      },
      {
        "dow": 2,
        "t": "2022-12-07T00:00:00",
        "wom": 2,
        "y": 0.41871964951333474,
        "y_hat": 0.47312547912871106
      },
      {
        "dow": 3,
        "t": "2022-12-08T00:00:00",
        "wom": 2,
        "y": 0.966612891389878,
        "y_hat": 0.5172346281782415
      },
      {
        "dow": 4,
        "t": "2022-12-09T00:00:00",
        "wom": 2,
        "y": 2.5654467983822786,
        "y_hat": 2.4449775700971856
      },
      {
        "dow": 5,
        "t": "2022-12-10T00:00:00",
        "wom": 2,
        "y": 0.34560834727361145,
        "y_hat": 0.5606125859661195
      },
      {
        "dow": 6,
        "t": "2022-12-11T00:00:00",
        "wom": 2,
        "y": 0.7570085832854714,
        "y_hat": 0.5457018888866806
      },
      {
        "dow": 0,
        "t": "2022-12-12T00:00:00",
        "wom": 3,
        "y": 0.9652744314834171,
        "y_hat": 0.4710161178959069
      },
      {
        "dow": 1,
        "t": "2022-12-13T00:00:00",
        "wom": 3,
        "y": 0.5648309593161226,
        "y_hat": 0.5336837354630672
      },
      {
        "dow": 2,
        "t": "2022-12-14T00:00:00",
        "wom": 3,
        "y": 0.7595290379916108,
        "y_hat": 0.49828660444260364
      },
      {
        "dow": 3,
        "t": "2022-12-15T00:00:00",
        "wom": 3,
        "y": 0.4539146559253099,
        "y_hat": 0.5248891664811817
      },
      {
        "dow": 4,
        "t": "2022-12-16T00:00:00",
        "wom": 3,
        "y": 2.6626571950380864,
        "y_hat": 2.448838509795448
      },
      {
        "dow": 5,
        "t": "2022-12-17T00:00:00",
        "wom": 3,
        "y": 0.901352733645678,
        "y_hat": 0.5358235389156042
      },
      {
        "dow": 6,
        "t": "2022-12-18T00:00:00",
        "wom": 3,
        "y": 0.1389757185937477,
        "y_hat": 0.5669428825994003
      },
      {
        "dow": 0,
        "t": "2022-12-19T00:00:00",
        "wom": 4,
        "y": 0.523376913529589,
        "y_hat": 0.4834707916350209
      },
      {
        "dow": 1,
        "t": "2022-12-20T00:00:00",
        "wom": 4,
        "y": 0.22793475704817479,
        "y_hat": 0.48780270680743293
      },
      {
        "dow": 2,
        "t": "2022-12-21T00:00:00",
        "wom": 4,
        "y": 0.3918232223846172,
        "y_hat": 0.46491507924850445
      },
      {
        "dow": 3,
        "t": "2022-12-22T00:00:00",
        "wom": 4,
        "y": 0.47576866552495234,
        "y_hat": 0.4963059605100148
      },
      {
        "dow": 4,
        "t": "2022-12-23T00:00:00",
        "wom": 4,
        "y": 2.3131731287518735,
        "y_hat": 2.4297238410504325
      },
      {
        "dow": 5,
        "t": "2022-12-24T00:00:00",
        "wom": 4,
        "y": 0.11547575723384762,
        "y_hat": 0.5271289193762412
      },
      {
        "dow": 6,
        "t": "2022-12-25T00:00:00",
        "wom": 4,
        "y": 0.880622203620008,
        "y_hat": 0.5257306518423508
      },
      {
        "dow": 0,
        "t": "2022-12-26T00:00:00",
        "wom": 5,
        "y": 0.4962845210256144,
        "y_hat": 0.4623486822690522
      },
      {
        "dow": 1,
        "t": "2022-12-27T00:00:00",
        "wom": 5,
        "y": 0.5406198774645495,
        "y_hat": 0.5271801081764019
      },
      {
        "dow": 2,
        "t": "2022-12-28T00:00:00",
        "wom": 5,
        "y": 0.26315431946537116,
        "y_hat": 0.4724997876723675
      },
      {
        "dow": 3,
        "t": "2022-12-29T00:00:00",
        "wom": 5,
        "y": 0.0927341711398395,
        "y_hat": 0.5096223750049854
      },
      {
        "dow": 4,
        "t": "2022-12-30T00:00:00",
        "wom": 5,
        "y": 2.42594516701449,
        "y_hat": 2.4119442031524905
      },
      {
        "dow": 5,
        "t": "2022-12-31T00:00:00",
        "wom": 5,
        "y": 0.9191390581073097,
        "y_hat": 0.5097847258328849
      }
    ]
  },
  "layer": [
    {
      "encoding": {
        "tooltip": {
          "field": "t",
          "type": "temporal"
        },
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "darkblue",
        "opacity": 0.6,
        "type": "circle"
      },
      "selection": {
        "selector002": {
          "bind": "scales",
          "type": "interval"
        }
      },
      "title": "",
      "width": 400
    },
    {
      "encoding": {
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y_hat",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "t",
        "type": "line"
      },
      "title": "4 lags, period: 7, MAE: 0.243",
      "width": 400
    }
  ]
}

If we increase `lags=7` to include the weekly effect, we get
almost as good a model as with seasonal terms:

```python
_, s3params, _, stage3 = run_autoreg(ts.y,
                     lags_=7,
                     seasonal_=False,
                    )
```

var spec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
  "config": {
    "view": {
      "continuousHeight": 300,
      "continuousWidth": 400
    }
  },
  "data": {
    "name": "data-460851078ef436e285d50a6cff3c9a96"
  },
  "datasets": {
    "data-460851078ef436e285d50a6cff3c9a96": [
      {
        "dow": 5,
        "t": "2022-01-01T00:00:00",
        "wom": 1,
        "y": 0.9955981828907805,
        "y_hat": null
      },
      {
        "dow": 6,
        "t": "2022-01-02T00:00:00",
        "wom": 1,
        "y": 0.6234109104925726,
        "y_hat": null
      },
      {
        "dow": 0,
        "t": "2022-01-03T00:00:00",
        "wom": 2,
        "y": 0.8760936874425284,
        "y_hat": 0.4967907586604228
      },
      {
        "dow": 1,
        "t": "2022-01-04T00:00:00",
        "wom": 2,
        "y": 0.8893778611747504,
        "y_hat": 0.5184239775556261
      },
      {
        "dow": 2,
        "t": "2022-01-05T00:00:00",
        "wom": 2,
        "y": 0.3867624782099456,
        "y_hat": 0.4972472553191131
      },
      {
        "dow": 3,
        "t": "2022-01-06T00:00:00",
        "wom": 2,
        "y": 0.6899198960223786,
        "y_hat": 0.5262597140421478
      },
      {
        "dow": 4,
        "t": "2022-01-07T00:00:00",
        "wom": 2,
        "y": 2.992852550566659,
        "y_hat": 2.4299590582515473
      },
      {
        "dow": 5,
        "t": "2022-01-08T00:00:00",
        "wom": 2,
        "y": 0.46804208563249405,
        "y_hat": 0.5525014522166138
      },
      {
        "dow": 6,
        "t": "2022-01-09T00:00:00",
        "wom": 2,
        "y": 0.9455841386245034,
        "y_hat": 0.5668882974435845
      },
      {
        "dow": 0,
        "t": "2022-01-10T00:00:00",
        "wom": 3,
        "y": 0.6381233538848571,
        "y_hat": 0.4776279996404177
      },
      {
        "dow": 1,
        "t": "2022-01-11T00:00:00",
        "wom": 3,
        "y": 0.7595024746539746,
        "y_hat": 0.5291292866690476
      },
      {
        "dow": 2,
        "t": "2022-01-12T00:00:00",
        "wom": 3,
        "y": 0.23738191602671632,
        "y_hat": 0.4808008912256273
      },
      {
        "dow": 3,
        "t": "2022-01-13T00:00:00",
        "wom": 3,
        "y": 0.09680924548766545,
        "y_hat": 0.515091715171366
      },
      {
        "dow": 4,
        "t": "2022-01-14T00:00:00",
        "wom": 3,
        "y": 2.588571863863769,
        "y_hat": 2.405058230192902
      },
      {
        "dow": 5,
        "t": "2022-01-15T00:00:00",
        "wom": 3,
        "y": 0.16573502501513393,
        "y_hat": 0.5090408212208333
      },
      {
        "dow": 6,
        "t": "2022-01-16T00:00:00",
        "wom": 3,
        "y": 0.38942124330865624,
        "y_hat": 0.5365359980531161
      },
      {
        "dow": 0,
        "t": "2022-01-17T00:00:00",
        "wom": 4,
        "y": 0.688787521402102,
        "y_hat": 0.44552691006817247
      },
      {
        "dow": 1,
        "t": "2022-01-18T00:00:00",
        "wom": 4,
        "y": 0.6697600544042848,
        "y_hat": 0.5006636268518205
      },
      {
        "dow": 2,
        "t": "2022-01-19T00:00:00",
        "wom": 4,
        "y": 0.36885391553882474,
        "y_hat": 0.4810838917599954
      },
      {
        "dow": 3,
        "t": "2022-01-20T00:00:00",
        "wom": 4,
        "y": 0.9325008856726638,
        "y_hat": 0.5141158767799752
      },
      {
        "dow": 4,
        "t": "2022-01-21T00:00:00",
        "wom": 4,
        "y": 2.141135576285614,
        "y_hat": 2.436148536768076
      },
      {
        "dow": 5,
        "t": "2022-01-22T00:00:00",
        "wom": 4,
        "y": 0.9772509899400809,
        "y_hat": 0.5414839206074327
      },
      {
        "dow": 6,
        "t": "2022-01-23T00:00:00",
        "wom": 4,
        "y": 0.5683493004574759,
        "y_hat": 0.5356818812038745
      },
      {
        "dow": 0,
        "t": "2022-01-24T00:00:00",
        "wom": 5,
        "y": 0.9649672826390342,
        "y_hat": 0.4945634970531389
      },
      {
        "dow": 1,
        "t": "2022-01-25T00:00:00",
        "wom": 5,
        "y": 0.22830327875114365,
        "y_hat": 0.5183276280490623
      },
      {
        "dow": 2,
        "t": "2022-01-26T00:00:00",
        "wom": 5,
        "y": 0.5236014315750456,
        "y_hat": 0.4834914229513924
      },
      {
        "dow": 3,
        "t": "2022-01-27T00:00:00",
        "wom": 5,
        "y": 0.5771602890446812,
        "y_hat": 0.49481710194259054
      },
      {
        "dow": 4,
        "t": "2022-01-28T00:00:00",
        "wom": 5,
        "y": 2.3198686688954586,
        "y_hat": 2.4344611018946396
      },
      {
        "dow": 5,
        "t": "2022-01-29T00:00:00",
        "wom": 5,
        "y": 0.4375103099359736,
        "y_hat": 0.5275197097634949
      },
      {
        "dow": 6,
        "t": "2022-01-30T00:00:00",
        "wom": 5,
        "y": 0.7071449184091295,
        "y_hat": 0.5300200373997083
      },
      {
        "dow": 0,
        "t": "2022-01-31T00:00:00",
        "wom": 6,
        "y": 0.1524637438937454,
        "y_hat": 0.46950282399087084
      },
      {
        "dow": 1,
        "t": "2022-02-01T00:00:00",
        "wom": 1,
        "y": 0.011748418393797033,
        "y_hat": 0.5027153449515473
      },
      {
        "dow": 2,
        "t": "2022-02-02T00:00:00",
        "wom": 1,
        "y": 0.25517769811183577,
        "y_hat": 0.4335507573741657
      },
      {
        "dow": 3,
        "t": "2022-02-03T00:00:00",
        "wom": 1,
        "y": 0.5156949359371945,
        "y_hat": 0.47556781187960434
      },
      {
        "dow": 4,
        "t": "2022-02-04T00:00:00",
        "wom": 1,
        "y": 2.6710173501944316,
        "y_hat": 2.418325496049474
      },
      {
        "dow": 5,
        "t": "2022-02-05T00:00:00",
        "wom": 1,
        "y": 0.8240477378755111,
        "y_hat": 0.5343478357671186
      },
      {
        "dow": 6,
        "t": "2022-02-06T00:00:00",
        "wom": 1,
        "y": 0.9508838016129778,
        "y_hat": 0.5601352577647144
      },
      {
        "dow": 0,
        "t": "2022-02-07T00:00:00",
        "wom": 2,
        "y": 0.05520043850117129,
        "y_hat": 0.4974480437451285
      },
      {
        "dow": 1,
        "t": "2022-02-08T00:00:00",
        "wom": 2,
        "y": 0.36173925787444183,
        "y_hat": 0.513206980650459
      },
      {
        "dow": 2,
        "t": "2022-02-09T00:00:00",
        "wom": 2,
        "y": 0.3287722160717247,
        "y_hat": 0.4384131691013086
      },
      {
        "dow": 3,
        "t": "2022-02-10T00:00:00",
        "wom": 2,
        "y": 0.26344459815995447,
        "y_hat": 0.49667802938134803
      },
      {
        "dow": 4,
        "t": "2022-02-11T00:00:00",
        "wom": 2,
        "y": 2.105522116894078,
        "y_hat": 2.415202698211017
      },
      {
        "dow": 5,
        "t": "2022-02-12T00:00:00",
        "wom": 2,
        "y": 0.9004373252539722,
        "y_hat": 0.5046824885073314
      },
      {
        "dow": 6,
        "t": "2022-02-13T00:00:00",
        "wom": 2,
        "y": 0.3083956070489431,
        "y_hat": 0.5319008881433882
      },
      {
        "dow": 0,
        "t": "2022-02-14T00:00:00",
        "wom": 3,
        "y": 0.8443045110438598,
        "y_hat": 0.483324909023404
      },
      {
        "dow": 1,
        "t": "2022-02-15T00:00:00",
        "wom": 3,
        "y": 0.9448650092640142,
        "y_hat": 0.5011821422416533
      },
      {
        "dow": 2,
        "t": "2022-02-16T00:00:00",
        "wom": 3,
        "y": 0.5570587072891441,
        "y_hat": 0.4977899462465503
      },
      {
        "dow": 3,
        "t": "2022-02-17T00:00:00",
        "wom": 3,
        "y": 0.5374558873035822,
        "y_hat": 0.5347949272417398
      },
      {
        "dow": 4,
        "t": "2022-02-18T00:00:00",
        "wom": 3,
        "y": 2.8223117437123824,
        "y_hat": 2.4354694540369994
      },
      {
        "dow": 5,
        "t": "2022-02-19T00:00:00",
        "wom": 3,
        "y": 0.02226637799800346,
        "y_hat": 0.5400703987126106
      },
      {
        "dow": 6,
        "t": "2022-02-20T00:00:00",
        "wom": 3,
        "y": 0.15725472652557881,
        "y_hat": 0.5456163598479873
      },
      {
        "dow": 0,
        "t": "2022-02-21T00:00:00",
        "wom": 4,
        "y": 0.9269915515225546,
        "y_hat": 0.43170828507018294
      },
      {
        "dow": 1,
        "t": "2022-02-22T00:00:00",
        "wom": 4,
        "y": 0.4501913471576654,
        "y_hat": 0.4954975240418244
      },
      {
        "dow": 2,
        "t": "2022-02-23T00:00:00",
        "wom": 4,
        "y": 0.9531441262485346,
        "y_hat": 0.48823066431783424
      },
      {
        "dow": 3,
        "t": "2022-02-24T00:00:00",
        "wom": 4,
        "y": 0.5844501255113494,
        "y_hat": 0.519519472671327
      },
      {
        "dow": 4,
        "t": "2022-02-25T00:00:00",
        "wom": 4,
        "y": 2.80397776091471,
        "y_hat": 2.458308069387769
      },
      {
        "dow": 5,
        "t": "2022-02-26T00:00:00",
        "wom": 4,
        "y": 0.26383392179603904,
        "y_hat": 0.542195800807761
      },
      {
        "dow": 6,
        "t": "2022-02-27T00:00:00",
        "wom": 4,
        "y": 0.7531464817944895,
        "y_hat": 0.5516416906073818
      },
      {
        "dow": 0,
        "t": "2022-02-28T00:00:00",
        "wom": 5,
        "y": 0.15885513834885823,
        "y_hat": 0.4618899359780873
      },
      {
        "dow": 1,
        "t": "2022-03-01T00:00:00",
        "wom": 1,
        "y": 0.14535499394438622,
        "y_hat": 0.5058303574877435
      },
      {
        "dow": 2,
        "t": "2022-03-02T00:00:00",
        "wom": 1,
        "y": 0.4410675050392091,
        "y_hat": 0.43816256596576625
      },
      {
        "dow": 3,
        "t": "2022-03-03T00:00:00",
        "wom": 1,
        "y": 0.7517726853328405,
        "y_hat": 0.48854162038072846
      },
      {
        "dow": 4,
        "t": "2022-03-04T00:00:00",
        "wom": 1,
        "y": 2.7945001756665624,
        "y_hat": 2.435556070318829
      },
      {
        "dow": 5,
        "t": "2022-03-05T00:00:00",
        "wom": 1,
        "y": 0.04870630413527299,
        "y_hat": 0.5510704584107008
      },
      {
        "dow": 6,
        "t": "2022-03-06T00:00:00",
        "wom": 1,
        "y": 0.2161041338955525,
        "y_hat": 0.5450948686636378
      },
      {
        "dow": 0,
        "t": "2022-03-07T00:00:00",
        "wom": 2,
        "y": 0.37010898460177155,
        "y_hat": 0.435041792407649
      },
      {
        "dow": 1,
        "t": "2022-03-08T00:00:00",
        "wom": 2,
        "y": 0.09520299889511219,
        "y_hat": 0.48298588836607886
      },
      {
        "dow": 2,
        "t": "2022-03-09T00:00:00",
        "wom": 2,
        "y": 0.11977796440076782,
        "y_hat": 0.44824665641173894
      },
      {
        "dow": 3,
        "t": "2022-03-10T00:00:00",
        "wom": 2,
        "y": 0.5000495405840498,
        "y_hat": 0.47676527608495756
      },
      {
        "dow": 4,
        "t": "2022-03-11T00:00:00",
        "wom": 2,
        "y": 2.453037332781335,
        "y_hat": 2.411129683739495
      },
      {
        "dow": 5,
        "t": "2022-03-12T00:00:00",
        "wom": 2,
        "y": 0.1454828110192531,
        "y_hat": 0.5278354061996601
      },
      {
        "dow": 6,
        "t": "2022-03-13T00:00:00",
        "wom": 2,
        "y": 0.37079700609072885,
        "y_hat": 0.5295379288986366
      },
      {
        "dow": 0,
        "t": "2022-03-14T00:00:00",
        "wom": 3,
        "y": 0.6368565984388843,
        "y_hat": 0.4447991215598311
      },
      {
        "dow": 1,
        "t": "2022-03-15T00:00:00",
        "wom": 3,
        "y": 0.5235082648923333,
        "y_hat": 0.4990719735426933
      },
      {
        "dow": 2,
        "t": "2022-03-16T00:00:00",
        "wom": 3,
        "y": 0.7192823846677754,
        "y_hat": 0.4749988421987305
      },
      {
        "dow": 3,
        "t": "2022-03-17T00:00:00",
        "wom": 3,
        "y": 0.6045077428498761,
        "y_hat": 0.5171315926773352
      },
      {
        "dow": 4,
        "t": "2022-03-18T00:00:00",
        "wom": 3,
        "y": 2.5404792929526683,
        "y_hat": 2.446592365808845
      },
      {
        "dow": 5,
        "t": "2022-03-19T00:00:00",
        "wom": 3,
        "y": 0.4830778428824394,
        "y_hat": 0.5360857207829948
      },
      {
        "dow": 6,
        "t": "2022-03-20T00:00:00",
        "wom": 3,
        "y": 0.9022692634866183,
        "y_hat": 0.5440178300153817
      },
      {
        "dow": 0,
        "t": "2022-03-21T00:00:00",
        "wom": 4,
        "y": 0.06386988650342573,
        "y_hat": 0.4783242019394053
      },
      {
        "dow": 1,
        "t": "2022-03-22T00:00:00",
        "wom": 4,
        "y": 0.2631672977407843,
        "y_hat": 0.5115034804148847
      },
      {
        "dow": 2,
        "t": "2022-03-23T00:00:00",
        "wom": 4,
        "y": 0.6967761906616411,
        "y_hat": 0.43673779150020503
      },
      {
        "dow": 3,
        "t": "2022-03-24T00:00:00",
        "wom": 4,
        "y": 0.6938964314225743,
        "y_hat": 0.5025456288446649
      },
      {
        "dow": 4,
        "t": "2022-03-25T00:00:00",
        "wom": 4,
        "y": 2.3402313329841156,
        "y_hat": 2.448043866842587
      },
      {
        "dow": 5,
        "t": "2022-03-26T00:00:00",
        "wom": 4,
        "y": 0.060442940833932424,
        "y_hat": 0.5353015977298601
      },
      {
        "dow": 6,
        "t": "2022-03-27T00:00:00",
        "wom": 4,
        "y": 0.8673761163808922,
        "y_hat": 0.5212422441741162
      },
      {
        "dow": 0,
        "t": "2022-03-28T00:00:00",
        "wom": 5,
        "y": 0.5544851701436769,
        "y_hat": 0.4546224926233655
      },
      {
        "dow": 1,
        "t": "2022-03-29T00:00:00",
        "wom": 5,
        "y": 0.8547133638306371,
        "y_hat": 0.5237514886317916
      },
      {
        "dow": 2,
        "t": "2022-03-30T00:00:00",
        "wom": 5,
        "y": 0.36375693048623126,
        "y_hat": 0.4802406327655775
      },
      {
        "dow": 3,
        "t": "2022-03-31T00:00:00",
        "wom": 5,
        "y": 0.7322858415787377,
        "y_hat": 0.5250775331729335
      },
      {
        "dow": 4,
        "t": "2022-04-01T00:00:00",
        "wom": 1,
        "y": 2.1005642322427573,
        "y_hat": 2.4312743324017627
      },
      {
        "dow": 5,
        "t": "2022-04-02T00:00:00",
        "wom": 1,
        "y": 0.6446246806801427,
        "y_hat": 0.530637749525594
      },
      {
        "dow": 6,
        "t": "2022-04-03T00:00:00",
        "wom": 1,
        "y": 0.523704664544059,
        "y_hat": 0.5251088026683788
      },
      {
        "dow": 0,
        "t": "2022-04-04T00:00:00",
        "wom": 2,
        "y": 0.6478284421164979,
        "y_hat": 0.47645242667294535
      },
      {
        "dow": 1,
        "t": "2022-04-05T00:00:00",
        "wom": 2,
        "y": 0.8035144383488594,
        "y_hat": 0.5079772060871229
      },
      {
        "dow": 2,
        "t": "2022-04-06T00:00:00",
        "wom": 2,
        "y": 0.06175772835110849,
        "y_hat": 0.4839291542047948
      },
      {
        "dow": 3,
        "t": "2022-04-07T00:00:00",
        "wom": 2,
        "y": 0.10101798118758876,
        "y_hat": 0.5137958988269249
      },
      {
        "dow": 4,
        "t": "2022-04-08T00:00:00",
        "wom": 2,
        "y": 2.328695425168449,
        "y_hat": 2.3970437026869673
      },
      {
        "dow": 5,
        "t": "2022-04-09T00:00:00",
        "wom": 2,
        "y": 0.7405205134959488,
        "y_hat": 0.503188594546576
      },
      {
        "dow": 6,
        "t": "2022-04-10T00:00:00",
        "wom": 2,
        "y": 0.08236194536441421,
        "y_hat": 0.5402774394697502
      },
      {
        "dow": 0,
        "t": "2022-04-11T00:00:00",
        "wom": 3,
        "y": 0.48567949453078907,
        "y_hat": 0.4691302067870829
      },
      {
        "dow": 1,
        "t": "2022-04-12T00:00:00",
        "wom": 3,
        "y": 0.5552603671504405,
        "y_hat": 0.4796291198149867
      },
      {
        "dow": 2,
        "t": "2022-04-13T00:00:00",
        "wom": 3,
        "y": 0.2511650942997945,
        "y_hat": 0.4681934405589045
      },
      {
        "dow": 3,
        "t": "2022-04-14T00:00:00",
        "wom": 3,
        "y": 0.4804870698202284,
        "y_hat": 0.5059180037110255
      },
      {
        "dow": 4,
        "t": "2022-04-15T00:00:00",
        "wom": 3,
        "y": 2.0257783736909825,
        "y_hat": 2.4182249938567857
      },
      {
        "dow": 5,
        "t": "2022-04-16T00:00:00",
        "wom": 3,
        "y": 0.9687966319838246,
        "y_hat": 0.5151312584930192
      },
      {
        "dow": 6,
        "t": "2022-04-17T00:00:00",
        "wom": 3,
        "y": 0.7151278400228451,
        "y_hat": 0.5305591407286968
      },
      {
        "dow": 0,
        "t": "2022-04-18T00:00:00",
        "wom": 4,
        "y": 0.4718442940718869,
        "y_hat": 0.49964798819154504
      },
      {
        "dow": 1,
        "t": "2022-04-19T00:00:00",
        "wom": 4,
        "y": 0.9026979968878186,
        "y_hat": 0.5135072021996053
      },
      {
        "dow": 2,
        "t": "2022-04-20T00:00:00",
        "wom": 4,
        "y": 0.39245084658370566,
        "y_hat": 0.47748693630179173
      },
      {
        "dow": 3,
        "t": "2022-04-21T00:00:00",
        "wom": 4,
        "y": 0.4827795920741036,
        "y_hat": 0.5288246903791284
      },
      {
        "dow": 4,
        "t": "2022-04-22T00:00:00",
        "wom": 4,
        "y": 2.551414450198103,
        "y_hat": 2.4260303356637705
      },
      {
        "dow": 5,
        "t": "2022-04-23T00:00:00",
        "wom": 4,
        "y": 0.650135286544272,
        "y_hat": 0.5303875499655886
      },
      {
        "dow": 6,
        "t": "2022-04-24T00:00:00",
        "wom": 4,
        "y": 0.467587186604593,
        "y_hat": 0.549943038786494
      },
      {
        "dow": 0,
        "t": "2022-04-25T00:00:00",
        "wom": 5,
        "y": 0.8420434940525285,
        "y_hat": 0.4754830070083764
      },
      {
        "dow": 1,
        "t": "2022-04-26T00:00:00",
        "wom": 5,
        "y": 0.44245377300469013,
        "y_hat": 0.5108340361542065
      },
      {
        "dow": 2,
        "t": "2022-04-27T00:00:00",
        "wom": 5,
        "y": 0.10472408289231805,
        "y_hat": 0.4844334459505021
      },
      {
        "dow": 3,
        "t": "2022-04-28T00:00:00",
        "wom": 5,
        "y": 0.07965539705381441,
        "y_hat": 0.4958677197277991
      },
      {
        "dow": 4,
        "t": "2022-04-29T00:00:00",
        "wom": 5,
        "y": 2.1204279487951343,
        "y_hat": 2.3990895488892217
      },
      {
        "dow": 5,
        "t": "2022-04-30T00:00:00",
        "wom": 5,
        "y": 0.09911422212585952,
        "y_hat": 0.4964206092595712
      },
      {
        "dow": 6,
        "t": "2022-05-01T00:00:00",
        "wom": 1,
        "y": 0.7653139992562019,
        "y_hat": 0.5110418469407004
      },
      {
        "dow": 0,
        "t": "2022-05-02T00:00:00",
        "wom": 2,
        "y": 0.6928765519708857,
        "y_hat": 0.45435487387021417
      },
      {
        "dow": 1,
        "t": "2022-05-03T00:00:00",
        "wom": 2,
        "y": 0.3293225432083243,
        "y_hat": 0.5227571533370909
      },
      {
        "dow": 2,
        "t": "2022-05-04T00:00:00",
        "wom": 2,
        "y": 0.6649514990871527,
        "y_hat": 0.47325979365624726
      },
      {
        "dow": 3,
        "t": "2022-05-05T00:00:00",
        "wom": 2,
        "y": 0.15101171079274767,
        "y_hat": 0.5058810792932257
      },
      {
        "dow": 4,
        "t": "2022-05-06T00:00:00",
        "wom": 2,
        "y": 2.1798428980656723,
        "y_hat": 2.431485876177647
      },
      {
        "dow": 5,
        "t": "2022-05-07T00:00:00",
        "wom": 2,
        "y": 0.2354835900991824,
        "y_hat": 0.5020829628448384
      },
      {
        "dow": 6,
        "t": "2022-05-08T00:00:00",
        "wom": 2,
        "y": 0.05741790510662548,
        "y_hat": 0.5182585301985556
      },
      {
        "dow": 0,
        "t": "2022-05-09T00:00:00",
        "wom": 3,
        "y": 0.23077326722872582,
        "y_hat": 0.4416008126128215
      },
      {
        "dow": 1,
        "t": "2022-05-10T00:00:00",
        "wom": 3,
        "y": 0.1944400629234596,
        "y_hat": 0.4714472781064954
      },
      {
        "dow": 2,
        "t": "2022-05-11T00:00:00",
        "wom": 3,
        "y": 0.7807354431480255,
        "y_hat": 0.44457005664505445
      },
      {
        "dow": 3,
        "t": "2022-05-12T00:00:00",
        "wom": 3,
        "y": 0.31021821068384303,
        "y_hat": 0.5020199652718704
      },
      {
        "dow": 4,
        "t": "2022-05-13T00:00:00",
        "wom": 3,
        "y": 2.1275853244649534,
        "y_hat": 2.4423983412290537
      },
      {
        "dow": 5,
        "t": "2022-05-14T00:00:00",
        "wom": 3,
        "y": 0.56813301154376,
        "y_hat": 0.5092969989732901
      },
      {
        "dow": 6,
        "t": "2022-05-15T00:00:00",
        "wom": 3,
        "y": 0.10167470187610927,
        "y_hat": 0.5250551324538184
      },
      {
        "dow": 0,
        "t": "2022-05-16T00:00:00",
        "wom": 4,
        "y": 0.37826726087578866,
        "y_hat": 0.46093647736965676
      },
      {
        "dow": 1,
        "t": "2022-05-17T00:00:00",
        "wom": 4,
        "y": 0.7138146434013722,
        "y_hat": 0.4781635078824496
      },
      {
        "dow": 2,
        "t": "2022-05-18T00:00:00",
        "wom": 4,
        "y": 0.12852670113262654,
        "y_hat": 0.46748643229002546
      },
      {
        "dow": 3,
        "t": "2022-05-19T00:00:00",
        "wom": 4,
        "y": 0.3585977195944233,
        "y_hat": 0.5115345457435468
      },
      {
        "dow": 4,
        "t": "2022-05-20T00:00:00",
        "wom": 4,
        "y": 2.22258022407796,
        "y_hat": 2.408682157424716
      },
      {
        "dow": 5,
        "t": "2022-05-21T00:00:00",
        "wom": 4,
        "y": 0.932770577180065,
        "y_hat": 0.5147356152220035
      },
      {
        "dow": 6,
        "t": "2022-05-22T00:00:00",
        "wom": 4,
        "y": 0.9336405466022045,
        "y_hat": 0.5407155484496845
      },
      {
        "dow": 0,
        "t": "2022-05-23T00:00:00",
        "wom": 5,
        "y": 0.7765773218651726,
        "y_hat": 0.5045082552262019
      },
      {
        "dow": 1,
        "t": "2022-05-24T00:00:00",
        "wom": 5,
        "y": 0.9092507321087694,
        "y_hat": 0.5345730713865291
      },
      {
        "dow": 2,
        "t": "2022-05-25T00:00:00",
        "wom": 5,
        "y": 0.9912972693958269,
        "y_hat": 0.4946869470039545
      },
      {
        "dow": 3,
        "t": "2022-05-26T00:00:00",
        "wom": 5,
        "y": 0.17273741694999156,
        "y_hat": 0.5468518827619808
      },
      {
        "dow": 4,
        "t": "2022-05-27T00:00:00",
        "wom": 5,
        "y": 2.7305657647776744,
        "y_hat": 2.4500618298869807
      },
      {
        "dow": 5,
        "t": "2022-05-28T00:00:00",
        "wom": 5,
        "y": 0.9331495324820412,
        "y_hat": 0.5193297047723543
      },
      {
        "dow": 6,
        "t": "2022-05-29T00:00:00",
        "wom": 5,
        "y": 0.9747799262042263,
        "y_hat": 0.5682632750846622
      },
      {
        "dow": 0,
        "t": "2022-05-30T00:00:00",
        "wom": 6,
        "y": 0.6159827699504155,
        "y_hat": 0.5058165123564484
      },
      {
        "dow": 1,
        "t": "2022-05-31T00:00:00",
        "wom": 6,
        "y": 0.18957160929478611,
        "y_hat": 0.5323172222676634
      },
      {
        "dow": 2,
        "t": "2022-06-01T00:00:00",
        "wom": 1,
        "y": 0.08329858559961756,
        "y_hat": 0.4655639528916069
      },
      {
        "dow": 3,
        "t": "2022-06-02T00:00:00",
        "wom": 1,
        "y": 0.7811845717468601,
        "y_hat": 0.48216420003370863
      },
      {
        "dow": 4,
        "t": "2022-06-03T00:00:00",
        "wom": 1,
        "y": 2.5856317189042306,
        "y_hat": 2.41854048611295
      },
      {
        "dow": 5,
        "t": "2022-06-04T00:00:00",
        "wom": 1,
        "y": 0.9915634439508605,
        "y_hat": 0.5481486757761187
      },
      {
        "dow": 6,
        "t": "2022-06-05T00:00:00",
        "wom": 1,
        "y": 0.9909525874581544,
        "y_hat": 0.5622201296718374
      },
      {
        "dow": 0,
        "t": "2022-06-06T00:00:00",
        "wom": 2,
        "y": 0.546656872738867,
        "y_hat": 0.509544473702861
      },
      {
        "dow": 1,
        "t": "2022-06-07T00:00:00",
        "wom": 2,
        "y": 0.10472218993110805,
        "y_hat": 0.5313215376478715
      },
      {
        "dow": 2,
        "t": "2022-06-08T00:00:00",
        "wom": 2,
        "y": 0.04250860920798383,
        "y_hat": 0.45950885511347206
      },
      {
        "dow": 3,
        "t": "2022-06-09T00:00:00",
        "wom": 2,
        "y": 0.3032781068949767,
        "y_hat": 0.4765300517785524
      },
      {
        "dow": 4,
        "t": "2022-06-10T00:00:00",
        "wom": 2,
        "y": 2.5924710028233178,
        "y_hat": 2.4027941662417835
      },
      {
        "dow": 5,
        "t": "2022-06-11T00:00:00",
        "wom": 2,
        "y": 0.1823450459040853,
        "y_hat": 0.5226555382078407
      },
      {
        "dow": 6,
        "t": "2022-06-12T00:00:00",
        "wom": 2,
        "y": 0.7714328053698969,
        "y_hat": 0.5395777784515343
      },
      {
        "dow": 0,
        "t": "2022-06-13T00:00:00",
        "wom": 3,
        "y": 0.7117102710960721,
        "y_hat": 0.4596964393488171
      },
      {
        "dow": 1,
        "t": "2022-06-14T00:00:00",
        "wom": 3,
        "y": 0.5078428447537274,
        "y_hat": 0.5242989977320872
      },
      {
        "dow": 2,
        "t": "2022-06-15T00:00:00",
        "wom": 3,
        "y": 0.4559806484621759,
        "y_hat": 0.48005120057372996
      },
      {
        "dow": 3,
        "t": "2022-06-16T00:00:00",
        "wom": 3,
        "y": 0.3663298759696507,
        "y_hat": 0.5102207588634888
      },
      {
        "dow": 4,
        "t": "2022-06-17T00:00:00",
        "wom": 3,
        "y": 2.036192987755432,
        "y_hat": 2.427030286845141
      },
      {
        "dow": 5,
        "t": "2022-06-18T00:00:00",
        "wom": 3,
        "y": 0.5857499825271564,
        "y_hat": 0.5102758304570155
      },
      {
        "dow": 6,
        "t": "2022-06-19T00:00:00",
        "wom": 3,
        "y": 0.050005294335790484,
        "y_hat": 0.5211856409518846
      },
      {
        "dow": 0,
        "t": "2022-06-20T00:00:00",
        "wom": 4,
        "y": 0.5061272547669947,
        "y_hat": 0.46097219295144776
      },
      {
        "dow": 1,
        "t": "2022-06-21T00:00:00",
        "wom": 4,
        "y": 0.7904390381333787,
        "y_hat": 0.4795887904276564
      },
      {
        "dow": 2,
        "t": "2022-06-22T00:00:00",
        "wom": 4,
        "y": 0.09323256170059835,
        "y_hat": 0.4771399084781224
      },
      {
        "dow": 3,
        "t": "2022-06-23T00:00:00",
        "wom": 4,
        "y": 0.40800965871918415,
        "y_hat": 0.5152238334334927
      },
      {
        "dow": 4,
        "t": "2022-06-24T00:00:00",
        "wom": 4,
        "y": 2.570342151877391,
        "y_hat": 2.4087498026209566
      },
      {
        "dow": 5,
        "t": "2022-06-25T00:00:00",
        "wom": 4,
        "y": 0.07936746817059592,
        "y_hat": 0.5279017923132528
      },
      {
        "dow": 6,
        "t": "2022-06-26T00:00:00",
        "wom": 4,
        "y": 0.017413769472798823,
        "y_hat": 0.5356649133238703
      },
      {
        "dow": 0,
        "t": "2022-06-27T00:00:00",
        "wom": 5,
        "y": 0.4856034209379412,
        "y_hat": 0.4328149611023629
      },
      {
        "dow": 1,
        "t": "2022-06-28T00:00:00",
        "wom": 5,
        "y": 0.10626439009089561,
        "y_hat": 0.47735501023779975
      },
      {
        "dow": 2,
        "t": "2022-06-29T00:00:00",
        "wom": 5,
        "y": 0.7531865011385493,
        "y_hat": 0.4565934866610085
      },
      {
        "dow": 3,
        "t": "2022-06-30T00:00:00",
        "wom": 5,
        "y": 0.2215890993417199,
        "y_hat": 0.4972579361761011
      },
      {
        "dow": 4,
        "t": "2022-07-01T00:00:00",
        "wom": 1,
        "y": 2.2444907708382646,
        "y_hat": 2.4391639795535345
      },
      {
        "dow": 5,
        "t": "2022-07-02T00:00:00",
        "wom": 1,
        "y": 0.41066189647640805,
        "y_hat": 0.5086383437322588
      },
      {
        "dow": 6,
        "t": "2022-07-03T00:00:00",
        "wom": 1,
        "y": 0.8875209104102515,
        "y_hat": 0.5276522539533361
      },
      {
        "dow": 0,
        "t": "2022-07-04T00:00:00",
        "wom": 2,
        "y": 0.29879768250458105,
        "y_hat": 0.4756765448644015
      },
      {
        "dow": 1,
        "t": "2022-07-05T00:00:00",
        "wom": 2,
        "y": 0.8642111944128015,
        "y_hat": 0.519103746855454
      },
      {
        "dow": 2,
        "t": "2022-07-06T00:00:00",
        "wom": 2,
        "y": 0.5035308154420571,
        "y_hat": 0.468279298304797
      },
      {
        "dow": 3,
        "t": "2022-07-07T00:00:00",
        "wom": 2,
        "y": 0.4608669940295842,
        "y_hat": 0.5311554916042507
      },
      {
        "dow": 4,
        "t": "2022-07-08T00:00:00",
        "wom": 2,
        "y": 2.1336407534117914,
        "y_hat": 2.43263548967435
      },
      {
        "dow": 5,
        "t": "2022-07-09T00:00:00",
        "wom": 2,
        "y": 0.07559350276009646,
        "y_hat": 0.5185008966658524
      },
      {
        "dow": 6,
        "t": "2022-07-10T00:00:00",
        "wom": 2,
        "y": 0.21152592170248208,
        "y_hat": 0.5122052608058499
      },
      {
        "dow": 0,
        "t": "2022-07-11T00:00:00",
        "wom": 3,
        "y": 0.7992198246512687,
        "y_hat": 0.4383825054702067
      },
      {
        "dow": 1,
        "t": "2022-07-12T00:00:00",
        "wom": 3,
        "y": 0.08134234250438865,
        "y_hat": 0.49702074794504336
      },
      {
        "dow": 2,
        "t": "2022-07-13T00:00:00",
        "wom": 3,
        "y": 0.8545474044325597,
        "y_hat": 0.4730370156455408
      },
      {
        "dow": 3,
        "t": "2022-07-14T00:00:00",
        "wom": 3,
        "y": 0.7192996694052958,
        "y_hat": 0.49903334487542134
      },
      {
        "dow": 4,
        "t": "2022-07-15T00:00:00",
        "wom": 3,
        "y": 2.839731861280379,
        "y_hat": 2.4590829237763088
      },
      {
        "dow": 5,
        "t": "2022-07-16T00:00:00",
        "wom": 3,
        "y": 0.6866567952654717,
        "y_hat": 0.5527420470316485
      },
      {
        "dow": 6,
        "t": "2022-07-17T00:00:00",
        "wom": 3,
        "y": 0.3025925308532783,
        "y_hat": 0.5678987356529254
      },
      {
        "dow": 0,
        "t": "2022-07-18T00:00:00",
        "wom": 4,
        "y": 0.35527032358475563,
        "y_hat": 0.4740865465555972
      },
      {
        "dow": 1,
        "t": "2022-07-19T00:00:00",
        "wom": 4,
        "y": 0.2998867051787012,
        "y_hat": 0.4893633221283946
      },
      {
        "dow": 2,
        "t": "2022-07-20T00:00:00",
        "wom": 4,
        "y": 0.32169056052383393,
        "y_hat": 0.4554266913463155
      },
      {
        "dow": 3,
        "t": "2022-07-21T00:00:00",
        "wom": 4,
        "y": 0.8817188595233318,
        "y_hat": 0.49571750586434926
      },
      {
        "dow": 4,
        "t": "2022-07-22T00:00:00",
        "wom": 4,
        "y": 2.6482046952131784,
        "y_hat": 2.4350689441557574
      },
      {
        "dow": 5,
        "t": "2022-07-23T00:00:00",
        "wom": 4,
        "y": 0.5392997474632014,
        "y_hat": 0.5561498314017669
      },
      {
        "dow": 6,
        "t": "2022-07-24T00:00:00",
        "wom": 4,
        "y": 0.28113347961902946,
        "y_hat": 0.5534601575845951
      },
      {
        "dow": 0,
        "t": "2022-07-25T00:00:00",
        "wom": 5,
        "y": 0.4047240553693535,
        "y_hat": 0.4656301818804722
      },
      {
        "dow": 1,
        "t": "2022-07-26T00:00:00",
        "wom": 5,
        "y": 0.09656379233844459,
        "y_hat": 0.4897301947692827
      },
      {
        "dow": 2,
        "t": "2022-07-27T00:00:00",
        "wom": 5,
        "y": 0.9545109848618508,
        "y_hat": 0.45239872668182046
      },
      {
        "dow": 3,
        "t": "2022-07-28T00:00:00",
        "wom": 5,
        "y": 0.2844751590965856,
        "y_hat": 0.5029360562621922
      },
      {
        "dow": 4,
        "t": "2022-07-29T00:00:00",
        "wom": 5,
        "y": 2.19828814692932,
        "y_hat": 2.4522787916765902
      },
      {
        "dow": 5,
        "t": "2022-07-30T00:00:00",
        "wom": 5,
        "y": 0.7872041161790215,
        "y_hat": 0.5111620022247799
      },
      {
        "dow": 6,
        "t": "2022-07-31T00:00:00",
        "wom": 5,
        "y": 0.8320061228006757,
        "y_hat": 0.5363666658917704
      },
      {
        "dow": 0,
        "t": "2022-08-01T00:00:00",
        "wom": 1,
        "y": 0.48050923278633095,
        "y_hat": 0.49486749692129434
      },
      {
        "dow": 1,
        "t": "2022-08-02T00:00:00",
        "wom": 1,
        "y": 0.9186312133845865,
        "y_hat": 0.5217480417049952
      },
      {
        "dow": 2,
        "t": "2022-08-03T00:00:00",
        "wom": 1,
        "y": 0.28383987157951585,
        "y_hat": 0.4800933434215706
      },
      {
        "dow": 3,
        "t": "2022-08-04T00:00:00",
        "wom": 1,
        "y": 0.6963665748778668,
        "y_hat": 0.5282645806595277
      },
      {
        "dow": 4,
        "t": "2022-08-05T00:00:00",
        "wom": 1,
        "y": 2.09245143951022,
        "y_hat": 2.427953405345737
      },
      {
        "dow": 5,
        "t": "2022-08-06T00:00:00",
        "wom": 1,
        "y": 0.7172188720261602,
        "y_hat": 0.5304867199912241
      },
      {
        "dow": 6,
        "t": "2022-08-07T00:00:00",
        "wom": 1,
        "y": 0.6862602995181284,
        "y_hat": 0.5287652055403764
      },
      {
        "dow": 0,
        "t": "2022-08-08T00:00:00",
        "wom": 2,
        "y": 0.12495616249592223,
        "y_hat": 0.4870366588398229
      },
      {
        "dow": 1,
        "t": "2022-08-09T00:00:00",
        "wom": 2,
        "y": 0.7803068072936611,
        "y_hat": 0.5038317649617358
      },
      {
        "dow": 2,
        "t": "2022-08-10T00:00:00",
        "wom": 2,
        "y": 0.1711236164496699,
        "y_hat": 0.45705759106594746
      },
      {
        "dow": 3,
        "t": "2022-08-11T00:00:00",
        "wom": 2,
        "y": 0.6418292052213659,
        "y_hat": 0.5176881437709107
      },
      {
        "dow": 4,
        "t": "2022-08-12T00:00:00",
        "wom": 2,
        "y": 2.398223140821866,
        "y_hat": 2.420421968469209
      },
      {
        "dow": 5,
        "t": "2022-08-13T00:00:00",
        "wom": 2,
        "y": 0.7667020108493909,
        "y_hat": 0.5363921988276147
      },
      {
        "dow": 6,
        "t": "2022-08-14T00:00:00",
        "wom": 2,
        "y": 0.41815335304653145,
        "y_hat": 0.5467991648029131
      },
      {
        "dow": 0,
        "t": "2022-08-15T00:00:00",
        "wom": 3,
        "y": 0.6891797613670014,
        "y_hat": 0.48215904682588656
      },
      {
        "dow": 1,
        "t": "2022-08-16T00:00:00",
        "wom": 3,
        "y": 0.5101490274593742,
        "y_hat": 0.5055926245703029
      },
      {
        "dow": 2,
        "t": "2022-08-17T00:00:00",
        "wom": 3,
        "y": 0.7796622447746596,
        "y_hat": 0.47991073110965105
      },
      {
        "dow": 3,
        "t": "2022-08-18T00:00:00",
        "wom": 3,
        "y": 0.5627361856480227,
        "y_hat": 0.5206046080919271
      },
      {
        "dow": 4,
        "t": "2022-08-19T00:00:00",
        "wom": 3,
        "y": 2.149269114111338,
        "y_hat": 2.4511273168906147
      },
      {
        "dow": 5,
        "t": "2022-08-20T00:00:00",
        "wom": 3,
        "y": 0.4028888263767394,
        "y_hat": 0.5251204340687972
      },
      {
        "dow": 6,
        "t": "2022-08-21T00:00:00",
        "wom": 3,
        "y": 0.29581445643538484,
        "y_hat": 0.5230749375317849
      },
      {
        "dow": 0,
        "t": "2022-08-22T00:00:00",
        "wom": 4,
        "y": 0.22454795399899974,
        "y_hat": 0.4591341483141852
      },
      {
        "dow": 1,
        "t": "2022-08-23T00:00:00",
        "wom": 4,
        "y": 0.5368502058660074,
        "y_hat": 0.48582304298727946
      },
      {
        "dow": 2,
        "t": "2022-08-24T00:00:00",
        "wom": 4,
        "y": 0.8352731932733497,
        "y_hat": 0.455701817281319
      },
      {
        "dow": 3,
        "t": "2022-08-25T00:00:00",
        "wom": 4,
        "y": 0.7156495255143432,
        "y_hat": 0.5237474583198937
      },
      {
        "dow": 4,
        "t": "2022-08-26T00:00:00",
        "wom": 4,
        "y": 2.081170707633651,
        "y_hat": 2.4586113850595304
      },
      {
        "dow": 5,
        "t": "2022-08-27T00:00:00",
        "wom": 4,
        "y": 0.9846573243966786,
        "y_hat": 0.5315420611782877
      },
      {
        "dow": 6,
        "t": "2022-08-28T00:00:00",
        "wom": 4,
        "y": 0.11556538882109113,
        "y_hat": 0.5361350363720931
      },
      {
        "dow": 0,
        "t": "2022-08-29T00:00:00",
        "wom": 5,
        "y": 0.26133230623941184,
        "y_hat": 0.4855036716924154
      },
      {
        "dow": 1,
        "t": "2022-08-30T00:00:00",
        "wom": 5,
        "y": 0.9006219041976188,
        "y_hat": 0.47725526614685454
      },
      {
        "dow": 2,
        "t": "2022-08-31T00:00:00",
        "wom": 5,
        "y": 0.2515146218161567,
        "y_hat": 0.4681948708253787
      },
      {
        "dow": 3,
        "t": "2022-09-01T00:00:00",
        "wom": 1,
        "y": 0.13098484463014348,
        "y_hat": 0.5268174837110191
      },
      {
        "dow": 4,
        "t": "2022-09-02T00:00:00",
        "wom": 1,
        "y": 2.9173298508007353,
        "y_hat": 2.410501086412229
      },
      {
        "dow": 5,
        "t": "2022-09-03T00:00:00",
        "wom": 1,
        "y": 0.902033118492389,
        "y_hat": 0.5239835691087095
      },
      {
        "dow": 6,
        "t": "2022-09-04T00:00:00",
        "wom": 1,
        "y": 0.6094812461745855,
        "y_hat": 0.579028055768711
      },
      {
        "dow": 0,
        "t": "2022-09-05T00:00:00",
        "wom": 2,
        "y": 0.21237629900528165,
        "y_hat": 0.4952691329353708
      },
      {
        "dow": 1,
        "t": "2022-09-06T00:00:00",
        "wom": 2,
        "y": 0.8674613314753005,
        "y_hat": 0.5026336407752172
      },
      {
        "dow": 2,
        "t": "2022-09-07T00:00:00",
        "wom": 2,
        "y": 0.7318690107796962,
        "y_hat": 0.46471651844447337
      },
      {
        "dow": 3,
        "t": "2022-09-08T00:00:00",
        "wom": 2,
        "y": 0.9273400869945371,
        "y_hat": 0.538865818625751
      },
      {
        "dow": 4,
        "t": "2022-09-09T00:00:00",
        "wom": 2,
        "y": 2.491296824471812,
        "y_hat": 2.4593024923288445
      },
      {
        "dow": 5,
        "t": "2022-09-10T00:00:00",
        "wom": 2,
        "y": 0.3736267511181386,
        "y_hat": 0.5549146181128354
      },
      {
        "dow": 6,
        "t": "2022-09-11T00:00:00",
        "wom": 2,
        "y": 0.6741226397964637,
        "y_hat": 0.5410405136378633
      },
      {
        "dow": 0,
        "t": "2022-09-12T00:00:00",
        "wom": 3,
        "y": 0.5690598209381145,
        "y_hat": 0.468701374121668
      },
      {
        "dow": 1,
        "t": "2022-09-13T00:00:00",
        "wom": 3,
        "y": 0.5845921058651229,
        "y_hat": 0.5164280677357482
      },
      {
        "dow": 2,
        "t": "2022-09-14T00:00:00",
        "wom": 3,
        "y": 0.06190141469816057,
        "y_hat": 0.47600193151568615
      },
      {
        "dow": 3,
        "t": "2022-09-15T00:00:00",
        "wom": 3,
        "y": 0.7726353296869438,
        "y_hat": 0.504562097693305
      },
      {
        "dow": 4,
        "t": "2022-09-16T00:00:00",
        "wom": 3,
        "y": 2.3982766116278027,
        "y_hat": 2.418824328436912
      },
      {
        "dow": 5,
        "t": "2022-09-17T00:00:00",
        "wom": 3,
        "y": 0.02485301435952192,
        "y_hat": 0.5440166771084493
      },
      {
        "dow": 6,
        "t": "2022-09-18T00:00:00",
        "wom": 3,
        "y": 0.3885550460991051,
        "y_hat": 0.5261644635357546
      },
      {
        "dow": 0,
        "t": "2022-09-19T00:00:00",
        "wom": 4,
        "y": 0.5124897271631259,
        "y_hat": 0.44182407349217223
      },
      {
        "dow": 1,
        "t": "2022-09-20T00:00:00",
        "wom": 4,
        "y": 0.6717284818449774,
        "y_hat": 0.499506798622325
      },
      {
        "dow": 2,
        "t": "2022-09-21T00:00:00",
        "wom": 4,
        "y": 0.8010250603087118,
        "y_hat": 0.47555005974701947
      },
      {
        "dow": 3,
        "t": "2022-09-22T00:00:00",
        "wom": 4,
        "y": 0.7215828184438076,
        "y_hat": 0.5304993746500212
      },
      {
        "dow": 4,
        "t": "2022-09-23T00:00:00",
        "wom": 4,
        "y": 2.467671973989098,
        "y_hat": 2.457380857040162
      },
      {
        "dow": 5,
        "t": "2022-09-24T00:00:00",
        "wom": 4,
        "y": 0.7177366636762655,
        "y_hat": 0.5433557257312887
      },
      {
        "dow": 6,
        "t": "2022-09-25T00:00:00",
        "wom": 4,
        "y": 0.48856096479123934,
        "y_hat": 0.5498226240838627
      },
      {
        "dow": 0,
        "t": "2022-09-26T00:00:00",
        "wom": 5,
        "y": 0.5960952391022855,
        "y_hat": 0.4822008051720972
      },
      {
        "dow": 1,
        "t": "2022-09-27T00:00:00",
        "wom": 5,
        "y": 0.8681440058433928,
        "y_hat": 0.5074071188774416
      },
      {
        "dow": 2,
        "t": "2022-09-28T00:00:00",
        "wom": 5,
        "y": 0.8669839542345549,
        "y_hat": 0.48578856966099204
      },
      {
        "dow": 3,
        "t": "2022-09-29T00:00:00",
        "wom": 5,
        "y": 0.8884123252558981,
        "y_hat": 0.5431003113470626
      },
      {
        "dow": 4,
        "t": "2022-09-30T00:00:00",
        "wom": 5,
        "y": 2.3463008214785965,
        "y_hat": 2.465821242163437
      },
      {
        "dow": 5,
        "t": "2022-10-01T00:00:00",
        "wom": 1,
        "y": 0.2954617156792265,
        "y_hat": 0.5490063522668106
      },
      {
        "dow": 6,
        "t": "2022-10-02T00:00:00",
        "wom": 1,
        "y": 0.633109100839611,
        "y_hat": 0.5313156558678173
      },
      {
        "dow": 0,
        "t": "2022-10-03T00:00:00",
        "wom": 2,
        "y": 0.3669341564363222,
        "y_hat": 0.46364614979584606
      },
      {
        "dow": 1,
        "t": "2022-10-04T00:00:00",
        "wom": 2,
        "y": 0.24876598531702476,
        "y_hat": 0.5087746891999738
      },
      {
        "dow": 2,
        "t": "2022-10-05T00:00:00",
        "wom": 2,
        "y": 0.16587282213408583,
        "y_hat": 0.4558300182291027
      },
      {
        "dow": 3,
        "t": "2022-10-06T00:00:00",
        "wom": 2,
        "y": 0.9919913685200391,
        "y_hat": 0.48973950777965536
      },
      {
        "dow": 4,
        "t": "2022-10-07T00:00:00",
        "wom": 2,
        "y": 2.3533296873989085,
        "y_hat": 2.4310423123676665
      },
      {
        "dow": 5,
        "t": "2022-10-08T00:00:00",
        "wom": 2,
        "y": 0.7213297581947791,
        "y_hat": 0.554911360704057
      },
      {
        "dow": 6,
        "t": "2022-10-09T00:00:00",
        "wom": 2,
        "y": 0.7224275916390493,
        "y_hat": 0.5439767292961218
      },
      {
        "dow": 0,
        "t": "2022-10-10T00:00:00",
        "wom": 3,
        "y": 0.29155319532221713,
        "y_hat": 0.4893020791489958
      },
      {
        "dow": 1,
        "t": "2022-10-11T00:00:00",
        "wom": 3,
        "y": 0.36991001906594234,
        "y_hat": 0.5115549109796871
      },
      {
        "dow": 2,
        "t": "2022-10-12T00:00:00",
        "wom": 3,
        "y": 0.28417739594797153,
        "y_hat": 0.4553343870788139
      },
      {
        "dow": 3,
        "t": "2022-10-13T00:00:00",
        "wom": 3,
        "y": 0.19840529083955494,
        "y_hat": 0.4997725564131675
      },
      {
        "dow": 4,
        "t": "2022-10-14T00:00:00",
        "wom": 3,
        "y": 2.4606866711518958,
        "y_hat": 2.4148643523441606
      },
      {
        "dow": 5,
        "t": "2022-10-15T00:00:00",
        "wom": 3,
        "y": 0.6335089949594493,
        "y_hat": 0.5152479312806922
      },
      {
        "dow": 6,
        "t": "2022-10-16T00:00:00",
        "wom": 3,
        "y": 0.5680543096824044,
        "y_hat": 0.5473753240467025
      },
      {
        "dow": 0,
        "t": "2022-10-17T00:00:00",
        "wom": 4,
        "y": 0.016790840072704594,
        "y_hat": 0.48026181895173703
      },
      {
        "dow": 1,
        "t": "2022-10-18T00:00:00",
        "wom": 4,
        "y": 0.15482355482083832,
        "y_hat": 0.4954814961576285
      },
      {
        "dow": 2,
        "t": "2022-10-19T00:00:00",
        "wom": 4,
        "y": 0.32294713218572024,
        "y_hat": 0.43446678008048273
      },
      {
        "dow": 3,
        "t": "2022-10-20T00:00:00",
        "wom": 4,
        "y": 0.09723518582862622,
        "y_hat": 0.48938074212176336
      },
      {
        "dow": 4,
        "t": "2022-10-21T00:00:00",
        "wom": 4,
        "y": 2.161951112634909,
        "y_hat": 2.4141786551814923
      },
      {
        "dow": 5,
        "t": "2022-10-22T00:00:00",
        "wom": 4,
        "y": 0.9308281099209696,
        "y_hat": 0.5013617569960774
      },
      {
        "dow": 6,
        "t": "2022-10-23T00:00:00",
        "wom": 4,
        "y": 0.3700513293335844,
        "y_hat": 0.5398557089374105
      },
      {
        "dow": 0,
        "t": "2022-10-24T00:00:00",
        "wom": 5,
        "y": 0.33730301215287395,
        "y_hat": 0.4907674124047824
      },
      {
        "dow": 1,
        "t": "2022-10-25T00:00:00",
        "wom": 5,
        "y": 0.2979813390418188,
        "y_hat": 0.49406289572899337
      },
      {
        "dow": 2,
        "t": "2022-10-26T00:00:00",
        "wom": 5,
        "y": 0.8989252622362512,
        "y_hat": 0.45597333301628634
      },
      {
        "dow": 3,
        "t": "2022-10-27T00:00:00",
        "wom": 5,
        "y": 0.9678609978612648,
        "y_hat": 0.5136804919832462
      },
      {
        "dow": 4,
        "t": "2022-10-28T00:00:00",
        "wom": 5,
        "y": 2.852867990371385,
        "y_hat": 2.47026482888401
      },
      {
        "dow": 5,
        "t": "2022-10-29T00:00:00",
        "wom": 5,
        "y": 0.09421902072623589,
        "y_hat": 0.5682198415030082
      },
      {
        "dow": 6,
        "t": "2022-10-30T00:00:00",
        "wom": 5,
        "y": 0.8581628861015397,
        "y_hat": 0.553362035505034
      },
      {
        "dow": 0,
        "t": "2022-10-31T00:00:00",
        "wom": 6,
        "y": 0.20114187846747755,
        "y_hat": 0.4596615396792977
      },
      {
        "dow": 1,
        "t": "2022-11-01T00:00:00",
        "wom": 1,
        "y": 0.6181376931339703,
        "y_hat": 0.516635984405509
      },
      {
        "dow": 2,
        "t": "2022-11-02T00:00:00",
        "wom": 1,
        "y": 0.19978354259674502,
        "y_hat": 0.4578832281890113
      },
      {
        "dow": 3,
        "t": "2022-11-03T00:00:00",
        "wom": 1,
        "y": 0.1768860986420725,
        "y_hat": 0.5110987214886396
      },
      {
        "dow": 4,
        "t": "2022-11-04T00:00:00",
        "wom": 1,
        "y": 2.4752377813951942,
        "y_hat": 2.4100299124147666
      },
      {
        "dow": 5,
        "t": "2022-11-05T00:00:00",
        "wom": 1,
        "y": 0.9055115735426204,
        "y_hat": 0.5148386395620688
      },
      {
        "dow": 6,
        "t": "2022-11-06T00:00:00",
        "wom": 1,
        "y": 0.6649858316616734,
        "y_hat": 0.5562701637739151
      },
      {
        "dow": 0,
        "t": "2022-11-07T00:00:00",
        "wom": 2,
        "y": 0.5304475702709496,
        "y_hat": 0.4980529825768307
      },
      {
        "dow": 1,
        "t": "2022-11-08T00:00:00",
        "wom": 2,
        "y": 0.1761904542362559,
        "y_hat": 0.5157292008835385
      },
      {
        "dow": 2,
        "t": "2022-11-09T00:00:00",
        "wom": 2,
        "y": 0.9624710144321974,
        "y_hat": 0.46314486020995904
      },
      {
        "dow": 3,
        "t": "2022-11-10T00:00:00",
        "wom": 2,
        "y": 0.7059542876896551,
        "y_hat": 0.5091456448776946
      },
      {
        "dow": 4,
        "t": "2022-11-11T00:00:00",
        "wom": 2,
        "y": 2.506728336965063,
        "y_hat": 2.46643581246614
      },
      {
        "dow": 5,
        "t": "2022-11-12T00:00:00",
        "wom": 2,
        "y": 0.8433091095294604,
        "y_hat": 0.5444135866449329
      },
      {
        "dow": 6,
        "t": "2022-11-13T00:00:00",
        "wom": 2,
        "y": 0.7560702301487369,
        "y_hat": 0.5563050037044083
      },
      {
        "dow": 0,
        "t": "2022-11-14T00:00:00",
        "wom": 3,
        "y": 0.06592714264449118,
        "y_hat": 0.4974098519973051
      },
      {
        "dow": 1,
        "t": "2022-11-15T00:00:00",
        "wom": 3,
        "y": 0.2504072631614749,
        "y_hat": 0.5074849113317751
      },
      {
        "dow": 2,
        "t": "2022-11-16T00:00:00",
        "wom": 3,
        "y": 0.5805948341045363,
        "y_hat": 0.44029974147134476
      },
      {
        "dow": 3,
        "t": "2022-11-17T00:00:00",
        "wom": 3,
        "y": 0.5803663521377741,
        "y_hat": 0.5023523218787471
      },
      {
        "dow": 4,
        "t": "2022-11-18T00:00:00",
        "wom": 3,
        "y": 2.044237322520029,
        "y_hat": 2.44234290195584
      },
      {
        "dow": 5,
        "t": "2022-11-19T00:00:00",
        "wom": 3,
        "y": 0.5560838530176727,
        "y_hat": 0.5245297729912423
      },
      {
        "dow": 6,
        "t": "2022-11-20T00:00:00",
        "wom": 3,
        "y": 0.28756910769465927,
        "y_hat": 0.5232410661077838
      },
      {
        "dow": 0,
        "t": "2022-11-21T00:00:00",
        "wom": 4,
        "y": 0.32658389395269083,
        "y_hat": 0.46862796297499915
      },
      {
        "dow": 1,
        "t": "2022-11-22T00:00:00",
        "wom": 4,
        "y": 0.9255075128439307,
        "y_hat": 0.4897524974527008
      },
      {
        "dow": 2,
        "t": "2022-11-23T00:00:00",
        "wom": 4,
        "y": 0.2351401173462816,
        "y_hat": 0.4737753846011047
      },
      {
        "dow": 3,
        "t": "2022-11-24T00:00:00",
        "wom": 4,
        "y": 0.6755009569121625,
        "y_hat": 0.5290397268854274
      },
      {
        "dow": 4,
        "t": "2022-11-25T00:00:00",
        "wom": 4,
        "y": 2.6532627672211397,
        "y_hat": 2.4265235342689375
      },
      {
        "dow": 5,
        "t": "2022-11-26T00:00:00",
        "wom": 4,
        "y": 0.13334860717528818,
        "y_hat": 0.5471812325695693
      },
      {
        "dow": 6,
        "t": "2022-11-27T00:00:00",
        "wom": 4,
        "y": 0.21224946729617233,
        "y_hat": 0.5441529542842961
      },
      {
        "dow": 0,
        "t": "2022-11-28T00:00:00",
        "wom": 5,
        "y": 0.10485005588135898,
        "y_hat": 0.4437656298775255
      },
      {
        "dow": 1,
        "t": "2022-11-29T00:00:00",
        "wom": 5,
        "y": 0.571653845644994,
        "y_hat": 0.4794622765680875
      },
      {
        "dow": 2,
        "t": "2022-11-30T00:00:00",
        "wom": 5,
        "y": 0.6455551020000664,
        "y_hat": 0.451805294954433
      },
      {
        "dow": 3,
        "t": "2022-12-01T00:00:00",
        "wom": 1,
        "y": 0.07205418156187893,
        "y_hat": 0.5217762126874631
      },
      {
        "dow": 4,
        "t": "2022-12-02T00:00:00",
        "wom": 1,
        "y": 2.8390812934919314,
        "y_hat": 2.4315490965073923
      },
      {
        "dow": 5,
        "t": "2022-12-03T00:00:00",
        "wom": 1,
        "y": 0.3495218764848188,
        "y_hat": 0.5200249530600702
      },
      {
        "dow": 6,
        "t": "2022-12-04T00:00:00",
        "wom": 1,
        "y": 0.9605832472674337,
        "y_hat": 0.5604742376767212
      },
      {
        "dow": 0,
        "t": "2022-12-05T00:00:00",
        "wom": 2,
        "y": 0.47976044149700026,
        "y_hat": 0.4769324212585106
      },
      {
        "dow": 1,
        "t": "2022-12-06T00:00:00",
        "wom": 2,
        "y": 0.6055161236619802,
        "y_hat": 0.5306881286657961
      },
      {
        "dow": 2,
        "t": "2022-12-07T00:00:00",
        "wom": 2,
        "y": 0.41871964951333474,
        "y_hat": 0.47312547912871106
      },
      {
        "dow": 3,
        "t": "2022-12-08T00:00:00",
        "wom": 2,
        "y": 0.966612891389878,
        "y_hat": 0.5172346281782415
      },
      {
        "dow": 4,
        "t": "2022-12-09T00:00:00",
        "wom": 2,
        "y": 2.5654467983822786,
        "y_hat": 2.4449775700971856
      },
      {
        "dow": 5,
        "t": "2022-12-10T00:00:00",
        "wom": 2,
        "y": 0.34560834727361145,
        "y_hat": 0.5606125859661195
      },
      {
        "dow": 6,
        "t": "2022-12-11T00:00:00",
        "wom": 2,
        "y": 0.7570085832854714,
        "y_hat": 0.5457018888866806
      },
      {
        "dow": 0,
        "t": "2022-12-12T00:00:00",
        "wom": 3,
        "y": 0.9652744314834171,
        "y_hat": 0.4710161178959069
      },
      {
        "dow": 1,
        "t": "2022-12-13T00:00:00",
        "wom": 3,
        "y": 0.5648309593161226,
        "y_hat": 0.5336837354630672
      },
      {
        "dow": 2,
        "t": "2022-12-14T00:00:00",
        "wom": 3,
        "y": 0.7595290379916108,
        "y_hat": 0.49828660444260364
      },
      {
        "dow": 3,
        "t": "2022-12-15T00:00:00",
        "wom": 3,
        "y": 0.4539146559253099,
        "y_hat": 0.5248891664811817
      },
      {
        "dow": 4,
        "t": "2022-12-16T00:00:00",
        "wom": 3,
        "y": 2.6626571950380864,
        "y_hat": 2.448838509795448
      },
      {
        "dow": 5,
        "t": "2022-12-17T00:00:00",
        "wom": 3,
        "y": 0.901352733645678,
        "y_hat": 0.5358235389156042
      },
      {
        "dow": 6,
        "t": "2022-12-18T00:00:00",
        "wom": 3,
        "y": 0.1389757185937477,
        "y_hat": 0.5669428825994003
      },
      {
        "dow": 0,
        "t": "2022-12-19T00:00:00",
        "wom": 4,
        "y": 0.523376913529589,
        "y_hat": 0.4834707916350209
      },
      {
        "dow": 1,
        "t": "2022-12-20T00:00:00",
        "wom": 4,
        "y": 0.22793475704817479,
        "y_hat": 0.48780270680743293
      },
      {
        "dow": 2,
        "t": "2022-12-21T00:00:00",
        "wom": 4,
        "y": 0.3918232223846172,
        "y_hat": 0.46491507924850445
      },
      {
        "dow": 3,
        "t": "2022-12-22T00:00:00",
        "wom": 4,
        "y": 0.47576866552495234,
        "y_hat": 0.4963059605100148
      },
      {
        "dow": 4,
        "t": "2022-12-23T00:00:00",
        "wom": 4,
        "y": 2.3131731287518735,
        "y_hat": 2.4297238410504325
      },
      {
        "dow": 5,
        "t": "2022-12-24T00:00:00",
        "wom": 4,
        "y": 0.11547575723384762,
        "y_hat": 0.5271289193762412
      },
      {
        "dow": 6,
        "t": "2022-12-25T00:00:00",
        "wom": 4,
        "y": 0.880622203620008,
        "y_hat": 0.5257306518423508
      },
      {
        "dow": 0,
        "t": "2022-12-26T00:00:00",
        "wom": 5,
        "y": 0.4962845210256144,
        "y_hat": 0.4623486822690522
      },
      {
        "dow": 1,
        "t": "2022-12-27T00:00:00",
        "wom": 5,
        "y": 0.5406198774645495,
        "y_hat": 0.5271801081764019
      },
      {
        "dow": 2,
        "t": "2022-12-28T00:00:00",
        "wom": 5,
        "y": 0.26315431946537116,
        "y_hat": 0.4724997876723675
      },
      {
        "dow": 3,
        "t": "2022-12-29T00:00:00",
        "wom": 5,
        "y": 0.0927341711398395,
        "y_hat": 0.5096223750049854
      },
      {
        "dow": 4,
        "t": "2022-12-30T00:00:00",
        "wom": 5,
        "y": 2.42594516701449,
        "y_hat": 2.4119442031524905
      },
      {
        "dow": 5,
        "t": "2022-12-31T00:00:00",
        "wom": 5,
        "y": 0.9191390581073097,
        "y_hat": 0.5097847258328849
      }
    ]
  },
  "layer": [
    {
      "encoding": {
        "tooltip": {
          "field": "t",
          "type": "temporal"
        },
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "darkblue",
        "opacity": 0.6,
        "type": "circle"
      },
      "selection": {
        "selector003": {
          "bind": "scales",
          "type": "interval"
        }
      },
      "title": "",
      "width": 400
    },
    {
      "encoding": {
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y_hat",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "t",
        "type": "line"
      },
      "title": "7 lags, period: 0, MAE: 0.312",
      "width": 400
    }
  ]
}

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

|        |           0 |
|:-------|------------:|
| trend  | 1.60315e-05 |
| s(1,7) | 0.429604    |
| s(2,7) | 0.391794    |
| s(3,7) | 0.425179    |
| s(4,7) | 0.459669    |
| s(5,7) | 0.424455    |
| s(6,7) | 0.467097    |
| s(7,7) | 2.38925     |
| y.L1   | 0.0285755   |
| y.L2   | 0.0539871   |

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

| t                   |         y |   dow |   wom |    y_hat |
|:--------------------|----------:|------:|------:|---------:|
| 2022-10-09 00:00:00 | 0.722428  |     6 |     2 | 0.543977 |
| 2022-10-10 00:00:00 | 0.291553  |     0 |     3 | 0.489302 |
| 2022-10-11 00:00:00 | 0.36991   |     1 |     3 | 0.511555 |
| 2022-10-13 00:00:00 | 0.198405  |     3 |     3 | 0.499773 |
| 2022-10-14 00:00:00 | 2.46069   |     4 |     3 | 2.41486  |
| 2022-10-15 00:00:00 | 0.633509  |     5 |     3 | 0.515248 |
| 2022-10-17 00:00:00 | 0.0167908 |     0 |     4 | 0.480262 |
| 2022-10-18 00:00:00 | 0.154824  |     1 |     4 | 0.495481 |
| 2022-10-19 00:00:00 | 0.322947  |     2 |     4 | 0.434467 |
| 2022-10-20 00:00:00 | 0.0972352 |     3 |     4 | 0.489381 |
| 2022-10-21 00:00:00 | 2.16195   |     4 |     4 | 2.41418  |
| 2022-10-22 00:00:00 | 0.930828  |     5 |     4 | 0.501362 |

__After__

```python
ts = ts.set_index('t').asfreq('d')
```

|           y |   dow |   wom |      y_hat |
|------------:|------:|------:|-----------:|
|   0.72133   |     5 |     2 |   0.554911 |
|   0.722428  |     6 |     2 |   0.543977 |
|   0.291553  |     0 |     3 |   0.489302 |
|   0.36991   |     1 |     3 |   0.511555 |
| nan         |   nan |   nan | nan        |
|   0.198405  |     3 |     3 |   0.499773 |
|   2.46069   |     4 |     3 |   2.41486  |
|   0.633509  |     5 |     3 |   0.515248 |
| nan         |   nan |   nan | nan        |
|   0.0167908 |     0 |     4 |   0.480262 |
|   0.154824  |     1 |     4 |   0.495481 |
|   0.322947  |     2 |     4 |   0.434467 |

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

|         y |   dow |   wom |
|----------:|------:|------:|
| 0.722428  |     6 |     2 |
| 0.291553  |     0 |     3 |
| 0.36991   |     1 |     3 |
| 0.198405  |     3 |     3 |
| 2.46069   |     4 |     3 |
| 0.633509  |     5 |     3 |
| 0.0167908 |     0 |     4 |
| 0.154824  |     1 |     4 |
| 0.322947  |     2 |     4 |
| 0.0972352 |     3 |     4 |
| 2.16195   |     4 |     4 |
| 0.930828  |     5 |     4 |

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

| t          |        y |   dow |   wom |      y_hat |
|:-----------|---------:|------:|------:|-----------:|
| 2022-01-01 | 0.995598 |     5 |     1 | nan        |
| 2022-01-02 | 0.623411 |     6 |     1 | nan        |
| 2022-01-03 | 0.876094 |     0 |     2 | nan        |
| 2022-01-04 | 0.889378 |     1 |     2 | nan        |
| 2022-01-05 | 0.386762 |     2 |     2 | nan        |
| 2022-01-06 | 0.68992  |     3 |     2 | nan        |
| 2022-01-07 | 2.99285  |     4 |     2 | nan        |
| 2022-01-08 | 0.468042 |     5 |     2 |   0.686514 |
| 2022-01-09 | 0.945584 |     6 |     2 |   0.655461 |

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

var spec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
  "config": {
    "view": {
      "continuousHeight": 300,
      "continuousWidth": 400
    }
  },
  "data": {
    "name": "data-0381f859545cf61fd1b32fed3c3bd645"
  },
  "datasets": {
    "data-0381f859545cf61fd1b32fed3c3bd645": [
      {
        "dow": 5,
        "t": "2022-01-01T00:00:00",
        "wom": 1,
        "y": 0.9955981828907805,
        "y_hat": null
      },
      {
        "dow": 6,
        "t": "2022-01-02T00:00:00",
        "wom": 1,
        "y": 0.6234109104925726,
        "y_hat": null
      },
      {
        "dow": 0,
        "t": "2022-01-03T00:00:00",
        "wom": 2,
        "y": 0.8760936874425284,
        "y_hat": null
      },
      {
        "dow": 1,
        "t": "2022-01-04T00:00:00",
        "wom": 2,
        "y": 0.8893778611747504,
        "y_hat": null
      },
      {
        "dow": 2,
        "t": "2022-01-05T00:00:00",
        "wom": 2,
        "y": 0.3867624782099456,
        "y_hat": null
      },
      {
        "dow": 3,
        "t": "2022-01-06T00:00:00",
        "wom": 2,
        "y": 0.6899198960223786,
        "y_hat": null
      },
      {
        "dow": 4,
        "t": "2022-01-07T00:00:00",
        "wom": 2,
        "y": 2.992852550566659,
        "y_hat": null
      },
      {
        "dow": 5,
        "t": "2022-01-08T00:00:00",
        "wom": 2,
        "y": 0.46804208563249405,
        "y_hat": 0.6865138627325476
      },
      {
        "dow": 6,
        "t": "2022-01-09T00:00:00",
        "wom": 2,
        "y": 0.9455841386245034,
        "y_hat": 0.6554608619568258
      },
      {
        "dow": 0,
        "t": "2022-01-10T00:00:00",
        "wom": 3,
        "y": 0.6381233538848571,
        "y_hat": 0.7590311163471878
      },
      {
        "dow": 1,
        "t": "2022-01-11T00:00:00",
        "wom": 3,
        "y": 0.7595024746539746,
        "y_hat": 0.7545480812731286
      },
      {
        "dow": 2,
        "t": "2022-01-12T00:00:00",
        "wom": 3,
        "y": 0.23738191602671632,
        "y_hat": 0.29762817695308785
      },
      {
        "dow": 3,
        "t": "2022-01-13T00:00:00",
        "wom": 3,
        "y": 0.09680924548766545,
        "y_hat": 0.8721926569984346
      },
      {
        "dow": 4,
        "t": "2022-01-14T00:00:00",
        "wom": 3,
        "y": 2.588571863863769,
        "y_hat": 2.4823231373528456
      },
      {
        "dow": 5,
        "t": "2022-01-15T00:00:00",
        "wom": 3,
        "y": 0.16573502501513393,
        "y_hat": 0.30191711710427926
      },
      {
        "dow": 6,
        "t": "2022-01-16T00:00:00",
        "wom": 3,
        "y": 0.38942124330865624,
        "y_hat": 0.9060525776561855
      },
      {
        "dow": 0,
        "t": "2022-01-17T00:00:00",
        "wom": 4,
        "y": 0.688787521402102,
        "y_hat": 0.5825102105125103
      },
      {
        "dow": 1,
        "t": "2022-01-18T00:00:00",
        "wom": 4,
        "y": 0.6697600544042848,
        "y_hat": 0.625498497244787
      },
      {
        "dow": 2,
        "t": "2022-01-19T00:00:00",
        "wom": 4,
        "y": 0.36885391553882474,
        "y_hat": 0.13454165379857652
      },
      {
        "dow": 3,
        "t": "2022-01-20T00:00:00",
        "wom": 4,
        "y": 0.9325008856726638,
        "y_hat": 0.3433370659240544
      },
      {
        "dow": 4,
        "t": "2022-01-21T00:00:00",
        "wom": 4,
        "y": 2.141135576285614,
        "y_hat": 2.0857489434922885
      },
      {
        "dow": 5,
        "t": "2022-01-22T00:00:00",
        "wom": 4,
        "y": 0.9772509899400809,
        "y_hat": 0.060031417905823434
      },
      {
        "dow": 6,
        "t": "2022-01-23T00:00:00",
        "wom": 4,
        "y": 0.5683493004574759,
        "y_hat": 0.39316954391267267
      },
      {
        "dow": 0,
        "t": "2022-01-24T00:00:00",
        "wom": 5,
        "y": 0.9649672826390342,
        "y_hat": 0.6332195170267185
      },
      {
        "dow": 1,
        "t": "2022-01-25T00:00:00",
        "wom": 5,
        "y": 0.22830327875114365,
        "y_hat": 0.5371182166381869
      },
      {
        "dow": 2,
        "t": "2022-01-26T00:00:00",
        "wom": 5,
        "y": 0.5236014315750456,
        "y_hat": 0.380278227095489
      },
      {
        "dow": 3,
        "t": "2022-01-27T00:00:00",
        "wom": 5,
        "y": 0.5771602890446812,
        "y_hat": 0.9418643355244579
      },
      {
        "dow": 4,
        "t": "2022-01-28T00:00:00",
        "wom": 5,
        "y": 2.3198686688954586,
        "y_hat": 1.829781219890305
      },
      {
        "dow": 5,
        "t": "2022-01-29T00:00:00",
        "wom": 5,
        "y": 0.4375103099359736,
        "y_hat": 0.7126278931332124
      },
      {
        "dow": 6,
        "t": "2022-01-30T00:00:00",
        "wom": 5,
        "y": 0.7071449184091295,
        "y_hat": 0.6222642203119366
      },
      {
        "dow": 0,
        "t": "2022-01-31T00:00:00",
        "wom": 6,
        "y": 0.1524637438937454,
        "y_hat": 0.7856041596956094
      },
      {
        "dow": 1,
        "t": "2022-02-01T00:00:00",
        "wom": 1,
        "y": 0.011748418393797033,
        "y_hat": 0.26192571705783035
      },
      {
        "dow": 2,
        "t": "2022-02-02T00:00:00",
        "wom": 1,
        "y": 0.25517769811183577,
        "y_hat": 0.455261034529315
      },
      {
        "dow": 3,
        "t": "2022-02-03T00:00:00",
        "wom": 1,
        "y": 0.5156949359371945,
        "y_hat": 0.6943467302124587
      },
      {
        "dow": 4,
        "t": "2022-02-04T00:00:00",
        "wom": 1,
        "y": 2.6710173501944316,
        "y_hat": 1.9140402274212918
      },
      {
        "dow": 5,
        "t": "2022-02-05T00:00:00",
        "wom": 1,
        "y": 0.8240477378755111,
        "y_hat": 0.2798862951312128
      },
      {
        "dow": 6,
        "t": "2022-02-06T00:00:00",
        "wom": 1,
        "y": 0.9508838016129778,
        "y_hat": 0.6477619325395739
      },
      {
        "dow": 0,
        "t": "2022-02-07T00:00:00",
        "wom": 2,
        "y": 0.05520043850117129,
        "y_hat": 0.10626548299883537
      },
      {
        "dow": 1,
        "t": "2022-02-08T00:00:00",
        "wom": 2,
        "y": 0.36173925787444183,
        "y_hat": 0.07902606725881982
      },
      {
        "dow": 2,
        "t": "2022-02-09T00:00:00",
        "wom": 2,
        "y": 0.3287722160717247,
        "y_hat": 0.19967626807227024
      },
      {
        "dow": 3,
        "t": "2022-02-10T00:00:00",
        "wom": 2,
        "y": 0.26344459815995447,
        "y_hat": 0.6815189188874452
      },
      {
        "dow": 4,
        "t": "2022-02-11T00:00:00",
        "wom": 2,
        "y": 2.105522116894078,
        "y_hat": 2.256862585427889
      },
      {
        "dow": 5,
        "t": "2022-02-12T00:00:00",
        "wom": 2,
        "y": 0.9004373252539722,
        "y_hat": 0.6556570684028041
      },
      {
        "dow": 6,
        "t": "2022-02-13T00:00:00",
        "wom": 2,
        "y": 0.3083956070489431,
        "y_hat": 0.8077855958589312
      },
      {
        "dow": 0,
        "t": "2022-02-14T00:00:00",
        "wom": 3,
        "y": 0.8443045110438598,
        "y_hat": 0.10747625190930281
      },
      {
        "dow": 1,
        "t": "2022-02-15T00:00:00",
        "wom": 3,
        "y": 0.9448650092640142,
        "y_hat": 0.3043468277197155
      },
      {
        "dow": 2,
        "t": "2022-02-16T00:00:00",
        "wom": 3,
        "y": 0.5570587072891441,
        "y_hat": 0.2379735463635557
      },
      {
        "dow": 3,
        "t": "2022-02-17T00:00:00",
        "wom": 3,
        "y": 0.5374558873035822,
        "y_hat": 0.42411381570534956
      },
      {
        "dow": 4,
        "t": "2022-02-18T00:00:00",
        "wom": 3,
        "y": 2.8223117437123824,
        "y_hat": 1.8122759691400105
      },
      {
        "dow": 5,
        "t": "2022-02-19T00:00:00",
        "wom": 3,
        "y": 0.02226637799800346,
        "y_hat": 0.6029464043261008
      },
      {
        "dow": 6,
        "t": "2022-02-20T00:00:00",
        "wom": 3,
        "y": 0.15725472652557881,
        "y_hat": 0.43637739407204956
      },
      {
        "dow": 0,
        "t": "2022-02-21T00:00:00",
        "wom": 4,
        "y": 0.9269915515225546,
        "y_hat": 0.7900107089095362
      },
      {
        "dow": 1,
        "t": "2022-02-22T00:00:00",
        "wom": 4,
        "y": 0.4501913471576654,
        "y_hat": 0.7910836031757595
      },
      {
        "dow": 2,
        "t": "2022-02-23T00:00:00",
        "wom": 4,
        "y": 0.9531441262485346,
        "y_hat": 0.47228267299910287
      },
      {
        "dow": 3,
        "t": "2022-02-24T00:00:00",
        "wom": 4,
        "y": 0.5844501255113494,
        "y_hat": 0.6981971342464581
      },
      {
        "dow": 4,
        "t": "2022-02-25T00:00:00",
        "wom": 4,
        "y": 2.80397776091471,
        "y_hat": 2.326977167881708
      },
      {
        "dow": 5,
        "t": "2022-02-26T00:00:00",
        "wom": 4,
        "y": 0.26383392179603904,
        "y_hat": -0.1267574960087076
      },
      {
        "dow": 6,
        "t": "2022-02-27T00:00:00",
        "wom": 4,
        "y": 0.7531464817944895,
        "y_hat": 0.3195987897096323
      },
      {
        "dow": 0,
        "t": "2022-02-28T00:00:00",
        "wom": 5,
        "y": 0.15885513834885823,
        "y_hat": 0.7710095367632724
      },
      {
        "dow": 1,
        "t": "2022-03-01T00:00:00",
        "wom": 1,
        "y": 0.14535499394438622,
        "y_hat": 0.5007516099900935
      },
      {
        "dow": 2,
        "t": "2022-03-02T00:00:00",
        "wom": 1,
        "y": 0.4410675050392091,
        "y_hat": 0.7993969341170383
      },
      {
        "dow": 3,
        "t": "2022-03-03T00:00:00",
        "wom": 1,
        "y": 0.7517726853328405,
        "y_hat": 0.7571567671045403
      },
      {
        "dow": 4,
        "t": "2022-03-04T00:00:00",
        "wom": 1,
        "y": 2.7945001756665624,
        "y_hat": 2.2943933815743853
      },
      {
        "dow": 5,
        "t": "2022-03-05T00:00:00",
        "wom": 1,
        "y": 0.04870630413527299,
        "y_hat": 0.15523369570751427
      },
      {
        "dow": 6,
        "t": "2022-03-06T00:00:00",
        "wom": 1,
        "y": 0.2161041338955525,
        "y_hat": 0.7523919188228312
      },
      {
        "dow": 0,
        "t": "2022-03-07T00:00:00",
        "wom": 2,
        "y": 0.37010898460177155,
        "y_hat": 0.15626470961583408
      },
      {
        "dow": 1,
        "t": "2022-03-08T00:00:00",
        "wom": 2,
        "y": 0.09520299889511219,
        "y_hat": 0.16680712124030556
      },
      {
        "dow": 2,
        "t": "2022-03-09T00:00:00",
        "wom": 2,
        "y": 0.11977796440076782,
        "y_hat": 0.4103293915372901
      },
      {
        "dow": 3,
        "t": "2022-03-10T00:00:00",
        "wom": 2,
        "y": 0.5000495405840498,
        "y_hat": 0.9192113734624194
      },
      {
        "dow": 4,
        "t": "2022-03-11T00:00:00",
        "wom": 2,
        "y": 2.453037332781335,
        "y_hat": 2.286457667370272
      },
      {
        "dow": 5,
        "t": "2022-03-12T00:00:00",
        "wom": 2,
        "y": 0.1454828110192531,
        "y_hat": -0.06220430368317223
      },
      {
        "dow": 6,
        "t": "2022-03-13T00:00:00",
        "wom": 2,
        "y": 0.37079700609072885,
        "y_hat": 0.31924298401229223
      },
      {
        "dow": 0,
        "t": "2022-03-14T00:00:00",
        "wom": 3,
        "y": 0.6368565984388843,
        "y_hat": 0.3259328400124133
      },
      {
        "dow": 1,
        "t": "2022-03-15T00:00:00",
        "wom": 3,
        "y": 0.5235082648923333,
        "y_hat": 0.08894246000633668
      },
      {
        "dow": 2,
        "t": "2022-03-16T00:00:00",
        "wom": 3,
        "y": 0.7192823846677754,
        "y_hat": 0.11426796841061274
      },
      {
        "dow": 3,
        "t": "2022-03-17T00:00:00",
        "wom": 3,
        "y": 0.6045077428498761,
        "y_hat": 0.6557431014718476
      },
      {
        "dow": 4,
        "t": "2022-03-18T00:00:00",
        "wom": 3,
        "y": 2.5404792929526683,
        "y_hat": 2.0317751774639357
      },
      {
        "dow": 5,
        "t": "2022-03-19T00:00:00",
        "wom": 3,
        "y": 0.4830778428824394,
        "y_hat": 0.02924182751184995
      },
      {
        "dow": 6,
        "t": "2022-03-20T00:00:00",
        "wom": 3,
        "y": 0.9022692634866183,
        "y_hat": 0.4482329020646049
      },
      {
        "dow": 0,
        "t": "2022-03-21T00:00:00",
        "wom": 4,
        "y": 0.06386988650342573,
        "y_hat": 0.553004364586114
      },
      {
        "dow": 1,
        "t": "2022-03-22T00:00:00",
        "wom": 4,
        "y": 0.2631672977407843,
        "y_hat": 0.5570706923884754
      },
      {
        "dow": 2,
        "t": "2022-03-23T00:00:00",
        "wom": 4,
        "y": 0.6967761906616411,
        "y_hat": 0.6150085858687979
      },
      {
        "dow": 3,
        "t": "2022-03-24T00:00:00",
        "wom": 4,
        "y": 0.6938964314225743,
        "y_hat": 0.7388755154259353
      },
      {
        "dow": 4,
        "t": "2022-03-25T00:00:00",
        "wom": 4,
        "y": 2.3402313329841156,
        "y_hat": 2.1194677520671177
      },
      {
        "dow": 5,
        "t": "2022-03-26T00:00:00",
        "wom": 4,
        "y": 0.060442940833932424,
        "y_hat": 0.3897243970155417
      },
      {
        "dow": 6,
        "t": "2022-03-27T00:00:00",
        "wom": 4,
        "y": 0.8673761163808922,
        "y_hat": 0.8555777482010288
      },
      {
        "dow": 0,
        "t": "2022-03-28T00:00:00",
        "wom": 5,
        "y": 0.5544851701436769,
        "y_hat": 0.05005113572567699
      },
      {
        "dow": 1,
        "t": "2022-03-29T00:00:00",
        "wom": 5,
        "y": 0.8547133638306371,
        "y_hat": 0.30799265087702066
      },
      {
        "dow": 2,
        "t": "2022-03-30T00:00:00",
        "wom": 5,
        "y": 0.36375693048623126,
        "y_hat": 0.5891060378294395
      },
      {
        "dow": 3,
        "t": "2022-03-31T00:00:00",
        "wom": 5,
        "y": 0.7322858415787377,
        "y_hat": 0.8476147016128437
      },
      {
        "dow": 4,
        "t": "2022-04-01T00:00:00",
        "wom": 1,
        "y": 2.1005642322427573,
        "y_hat": 1.9054960522642972
      },
      {
        "dow": 5,
        "t": "2022-04-02T00:00:00",
        "wom": 1,
        "y": 0.6446246806801427,
        "y_hat": 0.05183248410537335
      },
      {
        "dow": 6,
        "t": "2022-04-03T00:00:00",
        "wom": 1,
        "y": 0.523704664544059,
        "y_hat": 0.817652501111651
      },
      {
        "dow": 0,
        "t": "2022-04-04T00:00:00",
        "wom": 2,
        "y": 0.6478284421164979,
        "y_hat": 0.5636566345950416
      },
      {
        "dow": 1,
        "t": "2022-04-05T00:00:00",
        "wom": 2,
        "y": 0.8035144383488594,
        "y_hat": 0.7421666625984116
      },
      {
        "dow": 2,
        "t": "2022-04-06T00:00:00",
        "wom": 2,
        "y": 0.06175772835110849,
        "y_hat": 0.33720850094240656
      },
      {
        "dow": 3,
        "t": "2022-04-07T00:00:00",
        "wom": 2,
        "y": 0.10101798118758876,
        "y_hat": 0.862693862492971
      },
      {
        "dow": 4,
        "t": "2022-04-08T00:00:00",
        "wom": 2,
        "y": 2.328695425168449,
        "y_hat": 1.8107661262919517
      },
      {
        "dow": 5,
        "t": "2022-04-09T00:00:00",
        "wom": 2,
        "y": 0.7405205134959488,
        "y_hat": 0.45826743278332305
      },
      {
        "dow": 0,
        "t": "2022-04-11T00:00:00",
        "wom": 3,
        "y": 0.48567949453078907,
        "y_hat": 0.5501724833529308
      },
      {
        "dow": 1,
        "t": "2022-04-12T00:00:00",
        "wom": 3,
        "y": 0.5552603671504405,
        "y_hat": 0.6488247629745928
      },
      {
        "dow": 2,
        "t": "2022-04-13T00:00:00",
        "wom": 3,
        "y": 0.2511650942997945,
        "y_hat": 0.6929064770195568
      },
      {
        "dow": 3,
        "t": "2022-04-14T00:00:00",
        "wom": 3,
        "y": 0.4804870698202284,
        "y_hat": 0.05939116207607332
      },
      {
        "dow": 4,
        "t": "2022-04-15T00:00:00",
        "wom": 3,
        "y": 2.0257783736909825,
        "y_hat": 0.32315525307369836
      },
      {
        "dow": 5,
        "t": "2022-04-16T00:00:00",
        "wom": 3,
        "y": 0.9687966319838246,
        "y_hat": 1.895334962948804
      },
      {
        "dow": 6,
        "t": "2022-04-17T00:00:00",
        "wom": 3,
        "y": 0.7151278400228451,
        "y_hat": 0.6945263730736816
      },
      {
        "dow": 0,
        "t": "2022-04-18T00:00:00",
        "wom": 4,
        "y": 0.4718442940718869,
        "y_hat": 0.4832688892848366
      },
      {
        "dow": 1,
        "t": "2022-04-19T00:00:00",
        "wom": 4,
        "y": 0.9026979968878186,
        "y_hat": 0.516057965648621
      },
      {
        "dow": 2,
        "t": "2022-04-20T00:00:00",
        "wom": 4,
        "y": 0.39245084658370566,
        "y_hat": 0.21586693234238782
      },
      {
        "dow": 3,
        "t": "2022-04-21T00:00:00",
        "wom": 4,
        "y": 0.4827795920741036,
        "y_hat": 0.6296236528688083
      },
      {
        "dow": 4,
        "t": "2022-04-22T00:00:00",
        "wom": 4,
        "y": 2.551414450198103,
        "y_hat": 1.7689680855496177
      },
      {
        "dow": 5,
        "t": "2022-04-23T00:00:00",
        "wom": 4,
        "y": 0.650135286544272,
        "y_hat": 0.7509090694682378
      },
      {
        "dow": 6,
        "t": "2022-04-24T00:00:00",
        "wom": 4,
        "y": 0.467587186604593,
        "y_hat": 0.7077401201273404
      },
      {
        "dow": 0,
        "t": "2022-04-25T00:00:00",
        "wom": 5,
        "y": 0.8420434940525285,
        "y_hat": 0.5127879589787798
      },
      {
        "dow": 1,
        "t": "2022-04-26T00:00:00",
        "wom": 5,
        "y": 0.44245377300469013,
        "y_hat": 0.784983505808037
      },
      {
        "dow": 2,
        "t": "2022-04-27T00:00:00",
        "wom": 5,
        "y": 0.10472408289231805,
        "y_hat": 0.3653859716053267
      },
      {
        "dow": 3,
        "t": "2022-04-28T00:00:00",
        "wom": 5,
        "y": 0.07965539705381441,
        "y_hat": 0.6971877590283186
      },
      {
        "dow": 4,
        "t": "2022-04-29T00:00:00",
        "wom": 5,
        "y": 2.1204279487951343,
        "y_hat": 2.1928889744629707
      },
      {
        "dow": 5,
        "t": "2022-04-30T00:00:00",
        "wom": 5,
        "y": 0.09911422212585952,
        "y_hat": 0.47382371306305354
      },
      {
        "dow": 6,
        "t": "2022-05-01T00:00:00",
        "wom": 1,
        "y": 0.7653139992562019,
        "y_hat": 0.576383796758079
      },
      {
        "dow": 0,
        "t": "2022-05-02T00:00:00",
        "wom": 2,
        "y": 0.6928765519708857,
        "y_hat": 0.7380589968564099
      },
      {
        "dow": 1,
        "t": "2022-05-03T00:00:00",
        "wom": 2,
        "y": 0.3293225432083243,
        "y_hat": 0.4104935702124929
      },
      {
        "dow": 2,
        "t": "2022-05-04T00:00:00",
        "wom": 2,
        "y": 0.6649514990871527,
        "y_hat": 0.103696972835135
      },
      {
        "dow": 3,
        "t": "2022-05-05T00:00:00",
        "wom": 2,
        "y": 0.15101171079274767,
        "y_hat": 0.2992493056462366
      },
      {
        "dow": 4,
        "t": "2022-05-06T00:00:00",
        "wom": 2,
        "y": 2.1798428980656723,
        "y_hat": 1.796124340801475
      },
      {
        "dow": 5,
        "t": "2022-05-07T00:00:00",
        "wom": 2,
        "y": 0.2354835900991824,
        "y_hat": 0.0584430042257308
      },
      {
        "dow": 6,
        "t": "2022-05-08T00:00:00",
        "wom": 2,
        "y": 0.05741790510662548,
        "y_hat": 0.8050652577331223
      },
      {
        "dow": 0,
        "t": "2022-05-09T00:00:00",
        "wom": 3,
        "y": 0.23077326722872582,
        "y_hat": 0.6479360995178296
      },
      {
        "dow": 1,
        "t": "2022-05-10T00:00:00",
        "wom": 3,
        "y": 0.1944400629234596,
        "y_hat": 0.3823032530999987
      },
      {
        "dow": 2,
        "t": "2022-05-11T00:00:00",
        "wom": 3,
        "y": 0.7807354431480255,
        "y_hat": 0.5630555511048824
      },
      {
        "dow": 3,
        "t": "2022-05-12T00:00:00",
        "wom": 3,
        "y": 0.31021821068384303,
        "y_hat": 0.3484629822943033
      },
      {
        "dow": 4,
        "t": "2022-05-13T00:00:00",
        "wom": 3,
        "y": 2.1275853244649534,
        "y_hat": 1.8688760387460674
      },
      {
        "dow": 5,
        "t": "2022-05-14T00:00:00",
        "wom": 3,
        "y": 0.56813301154376,
        "y_hat": 0.12121453389528361
      },
      {
        "dow": 6,
        "t": "2022-05-15T00:00:00",
        "wom": 3,
        "y": 0.10167470187610927,
        "y_hat": 0.16288248147939066
      },
      {
        "dow": 0,
        "t": "2022-05-16T00:00:00",
        "wom": 4,
        "y": 0.37826726087578866,
        "y_hat": 0.2662954043038053
      },
      {
        "dow": 1,
        "t": "2022-05-17T00:00:00",
        "wom": 4,
        "y": 0.7138146434013722,
        "y_hat": 0.27506593622376013
      },
      {
        "dow": 2,
        "t": "2022-05-18T00:00:00",
        "wom": 4,
        "y": 0.12852670113262654,
        "y_hat": 0.6491855046987552
      },
      {
        "dow": 3,
        "t": "2022-05-19T00:00:00",
        "wom": 4,
        "y": 0.3585977195944233,
        "y_hat": 0.5310288224248504
      },
      {
        "dow": 4,
        "t": "2022-05-20T00:00:00",
        "wom": 4,
        "y": 2.22258022407796,
        "y_hat": 1.8367266236394506
      },
      {
        "dow": 5,
        "t": "2022-05-21T00:00:00",
        "wom": 4,
        "y": 0.932770577180065,
        "y_hat": 0.393419307798241
      },
      {
        "dow": 6,
        "t": "2022-05-22T00:00:00",
        "wom": 4,
        "y": 0.9336405466022045,
        "y_hat": 0.18229878325202417
      },
      {
        "dow": 0,
        "t": "2022-05-23T00:00:00",
        "wom": 5,
        "y": 0.7765773218651726,
        "y_hat": 0.41341030547172364
      },
      {
        "dow": 1,
        "t": "2022-05-24T00:00:00",
        "wom": 5,
        "y": 0.9092507321087694,
        "y_hat": 0.6391676954613696
      },
      {
        "dow": 2,
        "t": "2022-05-25T00:00:00",
        "wom": 5,
        "y": 0.9912972693958269,
        "y_hat": 0.12473936272598818
      },
      {
        "dow": 3,
        "t": "2022-05-26T00:00:00",
        "wom": 5,
        "y": 0.17273741694999156,
        "y_hat": 0.5269782597872781
      },
      {
        "dow": 4,
        "t": "2022-05-27T00:00:00",
        "wom": 5,
        "y": 2.7305657647776744,
        "y_hat": 1.979170042002642
      },
      {
        "dow": 5,
        "t": "2022-05-28T00:00:00",
        "wom": 5,
        "y": 0.9331495324820412,
        "y_hat": 0.7299035513510627
      },
      {
        "dow": 6,
        "t": "2022-05-29T00:00:00",
        "wom": 5,
        "y": 0.9747799262042263,
        "y_hat": 0.9206954334992564
      },
      {
        "dow": 0,
        "t": "2022-05-30T00:00:00",
        "wom": 6,
        "y": 0.6159827699504155,
        "y_hat": 0.7396754044567138
      },
      {
        "dow": 1,
        "t": "2022-05-31T00:00:00",
        "wom": 6,
        "y": 0.18957160929478611,
        "y_hat": 0.9060093053462557
      },
      {
        "dow": 2,
        "t": "2022-06-01T00:00:00",
        "wom": 1,
        "y": 0.08329858559961756,
        "y_hat": 0.845763492980497
      },
      {
        "dow": 3,
        "t": "2022-06-02T00:00:00",
        "wom": 1,
        "y": 0.7811845717468601,
        "y_hat": 0.46477426595591964
      },
      {
        "dow": 4,
        "t": "2022-06-03T00:00:00",
        "wom": 1,
        "y": 2.5856317189042306,
        "y_hat": 2.3230025540529873
      },
      {
        "dow": 5,
        "t": "2022-06-04T00:00:00",
        "wom": 1,
        "y": 0.9915634439508605,
        "y_hat": 0.768524548021911
      },
      {
        "dow": 6,
        "t": "2022-06-05T00:00:00",
        "wom": 1,
        "y": 0.9909525874581544,
        "y_hat": 0.9463365027972473
      },
      {
        "dow": 0,
        "t": "2022-06-06T00:00:00",
        "wom": 2,
        "y": 0.546656872738867,
        "y_hat": 0.5623363480471781
      },
      {
        "dow": 1,
        "t": "2022-06-07T00:00:00",
        "wom": 2,
        "y": 0.10472218993110805,
        "y_hat": 0.221555981820116
      },
      {
        "dow": 2,
        "t": "2022-06-08T00:00:00",
        "wom": 2,
        "y": 0.04250860920798383,
        "y_hat": 0.17429233540122444
      },
      {
        "dow": 3,
        "t": "2022-06-09T00:00:00",
        "wom": 2,
        "y": 0.3032781068949767,
        "y_hat": 0.9490771917255436
      },
      {
        "dow": 4,
        "t": "2022-06-10T00:00:00",
        "wom": 2,
        "y": 2.5924710028233178,
        "y_hat": 2.243413516993402
      },
      {
        "dow": 5,
        "t": "2022-06-11T00:00:00",
        "wom": 2,
        "y": 0.1823450459040853,
        "y_hat": 0.8051866860328697
      },
      {
        "dow": 6,
        "t": "2022-06-12T00:00:00",
        "wom": 2,
        "y": 0.7714328053698969,
        "y_hat": 1.01136349350275
      },
      {
        "dow": 0,
        "t": "2022-06-13T00:00:00",
        "wom": 3,
        "y": 0.7117102710960721,
        "y_hat": 0.4864442060277649
      },
      {
        "dow": 1,
        "t": "2022-06-14T00:00:00",
        "wom": 3,
        "y": 0.5078428447537274,
        "y_hat": 0.1422461286301691
      },
      {
        "dow": 2,
        "t": "2022-06-15T00:00:00",
        "wom": 3,
        "y": 0.4559806484621759,
        "y_hat": 0.07214413521763412
      },
      {
        "dow": 3,
        "t": "2022-06-16T00:00:00",
        "wom": 3,
        "y": 0.3663298759696507,
        "y_hat": 0.5660074381151942
      },
      {
        "dow": 4,
        "t": "2022-06-17T00:00:00",
        "wom": 3,
        "y": 2.036192987755432,
        "y_hat": 2.187137153842408
      },
      {
        "dow": 5,
        "t": "2022-06-18T00:00:00",
        "wom": 3,
        "y": 0.5857499825271564,
        "y_hat": 0.16249467094537484
      },
      {
        "dow": 6,
        "t": "2022-06-19T00:00:00",
        "wom": 3,
        "y": 0.050005294335790484,
        "y_hat": 0.7976307208570182
      },
      {
        "dow": 0,
        "t": "2022-06-20T00:00:00",
        "wom": 4,
        "y": 0.5061272547669947,
        "y_hat": 0.7173700635234476
      },
      {
        "dow": 1,
        "t": "2022-06-21T00:00:00",
        "wom": 4,
        "y": 0.7904390381333787,
        "y_hat": 0.5022676826514002
      },
      {
        "dow": 2,
        "t": "2022-06-22T00:00:00",
        "wom": 4,
        "y": 0.09323256170059835,
        "y_hat": 0.40609365787631546
      },
      {
        "dow": 3,
        "t": "2022-06-23T00:00:00",
        "wom": 4,
        "y": 0.40800965871918415,
        "y_hat": 0.587756511695825
      },
      {
        "dow": 4,
        "t": "2022-06-24T00:00:00",
        "wom": 4,
        "y": 2.570342151877391,
        "y_hat": 1.7762717834031059
      },
      {
        "dow": 5,
        "t": "2022-06-25T00:00:00",
        "wom": 4,
        "y": 0.07936746817059592,
        "y_hat": 0.39380647680979747
      },
      {
        "dow": 6,
        "t": "2022-06-26T00:00:00",
        "wom": 4,
        "y": 0.017413769472798823,
        "y_hat": 0.23570135842164824
      },
      {
        "dow": 0,
        "t": "2022-06-27T00:00:00",
        "wom": 5,
        "y": 0.4856034209379412,
        "y_hat": 0.5741140262802836
      },
      {
        "dow": 1,
        "t": "2022-06-28T00:00:00",
        "wom": 5,
        "y": 0.10626439009089561,
        "y_hat": 0.7003920976388153
      },
      {
        "dow": 2,
        "t": "2022-06-29T00:00:00",
        "wom": 5,
        "y": 0.7531865011385493,
        "y_hat": 0.14860690146029223
      },
      {
        "dow": 3,
        "t": "2022-06-30T00:00:00",
        "wom": 5,
        "y": 0.2215890993417199,
        "y_hat": 0.6214774197154729
      },
      {
        "dow": 4,
        "t": "2022-07-01T00:00:00",
        "wom": 1,
        "y": 2.2444907708382646,
        "y_hat": 2.200480811486483
      },
      {
        "dow": 5,
        "t": "2022-07-02T00:00:00",
        "wom": 1,
        "y": 0.41066189647640805,
        "y_hat": -0.005673179733373987
      },
      {
        "dow": 6,
        "t": "2022-07-03T00:00:00",
        "wom": 1,
        "y": 0.8875209104102515,
        "y_hat": 0.19356235565863536
      },
      {
        "dow": 0,
        "t": "2022-07-04T00:00:00",
        "wom": 2,
        "y": 0.29879768250458105,
        "y_hat": 0.42965708190220925
      },
      {
        "dow": 1,
        "t": "2022-07-05T00:00:00",
        "wom": 2,
        "y": 0.8642111944128015,
        "y_hat": 0.25646093555418714
      },
      {
        "dow": 2,
        "t": "2022-07-06T00:00:00",
        "wom": 2,
        "y": 0.5035308154420571,
        "y_hat": 0.6243655302007207
      },
      {
        "dow": 3,
        "t": "2022-07-07T00:00:00",
        "wom": 2,
        "y": 0.4608669940295842,
        "y_hat": 0.4776726161655248
      },
      {
        "dow": 4,
        "t": "2022-07-08T00:00:00",
        "wom": 2,
        "y": 2.1336407534117914,
        "y_hat": 1.9267086574935717
      },
      {
        "dow": 5,
        "t": "2022-07-09T00:00:00",
        "wom": 2,
        "y": 0.07559350276009646,
        "y_hat": 0.3769421655384924
      },
      {
        "dow": 6,
        "t": "2022-07-10T00:00:00",
        "wom": 2,
        "y": 0.21152592170248208,
        "y_hat": 0.8903739420885581
      },
      {
        "dow": 0,
        "t": "2022-07-11T00:00:00",
        "wom": 3,
        "y": 0.7992198246512687,
        "y_hat": 0.3956285063678675
      },
      {
        "dow": 1,
        "t": "2022-07-12T00:00:00",
        "wom": 3,
        "y": 0.08134234250438865,
        "y_hat": 0.7910512143289454
      },
      {
        "dow": 2,
        "t": "2022-07-13T00:00:00",
        "wom": 3,
        "y": 0.8545474044325597,
        "y_hat": 0.5177724958674339
      },
      {
        "dow": 3,
        "t": "2022-07-14T00:00:00",
        "wom": 3,
        "y": 0.7192996694052958,
        "y_hat": 0.620566249462581
      },
      {
        "dow": 4,
        "t": "2022-07-15T00:00:00",
        "wom": 3,
        "y": 2.839731861280379,
        "y_hat": 1.816150501123242
      },
      {
        "dow": 5,
        "t": "2022-07-16T00:00:00",
        "wom": 3,
        "y": 0.6866567952654717,
        "y_hat": -0.01276717253200424
      },
      {
        "dow": 6,
        "t": "2022-07-17T00:00:00",
        "wom": 3,
        "y": 0.3025925308532783,
        "y_hat": 0.39323564947278333
      },
      {
        "dow": 0,
        "t": "2022-07-18T00:00:00",
        "wom": 4,
        "y": 0.35527032358475563,
        "y_hat": 0.7382961242832644
      },
      {
        "dow": 1,
        "t": "2022-07-19T00:00:00",
        "wom": 4,
        "y": 0.2998867051787012,
        "y_hat": 0.21941760720480818
      },
      {
        "dow": 2,
        "t": "2022-07-20T00:00:00",
        "wom": 4,
        "y": 0.32169056052383393,
        "y_hat": 0.7898113131332755
      },
      {
        "dow": 3,
        "t": "2022-07-21T00:00:00",
        "wom": 4,
        "y": 0.8817188595233318,
        "y_hat": 0.9348772828820766
      },
      {
        "dow": 4,
        "t": "2022-07-22T00:00:00",
        "wom": 4,
        "y": 2.6482046952131784,
        "y_hat": 2.4248903471092027
      },
      {
        "dow": 5,
        "t": "2022-07-23T00:00:00",
        "wom": 4,
        "y": 0.5392997474632014,
        "y_hat": 0.5271321450531967
      },
      {
        "dow": 6,
        "t": "2022-07-24T00:00:00",
        "wom": 4,
        "y": 0.28113347961902946,
        "y_hat": 0.4233497058427694
      },
      {
        "dow": 0,
        "t": "2022-07-25T00:00:00",
        "wom": 5,
        "y": 0.4047240553693535,
        "y_hat": 0.40882549428677767
      },
      {
        "dow": 1,
        "t": "2022-07-26T00:00:00",
        "wom": 5,
        "y": 0.09656379233844459,
        "y_hat": 0.33944490226532564
      },
      {
        "dow": 2,
        "t": "2022-07-27T00:00:00",
        "wom": 5,
        "y": 0.9545109848618508,
        "y_hat": 0.39276582697742557
      },
      {
        "dow": 3,
        "t": "2022-07-28T00:00:00",
        "wom": 5,
        "y": 0.2844751590965856,
        "y_hat": 1.0053736522925893
      },
      {
        "dow": 4,
        "t": "2022-07-29T00:00:00",
        "wom": 5,
        "y": 2.19828814692932,
        "y_hat": 2.3190340584426465
      },
      {
        "dow": 5,
        "t": "2022-07-30T00:00:00",
        "wom": 5,
        "y": 0.7872041161790215,
        "y_hat": 0.41612529466428594
      },
      {
        "dow": 6,
        "t": "2022-07-31T00:00:00",
        "wom": 5,
        "y": 0.8320061228006757,
        "y_hat": 0.3873758547864451
      },
      {
        "dow": 0,
        "t": "2022-08-01T00:00:00",
        "wom": 1,
        "y": 0.48050923278633095,
        "y_hat": 0.3873271273730799
      },
      {
        "dow": 1,
        "t": "2022-08-02T00:00:00",
        "wom": 1,
        "y": 0.9186312133845865,
        "y_hat": 0.2657630729075549
      },
      {
        "dow": 2,
        "t": "2022-08-03T00:00:00",
        "wom": 1,
        "y": 0.28383987157951585,
        "y_hat": 0.812226397341909
      },
      {
        "dow": 3,
        "t": "2022-08-04T00:00:00",
        "wom": 1,
        "y": 0.6963665748778668,
        "y_hat": 0.545051274206895
      },
      {
        "dow": 4,
        "t": "2022-08-05T00:00:00",
        "wom": 1,
        "y": 2.09245143951022,
        "y_hat": 1.9173436713363692
      },
      {
        "dow": 5,
        "t": "2022-08-06T00:00:00",
        "wom": 1,
        "y": 0.7172188720261602,
        "y_hat": 0.6982186760753741
      },
      {
        "dow": 6,
        "t": "2022-08-07T00:00:00",
        "wom": 1,
        "y": 0.6862602995181284,
        "y_hat": 0.8294620154593446
      },
      {
        "dow": 0,
        "t": "2022-08-08T00:00:00",
        "wom": 2,
        "y": 0.12495616249592223,
        "y_hat": 0.5578165031270065
      },
      {
        "dow": 1,
        "t": "2022-08-09T00:00:00",
        "wom": 2,
        "y": 0.7803068072936611,
        "y_hat": 0.8828806245273373
      },
      {
        "dow": 2,
        "t": "2022-08-10T00:00:00",
        "wom": 2,
        "y": 0.1711236164496699,
        "y_hat": 0.3072981752448422
      },
      {
        "dow": 3,
        "t": "2022-08-11T00:00:00",
        "wom": 2,
        "y": 0.6418292052213659,
        "y_hat": 0.8776281703166817
      },
      {
        "dow": 4,
        "t": "2022-08-12T00:00:00",
        "wom": 2,
        "y": 2.398223140821866,
        "y_hat": 1.8286880827957883
      },
      {
        "dow": 5,
        "t": "2022-08-13T00:00:00",
        "wom": 2,
        "y": 0.7667020108493909,
        "y_hat": 0.6145027128796774
      },
      {
        "dow": 6,
        "t": "2022-08-14T00:00:00",
        "wom": 2,
        "y": 0.41815335304653145,
        "y_hat": 0.687838642784738
      },
      {
        "dow": 0,
        "t": "2022-08-15T00:00:00",
        "wom": 3,
        "y": 0.6891797613670014,
        "y_hat": 0.2788659947827048
      },
      {
        "dow": 1,
        "t": "2022-08-16T00:00:00",
        "wom": 3,
        "y": 0.5101490274593742,
        "y_hat": 0.7167571900739363
      },
      {
        "dow": 2,
        "t": "2022-08-17T00:00:00",
        "wom": 3,
        "y": 0.7796622447746596,
        "y_hat": 0.2435043736575076
      },
      {
        "dow": 3,
        "t": "2022-08-18T00:00:00",
        "wom": 3,
        "y": 0.5627361856480227,
        "y_hat": 0.8153113684881846
      },
      {
        "dow": 4,
        "t": "2022-08-19T00:00:00",
        "wom": 3,
        "y": 2.149269114111338,
        "y_hat": 2.120296010929617
      },
      {
        "dow": 5,
        "t": "2022-08-20T00:00:00",
        "wom": 3,
        "y": 0.4028888263767394,
        "y_hat": 0.6325429458532661
      },
      {
        "dow": 6,
        "t": "2022-08-21T00:00:00",
        "wom": 3,
        "y": 0.29581445643538484,
        "y_hat": 0.5516807517235476
      },
      {
        "dow": 0,
        "t": "2022-08-22T00:00:00",
        "wom": 4,
        "y": 0.22454795399899974,
        "y_hat": 0.6976284978396523
      },
      {
        "dow": 1,
        "t": "2022-08-23T00:00:00",
        "wom": 4,
        "y": 0.5368502058660074,
        "y_hat": 0.5873414452851247
      },
      {
        "dow": 2,
        "t": "2022-08-24T00:00:00",
        "wom": 4,
        "y": 0.8352731932733497,
        "y_hat": 0.7224014633390031
      },
      {
        "dow": 3,
        "t": "2022-08-25T00:00:00",
        "wom": 4,
        "y": 0.7156495255143432,
        "y_hat": 0.7332381829424769
      },
      {
        "dow": 4,
        "t": "2022-08-26T00:00:00",
        "wom": 4,
        "y": 2.081170707633651,
        "y_hat": 1.87659129600735
      },
      {
        "dow": 5,
        "t": "2022-08-27T00:00:00",
        "wom": 4,
        "y": 0.9846573243966786,
        "y_hat": 0.34638121191787496
      },
      {
        "dow": 6,
        "t": "2022-08-28T00:00:00",
        "wom": 4,
        "y": 0.11556538882109113,
        "y_hat": 0.3667682759429237
      },
      {
        "dow": 0,
        "t": "2022-08-29T00:00:00",
        "wom": 5,
        "y": 0.26133230623941184,
        "y_hat": 0.35503041593931184
      },
      {
        "dow": 1,
        "t": "2022-08-30T00:00:00",
        "wom": 5,
        "y": 0.9006219041976188,
        "y_hat": 0.6061647413332288
      },
      {
        "dow": 2,
        "t": "2022-08-31T00:00:00",
        "wom": 5,
        "y": 0.2515146218161567,
        "y_hat": 0.7661634229213348
      },
      {
        "dow": 3,
        "t": "2022-09-01T00:00:00",
        "wom": 1,
        "y": 0.13098484463014348,
        "y_hat": 0.8927774110621727
      },
      {
        "dow": 4,
        "t": "2022-09-02T00:00:00",
        "wom": 1,
        "y": 2.9173298508007353,
        "y_hat": 1.9056368098697876
      },
      {
        "dow": 5,
        "t": "2022-09-03T00:00:00",
        "wom": 1,
        "y": 0.902033118492389,
        "y_hat": 0.7302326136080878
      },
      {
        "dow": 6,
        "t": "2022-09-04T00:00:00",
        "wom": 1,
        "y": 0.6094812461745855,
        "y_hat": 0.25020246825690856
      },
      {
        "dow": 0,
        "t": "2022-09-05T00:00:00",
        "wom": 2,
        "y": 0.21237629900528165,
        "y_hat": 0.40071229674241443
      },
      {
        "dow": 1,
        "t": "2022-09-06T00:00:00",
        "wom": 2,
        "y": 0.8674613314753005,
        "y_hat": 0.8839620832813906
      },
      {
        "dow": 2,
        "t": "2022-09-07T00:00:00",
        "wom": 2,
        "y": 0.7318690107796962,
        "y_hat": 0.2160586019821361
      },
      {
        "dow": 3,
        "t": "2022-09-08T00:00:00",
        "wom": 2,
        "y": 0.9273400869945371,
        "y_hat": 0.4715389562157271
      },
      {
        "dow": 4,
        "t": "2022-09-09T00:00:00",
        "wom": 2,
        "y": 2.491296824471812,
        "y_hat": 2.5359284505627366
      },
      {
        "dow": 5,
        "t": "2022-09-10T00:00:00",
        "wom": 2,
        "y": 0.3736267511181386,
        "y_hat": 0.7727921780314946
      },
      {
        "dow": 6,
        "t": "2022-09-11T00:00:00",
        "wom": 2,
        "y": 0.6741226397964637,
        "y_hat": 0.6757791643455024
      },
      {
        "dow": 0,
        "t": "2022-09-12T00:00:00",
        "wom": 3,
        "y": 0.5690598209381145,
        "y_hat": 0.32910170992609866
      },
      {
        "dow": 1,
        "t": "2022-09-13T00:00:00",
        "wom": 3,
        "y": 0.5845921058651229,
        "y_hat": 0.8667305870601201
      },
      {
        "dow": 2,
        "t": "2022-09-14T00:00:00",
        "wom": 3,
        "y": 0.06190141469816057,
        "y_hat": 0.7304993399306224
      },
      {
        "dow": 3,
        "t": "2022-09-15T00:00:00",
        "wom": 3,
        "y": 0.7726353296869438,
        "y_hat": 1.1302618408265364
      },
      {
        "dow": 4,
        "t": "2022-09-16T00:00:00",
        "wom": 3,
        "y": 2.3982766116278027,
        "y_hat": 2.1245253377124134
      },
      {
        "dow": 5,
        "t": "2022-09-17T00:00:00",
        "wom": 3,
        "y": 0.02485301435952192,
        "y_hat": 0.3418451225617929
      },
      {
        "dow": 6,
        "t": "2022-09-18T00:00:00",
        "wom": 3,
        "y": 0.3885550460991051,
        "y_hat": 0.791546970444991
      },
      {
        "dow": 0,
        "t": "2022-09-19T00:00:00",
        "wom": 4,
        "y": 0.5124897271631259,
        "y_hat": 0.6167450786721291
      },
      {
        "dow": 1,
        "t": "2022-09-20T00:00:00",
        "wom": 4,
        "y": 0.6717284818449774,
        "y_hat": 0.5683455931097686
      },
      {
        "dow": 2,
        "t": "2022-09-21T00:00:00",
        "wom": 4,
        "y": 0.8010250603087118,
        "y_hat": 0.16348941838804557
      },
      {
        "dow": 3,
        "t": "2022-09-22T00:00:00",
        "wom": 4,
        "y": 0.7215828184438076,
        "y_hat": 0.9579218553270217
      },
      {
        "dow": 4,
        "t": "2022-09-23T00:00:00",
        "wom": 4,
        "y": 2.467671973989098,
        "y_hat": 2.0520034261598594
      },
      {
        "dow": 5,
        "t": "2022-09-24T00:00:00",
        "wom": 4,
        "y": 0.7177366636762655,
        "y_hat": 0.027309689043911595
      },
      {
        "dow": 6,
        "t": "2022-09-25T00:00:00",
        "wom": 4,
        "y": 0.48856096479123934,
        "y_hat": 0.5118161007784914
      },
      {
        "dow": 0,
        "t": "2022-09-26T00:00:00",
        "wom": 5,
        "y": 0.5960952391022855,
        "y_hat": 0.5831225175604509
      },
      {
        "dow": 1,
        "t": "2022-09-27T00:00:00",
        "wom": 5,
        "y": 0.8681440058433928,
        "y_hat": 0.7164890349274464
      },
      {
        "dow": 2,
        "t": "2022-09-28T00:00:00",
        "wom": 5,
        "y": 0.8669839542345549,
        "y_hat": 0.7558961040463579
      },
      {
        "dow": 3,
        "t": "2022-09-29T00:00:00",
        "wom": 5,
        "y": 0.8884123252558981,
        "y_hat": 0.9136258110680433
      },
      {
        "dow": 4,
        "t": "2022-09-30T00:00:00",
        "wom": 5,
        "y": 2.3463008214785965,
        "y_hat": 2.170395871059101
      },
      {
        "dow": 5,
        "t": "2022-10-01T00:00:00",
        "wom": 1,
        "y": 0.2954617156792265,
        "y_hat": 0.6193438855168791
      },
      {
        "dow": 6,
        "t": "2022-10-02T00:00:00",
        "wom": 1,
        "y": 0.633109100839611,
        "y_hat": 0.6250991594516301
      },
      {
        "dow": 0,
        "t": "2022-10-03T00:00:00",
        "wom": 2,
        "y": 0.3669341564363222,
        "y_hat": 0.6487485840081911
      },
      {
        "dow": 1,
        "t": "2022-10-04T00:00:00",
        "wom": 2,
        "y": 0.24876598531702476,
        "y_hat": 0.9024798454441081
      },
      {
        "dow": 2,
        "t": "2022-10-05T00:00:00",
        "wom": 2,
        "y": 0.16587282213408583,
        "y_hat": 0.864694771993483
      },
      {
        "dow": 3,
        "t": "2022-10-06T00:00:00",
        "wom": 2,
        "y": 0.9919913685200391,
        "y_hat": 1.0759773360016518
      },
      {
        "dow": 4,
        "t": "2022-10-07T00:00:00",
        "wom": 2,
        "y": 2.3533296873989085,
        "y_hat": 1.996269605357021
      },
      {
        "dow": 5,
        "t": "2022-10-08T00:00:00",
        "wom": 2,
        "y": 0.7213297581947791,
        "y_hat": 0.29729506705732356
      },
      {
        "dow": 6,
        "t": "2022-10-09T00:00:00",
        "wom": 2,
        "y": 0.7224275916390493,
        "y_hat": 0.7068588675971959
      },
      {
        "dow": 0,
        "t": "2022-10-10T00:00:00",
        "wom": 3,
        "y": 0.29155319532221713,
        "y_hat": 0.42728427875394515
      },
      {
        "dow": 1,
        "t": "2022-10-11T00:00:00",
        "wom": 3,
        "y": 0.36991001906594234,
        "y_hat": 0.33568224902855803
      },
      {
        "dow": 3,
        "t": "2022-10-13T00:00:00",
        "wom": 3,
        "y": 0.19840529083955494,
        "y_hat": 0.29565960994664947
      },
      {
        "dow": 4,
        "t": "2022-10-14T00:00:00",
        "wom": 3,
        "y": 2.4606866711518958,
        "y_hat": 1.1571579058213466
      },
      {
        "dow": 5,
        "t": "2022-10-15T00:00:00",
        "wom": 3,
        "y": 0.6335089949594493,
        "y_hat": 1.9483070568198397
      },
      {
        "dow": 0,
        "t": "2022-10-17T00:00:00",
        "wom": 4,
        "y": 0.016790840072704594,
        "y_hat": 0.8254696800585881
      },
      {
        "dow": 1,
        "t": "2022-10-18T00:00:00",
        "wom": 4,
        "y": 0.15482355482083832,
        "y_hat": 0.7621584712731398
      },
      {
        "dow": 2,
        "t": "2022-10-19T00:00:00",
        "wom": 4,
        "y": 0.32294713218572024,
        "y_hat": 0.3962266241656257
      },
      {
        "dow": 3,
        "t": "2022-10-20T00:00:00",
        "wom": 4,
        "y": 0.09723518582862622,
        "y_hat": 0.3806986869586265
      },
      {
        "dow": 4,
        "t": "2022-10-21T00:00:00",
        "wom": 4,
        "y": 2.161951112634909,
        "y_hat": 0.5267996251758436
      },
      {
        "dow": 5,
        "t": "2022-10-22T00:00:00",
        "wom": 4,
        "y": 0.9308281099209696,
        "y_hat": 2.0615971193726033
      },
      {
        "dow": 6,
        "t": "2022-10-23T00:00:00",
        "wom": 4,
        "y": 0.3700513293335844,
        "y_hat": 0.6583394087430087
      },
      {
        "dow": 0,
        "t": "2022-10-24T00:00:00",
        "wom": 5,
        "y": 0.33730301215287395,
        "y_hat": 0.16148730610734813
      },
      {
        "dow": 1,
        "t": "2022-10-25T00:00:00",
        "wom": 5,
        "y": 0.2979813390418188,
        "y_hat": 0.28412188113461
      },
      {
        "dow": 2,
        "t": "2022-10-26T00:00:00",
        "wom": 5,
        "y": 0.8989252622362512,
        "y_hat": 0.3507639467023297
      },
      {
        "dow": 3,
        "t": "2022-10-27T00:00:00",
        "wom": 5,
        "y": 0.9678609978612648,
        "y_hat": 0.3570138530344828
      },
      {
        "dow": 4,
        "t": "2022-10-28T00:00:00",
        "wom": 5,
        "y": 2.852867990371385,
        "y_hat": 1.950251357416163
      },
      {
        "dow": 5,
        "t": "2022-10-29T00:00:00",
        "wom": 5,
        "y": 0.09421902072623589,
        "y_hat": 0.7659453932958807
      },
      {
        "dow": 6,
        "t": "2022-10-30T00:00:00",
        "wom": 5,
        "y": 0.8581628861015397,
        "y_hat": 0.5576703327595525
      },
      {
        "dow": 0,
        "t": "2022-10-31T00:00:00",
        "wom": 6,
        "y": 0.20114187846747755,
        "y_hat": 0.36978132597398045
      },
      {
        "dow": 1,
        "t": "2022-11-01T00:00:00",
        "wom": 1,
        "y": 0.6181376931339703,
        "y_hat": 0.4695812294355507
      },
      {
        "dow": 2,
        "t": "2022-11-02T00:00:00",
        "wom": 1,
        "y": 0.19978354259674502,
        "y_hat": 0.868099924430836
      },
      {
        "dow": 3,
        "t": "2022-11-03T00:00:00",
        "wom": 1,
        "y": 0.1768860986420725,
        "y_hat": 1.2184194578067036
      },
      {
        "dow": 4,
        "t": "2022-11-04T00:00:00",
        "wom": 1,
        "y": 2.4752377813951942,
        "y_hat": 2.4522877379814707
      },
      {
        "dow": 5,
        "t": "2022-11-05T00:00:00",
        "wom": 1,
        "y": 0.9055115735426204,
        "y_hat": 0.13519463876149423
      },
      {
        "dow": 6,
        "t": "2022-11-06T00:00:00",
        "wom": 1,
        "y": 0.6649858316616734,
        "y_hat": 0.868497087092654
      },
      {
        "dow": 0,
        "t": "2022-11-07T00:00:00",
        "wom": 2,
        "y": 0.5304475702709496,
        "y_hat": 0.347290679389227
      },
      {
        "dow": 1,
        "t": "2022-11-08T00:00:00",
        "wom": 2,
        "y": 0.1761904542362559,
        "y_hat": 0.6530212188928876
      },
      {
        "dow": 2,
        "t": "2022-11-09T00:00:00",
        "wom": 2,
        "y": 0.9624710144321974,
        "y_hat": 0.27238690081586125
      },
      {
        "dow": 3,
        "t": "2022-11-10T00:00:00",
        "wom": 2,
        "y": 0.7059542876896551,
        "y_hat": 0.45276494790597416
      },
      {
        "dow": 4,
        "t": "2022-11-11T00:00:00",
        "wom": 2,
        "y": 2.506728336965063,
        "y_hat": 2.223550860170452
      },
      {
        "dow": 5,
        "t": "2022-11-12T00:00:00",
        "wom": 2,
        "y": 0.8433091095294604,
        "y_hat": 0.790427963771611
      },
      {
        "dow": 6,
        "t": "2022-11-13T00:00:00",
        "wom": 2,
        "y": 0.7560702301487369,
        "y_hat": 0.7645748747103471
      },
      {
        "dow": 0,
        "t": "2022-11-14T00:00:00",
        "wom": 3,
        "y": 0.06592714264449118,
        "y_hat": 0.551785744424038
      },
      {
        "dow": 1,
        "t": "2022-11-15T00:00:00",
        "wom": 3,
        "y": 0.2504072631614749,
        "y_hat": 0.3937051159443683
      },
      {
        "dow": 2,
        "t": "2022-11-16T00:00:00",
        "wom": 3,
        "y": 0.5805948341045363,
        "y_hat": 0.9297852525288297
      },
      {
        "dow": 3,
        "t": "2022-11-17T00:00:00",
        "wom": 3,
        "y": 0.5803663521377741,
        "y_hat": 0.9206309299876527
      },
      {
        "dow": 4,
        "t": "2022-11-18T00:00:00",
        "wom": 3,
        "y": 2.044237322520029,
        "y_hat": 2.2380066381976236
      },
      {
        "dow": 5,
        "t": "2022-11-19T00:00:00",
        "wom": 3,
        "y": 0.5560838530176727,
        "y_hat": 0.7884958398173992
      },
      {
        "dow": 6,
        "t": "2022-11-20T00:00:00",
        "wom": 3,
        "y": 0.28756910769465927,
        "y_hat": 0.7959620405490364
      },
      {
        "dow": 0,
        "t": "2022-11-21T00:00:00",
        "wom": 4,
        "y": 0.32658389395269083,
        "y_hat": 0.21166254932199938
      },
      {
        "dow": 1,
        "t": "2022-11-22T00:00:00",
        "wom": 4,
        "y": 0.9255075128439307,
        "y_hat": 0.38665377704660625
      },
      {
        "dow": 2,
        "t": "2022-11-23T00:00:00",
        "wom": 4,
        "y": 0.2351401173462816,
        "y_hat": 0.5808800334736386
      },
      {
        "dow": 3,
        "t": "2022-11-24T00:00:00",
        "wom": 4,
        "y": 0.6755009569121625,
        "y_hat": 0.8266328716447103
      },
      {
        "dow": 4,
        "t": "2022-11-25T00:00:00",
        "wom": 4,
        "y": 2.6532627672211397,
        "y_hat": 1.8280398649315686
      },
      {
        "dow": 5,
        "t": "2022-11-26T00:00:00",
        "wom": 4,
        "y": 0.13334860717528818,
        "y_hat": 0.4683779755412475
      },
      {
        "dow": 6,
        "t": "2022-11-27T00:00:00",
        "wom": 4,
        "y": 0.21224946729617233,
        "y_hat": 0.47571819591669623
      },
      {
        "dow": 0,
        "t": "2022-11-28T00:00:00",
        "wom": 5,
        "y": 0.10485005588135898,
        "y_hat": 0.4942734566618337
      },
      {
        "dow": 1,
        "t": "2022-11-29T00:00:00",
        "wom": 5,
        "y": 0.571653845644994,
        "y_hat": 0.9184085267852931
      },
      {
        "dow": 2,
        "t": "2022-11-30T00:00:00",
        "wom": 5,
        "y": 0.6455551020000664,
        "y_hat": 0.31231384650023997
      },
      {
        "dow": 3,
        "t": "2022-12-01T00:00:00",
        "wom": 1,
        "y": 0.07205418156187893,
        "y_hat": 0.9374594516565264
      },
      {
        "dow": 4,
        "t": "2022-12-02T00:00:00",
        "wom": 1,
        "y": 2.8390812934919314,
        "y_hat": 2.341819341243384
      },
      {
        "dow": 5,
        "t": "2022-12-03T00:00:00",
        "wom": 1,
        "y": 0.3495218764848188,
        "y_hat": 0.08944039048333478
      },
      {
        "dow": 6,
        "t": "2022-12-04T00:00:00",
        "wom": 1,
        "y": 0.9605832472674337,
        "y_hat": 0.3954251468739636
      },
      {
        "dow": 0,
        "t": "2022-12-05T00:00:00",
        "wom": 2,
        "y": 0.47976044149700026,
        "y_hat": 0.22742778707659686
      },
      {
        "dow": 1,
        "t": "2022-12-06T00:00:00",
        "wom": 2,
        "y": 0.6055161236619802,
        "y_hat": 0.6882974895030173
      },
      {
        "dow": 2,
        "t": "2022-12-07T00:00:00",
        "wom": 2,
        "y": 0.41871964951333474,
        "y_hat": 0.5985953774279416
      },
      {
        "dow": 3,
        "t": "2022-12-08T00:00:00",
        "wom": 2,
        "y": 0.966612891389878,
        "y_hat": 0.4802135456559535
      },
      {
        "dow": 4,
        "t": "2022-12-09T00:00:00",
        "wom": 2,
        "y": 2.5654467983822786,
        "y_hat": 2.4351431857753907
      },
      {
        "dow": 5,
        "t": "2022-12-10T00:00:00",
        "wom": 2,
        "y": 0.34560834727361145,
        "y_hat": 0.3847875951155271
      },
      {
        "dow": 0,
        "t": "2022-12-12T00:00:00",
        "wom": 3,
        "y": 0.9652744314834171,
        "y_hat": 1.0379068963198783
      },
      {
        "dow": 1,
        "t": "2022-12-13T00:00:00",
        "wom": 3,
        "y": 0.5648309593161226,
        "y_hat": 0.5472366758568219
      },
      {
        "dow": 2,
        "t": "2022-12-14T00:00:00",
        "wom": 3,
        "y": 0.7595290379916108,
        "y_hat": 0.670178606302035
      },
      {
        "dow": 3,
        "t": "2022-12-15T00:00:00",
        "wom": 3,
        "y": 0.4539146559253099,
        "y_hat": 0.5043741386101946
      },
      {
        "dow": 4,
        "t": "2022-12-16T00:00:00",
        "wom": 3,
        "y": 2.6626571950380864,
        "y_hat": 1.190177730072604
      },
      {
        "dow": 5,
        "t": "2022-12-17T00:00:00",
        "wom": 3,
        "y": 0.901352733645678,
        "y_hat": 2.102127404387424
      },
      {
        "dow": 6,
        "t": "2022-12-18T00:00:00",
        "wom": 3,
        "y": 0.1389757185937477,
        "y_hat": 0.5539582514000876
      },
      {
        "dow": 0,
        "t": "2022-12-19T00:00:00",
        "wom": 4,
        "y": 0.523376913529589,
        "y_hat": 1.0081920062672491
      },
      {
        "dow": 1,
        "t": "2022-12-20T00:00:00",
        "wom": 4,
        "y": 0.22793475704817479,
        "y_hat": 0.6600944163699223
      },
      {
        "dow": 2,
        "t": "2022-12-21T00:00:00",
        "wom": 4,
        "y": 0.3918232223846172,
        "y_hat": 0.7670472910419969
      },
      {
        "dow": 3,
        "t": "2022-12-22T00:00:00",
        "wom": 4,
        "y": 0.47576866552495234,
        "y_hat": 0.7544336162462358
      },
      {
        "dow": 4,
        "t": "2022-12-23T00:00:00",
        "wom": 4,
        "y": 2.3131731287518735,
        "y_hat": 2.401691369014163
      },
      {
        "dow": 5,
        "t": "2022-12-24T00:00:00",
        "wom": 4,
        "y": 0.11547575723384762,
        "y_hat": 0.7571463560630372
      },
      {
        "dow": 6,
        "t": "2022-12-25T00:00:00",
        "wom": 4,
        "y": 0.880622203620008,
        "y_hat": 0.390952255706321
      },
      {
        "dow": 0,
        "t": "2022-12-26T00:00:00",
        "wom": 5,
        "y": 0.4962845210256144,
        "y_hat": 0.5473976671154774
      },
      {
        "dow": 1,
        "t": "2022-12-27T00:00:00",
        "wom": 5,
        "y": 0.5406198774645495,
        "y_hat": 0.376157583818229
      },
      {
        "dow": 3,
        "t": "2022-12-29T00:00:00",
        "wom": 5,
        "y": 0.0927341711398395,
        "y_hat": 0.45615700825385697
      },
      {
        "dow": 4,
        "t": "2022-12-30T00:00:00",
        "wom": 5,
        "y": 2.42594516701449,
        "y_hat": 0.7909413138327701
      },
      {
        "dow": 5,
        "t": "2022-12-31T00:00:00",
        "wom": 5,
        "y": 0.9191390581073097,
        "y_hat": 1.8834550368539449
      }
    ]
  },
  "layer": [
    {
      "encoding": {
        "tooltip": {
          "field": "t",
          "type": "temporal"
        },
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "darkblue",
        "opacity": 0.6,
        "type": "circle"
      },
      "selection": {
        "selector006": {
          "bind": "scales",
          "type": "interval"
        }
      },
      "title": "",
      "width": 400
    },
    {
      "encoding": {
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y_hat",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "t",
        "type": "line"
      },
      "title": "7 lags, period: 0, MAE: 0.349",
      "width": 400
    }
  ]
}

|       |            0 |
|:------|-------------:|
| trend |  0.000439606 |
| y.L1  | -0.0669961   |
| y.L2  |  0.0346199   |
| y.L3  |  0.00114962  |
| y.L4  |  0.00301319  |
| y.L5  | -0.0221748   |
| y.L6  |  0.0999255   |
| y.L7  |  0.817227    |

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

|        |            0 |
|:-------|-------------:|
| trend  | -7.70677e-05 |
| s(1,7) |  0.398445    |
| s(2,7) |  0.338926    |
| s(3,7) |  0.424741    |
| s(4,7) |  0.463767    |
| s(5,7) |  0.387178    |
| s(6,7) |  0.471761    |
| s(7,7) |  2.39489     |
| y.L1   |  0.0436667   |
| y.L2   |  0.0682647   |

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

var spec = {
  "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
  "config": {
    "view": {
      "continuousHeight": 300,
      "continuousWidth": 400
    }
  },
  "data": {
    "name": "data-fd8763b902e03308481e7cc011e059f1"
  },
  "datasets": {
    "data-fd8763b902e03308481e7cc011e059f1": [
      {
        "dow": 5,
        "t": "2022-01-01T00:00:00",
        "wom": 1,
        "y": 0.9955981828907805,
        "y_hat": null
      },
      {
        "dow": 6,
        "t": "2022-01-02T00:00:00",
        "wom": 1,
        "y": 0.6234109104925726,
        "y_hat": null
      },
      {
        "dow": 0,
        "t": "2022-01-03T00:00:00",
        "wom": 2,
        "y": 0.8760936874425284,
        "y_hat": 0.46314410735923833
      },
      {
        "dow": 1,
        "t": "2022-01-04T00:00:00",
        "wom": 2,
        "y": 0.8893778611747504,
        "y_hat": 0.5208380786254705
      },
      {
        "dow": 2,
        "t": "2022-01-05T00:00:00",
        "wom": 2,
        "y": 0.3867624782099456,
        "y_hat": 0.47121516013853365
      },
      {
        "dow": 3,
        "t": "2022-01-06T00:00:00",
        "wom": 2,
        "y": 0.6899198960223786,
        "y_hat": 0.49843717875709764
      },
      {
        "dow": 4,
        "t": "2022-01-07T00:00:00",
        "wom": 2,
        "y": 4.992852550566659,
        "y_hat": 3.367887687726011
      },
      {
        "dow": 5,
        "t": "2022-01-08T00:00:00",
        "wom": 2,
        "y": 0.46804208563249405,
        "y_hat": 0.586440752594935
      },
      {
        "dow": 6,
        "t": "2022-01-09T00:00:00",
        "wom": 2,
        "y": 0.9455841386245034,
        "y_hat": 0.4880599574075523
      },
      {
        "dow": 0,
        "t": "2022-01-10T00:00:00",
        "wom": 3,
        "y": 0.6381233538848571,
        "y_hat": 0.492395365389997
      },
      {
        "dow": 1,
        "t": "2022-01-11T00:00:00",
        "wom": 3,
        "y": 0.7595024746539746,
        "y_hat": 0.5012581501601908
      },
      {
        "dow": 2,
        "t": "2022-01-12T00:00:00",
        "wom": 3,
        "y": 0.23738191602671632,
        "y_hat": 0.4743791950744365
      },
      {
        "dow": 3,
        "t": "2022-01-13T00:00:00",
        "wom": 3,
        "y": 0.09680924548766545,
        "y_hat": 0.4972611446738794
      },
      {
        "dow": 4,
        "t": "2022-01-14T00:00:00",
        "wom": 3,
        "y": 2.588571863863769,
        "y_hat": 3.351318177512937
      },
      {
        "dow": 5,
        "t": "2022-01-15T00:00:00",
        "wom": 3,
        "y": 0.16573502501513393,
        "y_hat": 0.5192828313652479
      },
      {
        "dow": 6,
        "t": "2022-01-16T00:00:00",
        "wom": 3,
        "y": 0.38942124330865624,
        "y_hat": 0.5578273103618593
      },
      {
        "dow": 0,
        "t": "2022-01-17T00:00:00",
        "wom": 4,
        "y": 0.688787521402102,
        "y_hat": 0.48230422263498074
      },
      {
        "dow": 1,
        "t": "2022-01-18T00:00:00",
        "wom": 4,
        "y": 0.6697600544042848,
        "y_hat": 0.52165104465011
      },
      {
        "dow": 2,
        "t": "2022-01-19T00:00:00",
        "wom": 4,
        "y": 0.36885391553882474,
        "y_hat": 0.4692897572013656
      },
      {
        "dow": 3,
        "t": "2022-01-20T00:00:00",
        "wom": 4,
        "y": 0.9325008856726638,
        "y_hat": 0.5048938642951577
      },
      {
        "dow": 4,
        "t": "2022-01-21T00:00:00",
        "wom": 4,
        "y": 4.141135576285614,
        "y_hat": 3.3769837746773206
      },
      {
        "dow": 5,
        "t": "2022-01-22T00:00:00",
        "wom": 4,
        "y": 0.9772509899400809,
        "y_hat": 0.5471983662075591
      },
      {
        "dow": 6,
        "t": "2022-01-23T00:00:00",
        "wom": 4,
        "y": 0.5683493004574759,
        "y_hat": 0.5348357091535335
      },
      {
        "dow": 0,
        "t": "2022-01-24T00:00:00",
        "wom": 5,
        "y": 0.9649672826390342,
        "y_hat": 0.46134947437514306
      },
      {
        "dow": 1,
        "t": "2022-01-25T00:00:00",
        "wom": 5,
        "y": 0.22830327875114365,
        "y_hat": 0.52548393804881
      },
      {
        "dow": 2,
        "t": "2022-01-26T00:00:00",
        "wom": 5,
        "y": 0.5236014315750456,
        "y_hat": 0.44389639144716336
      },
      {
        "dow": 3,
        "t": "2022-01-27T00:00:00",
        "wom": 5,
        "y": 0.5771602890446812,
        "y_hat": 0.5251944814497006
      },
      {
        "dow": 4,
        "t": "2022-01-28T00:00:00",
        "wom": 5,
        "y": 2.3198686688954586,
        "y_hat": 3.3587881913094093
      },
      {
        "dow": 5,
        "t": "2022-01-29T00:00:00",
        "wom": 5,
        "y": 0.4375103099359736,
        "y_hat": 0.4931327489267444
      },
      {
        "dow": 6,
        "t": "2022-01-30T00:00:00",
        "wom": 5,
        "y": 0.7071449184091295,
        "y_hat": 0.5764120471080875
      },
      {
        "dow": 0,
        "t": "2022-01-31T00:00:00",
        "wom": 6,
        "y": 0.1524637438937454,
        "y_hat": 0.4843778236005443
      },
      {
        "dow": 1,
        "t": "2022-02-01T00:00:00",
        "wom": 1,
        "y": 0.011748418393797033,
        "y_hat": 0.4912895489133048
      },
      {
        "dow": 2,
        "t": "2022-02-02T00:00:00",
        "wom": 1,
        "y": 0.25517769811183577,
        "y_hat": 0.4632434115482187
      },
      {
        "dow": 3,
        "t": "2022-02-03T00:00:00",
        "wom": 1,
        "y": 0.5156949359371945,
        "y_hat": 0.5226272734663433
      },
      {
        "dow": 4,
        "t": "2022-02-04T00:00:00",
        "wom": 1,
        "y": 2.6710173501944316,
        "y_hat": 3.365450520694246
      },
      {
        "dow": 5,
        "t": "2022-02-05T00:00:00",
        "wom": 1,
        "y": 0.8240477378755111,
        "y_hat": 0.5077601884451451
      },
      {
        "dow": 6,
        "t": "2022-02-06T00:00:00",
        "wom": 1,
        "y": 0.9508838016129778,
        "y_hat": 0.5784457319902714
      },
      {
        "dow": 0,
        "t": "2022-02-07T00:00:00",
        "wom": 2,
        "y": 0.05520043850117129,
        "y_hat": 0.4800567024042465
      },
      {
        "dow": 1,
        "t": "2022-02-08T00:00:00",
        "wom": 2,
        "y": 0.36173925787444183,
        "y_hat": 0.47943614339563645
      },
      {
        "dow": 2,
        "t": "2022-02-09T00:00:00",
        "wom": 2,
        "y": 0.3287722160717247,
        "y_hat": 0.4790326432716758
      },
      {
        "dow": 3,
        "t": "2022-02-10T00:00:00",
        "wom": 2,
        "y": 0.26344459815995447,
        "y_hat": 0.5133810260789565
      },
      {
        "dow": 4,
        "t": "2022-02-11T00:00:00",
        "wom": 2,
        "y": 4.105522116894078,
        "y_hat": 3.3537123075565805
      },
      {
        "dow": 5,
        "t": "2022-02-12T00:00:00",
        "wom": 2,
        "y": 0.9004373252539722,
        "y_hat": 0.5679865971047097
      },
      {
        "dow": 6,
        "t": "2022-02-13T00:00:00",
        "wom": 2,
        "y": 0.3083956070489431,
        "y_hat": 0.5328348848409343
      },
      {
        "dow": 0,
        "t": "2022-02-14T00:00:00",
        "wom": 3,
        "y": 0.8443045110438598,
        "y_hat": 0.45410995562044587
      },
      {
        "dow": 1,
        "t": "2022-02-15T00:00:00",
        "wom": 3,
        "y": 0.9448650092640142,
        "y_hat": 0.5294403467274962
      },
      {
        "dow": 2,
        "t": "2022-02-16T00:00:00",
        "wom": 3,
        "y": 0.5570587072891441,
        "y_hat": 0.4734509394512619
      },
      {
        "dow": 3,
        "t": "2022-02-17T00:00:00",
        "wom": 3,
        "y": 0.5374558873035822,
        "y_hat": 0.5018909202274727
      },
      {
        "dow": 4,
        "t": "2022-02-18T00:00:00",
        "wom": 3,
        "y": 2.8223117437123824,
        "y_hat": 3.355807141346047
      },
      {
        "dow": 5,
        "t": "2022-02-19T00:00:00",
        "wom": 3,
        "y": 0.02226637799800346,
        "y_hat": 0.5122206736430908
      },
      {
        "dow": 6,
        "t": "2022-02-20T00:00:00",
        "wom": 3,
        "y": 0.15725472652557881,
        "y_hat": 0.5440788922476947
      },
      {
        "dow": 0,
        "t": "2022-02-21T00:00:00",
        "wom": 4,
        "y": 0.9269915515225546,
        "y_hat": 0.47803094138813657
      },
      {
        "dow": 1,
        "t": "2022-02-22T00:00:00",
        "wom": 4,
        "y": 0.4501913471576654,
        "y_hat": 0.5373730200158857
      },
      {
        "dow": 2,
        "t": "2022-02-23T00:00:00",
        "wom": 4,
        "y": 0.9531441262485346,
        "y_hat": 0.4526387657826117
      },
      {
        "dow": 3,
        "t": "2022-02-24T00:00:00",
        "wom": 4,
        "y": 0.5844501255113494,
        "y_hat": 0.5327098767539253
      },
      {
        "dow": 4,
        "t": "2022-02-25T00:00:00",
        "wom": 4,
        "y": 4.8039777609147105,
        "y_hat": 3.344048901686988
      },
      {
        "dow": 5,
        "t": "2022-02-26T00:00:00",
        "wom": 4,
        "y": 0.26383392179603904,
        "y_hat": 0.5821756132840402
      },
      {
        "dow": 6,
        "t": "2022-02-27T00:00:00",
        "wom": 4,
        "y": 0.7531464817944895,
        "y_hat": 0.4860446340032133
      },
      {
        "dow": 0,
        "t": "2022-02-28T00:00:00",
        "wom": 5,
        "y": 0.15885513834885823,
        "y_hat": 0.4913213399898057
      },
      {
        "dow": 1,
        "t": "2022-03-01T00:00:00",
        "wom": 1,
        "y": 0.14535499394438622,
        "y_hat": 0.4894139499618042
      },
      {
        "dow": 2,
        "t": "2022-03-02T00:00:00",
        "wom": 1,
        "y": 0.4410675050392091,
        "y_hat": 0.4673009235282812
      },
      {
        "dow": 3,
        "t": "2022-03-03T00:00:00",
        "wom": 1,
        "y": 0.7517726853328405,
        "y_hat": 0.5242983398058877
      },
      {
        "dow": 4,
        "t": "2022-03-04T00:00:00",
        "wom": 1,
        "y": 2.7945001756665624,
        "y_hat": 3.36717886846027
      },
      {
        "dow": 5,
        "t": "2022-03-05T00:00:00",
        "wom": 1,
        "y": 0.04870630413527299,
        "y_hat": 0.5037285597856613
      },
      {
        "dow": 6,
        "t": "2022-03-06T00:00:00",
        "wom": 1,
        "y": 0.2161041338955525,
        "y_hat": 0.5456903300048429
      },
      {
        "dow": 0,
        "t": "2022-03-07T00:00:00",
        "wom": 2,
        "y": 0.37010898460177155,
        "y_hat": 0.4789904487556662
      },
      {
        "dow": 1,
        "t": "2022-03-08T00:00:00",
        "wom": 2,
        "y": 0.09520299889511219,
        "y_hat": 0.5149723194086877
      },
      {
        "dow": 2,
        "t": "2022-03-09T00:00:00",
        "wom": 2,
        "y": 0.11977796440076782,
        "y_hat": 0.45824376856712445
      },
      {
        "dow": 3,
        "t": "2022-03-10T00:00:00",
        "wom": 2,
        "y": 0.5000495405840498,
        "y_hat": 0.5142238934019195
      },
      {
        "dow": 4,
        "t": "2022-03-11T00:00:00",
        "wom": 2,
        "y": 4.453037332781335,
        "y_hat": 3.3687373045039117
      },
      {
        "dow": 5,
        "t": "2022-03-12T00:00:00",
        "wom": 2,
        "y": 0.1454828110192531,
        "y_hat": 0.5720402921370701
      },
      {
        "dow": 6,
        "t": "2022-03-13T00:00:00",
        "wom": 2,
        "y": 0.37079700609072885,
        "y_hat": 0.4932839945844801
      },
      {
        "dow": 0,
        "t": "2022-03-14T00:00:00",
        "wom": 3,
        "y": 0.6368565984388843,
        "y_hat": 0.48119153428465444
      },
      {
        "dow": 1,
        "t": "2022-03-15T00:00:00",
        "wom": 3,
        "y": 0.5235082648923333,
        "y_hat": 0.5192789443488037
      },
      {
        "dow": 2,
        "t": "2022-03-16T00:00:00",
        "wom": 3,
        "y": 0.7192823846677754,
        "y_hat": 0.46462605996387346
      },
      {
        "dow": 3,
        "t": "2022-03-17T00:00:00",
        "wom": 3,
        "y": 0.6045077428498761,
        "y_hat": 0.5213660899131913
      },
      {
        "dow": 4,
        "t": "2022-03-18T00:00:00",
        "wom": 3,
        "y": 2.5404792929526683,
        "y_hat": 3.352217724491186
      },
      {
        "dow": 5,
        "t": "2022-03-19T00:00:00",
        "wom": 3,
        "y": 0.4830778428824394,
        "y_hat": 0.4992124824894985
      },
      {
        "dow": 6,
        "t": "2022-03-20T00:00:00",
        "wom": 3,
        "y": 0.9022692634866183,
        "y_hat": 0.5696623489374741
      },
      {
        "dow": 0,
        "t": "2022-03-21T00:00:00",
        "wom": 4,
        "y": 0.06386988650342573,
        "y_hat": 0.4889231094613167
      },
      {
        "dow": 1,
        "t": "2022-03-22T00:00:00",
        "wom": 4,
        "y": 0.2631672977407843,
        "y_hat": 0.48054430455311054
      },
      {
        "dow": 2,
        "t": "2022-03-23T00:00:00",
        "wom": 4,
        "y": 0.6967761906616411,
        "y_hat": 0.47433586249083914
      },
      {
        "dow": 3,
        "t": "2022-03-24T00:00:00",
        "wom": 4,
        "y": 0.6938964314225743,
        "y_hat": 0.529165762112242
      },
      {
        "dow": 4,
        "t": "2022-03-25T00:00:00",
        "wom": 4,
        "y": 4.340231332984116,
        "y_hat": 3.3560675778268334
      },
      {
        "dow": 5,
        "t": "2022-03-26T00:00:00",
        "wom": 4,
        "y": 0.060442940833932424,
        "y_hat": 0.5611622862745577
      },
      {
        "dow": 6,
        "t": "2022-03-27T00:00:00",
        "wom": 4,
        "y": 0.8673761163808922,
        "y_hat": 0.49372116105702163
      },
      {
        "dow": 0,
        "t": "2022-03-28T00:00:00",
        "wom": 5,
        "y": 0.5544851701436769,
        "y_hat": 0.5017317277700397
      },
      {
        "dow": 1,
        "t": "2022-03-29T00:00:00",
        "wom": 5,
        "y": 0.8547133638306371,
        "y_hat": 0.4993226617981035
      },
      {
        "dow": 2,
        "t": "2022-03-30T00:00:00",
        "wom": 5,
        "y": 0.36375693048623126,
        "y_hat": 0.47909510053665916
      },
      {
        "dow": 3,
        "t": "2022-03-31T00:00:00",
        "wom": 5,
        "y": 0.7322858415787377,
        "y_hat": 0.49709060751326334
      },
      {
        "dow": 4,
        "t": "2022-04-01T00:00:00",
        "wom": 1,
        "y": 2.1005642322427573,
        "y_hat": 3.3685135240809965
      },
      {
        "dow": 5,
        "t": "2022-04-02T00:00:00",
        "wom": 1,
        "y": 0.6446246806801427,
        "y_hat": 0.47872473313152303
      },
      {
        "dow": 6,
        "t": "2022-04-03T00:00:00",
        "wom": 1,
        "y": 0.523704664544059,
        "y_hat": 0.590017049464721
      },
      {
        "dow": 0,
        "t": "2022-04-04T00:00:00",
        "wom": 2,
        "y": 0.6478284421164979,
        "y_hat": 0.4695189110772521
      },
      {
        "dow": 1,
        "t": "2022-04-05T00:00:00",
        "wom": 2,
        "y": 0.8035144383488594,
        "y_hat": 0.5141144090784562
      },
      {
        "dow": 2,
        "t": "2022-04-06T00:00:00",
        "wom": 2,
        "y": 0.06175772835110849,
        "y_hat": 0.4739647040430617
      },
      {
        "dow": 3,
        "t": "2022-04-07T00:00:00",
        "wom": 2,
        "y": 0.10101798118758876,
        "y_hat": 0.4877490772875721
      },
      {
        "dow": 4,
        "t": "2022-04-08T00:00:00",
        "wom": 2,
        "y": 4.3286954251684495,
        "y_hat": 3.355695550867808
      },
      {
        "dow": 5,
        "t": "2022-04-09T00:00:00",
        "wom": 2,
        "y": 0.7405205134959488,
        "y_hat": 0.5803999804068959
      },
      {
        "dow": 6,
        "t": "2022-04-10T00:00:00",
        "wom": 2,
        "y": 0.08236194536441421,
        "y_hat": 0.518426831720393
      },
      {
        "dow": 0,
        "t": "2022-04-11T00:00:00",
        "wom": 3,
        "y": 0.48567949453078907,
        "y_hat": 0.4501915372228665
      },
      {
        "dow": 1,
        "t": "2022-04-12T00:00:00",
        "wom": 3,
        "y": 0.5552603671504405,
        "y_hat": 0.5229493311781209
      },
      {
        "dow": 2,
        "t": "2022-04-13T00:00:00",
        "wom": 3,
        "y": 0.2511650942997945,
        "y_hat": 0.47029767025810737
      },
      {
        "dow": 3,
        "t": "2022-04-14T00:00:00",
        "wom": 3,
        "y": 0.4804870698202284,
        "y_hat": 0.5028070625459697
      },
      {
        "dow": 4,
        "t": "2022-04-15T00:00:00",
        "wom": 3,
        "y": 2.0257783736909825,
        "y_hat": 3.3629119700571724
      },
      {
        "dow": 5,
        "t": "2022-04-16T00:00:00",
        "wom": 3,
        "y": 0.9687966319838246,
        "y_hat": 0.48420627076821776
      },
      {
        "dow": 6,
        "t": "2022-04-17T00:00:00",
        "wom": 3,
        "y": 0.7151278400228451,
        "y_hat": 0.6039766502269391
      },
      {
        "dow": 0,
        "t": "2022-04-18T00:00:00",
        "wom": 4,
        "y": 0.4718442940718869,
        "y_hat": 0.4652625511432721
      },
      {
        "dow": 1,
        "t": "2022-04-19T00:00:00",
        "wom": 4,
        "y": 0.9026979968878186,
        "y_hat": 0.5010328014288734
      },
      {
        "dow": 2,
        "t": "2022-04-20T00:00:00",
        "wom": 4,
        "y": 0.39245084658370566,
        "y_hat": 0.48318936818782454
      },
      {
        "dow": 3,
        "t": "2022-04-21T00:00:00",
        "wom": 4,
        "y": 0.4827795920741036,
        "y_hat": 0.49609499533255863
      },
      {
        "dow": 4,
        "t": "2022-04-22T00:00:00",
        "wom": 4,
        "y": 4.551414450198103,
        "y_hat": 3.358104293212166
      },
      {
        "dow": 5,
        "t": "2022-04-23T00:00:00",
        "wom": 4,
        "y": 0.650135286544272,
        "y_hat": 0.5753391685579974
      },
      {
        "dow": 6,
        "t": "2022-04-24T00:00:00",
        "wom": 4,
        "y": 0.467587186604593,
        "y_hat": 0.5073889684394609
      },
      {
        "dow": 0,
        "t": "2022-04-25T00:00:00",
        "wom": 5,
        "y": 0.8420434940525285,
        "y_hat": 0.46688389370389355
      },
      {
        "dow": 1,
        "t": "2022-04-26T00:00:00",
        "wom": 5,
        "y": 0.44245377300469013,
        "y_hat": 0.5226058767316231
      },
      {
        "dow": 2,
        "t": "2022-04-27T00:00:00",
        "wom": 5,
        "y": 0.10472408289231805,
        "y_hat": 0.45395516784302603
      },
      {
        "dow": 3,
        "t": "2022-04-28T00:00:00",
        "wom": 5,
        "y": 0.07965539705381441,
        "y_hat": 0.5010234249592131
      },
      {
        "dow": 4,
        "t": "2022-04-29T00:00:00",
        "wom": 5,
        "y": 2.1204279487951343,
        "y_hat": 3.353058174693261
      },
      {
        "dow": 5,
        "t": "2022-04-30T00:00:00",
        "wom": 5,
        "y": 0.09911422212585952,
        "y_hat": 0.5008272176958237
      },
      {
        "dow": 6,
        "t": "2022-05-01T00:00:00",
        "wom": 1,
        "y": 0.7653139992562019,
        "y_hat": 0.569058515890244
      },
      {
        "dow": 0,
        "t": "2022-05-02T00:00:00",
        "wom": 2,
        "y": 0.6928765519708857,
        "y_hat": 0.49603993290380144
      },
      {
        "dow": 1,
        "t": "2022-05-03T00:00:00",
        "wom": 2,
        "y": 0.3293225432083243,
        "y_hat": 0.5070598732945819
      },
      {
        "dow": 2,
        "t": "2022-05-04T00:00:00",
        "wom": 2,
        "y": 0.6649514990871527,
        "y_hat": 0.4547388987432257
      },
      {
        "dow": 3,
        "t": "2022-05-05T00:00:00",
        "wom": 2,
        "y": 0.15101171079274767,
        "y_hat": 0.5249502676108607
      },
      {
        "dow": 4,
        "t": "2022-05-06T00:00:00",
        "wom": 2,
        "y": 4.179842898065672,
        "y_hat": 3.3366619790847585
      },
      {
        "dow": 5,
        "t": "2022-05-07T00:00:00",
        "wom": 2,
        "y": 0.2354835900991824,
        "y_hat": 0.5727751158727903
      },
      {
        "dow": 6,
        "t": "2022-05-08T00:00:00",
        "wom": 2,
        "y": 0.05741790510662548,
        "y_hat": 0.5046051042659013
      },
      {
        "dow": 0,
        "t": "2022-05-09T00:00:00",
        "wom": 3,
        "y": 0.23077326722872582,
        "y_hat": 0.46571068116428427
      },
      {
        "dow": 1,
        "t": "2022-05-10T00:00:00",
        "wom": 3,
        "y": 0.1944400629234596,
        "y_hat": 0.5140083071885236
      },
      {
        "dow": 2,
        "t": "2022-05-11T00:00:00",
        "wom": 3,
        "y": 0.7807354431480255,
        "y_hat": 0.4652580776672461
      },
      {
        "dow": 3,
        "t": "2022-05-12T00:00:00",
        "wom": 3,
        "y": 0.31021821068384303,
        "y_hat": 0.5335333537000669
      },
      {
        "dow": 4,
        "t": "2022-05-13T00:00:00",
        "wom": 3,
        "y": 2.1275853244649534,
        "y_hat": 3.3383872124237617
      },
      {
        "dow": 5,
        "t": "2022-05-14T00:00:00",
        "wom": 3,
        "y": 0.56813301154376,
        "y_hat": 0.49305364306785965
      },
      {
        "dow": 6,
        "t": "2022-05-15T00:00:00",
        "wom": 3,
        "y": 0.10167470187610927,
        "y_hat": 0.5855018372846994
      },
      {
        "dow": 0,
        "t": "2022-05-16T00:00:00",
        "wom": 4,
        "y": 0.37826726087578866,
        "y_hat": 0.455986402610608
      },
      {
        "dow": 1,
        "t": "2022-05-17T00:00:00",
        "wom": 4,
        "y": 0.7138146434013722,
        "y_hat": 0.5177149384223318
      },
      {
        "dow": 2,
        "t": "2022-05-18T00:00:00",
        "wom": 4,
        "y": 0.12852670113262654,
        "y_hat": 0.47894404781945854
      },
      {
        "dow": 3,
        "t": "2022-05-19T00:00:00",
        "wom": 4,
        "y": 0.3585977195944233,
        "y_hat": 0.49234009661506006
      },
      {
        "dow": 4,
        "t": "2022-05-20T00:00:00",
        "wom": 4,
        "y": 4.22258022407796,
        "y_hat": 3.361926918294259
      },
      {
        "dow": 5,
        "t": "2022-05-21T00:00:00",
        "wom": 4,
        "y": 0.932770577180065,
        "y_hat": 0.5670610080985306
      },
      {
        "dow": 6,
        "t": "2022-05-22T00:00:00",
        "wom": 4,
        "y": 0.9336405466022045,
        "y_hat": 0.5281083408226585
      },
      {
        "dow": 0,
        "t": "2022-05-23T00:00:00",
        "wom": 5,
        "y": 0.7765773218651726,
        "y_hat": 0.47367728240624635
      },
      {
        "dow": 1,
        "t": "2022-05-24T00:00:00",
        "wom": 5,
        "y": 0.9092507321087694,
        "y_hat": 0.5040074331605743
      },
      {
        "dow": 2,
        "t": "2022-05-25T00:00:00",
        "wom": 5,
        "y": 0.9912972693958269,
        "y_hat": 0.47248001499905545
      },
      {
        "dow": 3,
        "t": "2022-05-26T00:00:00",
        "wom": 5,
        "y": 0.17273741694999156,
        "y_hat": 0.5168343857950793
      },
      {
        "dow": 4,
        "t": "2022-05-27T00:00:00",
        "wom": 5,
        "y": 2.7305657647776744,
        "y_hat": 3.326054718494221
      },
      {
        "dow": 5,
        "t": "2022-05-28T00:00:00",
        "wom": 5,
        "y": 0.9331495324820412,
        "y_hat": 0.5192055504918722
      },
      {
        "dow": 6,
        "t": "2022-05-29T00:00:00",
        "wom": 5,
        "y": 0.9747799262042263,
        "y_hat": 0.5781495362178678
      },
      {
        "dow": 0,
        "t": "2022-05-30T00:00:00",
        "wom": 6,
        "y": 0.6159827699504155,
        "y_hat": 0.47501251140861733
      },
      {
        "dow": 1,
        "t": "2022-05-31T00:00:00",
        "wom": 6,
        "y": 0.18957160929478611,
        "y_hat": 0.49667559887803203
      },
      {
        "dow": 2,
        "t": "2022-06-01T00:00:00",
        "wom": 1,
        "y": 0.08329858559961756,
        "y_hat": 0.45170969617669915
      },
      {
        "dow": 3,
        "t": "2022-06-02T00:00:00",
        "wom": 1,
        "y": 0.7811845717468601,
        "y_hat": 0.5080513767580573
      },
      {
        "dow": 4,
        "t": "2022-06-03T00:00:00",
        "wom": 1,
        "y": 2.5856317189042306,
        "y_hat": 3.378452247975909
      },
      {
        "dow": 5,
        "t": "2022-06-04T00:00:00",
        "wom": 1,
        "y": 0.9915634439508605,
        "y_hat": 0.49336498671409934
      },
      {
        "dow": 6,
        "t": "2022-06-05T00:00:00",
        "wom": 1,
        "y": 0.9909525874581544,
        "y_hat": 0.5849955736648643
      },
      {
        "dow": 0,
        "t": "2022-06-06T00:00:00",
        "wom": 2,
        "y": 0.546656872738867,
        "y_hat": 0.47349334937524135
      },
      {
        "dow": 1,
        "t": "2022-06-07T00:00:00",
        "wom": 2,
        "y": 0.10472218993110805,
        "y_hat": 0.49348435063995416
      },
      {
        "dow": 2,
        "t": "2022-06-08T00:00:00",
        "wom": 2,
        "y": 0.04250860920798383,
        "y_hat": 0.4508317795278227
      },
      {
        "dow": 3,
        "t": "2022-06-09T00:00:00",
        "wom": 2,
        "y": 0.3032781068949767,
        "y_hat": 0.5092890134162211
      },
      {
        "dow": 4,
        "t": "2022-06-10T00:00:00",
        "wom": 2,
        "y": 4.592471002823318,
        "y_hat": 3.3623983198068546
      },
      {
        "dow": 5,
        "t": "2022-06-11T00:00:00",
        "wom": 2,
        "y": 0.1823450459040853,
        "y_hat": 0.5818796813387966
      },
      {
        "dow": 6,
        "t": "2022-06-12T00:00:00",
        "wom": 2,
        "y": 0.7714328053698969,
        "y_hat": 0.4881089072663307
      },
      {
        "dow": 0,
        "t": "2022-06-13T00:00:00",
        "wom": 3,
        "y": 0.7117102710960721,
        "y_hat": 0.4926226713577646
      },
      {
        "dow": 1,
        "t": "2022-06-14T00:00:00",
        "wom": 3,
        "y": 0.5078428447537274,
        "y_hat": 0.5066953130980829
      },
      {
        "dow": 2,
        "t": "2022-06-15T00:00:00",
        "wom": 3,
        "y": 0.4559806484621759,
        "y_hat": 0.45972253236545024
      },
      {
        "dow": 3,
        "t": "2022-06-16T00:00:00",
        "wom": 3,
        "y": 0.3663298759696507,
        "y_hat": 0.5105493933886052
      },
      {
        "dow": 4,
        "t": "2022-06-17T00:00:00",
        "wom": 3,
        "y": 2.036192987755432,
        "y_hat": 3.350636258895791
      },
      {
        "dow": 5,
        "t": "2022-06-18T00:00:00",
        "wom": 3,
        "y": 0.5857499825271564,
        "y_hat": 0.48716134911186165
      },
      {
        "dow": 6,
        "t": "2022-06-19T00:00:00",
        "wom": 3,
        "y": 0.050005294335790484,
        "y_hat": 0.588511986302321
      },
      {
        "dow": 0,
        "t": "2022-06-20T00:00:00",
        "wom": 4,
        "y": 0.5061272547669947,
        "y_hat": 0.45282519682152267
      },
      {
        "dow": 1,
        "t": "2022-06-21T00:00:00",
        "wom": 4,
        "y": 0.7904390381333787,
        "y_hat": 0.5233768346879093
      },
      {
        "dow": 2,
        "t": "2022-06-22T00:00:00",
        "wom": 4,
        "y": 0.09323256170059835,
        "y_hat": 0.47671630218797884
      },
      {
        "dow": 3,
        "t": "2022-06-23T00:00:00",
        "wom": 4,
        "y": 0.40800965871918415,
        "y_hat": 0.48778710400180914
      },
      {
        "dow": 4,
        "t": "2022-06-24T00:00:00",
        "wom": 4,
        "y": 4.570342151877391,
        "y_hat": 3.364200816994047
      },
      {
        "dow": 5,
        "t": "2022-06-25T00:00:00",
        "wom": 4,
        "y": 0.07936746817059592,
        "y_hat": 0.5772778004258172
      },
      {
        "dow": 6,
        "t": "2022-06-26T00:00:00",
        "wom": 4,
        "y": 0.017413769472798823,
        "y_hat": 0.4848483546554289
      },
      {
        "dow": 0,
        "t": "2022-06-27T00:00:00",
        "wom": 5,
        "y": 0.4856034209379412,
        "y_hat": 0.4685329923459605
      },
      {
        "dow": 1,
        "t": "2022-06-28T00:00:00",
        "wom": 5,
        "y": 0.10626439009089561,
        "y_hat": 0.5235903601136654
      },
      {
        "dow": 2,
        "t": "2022-06-29T00:00:00",
        "wom": 5,
        "y": 0.7531865011385493,
        "y_hat": 0.4525204106328804
      },
      {
        "dow": 3,
        "t": "2022-06-30T00:00:00",
        "wom": 5,
        "y": 0.2215890993417199,
        "y_hat": 0.5345217284569447
      },
      {
        "dow": 4,
        "t": "2022-07-01T00:00:00",
        "wom": 1,
        "y": 2.2444907708382646,
        "y_hat": 3.335127859281444
      },
      {
        "dow": 5,
        "t": "2022-07-02T00:00:00",
        "wom": 1,
        "y": 0.41066189647640805,
        "y_hat": 0.4992820495222629
      },
      {
        "dow": 6,
        "t": "2022-07-03T00:00:00",
        "wom": 1,
        "y": 0.8875209104102515,
        "y_hat": 0.574895396624389
      },
      {
        "dow": 0,
        "t": "2022-07-04T00:00:00",
        "wom": 2,
        "y": 0.29879768250458105,
        "y_hat": 0.48872453812214106
      },
      {
        "dow": 1,
        "t": "2022-07-05T00:00:00",
        "wom": 2,
        "y": 0.8642111944128015,
        "y_hat": 0.48743727984136237
      },
      {
        "dow": 2,
        "t": "2022-07-06T00:00:00",
        "wom": 2,
        "y": 0.5035308154420571,
        "y_hat": 0.4860758172762325
      },
      {
        "dow": 3,
        "t": "2022-07-07T00:00:00",
        "wom": 2,
        "y": 0.4608669940295842,
        "y_hat": 0.49986670114536924
      },
      {
        "dow": 4,
        "t": "2022-07-08T00:00:00",
        "wom": 2,
        "y": 4.133640753411791,
        "y_hat": 3.352036744586642
      },
      {
        "dow": 5,
        "t": "2022-07-09T00:00:00",
        "wom": 2,
        "y": 0.07559350276009646,
        "y_hat": 0.559425421994145
      },
      {
        "dow": 6,
        "t": "2022-07-10T00:00:00",
        "wom": 2,
        "y": 0.21152592170248208,
        "y_hat": 0.49911549427227236
      },
      {
        "dow": 0,
        "t": "2022-07-11T00:00:00",
        "wom": 3,
        "y": 0.7992198246512687,
        "y_hat": 0.4754007433683476
      },
      {
        "dow": 1,
        "t": "2022-07-12T00:00:00",
        "wom": 3,
        "y": 0.08134234250438865,
        "y_hat": 0.5281267498426745
      },
      {
        "dow": 2,
        "t": "2022-07-13T00:00:00",
        "wom": 3,
        "y": 0.8545474044325597,
        "y_hat": 0.4407939583160168
      },
      {
        "dow": 3,
        "t": "2022-07-14T00:00:00",
        "wom": 3,
        "y": 0.7192996694052958,
        "y_hat": 0.5387458333292984
      },
      {
        "dow": 4,
        "t": "2022-07-15T00:00:00",
        "wom": 3,
        "y": 2.839731861280379,
        "y_hat": 3.3494414372787884
      },
      {
        "dow": 5,
        "t": "2022-07-16T00:00:00",
        "wom": 3,
        "y": 0.6866567952654717,
        "y_hat": 0.503796358015635
      },
      {
        "dow": 6,
        "t": "2022-07-17T00:00:00",
        "wom": 3,
        "y": 0.3025925308532783,
        "y_hat": 0.5645834914553769
      },
      {
        "dow": 0,
        "t": "2022-07-18T00:00:00",
        "wom": 4,
        "y": 0.35527032358475563,
        "y_hat": 0.45800814597383194
      },
      {
        "dow": 1,
        "t": "2022-07-19T00:00:00",
        "wom": 4,
        "y": 0.2998867051787012,
        "y_hat": 0.5088674688427771
      },
      {
        "dow": 2,
        "t": "2022-07-20T00:00:00",
        "wom": 4,
        "y": 0.32169056052383393,
        "y_hat": 0.4634858654201598
      },
      {
        "dow": 3,
        "t": "2022-07-21T00:00:00",
        "wom": 4,
        "y": 0.8817188595233318,
        "y_hat": 0.5119845418945297
      },
      {
        "dow": 4,
        "t": "2022-07-22T00:00:00",
        "wom": 4,
        "y": 4.648204695213178,
        "y_hat": 3.3730927677623934
      },
      {
        "dow": 5,
        "t": "2022-07-23T00:00:00",
        "wom": 4,
        "y": 0.5392997474632014,
        "y_hat": 0.5636060075801738
      },
      {
        "dow": 6,
        "t": "2022-07-24T00:00:00",
        "wom": 4,
        "y": 0.28113347961902946,
        "y_hat": 0.49830562796957933
      },
      {
        "dow": 0,
        "t": "2022-07-25T00:00:00",
        "wom": 5,
        "y": 0.4047240553693535,
        "y_hat": 0.46204672122457113
      },
      {
        "dow": 1,
        "t": "2022-07-26T00:00:00",
        "wom": 5,
        "y": 0.09656379233844459,
        "y_hat": 0.5112377015882688
      },
      {
        "dow": 2,
        "t": "2022-07-27T00:00:00",
        "wom": 5,
        "y": 0.9545109848618508,
        "y_hat": 0.4543290229926863
      },
      {
        "dow": 3,
        "t": "2022-07-28T00:00:00",
        "wom": 5,
        "y": 0.2844751590965856,
        "y_hat": 0.5415696133844426
      },
      {
        "dow": 4,
        "t": "2022-07-29T00:00:00",
        "wom": 5,
        "y": 2.19828814692932,
        "y_hat": 3.3300730438774453
      },
      {
        "dow": 5,
        "t": "2022-07-30T00:00:00",
        "wom": 5,
        "y": 0.7872041161790215,
        "y_hat": 0.49493644795384756
      },
      {
        "dow": 6,
        "t": "2022-07-31T00:00:00",
        "wom": 5,
        "y": 0.8320061228006757,
        "y_hat": 0.5895080935234995
      },
      {
        "dow": 0,
        "t": "2022-08-01T00:00:00",
        "wom": 1,
        "y": 0.48050923278633095,
        "y_hat": 0.47349574964670255
      },
      {
        "dow": 1,
        "t": "2022-08-02T00:00:00",
        "wom": 1,
        "y": 0.9186312133845865,
        "y_hat": 0.49531623887083803
      },
      {
        "dow": 2,
        "t": "2022-08-03T00:00:00",
        "wom": 1,
        "y": 0.28383987157951585,
        "y_hat": 0.481374258041942
      },
      {
        "dow": 3,
        "t": "2022-08-04T00:00:00",
        "wom": 1,
        "y": 0.6963665748778668,
        "y_hat": 0.4895308452528114
      },
      {
        "dow": 4,
        "t": "2022-08-05T00:00:00",
        "wom": 1,
        "y": 2.09245143951022,
        "y_hat": 3.367381427164757
      },
      {
        "dow": 5,
        "t": "2022-08-06T00:00:00",
        "wom": 1,
        "y": 0.7172188720261602,
        "y_hat": 0.47711898965585053
      },
      {
        "dow": 6,
        "t": "2022-08-07T00:00:00",
        "wom": 1,
        "y": 0.6862602995181284,
        "y_hat": 0.5903954401566516
      },
      {
        "dow": 0,
        "t": "2022-08-08T00:00:00",
        "wom": 2,
        "y": 0.12495616249592223,
        "y_hat": 0.47043743472908495
      },
      {
        "dow": 1,
        "t": "2022-08-09T00:00:00",
        "wom": 2,
        "y": 0.7803068072936611,
        "y_hat": 0.4872167581804853
      },
      {
        "dow": 2,
        "t": "2022-08-10T00:00:00",
        "wom": 2,
        "y": 0.1711236164496699,
        "y_hat": 0.48818630133068247
      },
      {
        "dow": 3,
        "t": "2022-08-11T00:00:00",
        "wom": 2,
        "y": 0.6418292052213659,
        "y_hat": 0.48996501599705045
      },
      {
        "dow": 4,
        "t": "2022-08-12T00:00:00",
        "wom": 2,
        "y": 4.398223140821866,
        "y_hat": 3.3690588274255413
      },
      {
        "dow": 5,
        "t": "2022-08-13T00:00:00",
        "wom": 2,
        "y": 0.7667020108493909,
        "y_hat": 0.5622104389904248
      },
      {
        "dow": 6,
        "t": "2022-08-14T00:00:00",
        "wom": 2,
        "y": 0.41815335304653145,
        "y_hat": 0.5145159326667085
      },
      {
        "dow": 0,
        "t": "2022-08-15T00:00:00",
        "wom": 3,
        "y": 0.6891797613670014,
        "y_hat": 0.4589364233004018
      },
      {
        "dow": 1,
        "t": "2022-08-16T00:00:00",
        "wom": 3,
        "y": 0.5101490274593742,
        "y_hat": 0.5164990424649447
      },
      {
        "dow": 2,
        "t": "2022-08-17T00:00:00",
        "wom": 3,
        "y": 0.7796622447746596,
        "y_hat": 0.4593034805306277
      },
      {
        "dow": 3,
        "t": "2022-08-18T00:00:00",
        "wom": 3,
        "y": 0.5627361856480227,
        "y_hat": 0.520919092250974
      },
      {
        "dow": 4,
        "t": "2022-08-19T00:00:00",
        "wom": 3,
        "y": 2.149269114111338,
        "y_hat": 3.3455965986349505
      },
      {
        "dow": 5,
        "t": "2022-08-20T00:00:00",
        "wom": 3,
        "y": 0.4028888263767394,
        "y_hat": 0.48338721515697775
      },
      {
        "dow": 6,
        "t": "2022-08-21T00:00:00",
        "wom": 3,
        "y": 0.29581445643538484,
        "y_hat": 0.5768359610320534
      },
      {
        "dow": 0,
        "t": "2022-08-22T00:00:00",
        "wom": 4,
        "y": 0.22454795399899974,
        "y_hat": 0.4666043861557675
      },
      {
        "dow": 1,
        "t": "2022-08-23T00:00:00",
        "wom": 4,
        "y": 0.5368502058660074,
        "y_hat": 0.5036672475053295
      },
      {
        "dow": 2,
        "t": "2022-08-24T00:00:00",
        "wom": 4,
        "y": 0.8352731932733497,
        "y_hat": 0.4757520140760282
      },
      {
        "dow": 3,
        "t": "2022-08-25T00:00:00",
        "wom": 4,
        "y": 0.7156495255143432,
        "y_hat": 0.5218926864860483
      },
      {
        "dow": 4,
        "t": "2022-08-26T00:00:00",
        "wom": 4,
        "y": 4.081170707633651,
        "y_hat": 3.3491174707631624
      },
      {
        "dow": 5,
        "t": "2022-08-27T00:00:00",
        "wom": 4,
        "y": 0.9846573243966786,
        "y_hat": 0.547980796620411
      },
      {
        "dow": 6,
        "t": "2022-08-28T00:00:00",
        "wom": 4,
        "y": 0.11556538882109113,
        "y_hat": 0.5327797441872997
      },
      {
        "dow": 0,
        "t": "2022-08-29T00:00:00",
        "wom": 5,
        "y": 0.26133230623941184,
        "y_hat": 0.44038355293130604
      },
      {
        "dow": 1,
        "t": "2022-08-30T00:00:00",
        "wom": 5,
        "y": 0.9006219041976188,
        "y_hat": 0.5109183954983006
      },
      {
        "dow": 2,
        "t": "2022-08-31T00:00:00",
        "wom": 5,
        "y": 0.2515146218161567,
        "y_hat": 0.4875324687977037
      },
      {
        "dow": 3,
        "t": "2022-09-01T00:00:00",
        "wom": 1,
        "y": 0.13098484463014348,
        "y_hat": 0.4884072016059771
      },
      {
        "dow": 4,
        "t": "2022-09-02T00:00:00",
        "wom": 1,
        "y": 2.9173298508007353,
        "y_hat": 3.3474589693056096
      },
      {
        "dow": 5,
        "t": "2022-09-03T00:00:00",
        "wom": 1,
        "y": 0.902033118492389,
        "y_hat": 0.5254044809077754
      },
      {
        "dow": 6,
        "t": "2022-09-04T00:00:00",
        "wom": 1,
        "y": 0.6094812461745855,
        "y_hat": 0.5687842811167654
      },
      {
        "dow": 0,
        "t": "2022-09-05T00:00:00",
        "wom": 2,
        "y": 0.21237629900528165,
        "y_hat": 0.4608861980187175
      },
      {
        "dow": 1,
        "t": "2022-09-06T00:00:00",
        "wom": 2,
        "y": 0.8674613314753005,
        "y_hat": 0.49240027123906716
      },
      {
        "dow": 2,
        "t": "2022-09-07T00:00:00",
        "wom": 2,
        "y": 0.7318690107796962,
        "y_hat": 0.4878391740796804
      },
      {
        "dow": 3,
        "t": "2022-09-08T00:00:00",
        "wom": 2,
        "y": 0.9273400869945371,
        "y_hat": 0.506756175441995
      },
      {
        "dow": 4,
        "t": "2022-09-09T00:00:00",
        "wom": 2,
        "y": 4.491296824471812,
        "y_hat": 3.359970983327612
      },
      {
        "dow": 5,
        "t": "2022-09-10T00:00:00",
        "wom": 2,
        "y": 0.3736267511181386,
        "y_hat": 0.555416799446095
      },
      {
        "dow": 6,
        "t": "2022-09-11T00:00:00",
        "wom": 2,
        "y": 0.6741226397964637,
        "y_hat": 0.4966092312677435
      },
      {
        "dow": 0,
        "t": "2022-09-12T00:00:00",
        "wom": 3,
        "y": 0.5690598209381145,
        "y_hat": 0.4808513507432339
      },
      {
        "dow": 1,
        "t": "2022-09-13T00:00:00",
        "wom": 3,
        "y": 0.5845921058651229,
        "y_hat": 0.5029876878727212
      },
      {
        "dow": 2,
        "t": "2022-09-14T00:00:00",
        "wom": 3,
        "y": 0.06190141469816057,
        "y_hat": 0.46547491900350696
      },
      {
        "dow": 3,
        "t": "2022-09-15T00:00:00",
        "wom": 3,
        "y": 0.7726353296869438,
        "y_hat": 0.49189524129549633
      },
      {
        "dow": 4,
        "t": "2022-09-16T00:00:00",
        "wom": 3,
        "y": 2.3982766116278027,
        "y_hat": 3.376762437663063
      },
      {
        "dow": 5,
        "t": "2022-09-17T00:00:00",
        "wom": 3,
        "y": 0.02485301435952192,
        "y_hat": 0.4847759303608996
      },
      {
        "dow": 6,
        "t": "2022-09-18T00:00:00",
        "wom": 3,
        "y": 0.3885550460991051,
        "y_hat": 0.554230104568961
      },
      {
        "dow": 0,
        "t": "2022-09-19T00:00:00",
        "wom": 4,
        "y": 0.5124897271631259,
        "y_hat": 0.4821097920890602
      },
      {
        "dow": 1,
        "t": "2022-09-20T00:00:00",
        "wom": 4,
        "y": 0.6717284818449774,
        "y_hat": 0.5104035195187014
      },
      {
        "dow": 2,
        "t": "2022-09-21T00:00:00",
        "wom": 4,
        "y": 0.8010250603087118,
        "y_hat": 0.47038867274124535
      },
      {
        "dow": 3,
        "t": "2022-09-22T00:00:00",
        "wom": 4,
        "y": 0.7215828184438076,
        "y_hat": 0.5155588019145542
      },
      {
        "dow": 4,
        "t": "2022-09-23T00:00:00",
        "wom": 4,
        "y": 4.467671973989098,
        "y_hat": 3.3499236183795036
      },
      {
        "dow": 5,
        "t": "2022-09-24T00:00:00",
        "wom": 4,
        "y": 0.7177366636762655,
        "y_hat": 0.5612006852767245
      },
      {
        "dow": 6,
        "t": "2022-09-25T00:00:00",
        "wom": 4,
        "y": 0.48856096479123934,
        "y_hat": 0.50956973248284
      },
      {
        "dow": 0,
        "t": "2022-09-26T00:00:00",
        "wom": 5,
        "y": 0.5960952391022855,
        "y_hat": 0.46228939390587576
      },
      {
        "dow": 1,
        "t": "2022-09-27T00:00:00",
        "wom": 5,
        "y": 0.8681440058433928,
        "y_hat": 0.5099248534364216
      },
      {
        "dow": 2,
        "t": "2022-09-28T00:00:00",
        "wom": 5,
        "y": 0.8669839542345549,
        "y_hat": 0.47454169057096995
      },
      {
        "dow": 3,
        "t": "2022-09-29T00:00:00",
        "wom": 5,
        "y": 0.8884123252558981,
        "y_hat": 0.5112001970869061
      },
      {
        "dow": 4,
        "t": "2022-09-30T00:00:00",
        "wom": 5,
        "y": 2.3463008214785965,
        "y_hat": 3.353599886443772
      },
      {
        "dow": 5,
        "t": "2022-10-01T00:00:00",
        "wom": 1,
        "y": 0.2954617156792265,
        "y_hat": 0.4787231205197918
      },
      {
        "dow": 6,
        "t": "2022-10-02T00:00:00",
        "wom": 1,
        "y": 0.633109100839611,
        "y_hat": 0.5654854050519611
      },
      {
        "dow": 0,
        "t": "2022-10-03T00:00:00",
        "wom": 2,
        "y": 0.3669341564363222,
        "y_hat": 0.48157613891257134
      },
      {
        "dow": 1,
        "t": "2022-10-04T00:00:00",
        "wom": 2,
        "y": 0.24876598531702476,
        "y_hat": 0.4966360135215404
      },
      {
        "dow": 2,
        "t": "2022-10-05T00:00:00",
        "wom": 2,
        "y": 0.16587282213408583,
        "y_hat": 0.4597046533336795
      },
      {
        "dow": 3,
        "t": "2022-10-06T00:00:00",
        "wom": 2,
        "y": 0.9919913685200391,
        "y_hat": 0.5065276005502084
      },
      {
        "dow": 4,
        "t": "2022-10-07T00:00:00",
        "wom": 2,
        "y": 4.353329687398908,
        "y_hat": 3.3807804037182327
      },
      {
        "dow": 5,
        "t": "2022-10-08T00:00:00",
        "wom": 2,
        "y": 0.7213297581947791,
        "y_hat": 0.5476928041701374
      },
      {
        "dow": 6,
        "t": "2022-10-09T00:00:00",
        "wom": 2,
        "y": 0.7224275916390493,
        "y_hat": 0.5132643296627557
      },
      {
        "dow": 0,
        "t": "2022-10-10T00:00:00",
        "wom": 3,
        "y": 0.29155319532221713,
        "y_hat": 0.47034731849978
      },
      {
        "dow": 1,
        "t": "2022-10-11T00:00:00",
        "wom": 3,
        "y": 0.36991001906594234,
        "y_hat": 0.49076630378949176
      },
      {
        "dow": 2,
        "t": "2022-10-12T00:00:00",
        "wom": 3,
        "y": 0.28417739594797153,
        "y_hat": 0.4664809300402862
      },
      {
        "dow": 3,
        "t": "2022-10-13T00:00:00",
        "wom": 3,
        "y": 0.19840529083955494,
        "y_hat": 0.5065932203681017
      },
      {
        "dow": 4,
        "t": "2022-10-14T00:00:00",
        "wom": 3,
        "y": 2.4606866711518958,
        "y_hat": 3.3479592329739587
      },
      {
        "dow": 5,
        "t": "2022-10-15T00:00:00",
        "wom": 3,
        "y": 0.6335089949594493,
        "y_hat": 0.50578114273656
      },
      {
        "dow": 6,
        "t": "2022-10-16T00:00:00",
        "wom": 3,
        "y": 0.5680543096824044,
        "y_hat": 0.5735861546687405
      },
      {
        "dow": 0,
        "t": "2022-10-17T00:00:00",
        "wom": 4,
        "y": 0.016790840072704594,
        "y_hat": 0.4675766576338107
      },
      {
        "dow": 1,
        "t": "2022-10-18T00:00:00",
        "wom": 4,
        "y": 0.15482355482083832,
        "y_hat": 0.4858790402451255
      },
      {
        "dow": 2,
        "t": "2022-10-19T00:00:00",
        "wom": 4,
        "y": 0.32294713218572024,
        "y_hat": 0.4678000518383632
      },
      {
        "dow": 3,
        "t": "2022-10-20T00:00:00",
        "wom": 4,
        "y": 0.09723518582862622,
        "y_hat": 0.5150875504942485
      },
      {
        "dow": 4,
        "t": "2022-10-21T00:00:00",
        "wom": 4,
        "y": 4.16195111263491,
        "y_hat": 3.342856404365225
      },
      {
        "dow": 5,
        "t": "2022-10-22T00:00:00",
        "wom": 4,
        "y": 0.9308281099209696,
        "y_hat": 0.5705760873110616
      },
      {
        "dow": 6,
        "t": "2022-10-23T00:00:00",
        "wom": 4,
        "y": 0.3700513293335844,
        "y_hat": 0.5269966028261652
      },
      {
        "dow": 0,
        "t": "2022-10-24T00:00:00",
        "wom": 5,
        "y": 0.33730301215287395,
        "y_hat": 0.4502780070504614
      },
      {
        "dow": 1,
        "t": "2022-10-25T00:00:00",
        "wom": 5,
        "y": 0.2979813390418188,
        "y_hat": 0.5039893257892024
      },
      {
        "dow": 2,
        "t": "2022-10-26T00:00:00",
        "wom": 5,
        "y": 0.8989252622362512,
        "y_hat": 0.46206103589009695
      },
      {
        "dow": 3,
        "t": "2022-10-27T00:00:00",
        "wom": 5,
        "y": 0.9678609978612648,
        "y_hat": 0.5309666208161398
      },
      {
        "dow": 4,
        "t": "2022-10-28T00:00:00",
        "wom": 5,
        "y": 2.852867990371385,
        "y_hat": 3.354839465532302
      },
      {
        "dow": 5,
        "t": "2022-10-29T00:00:00",
        "wom": 5,
        "y": 0.09421902072623589,
        "y_hat": 0.49381380383111323
      },
      {
        "dow": 6,
        "t": "2022-10-30T00:00:00",
        "wom": 5,
        "y": 0.8581628861015397,
        "y_hat": 0.5406138105335944
      },
      {
        "dow": 0,
        "t": "2022-10-31T00:00:00",
        "wom": 6,
        "y": 0.20114187846747755,
        "y_hat": 0.495922702563148
      },
      {
        "dow": 1,
        "t": "2022-11-01T00:00:00",
        "wom": 1,
        "y": 0.6181376931339703,
        "y_hat": 0.48251223434842866
      },
      {
        "dow": 2,
        "t": "2022-11-02T00:00:00",
        "wom": 1,
        "y": 0.19978354259674502,
        "y_hat": 0.47807908164412843
      },
      {
        "dow": 3,
        "t": "2022-11-03T00:00:00",
        "wom": 1,
        "y": 0.1768860986420725,
        "y_hat": 0.49477435354018034
      },
      {
        "dow": 4,
        "t": "2022-11-04T00:00:00",
        "wom": 1,
        "y": 2.4752377813951942,
        "y_hat": 3.3495985518947013
      },
      {
        "dow": 5,
        "t": "2022-11-05T00:00:00",
        "wom": 1,
        "y": 0.9055115735426204,
        "y_hat": 0.5066109945223868
      },
      {
        "dow": 6,
        "t": "2022-11-06T00:00:00",
        "wom": 1,
        "y": 0.6649858316616734,
        "y_hat": 0.5825149666666662
      },
      {
        "dow": 0,
        "t": "2022-11-07T00:00:00",
        "wom": 2,
        "y": 0.5304475702709496,
        "y_hat": 0.4615167591584294
      },
      {
        "dow": 1,
        "t": "2022-11-08T00:00:00",
        "wom": 2,
        "y": 0.1761904542362559,
        "y_hat": 0.5007783066639323
      },
      {
        "dow": 2,
        "t": "2022-11-09T00:00:00",
        "wom": 2,
        "y": 0.9624710144321974,
        "y_hat": 0.4508816700538215
      },
      {
        "dow": 3,
        "t": "2022-11-10T00:00:00",
        "wom": 2,
        "y": 0.7059542876896551,
        "y_hat": 0.537080103349834
      },
      {
        "dow": 4,
        "t": "2022-11-11T00:00:00",
        "wom": 2,
        "y": 4.506728336965063,
        "y_hat": 3.34294986642124
      },
      {
        "dow": 5,
        "t": "2022-11-12T00:00:00",
        "wom": 2,
        "y": 0.8433091095294604,
        "y_hat": 0.5621587889861992
      },
      {
        "dow": 6,
        "t": "2022-11-13T00:00:00",
        "wom": 2,
        "y": 0.7560702301487369,
        "y_hat": 0.5118183202859423
      },
      {
        "dow": 0,
        "t": "2022-11-14T00:00:00",
        "wom": 3,
        "y": 0.06592714264449118,
        "y_hat": 0.4667626921284692
      },
      {
        "dow": 1,
        "t": "2022-11-15T00:00:00",
        "wom": 3,
        "y": 0.2504072631614749,
        "y_hat": 0.48077439560663243
      },
      {
        "dow": 2,
        "t": "2022-11-16T00:00:00",
        "wom": 3,
        "y": 0.5805948341045363,
        "y_hat": 0.46904505617938125
      },
      {
        "dow": 3,
        "t": "2022-11-17T00:00:00",
        "wom": 3,
        "y": 0.5803663521377741,
        "y_hat": 0.5206325161833976
      },
      {
        "dow": 4,
        "t": "2022-11-18T00:00:00",
        "wom": 3,
        "y": 2.044237322520029,
        "y_hat": 3.3511076646104807
      },
      {
        "dow": 5,
        "t": "2022-11-19T00:00:00",
        "wom": 3,
        "y": 0.5560838530176727,
        "y_hat": 0.47717546939253397
      },
      {
        "dow": 6,
        "t": "2022-11-20T00:00:00",
        "wom": 3,
        "y": 0.28756910769465927,
        "y_hat": 0.5840884325135942
      },
      {
        "dow": 0,
        "t": "2022-11-21T00:00:00",
        "wom": 4,
        "y": 0.32658389395269083,
        "y_hat": 0.4593350862783285
      },
      {
        "dow": 1,
        "t": "2022-11-22T00:00:00",
        "wom": 4,
        "y": 0.9255075128439307,
        "y_hat": 0.5058149936111085
      },
      {
        "dow": 2,
        "t": "2022-11-23T00:00:00",
        "wom": 4,
        "y": 0.2351401173462816,
        "y_hat": 0.4845584992817537
      },
      {
        "dow": 3,
        "t": "2022-11-24T00:00:00",
        "wom": 4,
        "y": 0.6755009569121625,
        "y_hat": 0.4852981565435497
      },
      {
        "dow": 4,
        "t": "2022-11-25T00:00:00",
        "wom": 4,
        "y": 4.65326276722114,
        "y_hat": 3.3660241632842935
      },
      {
        "dow": 5,
        "t": "2022-11-26T00:00:00",
        "wom": 4,
        "y": 0.13334860717528818,
        "y_hat": 0.5682027635512094
      },
      {
        "dow": 6,
        "t": "2022-11-27T00:00:00",
        "wom": 4,
        "y": 0.21224946729617233,
        "y_hat": 0.48093260735979076
      },
      {
        "dow": 0,
        "t": "2022-11-28T00:00:00",
        "wom": 5,
        "y": 0.10485005588135898,
        "y_hat": 0.4706848886505186
      },
      {
        "dow": 1,
        "t": "2022-11-29T00:00:00",
        "wom": 5,
        "y": 0.571653845644994,
        "y_hat": 0.5001876242873315
      },
      {
        "dow": 2,
        "t": "2022-11-30T00:00:00",
        "wom": 5,
        "y": 0.6455551020000664,
        "y_hat": 0.47907550538503835
      },
      {
        "dow": 3,
        "t": "2022-12-01T00:00:00",
        "wom": 1,
        "y": 0.07205418156187893,
        "y_hat": 0.511900469241254
      },
      {
        "dow": 4,
        "t": "2022-12-02T00:00:00",
        "wom": 1,
        "y": 2.8390812934919314,
        "y_hat": 3.3302582399803033
      },
      {
        "dow": 5,
        "t": "2022-12-03T00:00:00",
        "wom": 1,
        "y": 0.3495218764848188,
        "y_hat": 0.5227357418696772
      },
      {
        "dow": 6,
        "t": "2022-12-04T00:00:00",
        "wom": 1,
        "y": 0.9605832472674337,
        "y_hat": 0.5496114322547002
      },
      {
        "dow": 0,
        "t": "2022-12-05T00:00:00",
        "wom": 2,
        "y": 0.47976044149700026,
        "y_hat": 0.49034283520720406
      },
      {
        "dow": 1,
        "t": "2022-12-06T00:00:00",
        "wom": 2,
        "y": 0.6055161236619802,
        "y_hat": 0.48844581825571254
      },
      {
        "dow": 2,
        "t": "2022-12-07T00:00:00",
        "wom": 2,
        "y": 0.41871964951333474,
        "y_hat": 0.46755428383313286
      },
      {
        "dow": 3,
        "t": "2022-12-08T00:00:00",
        "wom": 2,
        "y": 0.966612891389878,
        "y_hat": 0.502417446427469
      },
      {
        "dow": 4,
        "t": "2022-12-09T00:00:00",
        "wom": 2,
        "y": 4.565446798382279,
        "y_hat": 3.3701007354088697
      },
      {
        "dow": 5,
        "t": "2022-12-10T00:00:00",
        "wom": 2,
        "y": 0.34560834727361145,
        "y_hat": 0.5549581849247693
      },
      {
        "dow": 6,
        "t": "2022-12-11T00:00:00",
        "wom": 2,
        "y": 0.7570085832854714,
        "y_hat": 0.49128256852675606
      },
      {
        "dow": 0,
        "t": "2022-12-12T00:00:00",
        "wom": 3,
        "y": 0.9652744314834171,
        "y_hat": 0.48297130788320025
      },
      {
        "dow": 1,
        "t": "2022-12-13T00:00:00",
        "wom": 3,
        "y": 0.5648309593161226,
        "y_hat": 0.512711413406946
      },
      {
        "dow": 2,
        "t": "2022-12-14T00:00:00",
        "wom": 3,
        "y": 0.7595290379916108,
        "y_hat": 0.44961781289866215
      },
      {
        "dow": 3,
        "t": "2022-12-15T00:00:00",
        "wom": 3,
        "y": 0.4539146559253099,
        "y_hat": 0.515972209984941
      },
      {
        "dow": 4,
        "t": "2022-12-16T00:00:00",
        "wom": 3,
        "y": 2.6626571950380864,
        "y_hat": 3.3399575236355257
      },
      {
        "dow": 5,
        "t": "2022-12-17T00:00:00",
        "wom": 3,
        "y": 0.901352733645678,
        "y_hat": 0.5032349581142425
      },
      {
        "dow": 6,
        "t": "2022-12-18T00:00:00",
        "wom": 3,
        "y": 0.1389757185937477,
        "y_hat": 0.5752227348968366
      },
      {
        "dow": 0,
        "t": "2022-12-19T00:00:00",
        "wom": 4,
        "y": 0.523376913529589,
        "y_hat": 0.4417912632304645
      },
      {
        "dow": 1,
        "t": "2022-12-20T00:00:00",
        "wom": 4,
        "y": 0.22793475704817479,
        "y_hat": 0.5173691105963008
      },
      {
        "dow": 2,
        "t": "2022-12-21T00:00:00",
        "wom": 4,
        "y": 0.3918232223846172,
        "y_hat": 0.45215093854630534
      },
      {
        "dow": 3,
        "t": "2022-12-22T00:00:00",
        "wom": 4,
        "y": 0.47576866552495234,
        "y_hat": 0.5138604155283891
      },
      {
        "dow": 4,
        "t": "2022-12-23T00:00:00",
        "wom": 4,
        "y": 4.313173128751874,
        "y_hat": 3.3529717000219144
      },
      {
        "dow": 5,
        "t": "2022-12-24T00:00:00",
        "wom": 4,
        "y": 0.11547575723384762,
        "y_hat": 0.5620578121412411
      },
      {
        "dow": 6,
        "t": "2022-12-25T00:00:00",
        "wom": 4,
        "y": 0.880622203620008,
        "y_hat": 0.4911613225954918
      },
      {
        "dow": 0,
        "t": "2022-12-26T00:00:00",
        "wom": 5,
        "y": 0.4962845210256144,
        "y_hat": 0.49490027253140084
      },
      {
        "dow": 1,
        "t": "2022-12-27T00:00:00",
        "wom": 5,
        "y": 0.5406198774645495,
        "y_hat": 0.4913120793045443
      },
      {
        "dow": 2,
        "t": "2022-12-28T00:00:00",
        "wom": 5,
        "y": 0.26315431946537116,
        "y_hat": 0.4642314292844817
      },
      {
        "dow": 3,
        "t": "2022-12-29T00:00:00",
        "wom": 5,
        "y": 0.0927341711398395,
        "y_hat": 0.4985528496072842
      },
      {
        "dow": 4,
        "t": "2022-12-30T00:00:00",
        "wom": 5,
        "y": 2.42594516701449,
        "y_hat": 3.34330404659483
      },
      {
        "dow": 5,
        "t": "2022-12-31T00:00:00",
        "wom": 5,
        "y": 0.9191390581073097,
        "y_hat": 0.5065376224438971
      }
    ]
  },
  "layer": [
    {
      "encoding": {
        "tooltip": {
          "field": "t",
          "type": "temporal"
        },
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "darkblue",
        "opacity": 0.6,
        "type": "circle"
      },
      "selection": {
        "selector008": {
          "bind": "scales",
          "type": "interval"
        }
      },
      "title": "",
      "width": 400
    },
    {
      "encoding": {
        "x": {
          "axis": {
            "grid": false,
            "ticks": true
          },
          "field": "t",
          "title": "",
          "type": "temporal"
        },
        "y": {
          "axis": {
            "domain": false,
            "grid": true,
            "ticks": true
          },
          "field": "y_hat",
          "scale": {
            "zero": false
          },
          "title": "",
          "type": "quantitative"
        }
      },
      "height": 200,
      "mark": {
        "color": "t",
        "type": "line"
      },
      "title": "2 lags, period: 7, MAE: 0.353",
      "width": 400
    }
  ]
}

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