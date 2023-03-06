---
title: "Histograms? SQL has you covered"
date: 2023-03-05
draft: false
author: "Aaron Slowey"
math: true
tags: ["SQL", "warehouse"]
categories: ["technical"]
---

## TL;DR
Binning data with SQL to plot histograms may seem an odd choice to using a DataFrame method such as `df.hist()` in Pandas, but it can be done, and delegating the task to your data warehouse can save space and time when data sets are large and continually updating.  I extend other tutorials with a bin scalar that adapts to multiple groups within a table.

## SQL frequency distributions
Some, perhaps a majority, of data scientists are more comfortable programming in Python, and particularly pandas, than SQL.  That is certainly the case for me.  However, over the past several months, I have been 'forced' to tap into the power of SQL for analytical tasks such as selecting subjects having a groupwise statistical summary of an attribute exceeding a threshold.  Other fields are then joined to the IDs of the qualifying subjects provided by such a common table expression (CTE).

Another more incisive analytical task is to create the data required to visualize a histogram (frequency distribution).  The intent is to divide the range of a variable of interest into equally spaced intervals or bins, and then count how many instances of that variable fall into each bin.  Let's briefly cover how we might do this with Python.

Generally, it is good practice to store such data, rather than temporarily caching it en route to a chart.  For example, `numpy.histogram()` can return the bin boundaries and associated counts, which you would pass to a plotting function.  This is preferable to calling, for example, pandas' `df.col1.hist()`, which will return a static chart but no data.  A nice alternative would be to use Altair, which would store the frequency data in a JSON object while at the same time visualizing the histogram.  In sum, there's at least a couple good options to derive and persist frequency distributions within the Python ecosystem.  But the preliminary step of loading the underlying instances could be sufficiently cumbersome to warrant using a data warehouse to distill those instances down to just the data needed to plot the histograms.

What do I mean by cumbersome?  I cannot claim you will ever find loading your data to be so, given the massively parallel processing (MPP) databases and other computing resources you use.  It's fair to say, though, that many people work with more modest infrastructure, such that the following approach could be helpful.

Although this post is written to be self-contained, if you want to run the code I provide below, you'll need to obtain a data set that has the following characteristics:

- Has at least one continuously numeric variable
- Has one or more categorical variables
- Has more than one subject; i.e., unique combinations of the categorical variables

A simple example would be a table containing the prices of several firms' daily stock returns over an extended period of time.  Our objective is to obtain the data needed to plot the frequency distribution of each firm's daily returns.  Another example could be the sediment concentration of phosphorous in streams draining various agricultural locales.

Here's what each of the following CTEs is doing:

`quantiles` computes the quantile (percentile) of each stock's price over a specified period.  You can replace the double curly brackets and enclosed text with your chosen values.  I use the current syntax for jinja to insert values into the SQL query from a Python program, as this facilitates logging and other tasks that I use Python for.  Note that `percentile_cont` can be implemented in at least two ways; here, we rely on the `group by` clause to ensure the percentile is computed for each stock's prices, rather than all of the prices in the table.  The other method would have been to include `over (partition by ticker)` with the `percentile_cont` clause.  The pros and cons of each approach are beyond the scope of this post.

The next CTE `prices`, particularly the line `floor(t.price / nullif(round(q.price_quantile), 0)) * round(q.price_quantile)` is the heart of this query.  `FLOOR(n)` returns the largest integer equal to or less than `n`. By multiplying the `floor()` of the quotient by the divisor, we obtain the lower bound of the bin.  For example, the value 632 will be labeled with a `bin_floor` of `600` if the quantile was 100.  To obtain the upper bound, we add the same value, which would yield `700`.  

While you could hard-code the divisor, we employ the quantile to reasonably scale the width of the bins for each subject, which we do not know in advance.  I suggest using the $10^{th}$ percentile. Values less than 1 such as 0.6 will get floored to zero, in which case that instance will bin to the interval `0-Divisor`.

`nullif` prevents a divide-by-zero error; it is one of Oracle Database's shorthands for `case when`.  It produces `null` when two arguments are equal; in this context, when `price_quantile=0`.

In CTE `bins`, the `count(price)` and group by `bin_floor` counts all instances labeled with each whole number obtained by `floor(t.price / nullif(round(q.price_quantile), 0)) * round(q.price_quantile)` in CTE `prices`.  Depending on your database software, you may be able to combine this with the previous CTE.

CTE `combo` is largely for testing purposes to check that the bin floor is consistent with the price quantile, etc.  In the main query, the second clause creates a new field that can be used as an axis label.

```sql
with quantiles
as (
  SELECT
    ticker,
    date_,
    PERCENTILE_CONT({{ qtile }}) WITHIN GROUP (ORDER BY price) as price_quantile
  from stock_data
  WHERE
    AND date_ BETWEEN to_date({{ date_start }}, 'YYYY-MM-DD')
      AND to_date({{ date_end }}, 'YYYY-MM-DD')
  GROUP BY
    ticker
)

, prices as (
    SELECT
    t.ticker,
    t.date_,
    t.price,
    q.price_quantile,
    floor(t.price / nullif(round(q.price_quantile), 0)) * round(q.price_quantile) as bin_floor
  FROM stock_data t JOIN quantiles q
  ON t.ticker = q.ticker AND t.date_ = q.date_
)

, bins
as (
  select
    ticker,
    date_,
    bin_floor,
    count(price) as price_count
  from prices
  group by ticker, date_, bin_floor
)

, combo
as (
select
  b.ticker,
  b.date_,
  q.price_quantile,
  b.bin_floor,
  b.price_count
from quantiles q join bins b
  on q.ticker = b.ticker
  and q.date_ = b.date_
)

---- Main query: Tabulate the bin counts and create a column of bin labels
select 
  ticker,
  date_,
  bin_floor || ' - ' || (bin_floor + round(price_quantile)) as bin_range,
  price_count
from combo
order by ticker, date_, bin_floor
```
