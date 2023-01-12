---
title: "Adventures in aggregation: Part 1"
date: 2022-08-12T15:14:44-05:00
draft: false
author: "Aaron Slowey"
---

# It's impossible to include an _associated_ field value alongside an aggregate of another variable

The mind sometimes has a way of creating a sensible objective that is non-sensical to a machine.  We want to tabulate the make of the most expensive car in teh same row as its price.  When you use the `.agg` method with a dictionary, you can end up with misaligned columns

```python
df = pd.DataFrame({'Sector': ['auto', 'auto', 'auto'],
                   'c': ['ACME', 'GM', 'FORD'],
                   'am': [20.5, 900.10, 450.50]})
                   
>>> df.groupby('Sector').agg({'c': 'first', 'am': 'max'})
```

![[Pasted image 20220928143130.png]]

Note that the first company name was selected, as was the maximum amount; but ACME is not associated with $900.10!

As discovered in the more complicated example below, the grouping variable will replace any index, so we cannot simply set or add `'c'` to the index and then groupby.

It's possible to have the first instance of one field alongside a summary of a variable, but they may be associated with different records.  If the dataframe is sorted descending, `.agg('first')` will align with `'max'`; if sorted ascending, `'first'` aligns with `'min'`.  But a quantile will not align to `first` or `last`; `nth` could be determined by locating the quantile value, as long as `np.quantile` outputs an observed value and not an interpolation, which it may do by default.  Alternatively, merge tables on the quantile, again with NumPy instructed to output the observation nearest 'true' quantile.  To do that, use `interpolation='nearest'`.

```python
def sector_quantilians(data: pd.DataFrame,
                       quantile: float,
                       group_field: str = 'Sector',
                       metric: str = 'AMOUNT'
                      ):
    print(f"{quantile:%}-ile of {metric} for {group_field}")
    return data.groupby(group_field, observed=True)[metric].apply(
        lambda x: np.quantile(x, quantile, interpolation='nearest'))
```