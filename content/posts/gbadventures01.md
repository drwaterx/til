---
title: "That which is aggregated and its metadata"
date: 2022-08-12T15:14:44-05:00
draft: false
author: "Aaron Slowey"
tags: ["pandas", "numpy"]
categories: ["technical"]
---

# It's impossible to include an _associated_ field value alongside an aggregate of another variable

Unlike ndarrays, DataFrames are often heterogeneous.  They are a more 
complete map of how we think of a data __set__ as a whole.  When we alter 
the structure of tabular data, often through aggregation of one field, we 
want to include values from other fields.  This is an example of an issue 
that arises at the interface of pandas and 
scikit-learn, for which the `ColumnTransformer` was created.

In the following example of car makes and fictitious carbon footprints,
we want to tabulate the make of the most expensive car in the same row as its 
footprint. The following code misaligns the rows, in that `TSLA`, not `GM`, is 
associated with the maximum `cfp`.

```python
df = pd.DataFrame({'Sector': ['auto', 'auto', 'auto'],
                   'make': ['GM', 'TSLA', 'FORD'],
                   'cfp': [20.5, 900.10, 450.50]})
                   
>>> df.groupby('Sector').agg({'make': 'first', 'cfp': 'max'})
```

| Sector   | make   |   cfp |
|:---------|:-------|------:|
| auto     | GM     | 900.1 |

The grouping variable will replace any index, so we cannot simply set or add 
`'make'` to the index and then `groupby`.

We can obtain the correct result in a limited number of cases. If the 
dataframe is sorted descending, `.agg('first')` will align with `'max'`; if 
sorted ascending, `'first'` aligns with `'min'`.  

A quantile of one field will not align to the `first` or `last` instance of 
another, but it could be located as long as `np.quantile` outputs an 
observed value and not an interpolation.  
Alternatively, merge tables on the quantile, with NumPy outputting the 
observation nearest the quantile by setting `method='nearest'`.

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