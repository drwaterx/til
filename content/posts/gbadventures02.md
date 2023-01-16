---
title: "Aggregation: Implications of indexing"
date: 2022-07-22T14:34:27-05:00
draft: false
author: "Aaron Slowey"
tags: ["pandas", "data management"]
categories: ["technical"]
---

While there are multiple syntaxes and methods to produce the same aggregated
data, those variations produce different indices. The format and contents of the
index can impact other processes, such as serialization and deserialization.

Consider the following artificial transactional data.

```python
txns = pd.concat([pd.DataFrame({'dt': pd.date_range("2022", freq="D", periods=10),
                                'amount': np.random.random(10),
                                'segment': ['ex'] * 10})] * 10, axis=0)
```

|                                       | dt                  |   amount | segment   |
|:--------------------------------------|:--------------------|---------:|:----------|
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |

Our primary objective is to produce a table of the highest amounts for one or
more days. In this exercise, we try different approaches and pick one that
yields the desired index. With any approach, `.groupby` places the grouping
variable into the index of any aggregated DataFrame.

Apply a `lambda` function directly to the groupby object (gbo), in which `x`
   is the row of the _aggregated_ `DataFrame` per the gbo directive. In this
   approach, we specify the column(s) with which to rank the instances. If
   present, other columns will be returned...

```python
(txns.groupby('dt').apply(lambda x: x.nlargest(3, 'amount'))
 )[0:9]
```

|                                       | dt                  |   amount | segment   |
|:--------------------------------------|:--------------------|---------:|:----------|
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |

..._Unless_ we follow that operation with a selection:

```python
(txns.groupby('dt').apply(lambda x: x.nlargest(3, 'amount'))
 .loc[:, 'amount']
 )[0:9]
```

|                                       |   amount |
|:--------------------------------------|---------:|
| (Timestamp('2022-01-01 00:00:00'), 0) | 0.992821 |
| (Timestamp('2022-01-01 00:00:00'), 0) | 0.992821 |
| (Timestamp('2022-01-01 00:00:00'), 0) | 0.992821 |
| (Timestamp('2022-01-02 00:00:00'), 1) | 0.954333 |
| (Timestamp('2022-01-02 00:00:00'), 1) | 0.954333 |
| (Timestamp('2022-01-02 00:00:00'), 1) | 0.954333 |
| (Timestamp('2022-01-03 00:00:00'), 2) | 0.489229 |
| (Timestamp('2022-01-03 00:00:00'), 2) | 0.489229 |
| (Timestamp('2022-01-03 00:00:00'), 2) | 0.489229 |

Either way, `nlargest` provides `dt` as the `DataFrame` index. If we
want to serialize as JSON in a bundle of other objects (covered elsewhere), we
need to convert the data to a dictionary. Note that `.to_dict` ignores the
index, so `.reset_index()`.

Notice how, in both of the previous two examples, the grouping variable _
combines_ with the original index to form a `MultiIndex`:

```python
txnsg = (txns.groupby('dt').apply(lambda x: x.nlargest(3, 'amount'))
        )[0:9]
print(txnsg.index)
```

```
MultiIndex([('2022-01-01', 0),
            ('2022-01-01', 0),
            ('2022-01-01', 0),
            ('2022-01-02', 1),
            ('2022-01-02', 1),
            ('2022-01-02', 1),
            ('2022-01-03', 2),
            ('2022-01-03', 2),
            ('2022-01-03', 2)],
           names=['dt', None])
```

Before determining the downstream implications of this `MultiIndex`, note that
we can produce the same output by reversing the `lambda` application with the
column selection:

```python
txns.groupby('dt')['amount'].apply(lambda x: x.nlargest(3))
```

```table
dt           
2022-01-01  0    0.992821
            0    0.992821
            0    0.992821
2022-01-02  1    0.954333
            1    0.954333
            1    0.954333
2022-01-03  2    0.489229
            2    0.489229
            2    0.489229
2022-01-04  3    0.234701
            3    0.234701
            3    0.234701
2022-01-05  4    0.695798
            4    0.695798
            4    0.695798
```

If returning a multiIndexed `DataFrame` by a `groupby` seems unexpected, you're
not alone. It only happens when `.groupby` is used with `.apply`, via
the `group_keys=True` default argument. If `group_keys=False`, only the original
indices are retained:

```python
(txns.groupby('dt', group_keys=False)
     .apply(lambda x: x.nlargest(3, 'amount'))
     .loc[:, 'amount']
)[0:9]
```

```table
0    0.992821
0    0.992821
0    0.992821
1    0.954333
1    0.954333
1    0.954333
2    0.489229
2    0.489229
2    0.489229
```

Another variation is whether to retain the grouping variable
values `as_index=True` (default) or replace them with integer
indices `as_index=False`:

```python
(txns.groupby('dt', as_index=False, group_keys=True).apply(
    lambda x: x.nlargest(3, 'amount'))
 .loc[:, 'amount']
 )[0:9]
```

```table
0  0    0.992821
   0    0.992821
   0    0.992821
1  1    0.954333
   1    0.954333
   1    0.954333
2  2    0.489229
   2    0.489229
   2    0.489229
```

When we write a multi-indexed DataFrame or Series with `.to_csv`, it will put
commas after level 0 and the _nameless_ integer index (level 1). Remember, such
multiIndexing only happens when `.groupby` is used with `.apply`, via
the `group_keys=True` default, such that they will form two separate columns
when opened in Excel or loaded by `.read_csv`. The issue is that level 1 remains
nameless--a blank column header--in Excel. But notice that, upon reading the
file, pandas provides the name `Unnamed: 1` (note also `dt.1`):

```python
txnsg.to_csv('multi_indexed_csv_test.csv')
reload = pd.read_csv('multi_indexed_csv_test.csv')
```

|    | dt         |   Unnamed: 1 | dt.1       |   amount | segment   |
|---:|:-----------|-------------:|:-----------|---------:|:----------|
|  0 | 2022-01-01 |            0 | 2022-01-01 | 0.992821 | ex        |
|  1 | 2022-01-01 |            0 | 2022-01-01 | 0.992821 | ex        |
|  2 | 2022-01-01 |            0 | 2022-01-01 | 0.992821 | ex        |
|  3 | 2022-01-02 |            1 | 2022-01-02 | 0.954333 | ex        |
|  4 | 2022-01-02 |            1 | 2022-01-02 | 0.954333 | ex        |
|  5 | 2022-01-02 |            1 | 2022-01-02 | 0.954333 | ex        |
|  6 | 2022-01-03 |            2 | 2022-01-03 | 0.489229 | ex        |
|  7 | 2022-01-03 |            2 | 2022-01-03 | 0.489229 | ex        |
|  8 | 2022-01-03 |            2 | 2022-01-03 | 0.489229 | ex        |

To ensure index levels are consistently labeled, give any unnamed levels a name
with the `.rename_axis` method:

```python
(txns.rename_axis('txn_id')
 .groupby('dt')
 .apply(lambda x: x.nlargest(3, 'amount'))
 )[0:9]
```

|                                       | dt                  |   amount | segment   |
|:--------------------------------------|:--------------------|---------:|:----------|
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-01 00:00:00'), 0) | 2022-01-01 00:00:00 | 0.992821 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-02 00:00:00'), 1) | 2022-01-02 00:00:00 | 0.954333 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |
| (Timestamp('2022-01-03 00:00:00'), 2) | 2022-01-03 00:00:00 | 0.489229 | ex        |

The exported csv of the dataframe above will have all columns labeled. But what
if, before the groupby operation, we slice the data with values contained in a
non-index field (other than `txn_id`)? We
must `.set_index('slicer_col')` before slicing the dataframe with those values,
which typically would be in a `pd.Index` instance (`idx`, say) and applied
via `.loc[idx]`.

Whether we rename a pre-existing index or not, if we set the index with another
column, that previous index will be lost, unless it is reset into columns.