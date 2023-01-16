---
title: "Slice well"
date: 2022-11-15T11:21:31-05:00
draft: false
author: "Aaron Slowey"
tags: ["pandas", "exception handling"]
categories: ["technical"]
---

In this post, I briefly review a few methods to select rows and/or columns of a
DataFrame that satisfy one or more criteria. I then introduce two additional
requirements that arises frequently in practice--slicing with
previously unknown criteria and managing serialization and deserialization 
to recover the desired data structure.

## Lever multiIndexes

I often find pandas' multiIndex to be helpful, although I do not observe it 
used very often. With a multi-indexed DataFrame, pandas' `.xs` method is a 
clean way to select instances, as it can be executed
in a piecewise sequence of criteria. A downside is that you lose the 
multiIndex level(s) from your table.

When you have a multi-indexed dataframe, we can use `.loc` at varying depth. In
general, `.loc` expects a row indexer followed by a column indexer. To select
multiple _labels_ (values) of a _level_, list them; same for the columns you
want to tabulate:

```python
df.loc[['NA', 'EMEA', 'LA'], ['post_dt', 'swift_msg']]
```

Note that the row indexer in this example is _one_ list, as is that of the 
column indexer; this will look into level 0 of the multiIndex. To further 
utilize the multiIndex, provide multiple lists of labels, bounded
as __one__ indexer by a _tuple_:

```python
df.loc[(['NA', 'EMEA', 'LA'], ['Mining', 'Retail']), ['post_dt', 'swift_msg']]
```

Pushing further, we arrive at the pinacle of dataframe slicing: the IndexSlice:

```python
df.loc[pd.IndexSlice[:, ['Mining', 'Retail']], ['post_dt', 'swift_msg']]
```

`:` is interpreted as "include every label in this level," which in this example
implies "all geographies."

Note that all of the above are more organized and readable version of 
applying a boolean mask like `df[(df['A']==0) & (df['B']==1)]`.

## When slicing goes awry

Often, we try to progammatically select a (sub)category of something from a data
set containing multiple groups. With commercial payments, the entire data set
may contain multiple bank clients _and_ payment channels (paper check, ACH,
wires, etc.), but not every client may have made payments through all channels.
We do not know this in advance and yet ask the program to slice thousands of 
samples.

Pandas' `.xs()` method will slice a dataframe on the basis of one level 
of an index. If the value we specify is absent, a `KeyError` is produced:

```python
import pandas as pd

df = pd.DataFrame({'c': ['x', 'x', 'y'], 'b': [1, 2, 3]}).set_index('c')
df.xs('z')

KeyError: 'z'
```

How should we handle this error so that the program will proceed to slice
the next requested subject?

The following block provides `KeyError` (most specific)
versus `Exception` (next level up); `print(error)` will produce `'z'`.

```python
df = pd.DataFrame({'c': ['x', 'x', 'y'], 'b': [1, 2, 3]}).set_index('c')

try:
    df.xs('z')
except KeyError as error:
    print(error)
    print(f'custom message')
```

So with the code, we know to associate the output `'z'` as a key of some kind.
Let's try to use exceptions' internal attributes:

```python
try:
    df.xs('z')
except KeyError as error:
    print(error)
    print(error.__traceback__)

>> > 'z'
< traceback object at 0x7fea784e9308 >
```

`'z'` is the `__context__` and `__cause__`.

Let's say these operations are within a function that we call from an
external module. Where should we handle this potential error, _in situ_ or
in `main.py`? Intuitively in-line with the slicing operation:

```python
def slice_it(data, rail):
    try:
        slice = data.xs(rail)
        print(f'Sample successful')
    except KeyError as error:
        print(f'KeyError {error}; i.e., not found in data')
        slice = pd.DataFrame()
    return slice


df = pd.DataFrame({'rails': ['x', 'x', 'y', 'q'],
                   'b': [1, 2, 3, 9]}).set_index('rails')
channels = ['x', 'y', 'abc', 'q']

for channel in channels:
    slice = slice_it(df, channel)
    if slice.empty:
        continue
```

The 3rd channel `z` is not in the data; the error will be printed and the code 
will
attempt to sample the next rail, per the `continue` command. Any follow-on
operations will __only__ occur upon successful completion of the slice. Note
that you cannot return `None` and apply a pandas method to check if the slice
worked. Instead, we chose to return an _empty_ DataFrame and check it with
the `.empty` attribute.

If empty, we could log that the data were unavailable for that particular
subject. Error and exception handling intertwines with how you design the
scope of functions.  The ease of deciding where to handle exceptions 
(in the processing function itself or a driver function) indicates how 
well you designed your function's scope.