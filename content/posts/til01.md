---
title: "Handle non-sensical operations to avoid downstream errors"
date: 2023-01-05T05:15:02-05:00
draft: false
---

When attempting to log-transform an array of values with NumPy, keep in mind

- Given negative numbers and zeroes, NumPy will output `NaN` and `-inf`, respectively, along with a `RuntimeWarning`.  Such values can cause downstream processing to fail or behave unexpectedly.
- `numpy.log` provides an argument to handle this situation
- How that argument affects `numpy.log`'s behavior depends on whether the output goes to a preexisting container or if that container is created on the fly.

Consider this example:
```python
observations = (np.log10(subject_data, where=column_name > 0)
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
```

The `where=column_name > 0` argument will cause the logarithmic transformation to ignore rows where that column value is not greater than zero and instead place the original value.  Any condition that evaluates to `true` or `false` can be used.

If `observations` did not exist prior, meaning it was _uninitialized_, locations of `subject_data` where the condition is `False` will result in `observations` remaining uninitialized in the corresponding positions.  If you try this out, you will see `NaN` or maybe something like `6.952161e-310`.

It may be better to _initialize_ the output container, with zeroes, `NaN`, or whatever facilitates downstream use.

Let's say we have 6 values to transform, of which one is a zero and another is negative
```python
summary = pd.DataFrame({'cat': ['a', 'a', 'b', 'c', 'c', 'c'], 'z': [33, 22, 44, 0, 11, -8]})

output = np.zeros(6)

# Note that we just call np.log, without a literal assignment to output
np.log(summary.z, out=output, where=summary.z > 0)

>>> output
>>> array([3.49650756, 3.09104245, 3.78418963, 0.        , 2.39789527,
       0.        ])
```