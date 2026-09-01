# Effects of dataset shape on DP analyses

Differential privacy works best on datasets that are "tall and skinny":
They have a lot of rows, but only a small number of columns under analysis.
Having more rows means that the contribution of any particular row is smaller,
and having fewer columns reveals less information for every row.

But how tall is tall enough? And how many columns is too many columns?
The right balance between privacy and utility will be vary depening on the sitation,
but we can provide some comparisons.

```{toctree}
:maxdepth: 1
:glob:
*
```