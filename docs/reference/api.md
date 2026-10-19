# API reference

This page shows the signature and docstring of each public name in `starfold`, grouped by the job it does.

```{eval-rst}
.. index::
   single: API
```
```{eval-rst}
.. index::
   single: recover
```
```{eval-rst}
.. index::
   single: null_test
```
```{eval-rst}
.. index::
   single: save
```
```{eval-rst}
.. index::
   single: load
```
```{eval-rst}
.. index::
   single: compare_backends
```

:::{note}
`starfold.search` runs the search alone. It needs `config=`, `random_state=` and a resolved `engine` (`"cpu"` or `"cuml"`, not `"auto"`).
:::

## Run a recovery

`recover` runs the whole procedure and returns one `Result`. `Config` declares its budgets and `Space` the ranges of the settings it searches. Every `Config` argument is a keyword with a default; {doc}`config` lists each one.

```{eval-rst}
.. autofunction:: starfold.recover
```

```{eval-rst}
.. autoclass:: starfold.Config()
```

```{eval-rst}
.. autoclass:: starfold.Space
```

## Run one stage

`search` runs the search alone. `null_test` runs the calibration test for a finished search. `generate_null` draws one structureless copy of a table.

```{eval-rst}
.. autofunction:: starfold.search.search
```

```{eval-rst}
.. autofunction:: starfold.null_test
```

```{eval-rst}
.. autofunction:: starfold.generate_null
```

## Compare backends

`compare_backends` fits one setting on the CPU and on cuML and reports how well the two partitions agree. `default_params` returns the library defaults.

```{eval-rst}
.. autofunction:: starfold.compare_backends
```

```{eval-rst}
.. autofunction:: starfold.default_params
```

## Save and load

`save` writes a result to a directory with checksums. `load` checks them and rebuilds the result. The files are listed in {doc}`artifacts`.

```{eval-rst}
.. autofunction:: starfold.save
```

```{eval-rst}
.. autofunction:: starfold.load
```

## Result classes

`Result` holds everything `recover` returns. The other classes hold its parts. Each field is explained in {doc}`result`.

```{eval-rst}
.. autoclass:: starfold.Result
   :members: n_groups, support, unresolved_mass
```

```{eval-rst}
.. autoclass:: starfold.SearchResult
```

```{eval-rst}
.. autoclass:: starfold.WorldEvidence
```

```{eval-rst}
.. autoclass:: starfold.NullEvidence
```

```{eval-rst}
.. autoclass:: starfold.BackendAgreement
```

```{eval-rst}
.. autoclass:: starfold.Params
   :members: to_dict, from_dict
```
