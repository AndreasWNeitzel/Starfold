# Saved result files

This page lists the files that `starfold.save` writes, how they are checksummed, and what `starfold.load` verifies before it returns a result.

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
   single: manifest
```
```{eval-rst}
.. index::
   single: checksum
```
```{eval-rst}
.. index::
   single: allow_pickle
```

## What `save` writes

`starfold.save(result, path)` creates the directory `path` if it is absent and writes three files into it. It overwrites files of the same name. It returns the directory as a `Path`.

| File | Holds |
|---|---|
| `result.json` | The format tag, the three `Params`, the search (trial table, candidates, scores, counters, the plan's seeds), the scalar part of the world evidence, the scalar part of the consensus check, the null evidence, the backend agreement, the flags and the provenance. Written with sorted keys and indent 1. |
| `arrays.npz` | Every array, compressed. See the list below. |
| `manifest.json` | The SHA-256 checksum of `result.json` and of `arrays.npz`, keyed by file name. |

The format tag in `result.json` is `"starfold-result-2"`. A non-finite Python float is stored as `null` and read back as NaN. A non-finite NumPy scalar can be written as `NaN`, which Python's JSON reader also accepts.

The array file holds 18 arrays: `labels`, `strengths`, `default_labels`, `kept_rows`, `plan_indices`, `world_labels`, `n_sampled`, `support`, `unresolved`, `own_support`, `group_recurrence`, `group_recurrence_band`, `count_values`, `property_median`, `property_band`, `world_agreement`, `consensus_labels` and `consensus_rows`. Each array is written as its own member with `np.save(allow_pickle=False)` and read with `allow_pickle=False`, so the file holds numbers and no Python objects.

## What is not saved

`save` stores no executable object and no model. It does not store your input table, only its fingerprint in `provenance["input_fingerprint"]`. It does not store UMAP embeddings or the measurement draws used inside refits.

## What `load` verifies

`starfold.load(path)` runs these checks in order and raises `ValueError` at the first that fails.

1. **Checksums.** `manifest.json` must be valid JSON and hold a checksum for both `result.json` and `arrays.npz`. Then `load` hashes every file named in the manifest and compares the result with the stored value. Each error names the file.
2. **Format.** The format tag must be `"starfold-result-2"`. A file with any other tag does not load.
3. **Support identity.** On every row that at least one refit sampled, `sum_g support[i, g] + unresolved[i]` must equal 1 within 1e-4.
4. **Flags.** It derives the flags again from the loaded arrays and compares them with the stored list. It refuses a file whose flags differ.

Then it rebuilds every dataclass and returns the `Result`. Arrays come back in their original dtypes.

The checksums detect accidental change, such as a truncated copy. They do not protect against someone who edits a file and its manifest together, because the manifest sits beside the files.

## Related pages

{doc}`../howto/save-and-load` shows the steps. {doc}`result` lists the fields that come back. {doc}`../troubleshooting` covers a file that `load` refuses.
