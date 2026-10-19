# Save a result and load it back

This guide writes a result to disk as a checksummed directory and reads it back, so you can share it or read it later without running the fits again.

```{eval-rst}
.. index::
   single: save
   single: load
   single: checksum
   single: allow_pickle
   single: manifest.json
   pair: artifact; result directory
```

## Prerequisites

- A result from `starfold.recover` (or from {doc}`run-calibration-test`, step 6).

## Steps

1. Save the result to a directory. Starfold creates it if it does not exist and overwrites files of the same names.

   ```python
   import starfold

   path = starfold.save(result, "runs/digits")
   ```

   `save` returns the directory as a `Path`.

2. Load it back.

   ```python
   loaded = starfold.load(path)
   ```

3. Check that nothing changed.

   ```python
   import numpy as np

   assert np.array_equal(loaded.labels, result.labels) and loaded.flags == result.flags
   print(f"saved to {path}; labels and flags match after reloading")
   ```

## Expected result

The recorded Digits notebook saves and reloads its result, then prints the line below. The notebook writes the absolute path of its own directory where `<your directory>` stands.

```text
saved to <your directory>; labels and flags match after reloading
```

## What each file holds

The directory has three files. For the recorded Digits run they are `arrays.npz` (91,594 bytes), `result.json` (41,372 bytes) and `manifest.json` (169 bytes), read from `notebooks/runs/digits`. The saved runs in `notebooks/runs/` ship with the repository, and notebooks 01 and 02 write them.

| File | Holds |
|---|---|
| `result.json` | the format tag `starfold-result-2`, the selected, deployed and default settings, the trial table and candidates, scalar evidence of the refits, the consensus check and the null, the flags, and the provenance |
| `arrays.npz` | every array: labels, strengths, default labels, kept rows, the resample rows, the refit label matrix, support, unresolved mass, own support, group recurrence and its band, count values, property medians and bands, refit agreement, and the consensus labels and rows |
| `manifest.json` | the SHA-256 checksum of `result.json` and of `arrays.npz` |

The directory does not hold your table `X`, the embeddings, fitted models or the measurement draws. It holds `kept_rows` and the `input_fingerprint`, which identify the table without storing it. The field list is in {doc}`../reference/artifacts`.

## What load checks

`starfold.load` raises `ValueError` if any of these fails:

- the checksum of a file differs from `manifest.json`, for example after you edit `result.json` by hand;
- the format tag is unknown;
- the support of any row plus its unresolved share does not sum to 1 (within 1e-4);
- the flags stored in the file differ from the flags re-derived from the arrays.

`load` re-derives the flags instead of trusting the stored list, so a stored claim cannot disagree with the evidence beside it.

The arrays are written and read with `allow_pickle=False`, and the rest is JSON. The files hold numbers and text, never executable objects, so loading a result does not run code from the file.

## Variations

- **Attach a calibration test.** Replace the `null` field and the `flags` field of the result in one step, then save ({doc}`run-calibration-test`). `load` refuses a result whose stored flags differ from the flags derived from its evidence.
- **Move between machines.** Copy the whole directory. A missing or changed file fails the checksum.
- **Share a result.** Send the directory. The reader can load it with the same Starfold format, without your table.

## Related pages

- {doc}`reproduce-a-run`
- {doc}`../reference/artifacts`
- {doc}`../reference/api`
