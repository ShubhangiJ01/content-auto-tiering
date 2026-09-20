# content-auto-tiering

## Whisper Embedding Dataset

The released ZIP (whisper_embedding.zip) inside dataset folder contains the pre-computed Whisper-large-v3 representations and the files required to reproduce the embedding-based evaluation experiments reported in the paper.

After extracting the ZIP, the directory structure is:

```text
whisper_embedding/
├── <journey_id>.npy
└── window_embeddings/
    ├── <journey_id>_windows.npy
    ├── <journey_id>_durations.npy
    └── ...
```

### Files Included

| File / Directory                                                 | Description                                                                                       |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `whisper_embedding_metadata.csv`                                 | Metadata mapping each asset (`journey_id`) to its tier label and corresponding embedding paths.   |
| `whisper_embedding_eval.py`                                      | Evaluation code used to load the embeddings and reproduce the reported Whisper-based experiments. |
| `whisper_embedding/<journey_id>.npy`                             | Pre-computed asset-level Whisper representation with shape `(1280,)`.                             |
| `whisper_embedding/window_embeddings/<journey_id>_windows.npy`   | Window-level Whisper representations with shape `(N, 1280)`.                                      |
| `whisper_embedding/window_embeddings/<journey_id>_durations.npy` | Original duration, in seconds, of each corresponding window with shape `(N,)`.                    |

Here, `N` is the number of 30-second windows for an asset.

### Asset-Level Embeddings

The `<journey_id>.npy` files contain the pre-computed asset-level representations used by the evaluation code.

The asset-level representation is obtained using duration-weighted mean pooling across the window-level representations:

```text
e_asset = Σᵢ (dᵢ / Σⱼ dⱼ) eᵢ
```

where `eᵢ` is the embedding for window `i` and `dᵢ` is the original duration of that window.

The resulting representation has 1280 dimensions.

### Window-Level Embeddings

The `window_embeddings/` directory contains the underlying representations used for the temporal pooling experiments.

For each asset:

```text
<journey_id>_windows.npy
    shape: (N, 1280)

<journey_id>_durations.npy
    shape: (N,)
```

The files are aligned by index. For example:

```text
_windows.npy                 _durations.npy

row 0 → window 0 embedding   index 0 → window 0 duration
row 1 → window 1 embedding   index 1 → window 1 duration
row 2 → window 2 embedding   index 2 → window 2 duration
```

### Whisper Windowing

The complete audio waveform was resampled to **16 kHz** and explicitly partitioned into **non-overlapping 30-second windows**.

The final window was retained even when shorter than 30 seconds and was zero-padded to 30 seconds before Whisper feature extraction. Its original duration was retained in the corresponding `_durations.npy` file.

For each window, the Whisper-large-v3 encoder hidden states were mean-pooled over the temporal dimension to obtain a 1280-dimensional representation.

### Loading the Embeddings

`whisper_embedding_metadata.csv` contains the paths required to locate the embeddings and the corresponding `Tier` labels.

The evaluation code loads the asset-level representations as follows:

```python
import numpy as np
import pandas as pd

embedding_metadata = pd.read_csv(
    "whisper_embedding_metadata.csv"
)

valid_df = embedding_metadata[
    embedding_metadata["embedding_path"].notna()
].copy()

X_list = []
y_list = []

for _, row in valid_df.iterrows():

    embedding = np.load(
        row["embedding_path"]
    )

    X_list.append(embedding)
    y_list.append(row["Tier"])

X_whisper = np.vstack(X_list)
y_whisper = np.asarray(y_list)

print("X shape:", X_whisper.shape)
print("y shape:", y_whisper.shape)
```

For `M` valid assets, the resulting arrays have:

```text
X_whisper: (M, 1280)
y_whisper: (M,)
```

`X_whisper` contains one asset-level Whisper representation per asset, while `y_whisper` contains the corresponding tier labels.

### Temporal Pooling Experiments

The window-level representations are provided to reproduce the temporal pooling experiments. The evaluation considers:

* **Mean** — duration-weighted mean across windows.
* **P10** — 10th percentile per embedding dimension.
* **P90** — 90th percentile per embedding dimension.
* **Mean-Bottom3** — mean of the three lowest values per embedding dimension.
* **Mean-Top3** — mean of the three highest values per embedding dimension.

The pre-computed `<journey_id>.npy` asset-level representation corresponds to the **Mean** pooling strategy.

### Evaluation Protocol

The reported Whisper experiments use:

* Whisper-large-v3 encoder representations (1280 dimensions).
* Non-overlapping 30-second audio windows.
* Five temporal pooling strategies.
* PCA with 16, 32, 64, 128, or 256 components.
* HistGradientBoosting classification.
* Stratified 5-fold cross-validation.
* Ten independent random seeds, resulting in 50 validation folds.

PCA is fitted independently within each training fold and then applied to the corresponding validation fold to prevent data leakage.

The original audio is not required to reproduce the reported embedding-based experiments.

