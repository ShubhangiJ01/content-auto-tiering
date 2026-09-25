# Content Auto Tier

## Whisper Embedding Dataset

The released ZIP (whisper_embedding.zip) inside dataset folder contains the pre-computed Whisper-large-v3 representations and the files required to reproduce the embedding-based evaluation experiments reported in the paper.

After extracting the ZIP, the directory structure is:

```text
whisper_embedding/
├── <journey_id>.npy
└── whisper_embedding_metadata.csv
└── whisper_embedding_eval.py

```

### Files Included

| File / Directory                                                 | Description                                                                                       |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `whisper_embedding_metadata.csv`                                 | Metadata mapping each asset (`journey_id`) to its tier label and corresponding embedding paths.   |
| `whisper_embedding_eval.py`                                      | Evaluation code used to load the embeddings and reproduce the reported Whisper-based experiments. |
| `whisper_embedding/<journey_id>.npy`                             | Pre-computed asset-level Whisper representation with shape `(1280,)`.                             |

### Asset-Level Embeddings

The `<journey_id>.npy` files contain the pre-computed asset-level representations used by the evaluation code.

The asset-level representation is obtained using duration-weighted mean pooling across the window-level representations:

```text
e_asset = Σᵢ (dᵢ / Σⱼ dⱼ) eᵢ
```

where `eᵢ` is the embedding for window `i` and `dᵢ` is the original duration of that window.

The resulting representation has 1280 dimensions.


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

## Transformed Feature Dataset

The released ZIP (`content_auto_tiering_transformed_features.zip`) contains the task-specific features: acoustic, linguistic, speaker, and visual signals measured on chunks of each asset and aggregated to asset level. After extracting the ZIP, the directory structure is:

```text
content_auto_tiering_transformed_features/
├── README.md
├── MANIFEST.csv
├── DATA_DICTIONARY.csv
├── legacy_paid/
│   ├── legacy_467_asset_features.csv
│   ├── legacy_467_chunk_features.csv
│   ├── legacy_467_feature_report.json
│   ├── paid_379_asset_features.csv
│   ├── paid_379_chunk_features.csv
│   ├── paid_379_feature_report.json
│   ├── merged_846_asset_features.csv
│   ├── merged_846_chunk_features.csv
│   ├── merged_846_dataset_report.json
│   ├── dataset_coverage.json
│   ├── retraining_config.json
│   └── splits/
│       ├── legacy_467_split.csv
│       ├── split_manifest_paid.csv
│       ├── split_manifest_merged_train.csv
│       ├── split_manifest_val_original.csv
│       ├── split_manifest_val_paid.csv
│       ├── split_manifest_val_merged.csv
│       └── split_summary.json
├── wave2_research/
│   ├── merged_asset_features_wave2.csv
│   ├── selected21_feature_manifest.json
│   ├── build_wave2_report.json
│   └── phase4_eval_summary.json
└── exact_nonpaid_1197/
    ├── asset_features_selected21.csv
    ├── cohort_manifest.csv
    ├── completion_snapshot.csv
    ├── feature_family_lineage.csv
    ├── cohort_lock.json
    └── feature_matrix_lock.json
```

`MANIFEST.csv` lists each file's row and column counts, size, SHA-256 checksum, and how it was derived from its source. `DATA_DICTIONARY.csv` lists every column of the data CSVs with its dtype and role (`identifier`, `label`, `transformed feature`, or `metadata/provenance`).

### Chunk Signals and Asset Features

Each asset is divided into time chunks, and seven signals are computed per chunk:

| Signal | Definition |
| --- | --- |
| `wpm` | Words per minute |
| `pn_ratio` | Proper nouns as a fraction of words (`proper_noun_count / chunk_words_count`) |
| `stoi_score` | Short-Time Objective Intelligibility (STOI) score of the speech |
| `overlap_ratio` | Fraction of the chunk with overlapping speech (`overlap_duration / duration`) |
| `ocr_ratio` | Fraction of the chunk with on-screen text detected by OCR (`ocr_duration / duration`) |
| `unique_speakers_count` | Number of distinct speakers |
| `speakers_per_minute` | Distinct speakers per minute |

Each signal is aggregated over the asset's chunks with seven statistics, giving 49 asset-level features named `<statistic>_<signal>` (for example `p90_wpm`):

| Statistic | Definition |
| --- | --- |
| `mean` | Duration-weighted mean |
| `p10`, `p90` | 10th and 90th percentiles |
| `mean_bottom3`, `mean_top3` | Mean of the three lowest or three highest chunk values |
| `dur_frac_bad` | Fraction of the asset's duration in chunks whose value is above 0.8 (below 0.2 for `stoi_score`) |
| `frac_chunks_bad` | Fraction of chunks whose value is above 0.8 (below 0.2 for `stoi_score`) |

### `legacy_paid/`: Legacy and PAID Cohorts

This folder holds two labelled cohorts, separately and merged:

* **Legacy** (`cohort` = `original`): 467 assets from the original training data, with 151/249/67 assets in tiers 1/2/3 (`Asset Tier`). On the original chunk boundaries, WPM, proper-noun, OCR, and STOI signals were recomputed from production pipeline outputs, while speaker and overlap signals come from the original chunk data (`feature_source` = `partial_tier_a_hybrid`).
* **PAID** (`cohort` = `paid`): 379 assets whose tier was manually reviewed, with 46/180/153 assets in tiers 1/2/3 (`assigned_tier`). Chunking and all signals come from the production feature pipeline (`feature_source` = `full_production`). `PAID` is the asset identifier and `Show Title` the show.

Because the two cohorts' features come from different pipelines, keep `cohort` and `feature_source` when using the merged tables.

| File | Rows × columns | Contents |
| --- | --- | --- |
| `legacy_467_asset_features.csv` | 467 × 51 | `journey_id`, `Asset Tier`, and the 49 features |
| `legacy_467_chunk_features.csv` | 2,086 × 18 | Chunk signals, with chunk `start`, `end`, and `duration` in seconds |
| `paid_379_asset_features.csv` | 379 × 54 | The 49 features, `Asset Tier`, `PAID`, `Show Title`, and `assigned_tier` |
| `paid_379_chunk_features.csv` | 608 × 22 | Chunk signals |
| `merged_846_asset_features.csv` | 846 × 56 | Both cohorts, with `cohort` and `feature_source` |
| `merged_846_chunk_features.csv` | 2,694 × 24 | Chunks of both cohorts |
| `*_report.json`, `dataset_coverage.json` | — | Feature-generation reports and per-split coverage |
| `retraining_config.json` | — | Cohort definitions, split policy, HistGradientBoosting parameters, and tuning grid |

#### Splits

`splits/` holds the frozen journey-level splits. Each cohort was split separately, stratified by tier, with 30% held out for validation (`random_state=42`):

| File | Assets | Tiers 1/2/3 |
| --- | --- | --- |
| `split_manifest_merged_train.csv` | 591 (326 legacy + 265 PAID) | 138/299/154 |
| `split_manifest_val_original.csv` | 141 legacy | 46/75/20 |
| `split_manifest_val_paid.csv` | 114 PAID | 14/54/46 |
| `split_manifest_val_merged.csv` | 255 (141 legacy + 114 PAID) | 60/129/66 |

`legacy_467_split.csv` and `split_manifest_paid.csv` give each cohort's complete train/validation assignment, and `split_summary.json` records the counts and split parameters. For one legacy training asset, the tier in the split manifests (`asset_tier`) differs from the asset tables; `dataset_coverage.json` reports it as a label mismatch.

### `wave2_research/`: 84-Feature Research Pool

`merged_asset_features_wave2.csv` (843 × 91) adds five chunk signals, aggregated with the same seven statistics, for 84 features in total:

| Signal | Definition |
| --- | --- |
| `turns_per_minute` | Speaker turns per minute |
| `turn_transition_rate` | Rate of speaker transitions between consecutive words (0–1) |
| `type_token_ratio` | Distinct words divided by total words |
| `avg_word_length` | Mean word length in characters |
| `rare_word_rate` | Fraction of words with an English Zipf frequency below 3.0 in [`wordfreq`](https://github.com/rspeer/wordfreq) |

This pool uses different chunking (contiguous tiles of about five minutes) and a different overlap source (Sortformer speaker diarization), so its 49 base features differ from those in `legacy_paid/`; do not mix the two. It covers 464 legacy and 379 PAID assets. Three legacy assets are missing, so evaluations on this pool cover 140 of the 141 `val_original` assets.

`selected21_feature_manifest.json` lists the ordered 21-feature subset chosen from this pool by nested cross-validation on the training split (benchmark `phase4_wave2_lexical_v1`), with its HistGradientBoosting parameters and validation metrics. `build_wave2_report.json` and `phase4_eval_summary.json` are the build and evaluation reports.

### `exact_nonpaid_1197/`: Final 1,197-Asset Matrix

`asset_features_selected21.csv` (1,197 × 23) is a model-ready matrix: `journey_id`, the label `bq_tier` (251/376/570 assets in tiers 1/2/3), and the 21 selected features in manifest order, with no missing values. The cohort contains only exact matches—assets whose recorded tier equals the tier assigned from their caption-edit profile—and excludes all 379 PAID assets and every asset in the validation splits above.

| File | Contents |
| --- | --- |
| `cohort_manifest.csv` | Per-asset cohort metadata: `bq_tier`, edit-profile `assigned_tier`, `profile_outlier`, speaker-label source, availability partition, and backfill flags |
| `completion_snapshot.csv` | Feature-family completion status for each asset (all complete) |
| `feature_family_lineage.csv` | 4,788 rows (1,197 assets × 4 feature families) recording where each family's values came from |
| `cohort_lock.json`, `feature_matrix_lock.json` | Cohort and matrix summaries with SHA-256 checksums |

## Caption Edit Dataset

`edit_content_tier_data.csv` has one row per asset (2,587 assets) with the number of edits a human made to the machine-generated captions, broken down by edit type. Edits are detected automatically by comparing the machine-generated caption file with the post-edited (ground-truth) file.

| Column | Description |
| --- | --- |
| `Journey_Id` | Asset identifier (`journey_id` in the other datasets) |
| `Workflow_Type` | Captioning workflow: `closed_captioning` (2,446 assets), `closed_captioning_v2` (79), or `emt_dialog` (62) |
| `Total_Gt_Blocks` | Number of caption blocks in the post-edited file |
| Edit-count columns | Number of edits of each type (see the next table) |
| `Content_Duration` | Asset duration in minutes |
| `Tier` | Tier label: 424/1,098/1,063 assets in tiers 1/2/3, and `?` for two assets with an unknown tier |
| `Created_At` | When the edit record was created (May 2024 to August 2026) |

Edit-count columns, with the name each tag has in the [detection-metric workbooks](#edit-tag-detection-metrics):

| Column | Workbook name | Edit counted |
| --- | --- | --- |
| `Word_Edits` | Word Correction | Word substitutions, insertions, and deletions |
| `Punctuation_Edits` | Punctuation Correction | Punctuation changes |
| `Line_Break` | Line Wrapping | Line breaks moved within a block |
| `Split_Edits` | Block Split | A machine block split into several blocks |
| `Merge` | Block Merge | Several machine blocks merged into one |
| `Split_Merge` | Block Split & Merge | Blocks re-segmented by both splitting and merging |
| `Block_Match_Time_Drift` | Timecode Drift | Timing drift between matched machine and post-edited blocks |
| `Speaker_Transition` | Speaker Change | Speaker-change markers such as `-NAME` added, removed, or changed |
| `Missing_Block` | Missed Block | A post-edited block with no machine counterpart |
| `Extra_Block` | Extra Block | A machine block with no post-edited counterpart |
| `Closed_Caption` | Closed Captioning | Bracketed non-speech annotations such as `[APPLAUSE]` |
| `VTT_Style` | Caption Styling | Style tags such as `<i>` |
| `Musical_Lyrics` | — | Lyric lines marked with `♪` |
| `Alignment_Change` | Block Alignment | WebVTT `align` setting |
| `Line_Shift` | Block Vertical Positioning Shift | WebVTT `line` setting (vertical position) |
| `Position_Shift` | Block Horizontal Positioning Shift | WebVTT `position` setting (horizontal position) |
| `Size_Change` | — | WebVTT `size` setting |
| `Region_Change` | — | IMSC region |
| `Spanwise_Style_Change` | — | IMSC word-level styling |

Columns that do not apply to a workflow's caption format are blank. `Alignment_Change`, `Line_Shift`, `Position_Shift`, `Size_Change`, and `VTT_Style` are filled only for `closed_captioning`; `Region_Change` and `Spanwise_Style_Change` only for `closed_captioning_v2` and `emt_dialog`; and `Closed_Caption` is blank for `emt_dialog`.

## Edit-Tag Detection Metrics

These workbooks report how accurately the automatic edit-tag detection identifies each edit type, evaluated on 36 caption files. Every row has `True Positives`, `False Positives`, `False Negatives`, `Precision`, and `Recall`.

| File | Sheet | Rows |
| --- | --- | --- |
| `edit_tag_metrics.xlsx` | `Edit Tag Metrics` | One per edit tag (15), summed over all files |
| `file_level_metrics.xlsx` | `File Metrics` | One per file (36), summed over all tags, plus a `TOTAL` row |
| `file_level_metrics.xlsx` | `File by Edit Tag` | One per file and edit tag (473), plus a `TOTAL` row |

Across all files and tags there are 33,594 true positives, 512 false positives, and 201 false negatives: precision 0.985 and recall 0.994. Per-tag precision ranges from 0.951 (Block Split) to 1.000, and recall from 0.977 (Block Merge) to 1.000. Files are identified by asset file name (for example `259180_002_PMF_4419762`), not by `journey_id`.

## Labeller Comments and Difficulty Factors

`Labellers_comments_Factors.xlsx` (sheet `Sampled Comments`) contains 100 free-text comments sampled from labellers' notes on individual assets, covering 64 shows. Each comment is tagged with one or more difficulty factors.

| Column | Description |
| --- | --- |
| `Asset Name` | Asset file name (filled for 36 comments) |
| `Show` | Show title |
| `PAID` | Asset identifier |
| `Comment` | The labeller's comment |
| `Factors` | Comma-separated difficulty factors |

Factor counts, where 44 comments have more than one factor:

| Factor | Comments |
| --- | ---: |
| Visual & Contextual | 42 |
| Linguistic & Terminology | 36 |
| Model Output Quality | 34 |
| Speaker Interaction | 30 |
| Accent & Speech | 15 |
| Audio Complexity | 4 |

