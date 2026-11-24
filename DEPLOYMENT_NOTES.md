# Research Selection deployment notes

## Active deployment

`artifacts/thesis/models/ensemble_FD00x.json` now points to the Research Selection bundles. `aerolife_runtime.RULPredictor` validates their selection-config fingerprint, model metadata and artifact checksums before inference.

## Rollback assets

The previous deployment is preserved:
- `artifacts/thesis/models/ensemble_FD00x_FrozenR1.json`
- `artifacts/thesis/results/interval_table_FD00x_FrozenR1.csv`
- Original `final_CAT_T200.joblib`, `final_XGB_T200.joblib`, `final_LGB_T200.joblib`, and `final_feature_pipeline.joblib`
- `artifacts/thesis/results/FROZEN_FINAL_ENSEMBLES_R1_three_family.json`

Do not delete these backup files. The Research Selection artifacts use the `research_` filename prefix.

## Selection-to-deployment rule

The final deployment configuration is not a fold-specific model. It aggregates the 15 inner-only selections per dataset: choose the modal target cap, take the component-wise median of the three blend weights among records selecting that cap, enforce the declared 0.10 minimum member weight, and renormalize. The resulting members are fitted on all 709 development engines. Nested outer-validation metrics remain estimates for the selection procedure; they are not relabeled as the final refit model's test metrics.

## GitHub Desktop

Commit and push the contents of this ZIP's project root to the same repository as the previous dashboard. Keep the folder structure intact. After deployment, check all four NASA datasets and both Custom Data upload/manual-history workflows.
