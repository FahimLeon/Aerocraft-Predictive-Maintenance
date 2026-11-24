# AeroLife deployment notes

## Active deployment

`artifacts/thesis/models/ensemble_FD00x.json` points to the Primary Research Model bundles. `aerolife_runtime.RULPredictor` validates their selection-config fingerprint, model metadata and artifact checksums before inference.

## Historical baseline assets

The historical baseline artifacts remain preserved separately:
- `artifacts/thesis/models/ensemble_FD00x_FrozenR1.json`
- `artifacts/thesis/results/interval_table_FD00x_FrozenR1.csv`
- Original `final_CAT_T200.joblib`, `final_XGB_T200.joblib`, `final_LGB_T200.joblib`, and `final_feature_pipeline.joblib`
- `artifacts/thesis/results/FROZEN_FINAL_ENSEMBLES_R1_three_family.json`

Do not delete these backup files. The active Primary Research Model artifacts use the `research_` filename prefix where applicable.

## Selection-to-deployment rule

The final deployment configuration is not a fold-specific model. It aggregates the 15 inner-only selections per dataset: choose the modal target cap, take the component-wise median of the three blend weights among records selecting that cap, enforce the declared 0.10 minimum member weight, and renormalize. The resulting members are fitted on all 709 development engines. Nested outer-validation metrics remain estimates for the selection procedure; they are not relabeled as the final refit model's test metrics.

## Interface and navigation

- The navigation sidebar is permanently expanded on load; sidebar collapse/expand controls are hidden.
- Workspace options: Overview, NASA Benchmark Explorer, Custom Data Prediction, About AeroLife.
- Overview order: workspace summary, four-dataset Research Performance Snapshot, dataset-level detail expander, then Explore AeroLife.
- The snapshot displays unweighted four-subset averages: Primary Research Model mean R² 0.8992, mean RMSE 18.631 cycles and mean MAE 12.348 cycles (strict nested-validation estimates); mean NASA Score 1,763.1 is a separate historical official-test reference for the Historical Baseline Reference. These are unweighted dataset means, not pooled-engine metrics.
- The sidebar and About page identify MD FAHIM MUNTASIR LEON as Researcher and FATEMA TUJ JOHORA as Research Supervisor.
- User-facing labels use **Primary Research Model** and **Historical Baseline Reference**. Internal artifact names and runtime identifiers remain unchanged.
- No prediction, preprocessing, feature, model artifact, ensemble weight or inference-runtime files were changed for this interface refresh.

## GitHub Desktop

Commit and push the contents of this ZIP's project root to the same repository as the previous dashboard. Keep the folder structure intact. After deployment, check all four NASA datasets and both Custom Data upload/manual-history workflows.
