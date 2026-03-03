# PCM_pix

This repo contains a refactored pipeline for:
- loading materials + mesh simulation tables
- training or loading surrogate neural networks (am/cr)
- running optimization (PSO / DE / PSO→DE hybrid)
- saving results under `outputs/<run_name>/`

## Recommended entrypoint

Use `main_clean.ipynb` as the primary notebook UI.

## Data layout

Put input files under `data/`:
- `Sb2Se3_am.txt`, `Sb2Se3_cr.txt`
- `Frantz-amorphous.csv`, `Frantz-crystal.csv`
- mesh tables:
  - `Sb2Se3 - amorphous_mesh_sbse_2025.txt`
  - `Sb2Se3 - crystal_mesh12_sbse_2025.txt`
- legacy models (optional):
  - `Sb2Se3_am_model_bagel_2025_updANN`, `Sb2Se3_am_scaler_X_bagel_2025_updANN`, `Sb2Se3_am_scaler_y_bagel_2025_updANN`
  - `Sb2Se3_cr_model_bagel_2025_updANN`, `Sb2Se3_cr_scaler_X_bagel_2025_updANN`, `Sb2Se3_cr_scaler_y_bagel_2025_updANN`

## Outputs

Each run creates:
- `outputs/<run_name>/logs/run.log`
- `outputs/<run_name>/models/` (new-format models + scalers)
- `outputs/<run_name>/results/` (best_pos/best_cost, hyperopt tables, solutions catalog)

## Hyperparameter optimization (hyperopt)

`main_clean.ipynb` supports caching hyperopt results in `outputs/<run_name>/results/`:
- PSO: `pso_random_search.csv`, `pso_refine.csv`
- DE:  `de_random_search.csv`, `de_refine.csv`

Modes in CFG:
- `pso_hyperopt_mode`: `"auto" | "use_saved" | "run"`
- `de_hyperopt_mode`: `"auto" | "use_saved" | "run"`

