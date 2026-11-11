###############################################################
# DDS calibration for LASAM (+PET or +NOM) + T-route, 1 or 2 tiles
# Based on dds_calibration_lasam.py (Peter La Follette, Lynker, May 2025)
#
# OBSERVED TILE WEIGHT VERSION
#   With 2 tiles, the tile-1 weight is taken from land cover (NLCD forest +
#   woody wetland fraction) instead of being calibrated freely:
#     WEIGHT_MODE=fixed        weight held at the observed value      (variant 2)
#     WEIGHT_MODE=constrained  calibrated within obs +/- WEIGHT_BAND  (variant 3)
#     WEIGHT_MODE=free         original behaviour, bounds (0, 1)      (variant 1)
#
#   Env vars (in addition to the usual N_ITER, MAX_CORES, MODEL_ROOT1/2,
#   LOGGING_DIR, GAGES_FILE, RESUME):
#     TILE_WEIGHTS_CSV  tile_weights_by_gage.csv from compute_tile_weights.py
#     WEIGHT_COLUMN     column to use (default w_tile1_areawt)
#     WEIGHT_MODE       fixed | constrained | free (default fixed)
#     WEIGHT_BAND       half-width for constrained mode (default 0.1)
#
#   Changes vs. the original:
#     - the weight stays the LAST element of the parameter vector, as before
#     - parameters with zero-width bounds are never perturbed by DDS
#     - the weights passed to objective_function_tiled and
#       run_validation_with_best come from the parameter vector
#       (the original passed a fixed [0.5, 0.5])
###############################################################

import os
import subprocess
import pandas as pd
import numpy as np
import random
import multiprocessing
import traceback
import yaml
import json
import shutil
from datetime import datetime

# === Shared logic and config ===
from pso_calibration_lasam import (
    clear_terminal,
    check_for_stop_signal_or_low_disk,
    get_observed_q,
    extract_initial_params,
    update_lasam_files_for_divide,
    objective_function_tiled,
    run_validation_with_best,
    extract_tile_params,
    cfg,  # path_config
    spinup_start, cal_start, cal_end, val_start, val_end,
)

np.random.seed(42)
random.seed(42)

# === CONFIG ===
n_iterations = int(os.environ.get("N_ITER", 25))
max_cores_for_gages = int(os.environ.get("MAX_CORES", 3))
project_root = cfg.project_root
observed_q_root = cfg.observed_q_root
logging_dir = cfg.logging_dir
model_roots = cfg.model_roots  # List of model roots for tiles
n_tiles = len(model_roots)

# === Observed tile weight settings ===
TILE_WEIGHTS_CSV = os.environ.get("TILE_WEIGHTS_CSV", "")
WEIGHT_COLUMN = os.environ.get("WEIGHT_COLUMN", "w_tile1_areawt")
WEIGHT_MODE = os.environ.get("WEIGHT_MODE", "fixed").lower()
WEIGHT_BAND = float(os.environ.get("WEIGHT_BAND", 0.1))
if WEIGHT_MODE not in ("fixed", "constrained", "free"):
    raise ValueError(f"WEIGHT_MODE must be fixed, constrained or free (got {WEIGHT_MODE})")


def load_obs_weight(gage_id):
    """Observed tile-1 weight for one gage from the compute_tile_weights.py output."""
    if not TILE_WEIGHTS_CSV:
        raise ValueError("Set TILE_WEIGHTS_CSV to tile_weights_by_gage.csv")
    tw = pd.read_csv(TILE_WEIGHTS_CSV, dtype={"gage_id": str})
    tw["gage_id"] = tw["gage_id"].str.zfill(8)
    hit = tw.loc[tw["gage_id"] == str(gage_id).zfill(8), WEIGHT_COLUMN]
    if hit.empty:
        raise KeyError(f"gage {gage_id} not found in {TILE_WEIGHTS_CSV}")
    return float(hit.iloc[0])


def weight_init_and_bounds(gage_id):
    """(initial value, (low, high)) for the tile-1 weight under WEIGHT_MODE."""
    if WEIGHT_MODE == "free":
        return 0.8, (0.0, 1.0)
    w = load_obs_weight(gage_id)
    if WEIGHT_MODE == "fixed":
        return w, (w, w)
    return w, (max(0.0, w - WEIGHT_BAND), min(1.0, w + WEIGHT_BAND))


def tile_weights_from(vec, n_tiles):
    """Tile weights implied by a parameter vector (weight is the last element)."""
    if n_tiles == 2:
        w = float(vec[-1])
        return [w, 1.0 - w]
    return [1.0 / n_tiles] * n_tiles


# === Helper ===
def reflect_bounds(x, low, high):
    if x < low:
        return low + (low - x)
    elif x > high:
        return high - (x - high)
    else:
        return x

class DDS:
    def __init__(self, bounds, n_iterations, gage_id,
                 observed_q_root, base_roots, init_params,
                 metric_to_calibrate_on="kge", sigma=0.2,
                 include_nom=False, n_tiles=2,
                 param_names=None):

        self.bounds = bounds
        self.n_iterations = n_iterations
        self.gage_id = gage_id
        self.observed_q_root = observed_q_root
        self.base_roots = base_roots
        self.metric_to_calibrate_on = metric_to_calibrate_on
        self.sigma = sigma
        self.include_nom = include_nom
        self.n_tiles = n_tiles

        self.best_position = np.copy(init_params)
        self.best_value = float("inf")
        self.param_names = param_names if param_names else [f"param_{i}" for i in range(len(bounds))]

        # parameters with zero-width bounds (e.g. a fixed tile weight) are never perturbed
        self.free = np.array([high > low for (low, high) in self.bounds])

    def optimize(self):
        num_params = len(self.bounds)
        log_rows = []
        log_path = os.path.join(logging_dir, f"{self.gage_id}.csv")
        _resume = None
        if os.environ.get("RESUME", "0") == "1" and os.path.exists(log_path):
            try:
                _old = pd.read_csv(log_path)
                _kcol = f"{self.metric_to_calibrate_on}_calibration"
                if (_old["iteration"].astype(str) == "final").any():
                    print(f"Gage {self.gage_id}: already complete, skipping.")
                    return self.best_position, self.best_value, 0
                _num = _old[pd.to_numeric(_old["iteration"], errors="coerce").notna()].copy()
                if len(_num) > 1:
                    _num["iteration"] = _num["iteration"].astype(int)
                    _ib = _num[_kcol].astype(float).idxmax()
                    _resume = dict(rows=_num.to_dict("records"), last=int(_num["iteration"].max()),
                                   pos=np.array([_num.loc[_ib, _n] for _n in self.param_names], dtype=float),
                                   val=-float(_num.loc[_ib, _kcol]), kge=float(_num.loc[_ib, _kcol]))
                    print(f"Gage {self.gage_id}: resuming after iteration {_resume['last']} (best so far {_resume['kge']:.4f})")
            except Exception as _e:
                print(f"Gage {self.gage_id}: could not read old log ({_e}); starting fresh")
                _resume = None
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        start_time = datetime.now()

        print(f"\n--- Initial evaluation for gage {self.gage_id} ---")

        obj_val, val_val, cal_metrics, val_metrics = objective_function_tiled(
            (self.best_position, 0, self.gage_id, self.observed_q_root),
            metric_to_calibrate_on=self.metric_to_calibrate_on,
            base_roots=self.base_roots,
            include_nom=self.include_nom,
            n_tiles=self.n_tiles,
            weights=tile_weights_from(self.best_position, self.n_tiles)
        )
        self.best_value = obj_val

        best_cal_metrics = cal_metrics  # save for later
        row = {
            "iteration": 0,
            "particle": 0,
            f"{self.metric_to_calibrate_on}_calibration": cal_metrics.get(self.metric_to_calibrate_on, np.nan),
            f"{self.metric_to_calibrate_on}_validation": val_metrics.get(self.metric_to_calibrate_on, np.nan),
        }
        row.update({self.param_names[i]: val for i, val in enumerate(self.best_position)})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        start_iter = 1
        if _resume is not None:
            self.best_position = _resume["pos"]; self.best_value = _resume["val"]
            best_cal_metrics = {self.metric_to_calibrate_on: _resume["kge"]}
            log_rows = _resume["rows"]; start_iter = _resume["last"] + 1
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

        for iteration in range(start_iter, self.n_iterations + 1):
            print(f"\n--- DDS Iteration {iteration} for gage {self.gage_id} ---")
            clear_terminal()

            if check_for_stop_signal_or_low_disk(project_root):
                print("Terminating early due to STOP_NOW or low disk.")
                break

            p = 1 - np.log(iteration) / np.log(self.n_iterations)
            perturb_mask = (np.random.rand(num_params) < p) & self.free
            if not np.any(perturb_mask):
                perturb_mask[np.random.choice(np.flatnonzero(self.free))] = True

            candidate = np.copy(self.best_position)
            for i in range(num_params):
                if perturb_mask[i]:
                    low, high = self.bounds[i]
                    perturb = np.random.normal(0, self.sigma) * (high - low)
                    candidate[i] = reflect_bounds(candidate[i] + perturb, low, high)

            obj_val, val_val, cal_metrics, val_metrics = objective_function_tiled(
                (candidate, 0, self.gage_id, self.observed_q_root),
                metric_to_calibrate_on=self.metric_to_calibrate_on,
                base_roots=self.base_roots,
                include_nom=self.include_nom,
                n_tiles=self.n_tiles,
                weights=tile_weights_from(candidate, self.n_tiles)
            )

            if obj_val < self.best_value:
                self.best_position = candidate
                self.best_value = obj_val
                best_cal_metrics = cal_metrics

            row = {
                "iteration": iteration,
                "particle": 0,
                f"{self.metric_to_calibrate_on}_calibration": cal_metrics.get(self.metric_to_calibrate_on, np.nan),
                f"{self.metric_to_calibrate_on}_validation": val_metrics.get(self.metric_to_calibrate_on, np.nan),
            }
            row.update({self.param_names[i]: val for i, val in enumerate(candidate)})
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

        # === Final validation ===
        print("\n--- Final validation with best parameters ---")
        final_val_metrics = run_validation_with_best(
            gage_id=self.gage_id,
            logging_dir=logging_dir,
            observed_q_root=self.observed_q_root,
            base_roots=self.base_roots,
            best_params=self.best_position,
            n_tiles=self.n_tiles,
            include_nom=self.include_nom,
            weights=tile_weights_from(self.best_position, self.n_tiles),
            metric_to_calibrate_on=self.metric_to_calibrate_on
        )

        final_val = final_val_metrics.get(self.metric_to_calibrate_on, np.nan)
        final_row = {
            "iteration": "final",
            "particle": 0,
            f"{self.metric_to_calibrate_on}_calibration": best_cal_metrics.get(self.metric_to_calibrate_on, np.nan),
            f"{self.metric_to_calibrate_on}_validation": final_val,
        }
        final_row.update({self.param_names[i]: val for i, val in enumerate(self.best_position)})
        log_rows.append(final_row)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        print(f"\nDDS complete for {self.gage_id}. Best = {-self.best_value:.4f} | Time: {datetime.now() - start_time}")
        return self.best_position, self.best_value, datetime.now() - start_time

def calibrate_gage_dds(gage_id):
    try:
        all_init_params = []
        all_bounds = []
        param_names = []
        include_nom_flags = []

        for tile_idx, root in enumerate(model_roots):
            config_dir = os.path.join(root, f"out/{gage_id}/configs/lasam")
            if not os.path.exists(config_dir):
                print(f"Missing config dir for tile {tile_idx}: {config_dir}")
                return

            example_file = sorted(f for f in os.listdir(config_dir) if f.startswith("lasam_config_cat"))[0]
            example_path = os.path.join(config_dir, example_file)

            tile_init = extract_initial_params(example_path)
            include_nom = os.path.isdir(os.path.join(root, f"out/{gage_id}/configs/noahowp"))
            include_nom_flags.append(include_nom)

            # Convert 'a' to log10
            a_index = -12 if include_nom else -6
            tile_init[a_index] = np.log10(tile_init[a_index])
            all_init_params.extend(tile_init)

            # Count soil layers
            with open(example_path) as f:
                lines = f.readlines()
            num_layers = len([
                line for line in lines if line.startswith("layer_soil_type=")
            ][0].split("=")[1].split(","))

            # Define bounds and parameter names for this tile
            tile_bounds = []
            tile_param_names = []

            for layer in range(num_layers):
                tile_bounds.extend([(-4, 0.0), (1.02, 3.0), (-4, 2)])  # log_alpha, n, log_Ks
                tile_param_names.extend([
                    f"log_alpha_{tile_idx}_{layer + 1}",
                    f"n_{tile_idx}_{layer + 1}",
                    f"log_Ks_{tile_idx}_{layer + 1}",
                ])

            tile_bounds.extend([
                (-8, -1),         # log10_a
                (0.01, 5.0),      # b
                (1e-4, 1 - 1e-4), # frac_to_GW
                (10.0, 500.0),    # field_capacity_psi
                (0.1, 1.0),       # spf_factor
                (0.3, 0.6)        # theta_e_1
            ])
            tile_param_names.extend([
                f"log10_a_{tile_idx}",
                f"b_{tile_idx}",
                f"frac_to_GW_{tile_idx}",
                f"field_capacity_psi_{tile_idx}",
                f"spf_factor_{tile_idx}",
                f"theta_e_1_{tile_idx}"
            ])

            if include_nom:
                tile_bounds.extend([
                    (0.625, 5.0), (0.1, 100.0), (0.0, 20.0),
                    (0.18, 5.0), (0.0, 80.0), (3.6, 12.6)
                ])
                tile_param_names.extend([
                    f"MFSNO_{tile_idx}", f"RSURF_SNOW_{tile_idx}", f"HVT_{tile_idx}",
                    f"CWPVT_{tile_idx}", f"VCMX25_{tile_idx}", f"MP_{tile_idx}"
                ])

            all_bounds.extend(tile_bounds)
            param_names.extend(tile_param_names)

        # Tile weight (last element) from land cover, if two tiles
        if n_tiles == 2:
            w_init, w_bounds = weight_init_and_bounds(gage_id)
            all_init_params.append(w_init)
            all_bounds.append(w_bounds)
            param_names.append("tile1_weight")
            print(f"Gage {gage_id}: tile1_weight mode={WEIGHT_MODE}, init={w_init:.3f}, bounds={w_bounds}")

        include_nom = any(include_nom_flags)

        # === Run DDS optimization ===
        print(f"Final param_names: {param_names}")
        dds = DDS(
            bounds=all_bounds,
            n_iterations=n_iterations,
            gage_id=gage_id,
            observed_q_root=observed_q_root,
            base_roots=model_roots,
            init_params=all_init_params,
            metric_to_calibrate_on="kge",
            include_nom=include_nom,
            n_tiles=n_tiles,
            param_names=param_names
        )
        dds.optimize()

    except Exception as e:
        print(f" Error calibrating {gage_id}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print(f"WEIGHT_MODE={WEIGHT_MODE}  TILE_WEIGHTS_CSV={TILE_WEIGHTS_CSV or '(not set)'}  column={WEIGHT_COLUMN}")
    gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    overall_start = datetime.now()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(calibrate_gage_dds, gage_list)

    print(f"\n=== Total DDS wall time for all gages: {datetime.now() - overall_start} ===")
