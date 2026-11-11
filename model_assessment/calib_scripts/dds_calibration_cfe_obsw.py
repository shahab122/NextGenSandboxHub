###############################################################
# DDS calibration for CFE (+PET or +NOM) + T-route, 1 or 2 tiles
# Based on dds_calibration_cfe_prod.py (Peter La Follette, Lynker, May 2025)
#
# OBSERVED TILE WEIGHT VERSION
#   With 2 tiles, the tile-1 weight is taken from land cover (NLCD forest +
#   woody wetland fraction) instead of being calibrated freely:
#     WEIGHT_MODE=fixed        weight held at the observed value      (variant 2)
#     WEIGHT_MODE=constrained  calibrated within obs +/- WEIGHT_BAND  (variant 3)
#     WEIGHT_MODE=free         original behaviour, bounds (0, 1)      (variant 1)
#
#   Env vars (in addition to the usual N_ITER, MAX_CORES, MODEL_ROOT1/2,
#   LOGGING_DIR, GAGES_FILE, CONFIG_DIR):
#     TILE_WEIGHTS_CSV  tile_weights_by_gage.csv from compute_tile_weights.py
#     WEIGHT_COLUMN     column to use (default w_tile1_areawt)
#     WEIGHT_MODE       fixed | constrained | free (default fixed)
#     WEIGHT_BAND       half-width for constrained mode (default 0.1)
#
#   Changes vs. the original:
#     - the weight stays the LAST element of the parameter vector, as before
#     - parameters with zero-width bounds are never perturbed by DDS
#     - the weight passed to objective_function_tiled is taken from the
#       candidate vector (the original passed a fixed [0.5, 0.5])
###############################################################

import os
import pandas as pd
import numpy as np
import random
import json
import yaml
from datetime import datetime
import multiprocessing
import subprocess
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_assessment.configs import path_config as cfg
from model_assessment.util.metrics import compute_metrics
from model_assessment.util.update_NOM import update_mptable
import shutil

project_root = cfg.project_root
sandbox_path = cfg.sandbox_path
logging_dir = cfg.logging_dir
observed_q_root = cfg.observed_q_root

from pso_calibration_cfe import (
    extract_initial_cfe_params,
    regenerate_cfe_config,
    objective_function_tiled,
    param_bounds,
    param_names,
    log_scale_params,
    check_for_stop_signal_or_low_disk,
    transform_params,
    nom_param_names,
    nom_param_bounds,
    extract_initial_nom_params,
    extract_tile_params
)

print("param names:")
print(param_names)

with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

cal_start = pd.Timestamp(time_cfg["cal_start"])
cal_end   = pd.Timestamp(time_cfg["cal_end"])
val_start = pd.Timestamp(time_cfg["val_start"])
val_end   = pd.Timestamp(time_cfg["val_end"])
spinup_start = pd.Timestamp(time_cfg["spinup_start"])

np.random.seed(42)
random.seed(42)

n_iterations = int(os.environ.get("N_ITER", 25))
max_cores_for_gages = int(os.environ.get("MAX_CORES", 3))
metric_to_calibrate_on = "kge"

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
        return 0.7, (0.0, 1.0)
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


def _safe_objective(args):
    """Failed model run -> scored like the LASAM script (objective +10, KGE -10) instead of crashing the pool."""
    import traceback
    try:
        return objective_function_tiled(args)
    except Exception:
        print(f"[WARN] evaluation failed for gage {args[2]}; scoring as -10", flush=True)
        traceback.print_exc()
        m = {metric_to_calibrate_on: -10.0}
        return 10.0, dict(m), dict(m)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def reflect_bounds(x, low, high):
    if x < low:
        return low + (low - x)
    elif x > high:
        return high - (x - high)
    else:
        return x

class DDS:
    def __init__(self, bounds, n_iterations, gage_id, init_params,
                 metric_to_calibrate_on="kge", sigma=0.2,
                 include_nom=False, nom_file_paths=None, param_names=None):
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.gage_id = gage_id
        self.metric = metric_to_calibrate_on
        self.sigma = sigma
        self.best_position = np.copy(init_params)
        self.best_value = float("inf")

        self.model_roots = cfg.model_roots
        self.n_tiles = len(self.model_roots)
        self.include_nom = include_nom
        self.nom_file_paths = nom_file_paths
        self.param_names = param_names

        # parameters with zero-width bounds (e.g. a fixed tile weight) are never perturbed
        self.free = np.array([high > low for (low, high) in self.bounds])

        self.observed_path = os.path.join(cfg.observed_q_root, "successful_sites_resampled", f"{self.gage_id}.csv")

    def optimize(self):
        log_path = os.path.join(cfg.logging_dir, f"{self.gage_id}.csv")
        log_rows = []
        num_params = len(self.bounds)
        start_time = datetime.now()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        _resume = None
        if os.environ.get("RESUME", "0") == "1" and os.path.exists(log_path):
            try:
                _old = pd.read_csv(log_path)
                _kcol = f"{self.metric}_calibration"
                if (_old["iteration"].astype(str).str.lower() == "final").any():
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

        print(f"\n--- Initial evaluation for gage {self.gage_id} ---")

        obj_val, val_metrics, cal_metrics = objective_function_tiled((
            self.best_position, 0, self.gage_id,
            self.model_roots,
            cfg.observed_q_root,
            [self.include_nom] * self.n_tiles,
            self.nom_file_paths,
            tile_weights_from(self.best_position, self.n_tiles)
        ))

        self.best_value = obj_val
        best_val_metrics = val_metrics
        best_cal_metrics = cal_metrics

        row = {
            "iteration": 0,
            "particle": 0,
            **dict(zip(self.param_names, self.best_position)),
            f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
            f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        start_iter = 1
        if _resume is not None:
            self.best_position = _resume["pos"]; self.best_value = _resume["val"]
            best_cal_metrics = {self.metric: _resume["kge"]}
            log_rows = _resume["rows"]; start_iter = _resume["last"] + 1
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

        for iteration in range(start_iter, self.n_iterations + 1):
            clear_terminal()
            check_for_stop_signal_or_low_disk()
            print(f"\n--- DDS Iteration {iteration} for gage {self.gage_id} ---")

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
                    candidate[i] = max(low, min(high, candidate[i]))

            obj_val, val_metrics, cal_metrics = _safe_objective((
                candidate, 0, self.gage_id,
                self.model_roots,
                cfg.observed_q_root,
                [self.include_nom] * self.n_tiles,
                self.nom_file_paths,
                tile_weights_from(candidate, self.n_tiles)
            ))

            if obj_val < self.best_value:
                self.best_value = obj_val
                self.best_position = candidate
                best_val_metrics = val_metrics
                best_cal_metrics = cal_metrics

            row = {
                "iteration": iteration,
                "particle": 0,
                **dict(zip(self.param_names, candidate)),
                f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
                f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
            }
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

        # === Final full-period run ===
        print(f"\n[INFO] Running final weighted-routing validation for {self.gage_id}...")

        # Handle weights
        weights = tile_weights_from(self.best_position, self.n_tiles)
        param_vector = self.best_position[:-1] if self.n_tiles == 2 else self.best_position

        # === STEP 1: Run hydrology for each tile ===
        for tile_idx, tile_root in enumerate(self.model_roots):
            tile_params = extract_tile_params(param_vector, tile_idx, self.n_tiles)

            # Strip tile suffixes for config updates
            names_for_tile = self.param_names[tile_idx * len(tile_params): (tile_idx + 1) * len(tile_params)]
            base_names = [n.split("_tile")[0] for n in names_for_tile if n != "tile_weight"]

            # Transform log-scale params and regenerate configs
            true_best = transform_params(tile_params, base_names)
            config_dir = os.path.join(tile_root, f"out/{self.gage_id}/configs/cfe")
            regenerate_cfe_config(config_dir, true_best, base_names)

            # Update NOM params if present
            if self.include_nom and self.nom_file_paths[tile_idx]:
                nom_vals = tile_params[-6:]
                update_mptable(
                    original_file=self.nom_file_paths[tile_idx],
                    output_file=self.nom_file_paths[tile_idx],
                    updated_params=dict(zip(nom_param_names, nom_vals))
                )

            # Update realization JSON for full spinup+validation
            json_dir = os.path.join(tile_root, "out", self.gage_id, "json")
            realization_file = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
            realization_path = os.path.join(json_dir, realization_file)
            with open(realization_path, "r") as f:
                realization = json.load(f)
            realization["time"]["start_time"] = time_cfg["spinup_start"]
            realization["time"]["end_time"] = time_cfg["val_end"]
            with open(realization_path, "w") as f:
                json.dump(realization, f, indent=4)

            # Run hydrology only
            tile_sandbox_config = os.path.join(os.environ.get("CONFIG_DIR", os.path.join(cfg.project_root, "configs")), f"sandbox_config_tile{tile_idx+1}.yaml")
            subprocess.call(
                ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", self.gage_id],
                cwd=tile_root
            )

        # === STEP 2: Weighted averaging of divide outputs ===
        if self.n_tiles == 1:
            weighted_div_dir = os.path.join(self.model_roots[0], "out", self.gage_id, "outputs", "div")
        else:
            weighted_div_dir = os.path.join(self.model_roots[0], "out", self.gage_id, "outputs", "div_weighted")
            if os.path.exists(weighted_div_dir):
                shutil.rmtree(weighted_div_dir)
            os.makedirs(weighted_div_dir, exist_ok=True)

            div_dirs = [os.path.join(root, "out", self.gage_id, "outputs", "div") for root in self.model_roots]
            files = [f for f in os.listdir(div_dirs[0]) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]

            for fname in files:
                dfs = []
                for w, div_dir in zip(weights, div_dirs):
                    fpath = os.path.join(div_dir, fname)
                    df = pd.read_csv(fpath, header=None if fname.startswith("nex-") else 0)
                    if fname.startswith("nex-"):
                        df.columns = ["Time Step", "Time", "q_out"]
                    dfs.append(df["q_out"] * w)
                combined = sum(dfs)
                out_df = df.copy()
                out_df["q_out"] = combined
                out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                out_df.to_csv(os.path.join(weighted_div_dir, fname), index=False, header=not fname.startswith("nex-"))

        # === STEP 3: Run routing once ===
        troute_path = os.path.join(self.model_roots[0], "out", self.gage_id, "configs", "troute_config.yaml")
        with open(troute_path, "r") as f:
            troute_cfg = yaml.safe_load(f)
        nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
        troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
        troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
        troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir
        yaml.safe_dump(troute_cfg, open(troute_path, "w"))

        # Clear old routing outputs
        troute_dir = os.path.join(self.model_roots[0], "out", self.gage_id, "troute")
        if os.path.isdir(troute_dir):
            for fname in os.listdir(troute_dir):
                if fname.endswith(".nc"):
                    os.remove(os.path.join(troute_dir, fname))

        # Run routing
        subprocess.call(["python3", "-m", "nwm_routing", "-f", "-V4", troute_path])

        # === STEP 4: Extract final routed hydrograph ===
        postproc_dir = os.path.join(self.model_roots[0], "postproc")
        final_output_path = os.path.join(postproc_dir, f"{self.gage_id}_best.csv")
        get_hydrograph_path = os.path.join(cfg.project_root, "model_assessment", "util", "get_hydrograph.py")
        subprocess.call(
            ["python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", final_output_path, "--base_dir", self.model_roots[0]],
            cwd=postproc_dir
        )

        # === STEP 5: Compute metrics and log ===
        obs_df = pd.read_csv(self.observed_path, parse_dates=["value_time"]).set_index("value_time")["flow_m3_per_s"]
        sim_df = pd.read_csv(final_output_path, parse_dates=["current_time"]).set_index("current_time")["flow"].resample("1h").mean()
        sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
        sim_val, obs_val = sim_val.align(obs_val, join="inner")
        val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

        summary_row = {
            "iteration": "final",
            "particle": 0,
            **dict(zip(self.param_names, self.best_position)),
            f"{self.metric}_calibration": best_cal_metrics.get(self.metric, np.nan),
            f"{self.metric}_validation": val_metrics_final.get(self.metric, np.nan),
        }
        log_rows.append(summary_row)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)
        print(f" Final {self.metric.upper()} = {val_metrics_final.get(self.metric, np.nan):.4f}")

        return self.best_position, self.best_value, datetime.now() - start_time


def calibrate_gage_dds(gage_id):
    all_init, all_bounds, include_nom_flags, nom_file_paths, names = [], [], [], [], []

    for tile_idx, root in enumerate(cfg.model_roots):
        config_dir = os.path.join(root, f"out/{gage_id}/configs/cfe")
        config_file = sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0]
        config_path = os.path.join(config_dir, config_file)

        init = extract_initial_cfe_params(config_path)
        bounds = param_bounds.copy()

        include_nom = os.path.isdir(os.path.join(root, f"out/{gage_id}/configs/noahowp"))
        if include_nom:
            nom_path = os.path.join(root, f"out/{gage_id}/configs/noahowp/parameters/MPTABLE.TBL")
            init += extract_initial_nom_params(nom_path)
            bounds += nom_param_bounds
            nom_file_paths.append(nom_path)
        else:
            nom_file_paths.append("")

        all_init.extend(init)
        all_bounds.extend(bounds)
        include_nom_flags.append(include_nom)

        tile_suffix = f"_tile{tile_idx+1}"
        tile_names = [f"{name}{tile_suffix}" for name in param_names]
        if include_nom:
            tile_names += [f"{name}{tile_suffix}" for name in nom_param_names]
        names.extend(tile_names)

    # Tile weight (last element) from land cover, if two tiles
    if len(cfg.model_roots) == 2:
        w_init, w_bounds = weight_init_and_bounds(gage_id)
        all_init.append(w_init)
        all_bounds.append(w_bounds)
        names.append("tile_weight")
        print(f"Gage {gage_id}: tile_weight mode={WEIGHT_MODE}, init={w_init:.3f}, bounds={w_bounds}")

    print(f"Total: {len(all_init)} init, {len(names)} names")
    if len(all_init) != len(names):
        raise ValueError("Mismatch between total init params and names!")

    dds = DDS(
        all_bounds, n_iterations, gage_id, all_init,
        metric_to_calibrate_on=metric_to_calibrate_on,
        include_nom=any(include_nom_flags),
        nom_file_paths=nom_file_paths,
        param_names=names
    )
    dds.optimize()


def _calibrate_safe(gage_id):
    import traceback
    try:
        calibrate_gage_dds(gage_id)
    except Exception as e:
        print(f" Error calibrating {gage_id}: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    t0 = datetime.now()
    print(f"WEIGHT_MODE={WEIGHT_MODE}  TILE_WEIGHTS_CSV={TILE_WEIGHTS_CSV or '(not set)'}  column={WEIGHT_COLUMN}")
    gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(_calibrate_safe, gage_list, chunksize=1)
    print(f"\n=== Total DDS wall time for all gages: {datetime.now() - t0} ===")
