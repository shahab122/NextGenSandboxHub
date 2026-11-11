####tiled support 
###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | May 2025]
# Calibrates cfe+pet+t-route or cfe+t-route+NOM, if NOM is in the model formulation then some of its parameters will be calibrated
# only supports 1 tile

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

n_iterations = 2
max_cores_for_gages = 2
metric_to_calibrate_on = "kge"

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

        self.observed_path = os.path.join(cfg.observed_q_root, "successful_sites_resampled", f"{self.gage_id}.csv")

    def optimize(self):
        log_path = os.path.join(cfg.logging_dir, f"{self.gage_id}.csv")
        log_rows = []
        num_params = len(self.bounds)
        start_time = datetime.now()

        print(f"\n--- Initial evaluation for gage {self.gage_id} ---")

        obj_val, val_metrics, cal_metrics = objective_function_tiled((
            self.best_position, 0, self.gage_id,
            self.model_roots,
            cfg.observed_q_root,
            [self.include_nom] * self.n_tiles,
            self.nom_file_paths,
            [1.0 / self.n_tiles] * self.n_tiles
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

        for iteration in range(1, self.n_iterations + 1):
            clear_terminal()
            check_for_stop_signal_or_low_disk()
            print(f"\n--- DDS Iteration {iteration} for gage {self.gage_id} ---")

            p = 1 - np.log(iteration) / np.log(self.n_iterations)
            perturb_mask = np.random.rand(num_params) < p
            if not np.any(perturb_mask):
                perturb_mask[np.random.randint(0, num_params)] = True

            candidate = np.copy(self.best_position)
            for i in range(num_params):
                if perturb_mask[i]:
                    low, high = self.bounds[i]
                    perturb = np.random.normal(0, self.sigma) * (high - low)
                    candidate[i] += perturb
                    candidate[i] = max(low, min(high, candidate[i]))

            obj_val, val_metrics, cal_metrics = objective_function_tiled((
                candidate, 0, self.gage_id,
                self.model_roots,
                cfg.observed_q_root,
                [self.include_nom] * self.n_tiles,
                self.nom_file_paths,
                [1.0 / self.n_tiles] * self.n_tiles
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

        # === Final full-period run ===
        print(f"\n[INFO] Running final weighted-routing validation for {self.gage_id}...")

        # Handle weights
        if self.n_tiles == 2:
            tile_weight = self.best_position[-1]
            weights = [tile_weight, 1.0 - tile_weight]
            param_vector = self.best_position[:-1]
        else:
            weights = [1.0 / self.n_tiles] * self.n_tiles
            param_vector = self.best_position

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
            "iteration": "FINAL",
            "particle": "BEST",
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

    # Add tile weight if two tiles
    if len(cfg.model_roots) == 2:
        all_init.append(0.7)             # example initial guess
        all_bounds.append((0.0, 1.0))
        names.append("tile_weight")

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


if __name__ == "__main__":
    gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(calibrate_gage_dds, gage_list)







































#####no tiled support 
# ###############################################################
# # Author      : Peter La Follette [plafollette@lynker.com | May 2025]
# # Calibrates cfe+pet+t-route or cfe+t-route+NOM, if NOM is in the model formulation then some of its parameters will be calibrated
# # only supports 1 tile

# import os
# import pandas as pd
# import numpy as np
# import random
# import json
# import yaml
# from datetime import datetime
# import multiprocessing
# import subprocess
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from model_assessment.configs import path_config as cfg
# from model_assessment.util.metrics import compute_metrics
# from model_assessment.util.update_NOM import update_mptable

# from pso_calibration_cfe import (
#     extract_initial_cfe_params,
#     regenerate_cfe_config,
#     objective_function,
#     param_bounds,
#     param_names,
#     log_scale_params,
#     check_for_stop_signal_or_low_disk,
#     transform_params,
#     nom_param_names,
#     nom_param_bounds,
#     extract_initial_nom_params,
# )

# with open("model_assessment/configs/time_config.yaml", "r") as f:
#     time_cfg = yaml.safe_load(f)

# cal_start = pd.Timestamp(time_cfg["cal_start"])
# cal_end   = pd.Timestamp(time_cfg["cal_end"])
# val_start = pd.Timestamp(time_cfg["val_start"])
# val_end   = pd.Timestamp(time_cfg["val_end"])
# spinup_start = pd.Timestamp(time_cfg["spinup_start"])

# np.random.seed(42)
# random.seed(42)

# n_iterations = 2
# max_cores_for_gages = 2
# metric_to_calibrate_on = "kge"

# def clear_terminal():
#     os.system('cls' if os.name == 'nt' else 'clear')

# def reflect_bounds(x, low, high):
#     if x < low:
#         return low + (low - x)
#     elif x > high:
#         return high - (x - high)
#     else:
#         return x

# class DDS:
#     def __init__(self, bounds, n_iterations, gage_id, config_path, observed_path, postproc_base_path,
#                  init_params, metric_to_calibrate_on="kge", sigma=0.2,
#                  include_nom=False, nom_file_path=None):
#         self.bounds = bounds
#         self.n_iterations = n_iterations
#         self.gage_id = gage_id
#         self.config_path = config_path
#         self.observed_path = observed_path
#         self.postproc_base_path = postproc_base_path
#         self.metric = metric_to_calibrate_on
#         self.sigma = sigma
#         self.best_position = np.copy(init_params)
#         self.best_value = float("inf")

#         self.include_nom = include_nom
#         self.nom_file_path = nom_file_path


#     def optimize(self):
#         log_path = os.path.join(cfg.logging_dir, f"{self.gage_id}.csv")
#         log_rows = []
#         num_params = len(self.bounds)
#         start_time = datetime.now()

#         # === Initial run (spinup + calibration only)
#         print(f"\n--- Initial evaluation (iteration 0) for gage {self.gage_id} ---")
#         obj_val, val_metrics, cal_metrics = objective_function(
#             (self.best_position, 0, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path, self.include_nom, self.nom_file_path)
#         )
#         self.best_value = obj_val
#         best_val_metrics = val_metrics
#         best_cal_metrics = cal_metrics

#         row = {
#             "iteration": 0,
#             "particle": 0,
#             **dict(zip(param_names, self.best_position)),
#             f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
#             f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
#         }

#         log_rows.append(row)

#         # === DDS main loop
#         for iteration in range(1, self.n_iterations + 1):
#             clear_terminal()
#             print(f"\n--- Iteration {iteration} for gage {self.gage_id} ---")

#             ### As I was attempting my first calibration runs with 10s of gages running in parallel and hundreds of iterations on MacOS, I had spotlight indexing on.
#             ### It is actually the case that you can run out of disk space if spotlight indexing is on and it includes outputs from nextgen, because nextgen model outputs amount to a huge amount of data written per day
#             ### To address this, all directories that will contain nextgen outputs at the divide scale, as well as the t-route outputs, will have a .metadata_never_index file created with them during the -conf step in NextGenSandboxHub.
#             ### This should make spotlight indexing skip these files and avoid the issue where the available disk space goes to 0, but just to be sure, this function stops the calibration execution in the event that disk space gets too low 
#             ### if you have the ram to spare, writing the out directory on a ramdisk is a good idea anyway
#             if check_for_stop_signal_or_low_disk():
#                 break

#             p = 1 - np.log(iteration) / np.log(self.n_iterations)
#             perturb_mask = np.random.rand(num_params) < p
#             if not np.any(perturb_mask):
#                 perturb_mask[np.random.randint(0, num_params)] = True

#             candidate = np.copy(self.best_position)
#             for i in range(num_params):
#                 if perturb_mask[i]:
#                     low, high = self.bounds[i]
#                     perturb = np.random.normal(0, self.sigma) * (high - low)
#                     candidate[i] += perturb
#                     candidate[i] = reflect_bounds(reflect_bounds(candidate[i], low, high), low, high)
#                     candidate[i] = np.clip(candidate[i], low, high)

#             obj_val, val_metrics, cal_metrics = objective_function(
#                 (candidate, 0, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path, self.include_nom, self.nom_file_path)
#             )

#             if obj_val < self.best_value:
#                 if hasattr(self, 'include_nom') and self.include_nom:
#                     # extract just the NOM parameters from candidate (last N params)
#                     nom_param_count = len(nom_param_names)
#                     nom_params_only = candidate[-nom_param_count:]
#                     updated_params_dict = dict(zip(nom_param_names, nom_params_only))

#                     update_mptable(
#                         original_file=self.nom_file_path,
#                         output_file=self.nom_file_path,
#                         updated_params=updated_params_dict
#                     )


#                 self.best_value = obj_val
#                 self.best_position = candidate
#                 best_val_metrics = val_metrics
#                 best_cal_metrics = cal_metrics

#             row = {
#                 "iteration": iteration,
#                 "particle": 0,
#                 **dict(zip(param_names, candidate)),
#                 f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
#                 f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
#             }
#             log_rows.append(row)

#         # === Final full-period run ===
#         print(f"\nRunning final full-period validation for {self.gage_id}...")
#         from pso_calibration_cfe import transform_params
#         true_best_params = transform_params(self.best_position)
#         regenerate_cfe_config(self.config_path, true_best_params)
#         if hasattr(self, 'include_nom') and self.include_nom:
#             nom_param_count = len(nom_param_names)
#             nom_params_only = true_best_params[-nom_param_count:]
#             updated_params_dict = dict(zip(nom_param_names, nom_params_only))

#             update_mptable(
#                 original_file=self.nom_file_path,
#                 output_file=self.nom_file_path,
#                 updated_params=updated_params_dict
#             )


#         json_dir = os.path.join(cfg.model_roots[0], "out", self.gage_id, "json")
#         realization_path = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#         realization_path = os.path.join(json_dir, realization_path)

#         with open(realization_path, "r") as f:
#             realization_config = json.load(f)
#         realization_config["time"]["start_time"] = time_cfg["spinup_start"]
#         realization_config["time"]["end_time"]   = time_cfg["val_end"]
#         with open(realization_path, "w") as f:
#             json.dump(realization_config, f, indent=4)

#         troute_path = os.path.join(cfg.model_roots[0], "out", self.gage_id, "configs", "troute_config.yaml")
#         with open(troute_path, "r") as f:
#             troute_cfg = yaml.safe_load(f)
#         nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
#         troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#         troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
#         with open(troute_path, "w") as f:
#             yaml.dump(troute_cfg, f)

#         subprocess.call(["python", "sandbox.py", "-run", "--gage_id", self.gage_id], cwd=cfg.project_root)

#         get_hydrograph_path = os.path.join(cfg.project_root, "model_assessment", "util", "get_hydrograph.py")
#         output_path = os.path.join(self.postproc_base_path, f"{self.gage_id}_best.csv")
#         subprocess.call([
#             "python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", output_path, "--base_dir", cfg.model_roots[0]
#         ], cwd=self.postproc_base_path)

#         sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#         obs_df = pd.read_csv(self.observed_path, parse_dates=['value_time']).set_index('value_time')['flow_m3_per_s']
#         sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#         sim_val, obs_val = sim_val.align(obs_val, join='inner')
#         val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#         summary_row = {
#             "iteration": "FINAL",
#             "particle": "BEST",
#             **dict(zip(param_names, self.best_position)),
#             f"{self.metric}_calibration": best_cal_metrics.get(self.metric, np.nan),
#             f"{self.metric}_validation": val_metrics_final.get(self.metric, np.nan),
#         }
#         log_rows.append(summary_row)
#         pd.DataFrame(log_rows).to_csv(log_path, index=False)
#         print(f"\n Finished DDS for gage {self.gage_id}. Best obj = {-self.best_value:.4f} | Time: {datetime.now() - start_time}")
#         return self.best_position, self.best_value, datetime.now() - start_time

# def calibrate_gage_dds(gage_id):
#     global param_names  # must come first before any use

#     model_root = cfg.model_roots[0]
#     config_dir = os.path.join(model_root, f"out/{gage_id}/configs/cfe")
#     config_path = os.path.join(config_dir, sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0])
#     observed_path = os.path.join(cfg.observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
#     postproc_path = os.path.join(model_root, "postproc")

#     init_params = extract_initial_cfe_params(config_path)
#     bounds = param_bounds.copy()
#     names = param_names.copy()

#     # === Check for NOM parameters
#     nom_config_dir = os.path.join(model_root, f"out/{gage_id}/configs/noahowp")
#     include_nom = os.path.isdir(nom_config_dir)

#     nom_file_path = None
#     if include_nom:
#         nom_file_path = os.path.join(nom_config_dir, "parameters", "MPTABLE.TBL")
#         nom_init_vals = extract_initial_nom_params(nom_file_path)
#         init_params += nom_init_vals
#         bounds += nom_param_bounds
#         names += nom_param_names

#     param_names = names  # redefine after global

#     dds = DDS(bounds, n_iterations, gage_id, config_path, observed_path, postproc_path, init_params,
#             metric_to_calibrate_on=metric_to_calibrate_on,
#             include_nom=include_nom,
#             nom_file_path=nom_file_path)

#     # attach for use in final run
#     dds.include_nom = include_nom
#     dds.nom_file_path = nom_file_path

#     dds.optimize()

# if __name__ == "__main__":
#     gages_file = os.path.join(cfg.project_root, "basin_IDs/basin_IDs.csv")
#     gage_list = pd.read_csv(gages_file, dtype={"gage_id": str})["gage_id"].tolist()

#     ctx = multiprocessing.get_context("spawn")
#     with ctx.Pool(processes=max_cores_for_gages) as pool:
#         pool.map(calibrate_gage_dds, gage_list)











































###does not calibrate NOM 
# ###############################################################
# # Author      : Peter La Follette [plafollette@lynker.com | May 2025]
# # Calibrates cfe+pet+t-route or cfe+t-route+NOM, if NOM is in the model formulation then some of its parameters will be calibrated

# import os
# import pandas as pd
# import numpy as np
# import random
# import json
# import yaml
# from datetime import datetime
# import multiprocessing
# import subprocess
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from model_assessment.configs import path_config as cfg
# from model_assessment.util.metrics import compute_metrics
# from model_assessment.util.update_NOM import update_mptable

# from pso_calibration_cfe import (
#     check_for_stop_signal_or_low_disk,
#     extract_initial_cfe_params,
#     regenerate_cfe_config,
#     objective_function,
#     param_bounds,
#     param_names,
#     log_scale_params,
# )



# with open("model_assessment/configs/time_config.yaml", "r") as f:
#     time_cfg = yaml.safe_load(f)

# cal_start = pd.Timestamp(time_cfg["cal_start"])
# cal_end   = pd.Timestamp(time_cfg["cal_end"])
# val_start = pd.Timestamp(time_cfg["val_start"])
# val_end   = pd.Timestamp(time_cfg["val_end"])
# spinup_start = pd.Timestamp(time_cfg["spinup_start"])

# np.random.seed(42)
# random.seed(42)

# n_iterations = 2
# max_cores_for_gages = 2
# metric_to_calibrate_on = "kge"

# def clear_terminal():
#     os.system('cls' if os.name == 'nt' else 'clear')

# def reflect_bounds(x, low, high):
#     if x < low:
#         return low + (low - x)
#     elif x > high:
#         return high - (x - high)
#     else:
#         return x

# class DDS:
#     def __init__(self, bounds, n_iterations, gage_id, config_path, observed_path, postproc_base_path,
#                  init_params, metric_to_calibrate_on="kge", sigma=0.2):
#         self.bounds = bounds
#         self.n_iterations = n_iterations
#         self.gage_id = gage_id
#         self.config_path = config_path
#         self.observed_path = observed_path
#         self.postproc_base_path = postproc_base_path
#         self.metric = metric_to_calibrate_on
#         self.sigma = sigma
#         self.best_position = np.copy(init_params)
#         self.best_value = float("inf")

#     def optimize(self):
#         log_path = os.path.join(cfg.logging_dir, f"{self.gage_id}.csv")
#         log_rows = []
#         num_params = len(self.bounds)
#         start_time = datetime.now()

#         # === Initial run (spinup + calibration only)
#         print(f"\n--- Initial evaluation (iteration 0) for gage {self.gage_id} ---")
#         obj_val, val_metrics, cal_metrics = objective_function(
#             (self.best_position, 0, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path)
#         )
#         self.best_value = obj_val
#         best_val_metrics = val_metrics
#         best_cal_metrics = cal_metrics

#         row = {
#             "iteration": 0,
#             "particle": 0,
#             **dict(zip(param_names, self.best_position)),
#             f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
#             f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
#         }

#         log_rows.append(row)

#         # === DDS main loop
#         for iteration in range(1, self.n_iterations + 1):
#             # clear_terminal()
#             print(f"\n--- Iteration {iteration} for gage {self.gage_id} ---")

#             ### As I was attempting my first calibration runs with 10s of gages running in parallel and hundreds of iterations on MacOS, I had spotlight indexing on.
#             ### It is actually the case that you can run out of disk space if spotlight indexing is on and it includes outputs from nextgen, because nextgen model outputs amount to a huge amount of data written per day
#             ### To address this, all directories that will contain nextgen outputs at the divide scale, as well as the t-route outputs, will have a .metadata_never_index file created with them during the -conf step in NextGenSandboxHub.
#             ### This should make spotlight indexing skip these files and avoid the issue where the available disk space goes to 0, but just to be sure, this function stops the calibration execution in the event that disk space gets too low 
#             if check_for_stop_signal_or_low_disk():
#                 break

#             p = 1 - np.log(iteration) / np.log(self.n_iterations)
#             perturb_mask = np.random.rand(num_params) < p
#             if not np.any(perturb_mask):
#                 perturb_mask[np.random.randint(0, num_params)] = True

#             candidate = np.copy(self.best_position)
#             for i in range(num_params):
#                 if perturb_mask[i]:
#                     low, high = self.bounds[i]
#                     perturb = np.random.normal(0, self.sigma) * (high - low)
#                     candidate[i] += perturb
#                     candidate[i] = reflect_bounds(reflect_bounds(candidate[i], low, high), low, high)
#                     candidate[i] = np.clip(candidate[i], low, high)

#             obj_val, val_metrics, cal_metrics = objective_function(
#                 (candidate, 0, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path)
#             )

#             if obj_val < self.best_value:
#                 self.best_value = obj_val
#                 self.best_position = candidate
#                 best_val_metrics = val_metrics
#                 best_cal_metrics = cal_metrics

#             row = {
#                 "iteration": iteration,
#                 "particle": 0,
#                 **dict(zip(param_names, candidate)),
#                 f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
#                 f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
#             }
#             log_rows.append(row)

#         # === Final full-period run ===
#         print(f"\nRunning final full-period validation for {self.gage_id}...")
#         from pso_calibration_cfe import transform_params
#         true_best_params = transform_params(self.best_position)
#         regenerate_cfe_config(self.config_path, true_best_params)

#         json_dir = os.path.join(cfg.model_roots[0], "out", self.gage_id, "json")
#         realization_path = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#         realization_path = os.path.join(json_dir, realization_path)

#         with open(realization_path, "r") as f:
#             realization_config = json.load(f)
#         realization_config["time"]["start_time"] = time_cfg["spinup_start"]
#         realization_config["time"]["end_time"]   = time_cfg["val_end"]
#         with open(realization_path, "w") as f:
#             json.dump(realization_config, f, indent=4)

#         troute_path = os.path.join(cfg.model_roots[0], "out", self.gage_id, "configs", "troute_config.yaml")
#         with open(troute_path, "r") as f:
#             troute_cfg = yaml.safe_load(f)
#         nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
#         troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#         troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
#         with open(troute_path, "w") as f:
#             yaml.dump(troute_cfg, f)

#         subprocess.call(["python", "sandbox.py", "-run", "--gage_id", self.gage_id], cwd=cfg.project_root)

#         get_hydrograph_path = os.path.join(cfg.project_root, "model_assessment", "util", "get_hydrograph.py")
#         output_path = os.path.join(self.postproc_base_path, f"{self.gage_id}_best.csv")
#         subprocess.call([
#             "python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", output_path, "--base_dir", cfg.model_roots[0]
#         ], cwd=self.postproc_base_path)

#         sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#         obs_df = pd.read_csv(self.observed_path, parse_dates=['value_time']).set_index('value_time')['flow_m3_per_s']
#         sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#         sim_val, obs_val = sim_val.align(obs_val, join='inner')
#         val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#         summary_row = {
#             "iteration": "FINAL",
#             "particle": "BEST",
#             **dict(zip(param_names, self.best_position)),
#             f"{self.metric}_calibration": best_cal_metrics.get(self.metric, np.nan),
#             f"{self.metric}_validation": val_metrics_final.get(self.metric, np.nan),
#         }
#         log_rows.append(summary_row)
#         pd.DataFrame(log_rows).to_csv(log_path, index=False)
#         print(f"\n Finished DDS for gage {self.gage_id}. Best obj = {-self.best_value:.4f} | Time: {datetime.now() - start_time}")
#         return self.best_position, self.best_value, datetime.now() - start_time

# def calibrate_gage_dds(gage_id):
#     model_root = cfg.model_roots[0]
#     config_dir = os.path.join(model_root, f"out/{gage_id}/configs/cfe")
#     config_path = os.path.join(config_dir, sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0])
#     observed_path = os.path.join(cfg.observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
#     postproc_path = os.path.join(model_root, "postproc")

#     init_params = extract_initial_cfe_params(config_path)
#     dds = DDS(param_bounds, n_iterations, gage_id, config_path, observed_path, postproc_path, init_params)
#     dds.optimize()

# if __name__ == "__main__":
#     gages_file = os.path.join(cfg.project_root, "out/basins_passed_custom.csv")
#     gage_list = pd.read_csv(gages_file, dtype={"gage_id": str})["gage_id"].tolist()

#     ctx = multiprocessing.get_context("spawn")
#     with ctx.Pool(processes=max_cores_for_gages) as pool:
#         pool.map(calibrate_gage_dds, gage_list)


