###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# Calibrates lasam+pet+t-route or lasam+t-route+NOM, if NOM is in the model formulation then some of its parameters will be calibrated
# currently tiled formulations are supported (up to 2 tiles), however each tile will be a different instance of lasam+pet+t-route or lasam+t-route+NOM rather than able to be based on cfe

import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from hydroeval import kge
import random
import multiprocessing
import sys
import traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_assessment.util.metrics import compute_metrics
from model_assessment.util.update_NOM import update_mptable
import yaml
import json
import shutil 

np.random.seed(42)
random.seed(42)

# === CONFIGURATION ===
n_particles = 2 #15
n_iterations = 2 #50 #so total number of model runs at a gage will be equal to n_particles*n_iterations
max_cores_for_gages = 10 #11  # How many gages to calibrate in parallel. I recommend that this should be the number of cores your machine has or that number - 1.


from model_assessment.configs import path_config as cfg
with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

spinup_start = pd.Timestamp(time_cfg["spinup_start"])
cal_start    = pd.Timestamp(time_cfg["cal_start"])
cal_end      = pd.Timestamp(time_cfg["cal_end"])
val_start    = pd.Timestamp(time_cfg["val_start"])
val_end      = pd.Timestamp(time_cfg["val_end"])

project_root = cfg.project_root
logging_dir = cfg.logging_dir
sandbox_path = cfg.sandbox_path
observed_q_root = cfg.observed_q_root

def clear_terminal(): #this runs regularly because verbose output from nextgen can be a lot
    os.system('clear')  # 'cls' for Windows

### As I was attempting my first calibration runs with 10s of gages running in parallel and hundreds of iterations on MacOS, I had spotlight indexing on.
### It is actually the case that you can run out of disk space if spotlight indexing is on and it includes outputs from nextgen, because nextgen model outputs amount to a huge amount of data written per day
### To address this, all directories that will contain nextgen outputs at the divide scale, as well as the t-route outputs, will have a .metadata_never_index file created with them during the -conf step in NextGenSandboxHub.
### This should make spotlight indexing skip these files and avoid the issue where the available disk space goes to 0, but just to be sure, this function stops the calibration execution in the event that disk space gets too low 
### if you have the ram to spare, writing the out directory on a ramdisk is a good idea anyway
def check_for_stop_signal_or_low_disk(project_root, threshold_gb=50):
    # Check for stop file
    stop_file = os.path.join(project_root, "STOP_NOW.txt")
    if os.path.exists(stop_file):
        print("Detected STOP_NOW.txt")
        sys.exit(1)
        # return True

    # Check available disk space using statvfs
    stat = os.statvfs("/")
    free_bytes = stat.f_frsize * stat.f_bavail
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < threshold_gb:
        print(f"Free disk space below threshold: {free_gb:.2f} GB")
        sys.exit(1)
        # return True

    return False


def get_observed_q(observed_path):
    df = pd.read_csv(observed_path, parse_dates=['value_time'])
    df.set_index('value_time', inplace=True)
    
    # warning if data isn't hourly
    expected_freq = pd.infer_freq(df.index)
    if expected_freq != 'h':
        print(f"Warning: Observed data in {observed_path} is not hourly (inferred freq: {expected_freq})")
    
    return df['flow_m3_per_s']

def extract_tile_params(full_params, tile_idx, n_tiles):
    full_params = np.array(full_params)
    total_len = len(full_params)

    # Special case: weight parameter at the end
    if n_tiles == 2 and total_len % 2 == 1:
        param_slice = full_params[:-1]  # Exclude weight
        chunk_size = len(param_slice) // n_tiles
        return param_slice[tile_idx * chunk_size : (tile_idx + 1) * chunk_size]

    # Standard case
    if total_len % n_tiles != 0:
        raise ValueError(f"Parameter vector length ({total_len}) not divisible by number of tiles ({n_tiles})")
    chunk_size = total_len // n_tiles
    return full_params[tile_idx * chunk_size : (tile_idx + 1) * chunk_size]

# extracts parameters from lasam and NOM config files or tables 
def extract_initial_params(example_config_path):
    with open(example_config_path) as f:
        lines = f.readlines()

    soil_types_line = next(line for line in lines if line.startswith("layer_soil_type="))
    soil_types = list(map(int, soil_types_line.strip().split("=")[1].split(",")))
    num_layers = len(soil_types)

    a = float(next(line.split("=")[1] for line in lines if line.startswith("a=")))
    b = float(next(line.split("=")[1] for line in lines if line.startswith("b=")))
    frac_to_GW = float(next(line.split("=")[1] for line in lines if line.startswith("frac_to_GW=")))
    field_capacity_psi = float(next(line.split("=")[1].split("[")[0] for line in lines if line.startswith("field_capacity_psi=")))
    spf_factor = float(next(line.split("=")[1] for line in lines if line.startswith("spf_factor=")))

    soil_file = next(line.split("=")[1].strip() for line in lines if line.startswith("soil_params_file="))
    with open(soil_file) as f:
        soil_lines = f.readlines()

    layer_params = []
    for soil_type in soil_types:
        tokens = soil_lines[soil_type].split()
        alpha = float(tokens[3])
        n = float(tokens[4])
        Ks = float(tokens[5])
        layer_params.extend([
            np.log10(alpha),
            n,
            np.log10(Ks)
        ])

    first_soil_type = soil_types[0]
    first_layer_tokens = soil_lines[first_soil_type].split()
    theta_e_1 = float(first_layer_tokens[2])  # Assuming column 2 is theta_e

    # === NOM parameter extraction (new logic)
    config_root = os.path.dirname(os.path.dirname(example_config_path))  # trims off /lasam
    nom_dir = os.path.join(config_root, "noahowp")
    nom_params = []

    if os.path.isdir(nom_dir):
        mptable_path = os.path.join(nom_dir, "parameters", "MPTABLE.TBL")
        # print(f"Reading MPTABLE.TBL from: {mptable_path}")
        try:
            with open(mptable_path) as f:
                lines = f.readlines()

            nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
            nom_params_dict = {}

            for line in lines:
                if '=' not in line or line.strip().startswith('!'):
                    continue
                key, value = line.split('=', 1)
                param_name = key.strip()
                if param_name in nom_param_names:
                    value_str = value.split('!')[0]
                    values = [v.strip() for v in value_str.split(',') if v.strip()]
                    nom_params_dict[param_name] = float(values[0])

            if set(nom_params_dict.keys()) != set(nom_param_names):
                raise ValueError(f"Found NOM parameters {list(nom_params_dict.keys())}, expected {nom_param_names}. Check formatting in {mptable_path}")

            nom_params = [nom_params_dict[p] for p in nom_param_names]

        except Exception as e:
            print(f"Failed to parse NOM parameters from {mptable_path}: {e}")

    return layer_params + [a, b, frac_to_GW, field_capacity_psi, spf_factor, theta_e_1] + nom_params


#
def objective_function_tiled(args, metric_to_calibrate_on="kge", base_roots=None, include_nom=False, n_tiles=2, weights=None):
    params, particle_idx, gage_id, observed_q_root = args

    if base_roots is None or len(base_roots) != n_tiles:
        raise ValueError("Must provide one project root per tile")
    if weights is None:
        weights = [0.7, 0.3] if n_tiles == 2 else [1.0 / n_tiles] * n_tiles
    assert len(weights) == n_tiles

    obs_path = os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
    obs_series = get_observed_q(obs_path)

    # === CLEAR PREVIOUS RESULTS ===
    for tile_root in cfg.model_roots:
        outputs_dir = os.path.join(tile_root, "out", gage_id, "outputs")
        div_dir = os.path.join(outputs_dir, "div")
        div_weighted_dir = os.path.join(outputs_dir, "div_weighted")

        for target_dir in [div_dir, div_weighted_dir]:
            if os.path.isdir(target_dir):
                for item in os.listdir(target_dir):
                    if item.startswith("."):  # preserve hidden files like .metadata_never_index
                        continue
                    full_path = os.path.join(target_dir, item)
                    if os.path.isfile(full_path) or os.path.islink(full_path):
                        os.remove(full_path)
                    elif os.path.isdir(full_path):
                        shutil.rmtree(full_path)


        # Clear routed NetCDF files
        troute_dir = os.path.join(tile_root, "out", gage_id, "troute")
        if os.path.isdir(troute_dir):
            for f in os.listdir(troute_dir):
                if f.endswith(".nc"):
                    os.remove(os.path.join(troute_dir, f))

        # Clear postprocessing particle files
        postproc_dir = os.path.join(tile_root, "postproc")
        if os.path.isdir(postproc_dir):
            for f in os.listdir(postproc_dir):
                if f.startswith(f"{gage_id}_particle_") and f.endswith(".csv"):
                    os.remove(os.path.join(postproc_dir, f))

    try:
        # === STEP 1: Run hydrology per tile ===
        for tile in range(n_tiles):
            tile_params = extract_tile_params(params, tile, n_tiles)
            tile_root = base_roots[tile]

            config_dir = os.path.join(tile_root, f"out/{gage_id}/configs/lasam")
            config_files = sorted(f for f in os.listdir(config_dir) if f.startswith("lasam_config_cat"))
            for fname in config_files:
                update_lasam_files_for_divide(os.path.join(config_dir, fname), tile_params, include_nom)

            if include_nom:
                nom_template = os.path.join(tile_root, f"out/{gage_id}/configs/noahowp/parameters/MPTABLE.TBL")
                update_mptable(
                    original_file=nom_template,
                    output_file=nom_template,
                    updated_params={
                        "MFSNO": tile_params[-6],
                        "RSURF_SNOW": tile_params[-5],
                        "HVT": tile_params[-4],
                        "CWPVT": tile_params[-3],
                        "VCMX25": tile_params[-2],
                        "MP": tile_params[-1]
                    },
                    verbose=False
                )

            # Update realization.json for hydrology window
            json_dir = os.path.join(tile_root, "out", gage_id, "json")
            realization_path = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
            realization_path = os.path.join(json_dir, realization_path)

            with open(realization_path, "r") as f:
                realization_config = json.load(f)
            realization_config["time"]["start_time"] = spinup_start.strftime("%Y-%m-%d %H:%M:%S")
            realization_config["time"]["end_time"] = cal_end.strftime("%Y-%m-%d %H:%M:%S")
            with open(realization_path, "w") as f:
                json.dump(realization_config, f, indent=4)

            # Run hydrology only (divide scale)
            subprocess.call([
                "python", sandbox_path,
                "-i", os.path.join(os.environ.get("CONFIG_DIR", os.path.join(cfg.project_root, "configs")), f"sandbox_config_tile{tile+1}.yaml"),
                "-run",
                "--gage_id", gage_id
            ], cwd=tile_root)


        # === STEP 2: Weighted averaging of divide outputs ===
        if n_tiles == 1:
            # Skip weighting to save on I/O: use tile 0's existing divide outputs directly
            weighted_div_dir = os.path.join(base_roots[0], "out", gage_id, "outputs", "div")
        else:
            # Perform weighted averaging for multiple tiles
            weighted_div_dir = os.path.join(base_roots[0], "out", gage_id, "outputs", "div_weighted")
            if os.path.exists(weighted_div_dir):
                shutil.rmtree(weighted_div_dir)  # Explicitly clear old weighted outputs
            os.makedirs(weighted_div_dir, exist_ok=True)

            div_dirs = [os.path.join(root, "out", gage_id, "outputs", "div") for root in base_roots]
            files = [f for f in os.listdir(div_dirs[0]) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]

            for fname in files:
                dfs = []
                for tile_idx, div_dir in enumerate(div_dirs):
                    fpath = os.path.join(div_dir, fname)
                    if os.path.exists(fpath):
                        if fname.startswith("nex-"):
                            df = pd.read_csv(fpath, header=None)
                            df.columns = ["Time Step", "Time", "q_out"]
                        else:
                            df = pd.read_csv(fpath)
                        dfs.append(df["q_out"] * weights[tile_idx])
                if dfs:
                    combined = sum(dfs)
                    out_df = df.copy()
                    out_df["q_out"] = combined
                    out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

                    if fname.startswith("nex-"):
                        out_df.to_csv(os.path.join(weighted_div_dir, fname), index=False, header=False)
                    else:
                        out_df.to_csv(os.path.join(weighted_div_dir, fname), index=False)



        # === STEP 3: Update troute_config.yaml ===
        troute_path = os.path.join(base_roots[0], "out", gage_id, "configs", "troute_config.yaml")
        with open(troute_path, "r") as f:
            troute_cfg = yaml.safe_load(f)
        if n_tiles > 1:
            troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir
        nts = int((cal_end - spinup_start) / pd.Timedelta(seconds=300))
        troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
        troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts
        with open(troute_path, "w") as f:
            yaml.dump(troute_cfg, f)

        # === STEP 4: Run routing ===
        subprocess.call(["python3", "-m", "nwm_routing", "-f", "-V4", troute_path])

        # === STEP 5: Extract routed hydrograph ===
        postproc_dir = os.path.join(base_roots[0], "postproc")
        output_filename = f"{gage_id}_particle_{particle_idx}.csv"
        output_path = os.path.join(postproc_dir, output_filename)
        get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
        subprocess.call(["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", base_roots[0]])

        sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')

        # === STEP 6: Robust simulation length check ===
        window_start = spinup_start
        window_end = cal_end  # Will be val_end if in validation mode
        expected_length = int((window_end - window_start) / pd.Timedelta(hours=1)) + 1
        actual_length = len(sim_df)
        allowed_tolerance = 1
        incomplete_run = abs(actual_length - expected_length) > allowed_tolerance

        if incomplete_run:
            print(f"[WARN] Incomplete routed hydrograph for {gage_id}: expected {expected_length}, got {actual_length}")

        # === STEP 7: Compute metrics ===
        avg_sim = sim_df['flow'].resample('1h').mean()
        sim_cal, obs_cal = avg_sim[cal_start:cal_end].dropna(), obs_series[cal_start:cal_end].dropna()
        sim_val, obs_val = avg_sim[val_start:val_end].dropna(), obs_series[val_start:val_end].dropna()
        sim_cal, obs_cal = sim_cal.align(obs_cal, join='inner')
        sim_val, obs_val = sim_val.align(obs_val, join='inner')

        if len(sim_cal) > 0: sim_cal.iloc[-1] += 1e-8; obs_cal.iloc[-1] += 1e-8
        if len(sim_val) > 0: sim_val.iloc[-1] += 1e-8; obs_val.iloc[-1] += 1e-8

        cal_metrics = compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)
        val_metrics = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

        # Log incomplete run if necessary
        if incomplete_run:
            failed_row = {
                "gage_id": gage_id,
                "particle_idx": particle_idx,
                "params": params.tolist() if hasattr(params, "tolist") else list(params),
                "error_message": f"Incomplete routed hydrograph: expected {expected_length}, got {actual_length}",
                "cal_kge": cal_metrics.get("kge", np.nan),
                "val_kge": val_metrics.get("kge", np.nan)
            }
            log_path = os.path.join(logging_dir, "incomplete_simulations.csv")
            pd.DataFrame([failed_row]).to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

        return -cal_metrics[metric_to_calibrate_on], val_metrics[metric_to_calibrate_on], cal_metrics, val_metrics

    except Exception as e:
        print(f"Error in tiled objective function for {gage_id}: {e}")

        # === LOG the failed particle ===
        failed_row = {
            "gage_id": gage_id,
            "particle_idx": particle_idx,
            "params": params.tolist() if hasattr(params, "tolist") else list(params),
            "error_message": str(e)
        }
        log_path = os.path.join(logging_dir, "incomplete_simulations.csv")
        df = pd.DataFrame([failed_row])
        if os.path.exists(log_path):
            df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(log_path, index=False)

        placeholder = {
            "kge": -10.0, "log_kge": -10.0, "volume_error_percent": -999.0,
            "peak_time_error_hours": -999.0, "peak_flow_error_percent": -999.0,
            "event_kge": -10.0, "event_hours": 0, "total_hours": 0
        }
        return 10.0, -10.0, placeholder, placeholder



def run_validation_with_best(
    gage_id, logging_dir, observed_q_root, base_roots, best_params,
    n_tiles=2, include_nom=False, weights=None, metric_to_calibrate_on="kge"):
    global cal_end
    original_cal_end = cal_end
    cal_end = val_end  # Extend to validation window

    print(f"\nRunning final validation for gage {gage_id}...")

    # Run objective function for validation
    args = (best_params, "best", gage_id, observed_q_root)
    _, _, cal_metrics, val_metrics = objective_function_tiled(
        args,
        metric_to_calibrate_on=metric_to_calibrate_on,
        base_roots=base_roots,
        include_nom=include_nom,
        n_tiles=n_tiles,
        weights=weights
    )

    # === Copy final routed hydrograph ===
    postproc_dir = os.path.join(base_roots[0], "postproc")
    final_src = os.path.join(postproc_dir, f"{gage_id}_particle_best.csv")
    final_dst = os.path.join(postproc_dir, f"{gage_id}_best.csv")

    try:
        if os.path.exists(final_src):
            shutil.copy(final_src, final_dst)
            print(f"Final routed hydrograph saved to: {final_dst}")
        else:
            print(f"[WARN] Final routed hydrograph not found at {final_src}")
    except Exception as e:
        print(f"[ERROR] Could not save final routed hydrograph: {e}")

    cal_end = original_cal_end  # Restore calibration window
    return val_metrics



# === UPDATE LASAM FILES ===
def update_lasam_files_for_divide(config_path, params, include_nom):
    with open(config_path, 'r') as f:
        lines = f.readlines()

    # === Identify soil param file and soil types
    soil_file = None
    soil_types = []
    for line in lines:
        if line.startswith("soil_params_file="):
            soil_file = line.split("=")[1].strip()
        elif line.startswith("layer_soil_type="):
            soil_types = list(map(int, line.strip().split("=")[1].split(",")))

    num_layers = len(soil_types)

    # === Extract LASAM parameter slice
    if include_nom:
        lasam_slice = params[-12:-6]
    else:
        lasam_slice = params[-6:]

    log_a, b, frac_to_GW, field_capacity_psi, spf_factor, theta_e_1 = lasam_slice
    a = 10 ** log_a

    # debug log
    # print(f"LASAM params being written: a={a:.4e}, b={b}, frac_to_GW={frac_to_GW}, spf_factor={spf_factor}, theta_e_1={theta_e_1}, field_capacity_psi={field_capacity_psi}")

    # === Update config file (in-place)
    for i, line in enumerate(lines):
        if line.startswith("a="):
            lines[i] = f"a={a}\n"
        elif line.startswith("b="):
            lines[i] = f"b={b}\n"
        elif line.startswith("frac_to_GW="):
            lines[i] = f"frac_to_GW={frac_to_GW}\n"
        elif line.startswith("field_capacity_psi="):
            lines[i] = f"field_capacity_psi={field_capacity_psi}[cm]\n"
        elif line.startswith("spf_factor="):
            lines[i] = f"spf_factor={spf_factor}\n"

    with open(config_path, 'w') as f:
        f.writelines(lines)

    # === Update soil file ===
    if soil_file:
        with open(soil_file, 'r') as f:
            soil_lines = f.readlines()

        for i, soil_type in enumerate(soil_types):
            start = i * 3
            log_alpha, n, log_Ks = params[start:start + 3]
            alpha = 10 ** log_alpha
            Ks = 10 ** log_Ks
            tokens = soil_lines[soil_type].split()

            if i == 0:
                tokens[2] = str(theta_e_1)  # Only top layer

            tokens[3] = str(alpha)
            tokens[4] = str(n)
            tokens[5] = str(Ks)
            soil_lines[soil_type] = "\t".join(tokens) + "\n"

        with open(soil_file, 'w') as f:
            f.writelines(soil_lines)

# === PSO ===
class Particle:
    def __init__(self, bounds, init_position=None):
        if init_position is not None:
            self.position = np.array(init_position)
        else:
            self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([(high - low) * 0.1 * np.random.uniform(-1, 1) for low, high in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        self.current_value = float('inf')
        self.stagnation_counter = 0  # Tracks how many iterations without improvement

    def reset(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([(high - low) * 0.1 * np.random.uniform(-1, 1) for low, high in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        self.current_value = float('inf')
        self.stagnation_counter = 0


    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.random(len(self.position))
        r2 = np.random.random(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(len(self.position)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]; self.velocity[i] *= -0.5
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]; self.velocity[i] *= -0.5

from scipy.optimize import minimize  # Add this at the top if not already present

class PSO:
    def __init__(self, n_particles, bounds, n_iterations, gage_id,
                 init_params, base_roots, observed_q_root,
                 metric_to_calibrate_on="kge", w=0.9, c1=1.5, c2=1.5,
                 include_nom=False, n_tiles=2, weights=None):

        self.n_particles = n_particles
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.gage_id = gage_id
        self.include_nom = include_nom
        self.observed_q_root = observed_q_root
        self.base_roots = base_roots
        self.metric_to_calibrate_on = metric_to_calibrate_on
        self.weights = weights if weights is not None else [1.0 / n_tiles] * n_tiles
        self.n_tiles = n_tiles

        self.particles = [Particle(bounds, init_position=init_params if i == 0 else None) for i in range(n_particles)]
        self.global_best_position = self.particles[0].position
        self.global_best_value = float('inf')
        self.c1, self.c2 = c1, c2

    ###
    def optimize(self):
        from datetime import datetime
        best_calibration_metric = None
        best_validation_metric = None

        start_time = datetime.now()
        log_rows = []
        log_path = os.path.join(project_root, "logging", f"{self.gage_id}.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        stagnation_threshold = 10
        w_start, w_end = 0.9, 0.4  # Inertia decay range

        for iteration in range(self.n_iterations):
            if check_for_stop_signal_or_low_disk(project_root, threshold_gb=50):
                print(f"Early termination of PSO for gage {self.gage_id} at iteration {iteration + 1}")
                break

            clear_terminal()
            print(f"\n--- Iteration {iteration + 1} for gage {self.gage_id} ---")

            w = w_start - (w_start - w_end) * (iteration / self.n_iterations)

            results = []
            ###
            for i, p in enumerate(self.particles):
                if check_for_stop_signal_or_low_disk(project_root, threshold_gb=50):
                    print(f"Early termination of PSO for gage {self.gage_id} during particle {i} at iteration {iteration + 1}")
                    return self.global_best_position, self.global_best_value, best_validation_metric, datetime.now() - start_time

                full_position = p.position
                if self.n_tiles == 2:
                    weight_tile0 = full_position[-1]
                    weights = [weight_tile0, 1.0 - weight_tile0]
                    param_slice = full_position
                else:
                    weights = self.weights  # fallback
                    param_slice = full_position

                args = (param_slice, i, self.gage_id, self.observed_q_root)
                result = objective_function_tiled(
                    args,
                    metric_to_calibrate_on=self.metric_to_calibrate_on,
                    base_roots=self.base_roots,
                    include_nom=self.include_nom,
                    n_tiles=self.n_tiles,
                    weights=weights
                )
                results.append(result)



            global_best_particle_idx = np.argmin([r[0] for r in results])

            for idx, (objective_value, val_metric, cal_metrics, val_metrics) in enumerate(results):
                particle = self.particles[idx]
                particle.current_value = objective_value

                if objective_value < particle.best_value:
                    particle.best_value = objective_value
                    particle.best_position = np.copy(particle.position)
                    particle.stagnation_counter = 0
                else:
                    particle.stagnation_counter += 1

                if objective_value < self.global_best_value:
                    self.global_best_value = objective_value
                    self.global_best_position = np.copy(particle.position)
                    best_calibration_metric = cal_metrics[self.metric_to_calibrate_on]
                    best_validation_metric = val_metrics[self.metric_to_calibrate_on]
                    best_cal_metrics = cal_metrics

                    # Save best hydrograph live
                    try:
                        best_path_live = os.path.join(self.base_roots[0], "postproc", f"{self.gage_id}_particle_{idx}.csv")
                        if os.path.exists(best_path_live):
                            live_best_dst = os.path.join(self.base_roots[0], "postproc", f"{self.gage_id}_best.csv")
                            shutil.copy(best_path_live, live_best_dst)
                            print(f"Live best routed hydrograph saved: {live_best_dst}")
                        else:
                            print(f"[WARN] Could not find routed hydrograph for best particle {idx}")
                    except Exception as e:
                        print(f"Could not update best hydrograph for {self.gage_id} (particle {idx}): {e}")


                if particle.stagnation_counter >= stagnation_threshold and idx != global_best_particle_idx:
                    print(f"Resetting particle {idx} after {stagnation_threshold} stagnant iterations.")
                    particle.reset(self.bounds)

                param_dict = {}
                full_params = particle.position
                total_layers = 0
                for tile in range(self.n_tiles):
                    tile_params = extract_tile_params(full_params, tile, self.n_tiles)
                    config_dir = os.path.join(self.base_roots[tile], f"out/{self.gage_id}/configs/lasam")
                    config_file = sorted(f for f in os.listdir(config_dir) if f.startswith("lasam_config_cat"))[0]
                    config_path = os.path.join(config_dir, config_file)

                    with open(config_path) as f:
                        lines = f.readlines()
                    soil_type_line = next(line for line in lines if line.startswith("layer_soil_type="))
                    num_layers = len(soil_type_line.strip().split("=")[1].split(","))
                    total_layers += num_layers

                    for i in range(num_layers):
                        param_dict[f"log_alpha_{tile}_{i+1}"] = tile_params[3*i]
                        param_dict[f"n_{tile}_{i+1}"] = tile_params[3*i + 1]
                        param_dict[f"log_Ks_{tile}_{i+1}"] = tile_params[3*i + 2]

                    offset = 3 * num_layers
                    lasam_params = tile_params[offset:-6] if self.include_nom else tile_params[offset:]
                    log10_a, b, frac_to_GW, field_capacity_psi, spf_factor, theta_e_1 = lasam_params
                    param_dict.update({
                        f"log10_a_{tile}": log10_a,
                        f"b_{tile}": b,
                        f"frac_to_GW_{tile}": frac_to_GW,
                        f"field_capacity_psi_{tile}": field_capacity_psi,
                        f"spf_factor_{tile}": spf_factor,
                        f"theta_e_1_{tile}": theta_e_1
                    })

                    if self.include_nom:
                        nom_params = tile_params[-6:]
                        param_dict.update({
                            f"MFSNO_{tile}": nom_params[0],
                            f"RSURF_SNOW_{tile}": nom_params[1],
                            f"HVT_{tile}": nom_params[2],
                            f"CWPVT_{tile}": nom_params[3],
                            f"VCMX25_{tile}": nom_params[4],
                            f"MP_{tile}": nom_params[5]
                        })
                ###
                # add weights to log
                if self.n_tiles == 2 and len(p.position) == len(self.bounds):  # weight param is included
                    weight_tile0 = p.position[-1]
                    weight_tile1 = 1.0 - weight_tile0
                    param_dict["weight_tile0"] = weight_tile0
                    param_dict["weight_tile1"] = weight_tile1

                row = {
                    "iteration": iteration + 1,
                    "particle": idx,
                    f"{self.metric_to_calibrate_on}_calibration": cal_metrics[self.metric_to_calibrate_on],
                    f"{self.metric_to_calibrate_on}_validation": val_metrics[self.metric_to_calibrate_on],
                    **param_dict
                }
                log_rows.append(row)

            pd.DataFrame(log_rows).to_csv(log_path, index=False)

            for p in self.particles:
                p.update_velocity(self.global_best_position, w, self.c1, self.c2)
                p.update_position(self.bounds)

            print(f"Best objective so far: {-self.global_best_value:.4f}")

        print(f"PSO completed for {self.gage_id} in {datetime.now() - start_time}")

        # Now run final validation
        final_val_metrics = run_validation_with_best(
            gage_id=self.gage_id,
            logging_dir=logging_dir,
            observed_q_root=self.observed_q_root,
            base_roots=self.base_roots,
            best_params=self.global_best_position,
            n_tiles=self.n_tiles,
            include_nom=self.include_nom,
            weights=self.weights
        )


        best_position = self.global_best_position


        param_dict = {}
        for tile in range(self.n_tiles):
            tile_params = extract_tile_params(best_position, tile, self.n_tiles)
            config_dir = os.path.join(self.base_roots[tile], f"out/{self.gage_id}/configs/lasam")
            config_file = sorted(f for f in os.listdir(config_dir) if f.startswith("lasam_config_cat"))[0]
            config_path = os.path.join(config_dir, config_file)

            with open(config_path) as f:
                lines = f.readlines()
            soil_type_line = next(line for line in lines if line.startswith("layer_soil_type="))
            num_layers = len(soil_type_line.strip().split("=")[1].split(","))

            for i in range(num_layers):
                param_dict[f"log_alpha_{tile}_{i+1}"] = tile_params[3*i]
                param_dict[f"n_{tile}_{i+1}"] = tile_params[3*i + 1]
                param_dict[f"log_Ks_{tile}_{i+1}"] = tile_params[3*i + 2]

            offset = 3 * num_layers
            lasam_params = tile_params[offset:-6] if self.include_nom else tile_params[offset:]
            log10_a, b, frac_to_GW, field_capacity_psi, spf_factor, theta_e_1 = lasam_params
            param_dict.update({
                f"log10_a_{tile}": log10_a,
                f"b_{tile}": b,
                f"frac_to_GW_{tile}": frac_to_GW,
                f"field_capacity_psi_{tile}": field_capacity_psi,
                f"spf_factor_{tile}": spf_factor,
                f"theta_e_1_{tile}": theta_e_1
            })

            if self.include_nom:
                nom_params = tile_params[-6:]
                param_dict.update({
                    f"MFSNO_{tile}": nom_params[0],
                    f"RSURF_SNOW_{tile}": nom_params[1],
                    f"HVT_{tile}": nom_params[2],
                    f"CWPVT_{tile}": nom_params[3],
                    f"VCMX25_{tile}": nom_params[4],
                    f"MP_{tile}": nom_params[5]
                })

        # Also add weight if using weight parameter
        if self.n_tiles == 2 and len(best_position) == len(self.bounds):
            weight_tile0 = best_position[-1]
            param_dict["weight_tile0"] = weight_tile0
            param_dict["weight_tile1"] = 1.0 - weight_tile0

        # Lookup calibration metrics from global best
        global_best_idx = np.argmin([p.best_value for p in self.particles])
        global_best_particle = self.particles[global_best_idx]

        # After run_validation_with_best()
        final_row = {
            "iteration": "final",
            "particle": -1,
            f"{self.metric_to_calibrate_on}_calibration": best_cal_metrics[self.metric_to_calibrate_on],
            f"{self.metric_to_calibrate_on}_validation": final_val_metrics[self.metric_to_calibrate_on],
            **param_dict
        }


        pd.concat([
            pd.read_csv(log_path),
            pd.DataFrame([final_row])
        ]).to_csv(log_path, index=False)

        return self.global_best_position, self.global_best_value, best_validation_metric, datetime.now() - start_time



# === PER-GAGE WRAPPER ===
def calibrate_gage(gage_id):
    observed_q_root = cfg.observed_q_root
    model_roots = cfg.model_roots
    n_tiles = len(model_roots)

    # For now, assume equal weights per tile. Weights are set later as a part of calibration.
    weights = [1.0 / n_tiles] * n_tiles

    try:
        all_init_params = []
        all_bounds = []
        include_nom_flags = []

        for tile_idx, root in enumerate(model_roots):
            config_dir = os.path.join(root, f"out/{gage_id}/configs/lasam")
            if not os.path.exists(config_dir):
                print(f"Skipping tile {tile_idx} for {gage_id}: missing config dir {config_dir}")
                return

            # === Read initial parameters
            example_config = sorted(f for f in os.listdir(config_dir) if f.startswith("lasam_config_cat"))[0]
            example_path = os.path.join(config_dir, example_config)
            tile_init = extract_initial_params(example_path)

            # Convert 'a' to log10
            include_nom = os.path.isdir(os.path.join(root, f"out/{gage_id}/configs/noahowp"))
            a_index = -12 if include_nom else -6
            tile_init[a_index] = np.log10(tile_init[a_index])

            all_init_params.extend(tile_init)
            include_nom_flags.append(include_nom)

            # === Set bounds
            num_layers = len([
                line for line in open(example_path)
                if line.startswith("layer_soil_type=")
            ][0].split("=")[1].split(","))

            tile_bounds = []
            for _ in range(num_layers):
                tile_bounds.extend([(-4, 0.0), (1.02, 3.0), (-4, 2)])  # log_alpha, n, log_Ks

            tile_bounds.extend([
                (-8, -1),         # log10_a
                (0.01, 5.0),      # b
                (1e-4, 1 - 1e-4), # frac_to_GW
                (10.0, 500.0),    # field_capacity_psi
                (0.1, 1.0),       # spf_factor
                (0.3, 0.6)        # theta_e_1
            ])

            if include_nom:
                tile_bounds.extend([
                    (0.625, 5.0),     # MFSNO
                    (0.1, 100.0),     # RSURF_SNOW
                    (0.0, 20.0),      # HVT
                    (0.18, 5.0),      # CWPVT
                    (0.0, 80.0),      # VCMX25
                    (3.6, 12.6)       # MP
                ])

            all_bounds.extend(tile_bounds)

        # After all tile_bounds are appended
        if n_tiles == 2:
            # Add a parameter for weight of tile 0 (tile 1 weight is 1 - this)
            all_init_params.append(0.8)  # or any reasonable default
            all_bounds.append((0.0, 1.0))  # bounds for tile 0 weight


        # === Determine if any tile includes NOM
        include_nom = any(include_nom_flags)

        print(f"Length of all_init_params: {len(all_init_params)}")
        print(f"Length of all_bounds: {len(all_bounds)}")
        print(f"n_tiles: {n_tiles}")

        # === Run PSO
        pso = PSO(
            n_particles=n_particles,
            bounds=all_bounds,
            n_iterations=n_iterations,
            gage_id=gage_id,
            init_params=all_init_params,
            base_roots=model_roots,
            observed_q_root=observed_q_root,
            metric_to_calibrate_on="kge",
            include_nom=include_nom,
            n_tiles=n_tiles,
            weights=weights
        )

        best_pos, best_val, best_val_metric, runtime = pso.optimize()

    except Exception as e:
        print(f"Error calibrating {gage_id}: {e}")
        traceback.print_exc()


# === MAIN ===
if __name__ == "__main__":
    from multiprocessing import get_context
    from datetime import datetime

    start_time = datetime.now()

    # Adjust as needed
    gages_file = cfg.gages_file
    gage_list = pd.read_csv(gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    ctx = get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(calibrate_gage, gage_list)

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\n Total wall time for calibration: {total_duration}")

