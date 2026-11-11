###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# extracts the hydrograph for the nexus we are interested in for objective function calculation 
### Updated: extract only the flow time series for the correct nexus from a NetCDF file
import os
import sys
import argparse
import xarray as xr
import pandas as pd
from pathlib import Path
import traceback

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--gage_id", type=str, help="USGS gage ID to extract")
parser.add_argument("--output", type=str, help="Optional path to save output CSV")
parser.add_argument("--base_dir", type=str, default="/Users/peterlafollette/CIROH_project/NextGenSandboxHub", help="Root dir of the model (tile-specific)")

args = parser.parse_args()

# === CONFIGURATION ===
gage_id = args.gage_id or os.environ.get("GAGE_ID") or "08103900"
base_dir = args.base_dir
troute_dir = os.path.join(base_dir, "out", gage_id, "troute")

# UPDATED: summary CSV is now in model_assessment
# summary_csv = os.path.join(base_dir, "model_assessment", "util", "downstream_flowpath_summary.csv")

parent_dir = Path(base_dir).resolve().parent
summary_csv = os.path.join(parent_dir, "NextGenSandboxHub", "model_assessment", "util", "downstream_flowpath_summary.csv")

if not os.path.exists(summary_csv):
    print(f"ERROR: Required CSV not found at:\n  {summary_csv}")
    print("Please run get_penult_ids.py to generate it before continuing.")
    sys.exit(1)

# Handle output path
if args.output:
    output_csv = args.output
else:
    particle_id = os.environ.get("PARTICLE_ID")
    output_dir = os.path.join(base_dir, "postproc")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{gage_id}_particle_{particle_id}.csv" if particle_id else f"{gage_id}.csv")

# === Step 1: Get the nexus feature ID for this gage ===
def get_nexus_id(gage_id, summary_csv):
    df = pd.read_csv(summary_csv, dtype=str)
    
    if gage_id not in df["gage_id"].values:
        raise ValueError(f" Gage ID {gage_id} not found in summary file.")

    row = df[df["gage_id"] == gage_id].iloc[0]

    # Determine which column to use
    if "most_downstream_nexus" in df.columns:
        nexus_str = row["most_downstream_nexus"]
    elif "nexus_before_it" in df.columns:
        nexus_str = row["nexus_before_it"]
    else:
        raise ValueError(f" Neither 'most_downstream_nexus' nor 'nexus_before_it' found in summary file.")

    # Validate format
    if not isinstance(nexus_str, str) or not nexus_str.startswith("nex-"):
        raise ValueError(f" Invalid nexus ID format for gage {gage_id}: {nexus_str}")
    
    return int(nexus_str.replace("nex-", ""))


# === Step 2: Extract flow time series from NetCDF for specific nexus ===
def extract_nexus_flow_old(netcdf_path, nexus_id):
    ds = xr.open_dataset(netcdf_path)
    
    if "flow" not in ds or "feature_id" not in ds.coords:
        raise ValueError(" Expected variables 'flow' and coordinate 'feature_id' not found in NetCDF file.")
    
    # Get index of the matching feature_id
    feature_ids = ds["feature_id"].values
    if nexus_id not in feature_ids:
        raise ValueError(f" Nexus ID {nexus_id} not found in NetCDF file's feature_id list.")

    idx = int((feature_ids == nexus_id).nonzero()[0][0])
    
    flow = ds["flow"].isel(feature_id=idx).to_series()
    df = flow.reset_index().rename(columns={"time": "current_time", "flow": "flow"})
    return df

def get_wb_ids(gage_id, summary_csv):
    import pandas as pd
    df = pd.read_csv(summary_csv, dtype=str)
    v = df.loc[df["gage_id"] == gage_id, "wbs_into_that_nexus"].iloc[0]
    if not isinstance(v, str):
        return []
    return [int(x.strip().replace("wb-", "")) for x in v.split(",") if x.strip().startswith("wb-")]

def extract_outlet_flow(netcdf_path, nexus_id, wb_ids):
    ds = xr.open_dataset(netcdf_path)
    fids = ds["feature_id"].values
    if nexus_id in fids:   # downstream flowpath of the outlet nexus is in the t-route output
        return extract_nexus_flow_old(netcdf_path, nexus_id)
    present = [w for w in wb_ids if w in fids]
    if not wb_ids or len(present) != len(wb_ids):
        raise ValueError(f" Neither nexus {nexus_id} nor all inflow wbs {wb_ids} found in t-route output.")
    flow = ds["flow"].sel(feature_id=present).sum("feature_id").to_series()
    flow.name = "flow"
    return flow.reset_index().rename(columns={"time": "current_time"})

# === MAIN EXECUTION ===
try:
    netcdf_files = [f for f in os.listdir(troute_dir) if f.endswith(".nc")]
    if not netcdf_files:
        raise FileNotFoundError(f"No NetCDF file found in {troute_dir}")
    elif len(netcdf_files) > 1:
        print(f" Multiple NetCDF files found in {troute_dir}, using the first one: {netcdf_files[0]}")

    netcdf_path = os.path.join(troute_dir, netcdf_files[0])
    nexus_id = get_nexus_id(gage_id, summary_csv)
    flow_df = extract_outlet_flow(netcdf_path, nexus_id, get_wb_ids(gage_id, summary_csv))
    flow_df.to_csv(output_csv, index=False)
    print(f" Saved nexus flow time series to: {output_csv}")

except Exception as e:
    print(f" Error while processing {gage_id}: {e}")
    traceback.print_exc()
    sys.exit(1)
