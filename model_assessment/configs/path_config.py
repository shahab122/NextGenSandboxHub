import os

# Define base paths
project_root = "/anvil/projects/x-eve250009/malam3/NextGenSandboxHub"
model_assessment_root = os.path.join(project_root, "model_assessment")

# Define commonly used paths
logging_dir = os.environ.get("LOGGING_DIR", os.path.join(project_root, "logging")) ###this will be the directory where parameter logs will be stored as calibration progresses
sandbox_path = os.path.join(project_root, "sandbox.py") ###path to the sandbox script which is used to run models in the calibration scripts
calib_scripts_dir = os.path.join(model_assessment_root, "calib_scripts") ###path to where the calibration scripts live
observed_q_root = os.path.join(project_root, "model_assessment", "USGS_streamflow") ###where USGS streamflow observations live 
usgs_output_dir = os.path.join(observed_q_root, "successful_sites") ###where raw USGS streamflow data will go
gages_file = os.environ.get("GAGES_FILE", os.path.join(project_root, "basin_IDs", "basin_IDs_384.csv"))


# Tile model roots
### for each model root, "out" subdirectory must be created, and also each model root iteself must be created 
model_roots = [os.environ.get("MODEL_ROOT1", "/anvil/scratch/x-malam3/tile_calibration/model1")]
if os.environ.get("MODEL_ROOT2"):   # set MODEL_ROOT2 to run 2 tiles; unset = 1 tile
    model_roots.append(os.environ["MODEL_ROOT2"])

# Ensure logging directory exists
os.makedirs(logging_dir, exist_ok=True)
