# ODM Process Runners

Scripts to automate OpenDroneMap (ODM) processing for both Windows (PowerShell) and Linux/Unix (Bash) environments.

## Note for Deep Learning - Object Detection

The ODM is particularly valuable for preparing drone imagery for object detection model training. The ODM software:
- Generates undistorted images, removing lens distortion that could affect model accuracy
- Helps create higher-quality training datasets for computer vision tasks

**Key Benefits:**
- Reduced noise in training data by removing camera lens distortion
- More consistent object appearance across images
- Better model generalization due to standardized image geometry
- Improved accuracy in object size and position predictions

## Table of Contents

- [Prerequisites](#prerequisites)
- [Important: Working Directory](#important-working-directory)
- [Usage](#usage)
- [Output](#output)
- [Script Parameters](#script-parameters)
- [Error Handling](#error-handling)
- [Common Issues](#common-issues)

## Prerequisites

- OpenDroneMap installed using Docker: Installation guide [here](https://github.com/OpenDroneMap/ODM)
- For Windows system: PowerShell
- For Linux/Unix system: Bash shell
  
## Important: Working Directory

**The scripts must be run from the directory containing your drone images!**

- Example of directory structure:
  ```
  your_drone_project/         # This is where your images are
  ├── images/                 # Your drone images
  │   ├── IMG_0001.JPG
  │   ├── IMG_0002.JPG
  │   └── gcp_list.txt        # Optional: GCP file if you have ground control points
  ├── run_ODM_process.ps1 (Copy a script here if Using Window) Or run_ODM_process.sh (Copy a script here if using Linux/Unix shell)
  ```
- `gcp_list.txt` can be created by [1_gcp_finder](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/1_gcp_finder)

## Usage

### For Windows Users

- Copy `run_ODM_process.ps1` to your image directory
- Place your `gcp_list.txt` in the `images` folder
- Navigate to your image directory:
  ```PowerShell
  cd path\to\your\drone_project
  ```
- Run the script:
  ```PowerShell
  .\run_ODM_process.ps1 
  ```

### For Linux/Unix Users

- Copy `run_ODM_process.sh` to your image directory
- Place your `gcp_list.txt` in the `images` folder
- Navigate to your image directory:
  ```bash
  cd path/to/your/drone_project
  ```
- Make the script executable:
  ```bash
  chmod +x run_ODM_process.sh
  ```
- Run the script:
  ```bash
  ./run_ODM_process.sh
  ```

## Output

The script will create several output directories:
- `odm_orthophoto/`: Contains the orthophoto results
- `odm_dem/`: Contains Digital Elevation Model
- `odm_texturing/`: Contains the textured mesh
- `odm_georeferencing/`: Contains georeferenced results
- And other ODM-specific output folders

## Script Parameters

The scripts use these default parameters:
- Orthophoto resolution: 0.5 cm
- DSM generation enabled
- Uses GCP if `gcp_list.txt` is present in the images folder

## Error Handling

- The scripts will check for the presence of the GCP file and notify you
- Docker commands are executed with error-checking
- Progress messages are displayed during processing

## Common Issues

- Wrong Working Directory
   - Make sure you are running the script from the directory containing your images
   - The script uses the current directory path for all operations
- Docker not running
   - Ensure the Docker daemon is running before executing the scripts
- Permission Issues
   - Windows: Ensure PowerShell has the necessary permissions
   - Linux: Make sure the script is executable (`chmod +x`)
- Path Issues
   - Avoid spaces in folder and file names
   - Use the scripts from the directory containing your images
