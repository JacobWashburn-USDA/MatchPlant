# ODM Process Runners

Scripts to automate OpenDroneMap (ODM) processing for both Windows (PowerShell) and Linux/Unix (Bash) environments.

## Prerequisites

- [Docker](https://www.docker.com/get-started) installed and running
- For Windows: PowerShell
- For Linux/Unix: Bash shell
- OpenDroneMap Docker image (will be pulled automatically if not present)

## Directory Structure

Your project folder should look like this:
```
project_folder/
├── images/              # Place your drone images here
│   └── gcp_list.txt    # Optional: GCP file if you have ground control points
├── run_ODM_process.ps1 # Windows PowerShell script
└── run_ODM_process.sh  # Linux/Unix shell script
```

## Usage

### For Windows Users

1. Place your drone images in the `images` folder
2. If using ground control points, place your `gcp_list.txt` in the `images` folder
3. Open PowerShell in your project directory
4. Run the script:
```powershell
.\run_ODM_process.ps1
```

### For Linux/Unix Users

1. Place your drone images in the `images` folder
2. If using ground control points, place your `gcp_list.txt` in the `images` folder
3. Make the script executable:
```bash
chmod +x run_ODM_process.sh
```
4. Run the script:
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
- Orthophoto resolution: 0.5
- DSM generation enabled
- Uses GCP if `gcp_list.txt` is present in the images folder

## Error Handling

- The scripts will check for the presence of GCP file and notify you
- Docker commands are executed with error checking
- Progress messages are displayed during processing

## Common Issues

1. **Docker not running**
   - Ensure Docker daemon is running before executing the scripts
   
2. **Permission Issues**
   - Windows: Ensure PowerShell has necessary permissions
   - Linux: Make sure script is executable (`chmod +x`)

3. **Path Issues**
   - Avoid spaces in folder and file names
   - Use the scripts from the project root directory

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Add your chosen license here]
