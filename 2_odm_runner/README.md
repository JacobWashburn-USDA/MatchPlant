# ODM Process Runner

Runs OpenDroneMap and orthorectification for a local drone-image project. The dashboard uses [run_ODM_process.py](run_ODM_process.py) on Windows, macOS, and Linux. Python calls Docker directly; this workflow does not use PowerShell or Bash execution policies.

## Requirements

- Python 3.9+; the runner uses only the Python standard library.
- Docker installed and running with access to the project folder. For typical Windows/macOS desktop use, start Docker Desktop before running.
- Drone images in an `images/` subfolder. Optional ground control points belong in `images/gcp_list.txt`.

## Project structure

```text
your_drone_project/
  images/
    IMG_0001.JPG
    IMG_0002.JPG
    gcp_list.txt       # optional
```

Select `your_drone_project`, not its `images` subfolder. There is no need to copy the runner into the project folder. GCP data can be prepared with [GCP Finder](../1_gcp_finder/README.md).

## Using the dashboard

Open **ODM Runner** in the [local dashboard](../dashboard/README.md). Confirm that Docker is installed and running. Click **Browse…**, select your project folder, then click **Run**. The folder is required and must contain `images/`. Follow progress and errors in the live log.

## Using the terminal

From this module's folder:

```bash
python run_ODM_process.py --project-folder "path/to/your_drone_project"
```

On systems where Python is named `python3`, use that command instead. Without `--project-folder`, the runner uses the current working directory. Paths containing spaces are supported by passing arguments directly to Docker.

## Processing and outputs

1. Validate the project and check Docker readiness.
2. Run `opendronemap/odm` with `--orthophoto-resolution=0.5` and `--dsm`; add `--gcp images/gcp_list.txt` when that file exists.
3. Write a sorted `img_list.txt` from files in `opensfm/undistorted/images/`.
4. Run `/code/contrib/orthorectify/run.sh` in the ODM container using the project and image list.

Outputs are written inside the project, including `odm_orthophoto/`, `odm_dem/`, `odm_texturing/`, `opensfm/`, `odm_meshing/`, `odm_filterpoints/`, `odm_report/`, and `odm_georeferencing/`. Docker output is streamed without interactive terminal flags. The runner returns a nonzero exit status if a Docker stage fails and does not continue to orthorectification after failed ODM processing or missing undistorted images.

## Troubleshooting

- **Docker not found/not ready:** install or start Docker and ensure its command is available on PATH.
- **Missing images folder:** choose the project folder containing `images/`.
- **Mount or permissions error:** ensure Docker can access the selected location and that you can write output files there.
- **Processing error:** inspect the live Docker log. Requirements and resource needs depend on the dataset.

The Python runner's integration and error paths have been tested with mocked Docker execution. A complete real ODM run and native macOS execution still need verification.

## Legacy shell scripts

[run_ODM_process.ps1](run_ODM_process.ps1) and [run_ODM_process.sh](run_ODM_process.sh) remain for existing standalone workflows. They use the current working directory as the project folder. They are not launched by the dashboard; their platform-specific permissions and interactive Docker flags are separate from the Python runner. Prefer the Python runner for dashboard use.
