# 2 · ODM Runner

**Stage 1 · Prepare imagery** · **Type: Docker** · [← Back to MatchPlant](../README.md)

Run [OpenDroneMap](https://github.com/OpenDroneMap/ODM) (ODM) on a project folder to generate the orthomosaic, DSM (digital surface model), undistorted images, and orthorectified images used by the rest of the pipeline. ODM runs inside Docker, started by a Python runner that works on Windows, macOS, and Linux.

| | |
| --- | --- |
| **Requires** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running |
| **Input** | A project folder containing an `images/` subfolder (and optionally `images/gcp_list.txt`) |
| **Output** | ODM results inside the project folder: orthomosaic, DSM, undistorted and orthorectified images |
| **Previous step** | [1 GCP Finder](../1_gcp_finder/README.md) (optional) |
| **Next step** | [3 Minimum Image Finder](../3_min_img_finder/README.md); later, [8 Image-to-Ortho Projector](../8_img_to_ortho/README.md) uses this same project folder |

## Run it from the dashboard

1. Start **Docker Desktop** and wait until it is running.
2. On the dashboard home, select **ODM Runner**. The page shows whether Docker is installed and running.
3. Click **Browse…** and choose your **ODM project folder**: the folder that *contains* `images/`, not `images/` itself.
4. Click **Run** and follow the **Output** panel. ODM can take a long time on large image sets.
5. When it finishes, check the output folders listed below.

## Inputs

```text
your_drone_project/          ← choose this folder
└── images/
    ├── IMG_0001.JPG
    ├── IMG_0002.JPG
    └── gcp_list.txt         # optional, from GCP Finder
```

If `images/gcp_list.txt` exists, it is used automatically. You do not need to copy any script into the project folder.

## What it does

1. Checks the project folder and that Docker is ready.
2. Runs `opendronemap/odm` with `--orthophoto-resolution=0.5` and `--dsm`, plus `--gcp images/gcp_list.txt` when that file exists.
3. Lists the undistorted images in `img_list.txt`.
4. Runs ODM's orthorectification on those images.

If a step fails, the runner stops and reports the error.

## Outputs

Results are written inside the project folder. The ones used later in MatchPlant:

```text
your_drone_project/
├── odm_orthophoto/odm_orthophoto.tif      # orthomosaic → module 3
├── odm_dem/dsm.tif                        # DSM → module 8
├── odm_georeferencing/coords.txt          # → module 8
├── opensfm/undistorted/images/            # undistorted images → modules 3 and 8
├── opensfm/undistorted/reconstruction.json  # camera parameters → module 8
├── orthorectified/                        # orthorectified images → module 3
└── odm_texturing/, odm_meshing/, odm_filterpoints/, odm_report/, ...
```

## Run without the dashboard

The runner only needs Python 3.9+ (standard library) and Docker. From this folder:

```bash
python run_ODM_process.py --project-folder "path/to/your_drone_project"
```

Use `python3` if that is your Python command. Without `--project-folder`, the current folder is used as the project.

The older [run_ODM_process.ps1](run_ODM_process.ps1) (Windows PowerShell) and [run_ODM_process.sh](run_ODM_process.sh) (Bash) scripts are kept for existing workflows. They use the current folder as the project. The Python runner is recommended.

## Troubleshooting

- **Docker not found or not running:** install or start Docker Desktop, then reload the module page.
- **Missing images folder:** choose the project folder that contains `images/`.
- **Mount or permission error:** make sure Docker can access the folder's drive and that you can write to it.
- **Processing error or out of memory:** read the Docker log in the Output panel; increase Docker's memory limit or use fewer images.

## License

MIT License. See [LICENSE](../LICENSE).
