"""Run the MatchPlant ODM workflow without a shell or extra Python packages."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

IMAGE = "opendronemap/odm"
OUTPUTS = (
    "odm_orthophoto", "odm_texturing", "opensfm", "odm_dem",
    "odm_meshing", "odm_filterpoints", "odm_report", "odm_georeferencing",
)


def odm_command(project, docker="docker"):
    command = [docker, "run", "--rm"]
    for name in ("images",) + OUTPUTS:
        command.extend(["-v", f"{(project / name).as_posix()}:/code/{name}"])
    command.extend([IMAGE, "--orthophoto-resolution=0.5", "--dsm"])
    if (project / "images" / "gcp_list.txt").is_file():
        command.extend(["--gcp", "images/gcp_list.txt"])
    return command


def orthorectify_command(project, docker="docker"):
    return [docker, "run", "--rm", "-v", f"{project.as_posix()}:/datasets",
            "--entrypoint", "/code/contrib/orthorectify/run.sh", IMAGE,
            "/datasets", "--image-list", "img_list.txt"]


def run_project(project):
    project = Path(project).expanduser().resolve()
    if not (project / "images").is_dir():
        raise ValueError("Select a project folder containing an images subfolder.")
    docker = shutil.which("docker")
    if not docker:
        raise ValueError("Docker was not found. Install Docker Desktop and start it first.")
    try:
        subprocess.run([docker, "info"], check=True, capture_output=True, text=True, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Docker is not ready. Start Docker Desktop and wait until it is running.") from exc
    for name in OUTPUTS:
        (project / name).mkdir(exist_ok=True)
    print("Running ODM...", flush=True)
    subprocess.run(odm_command(project, docker), check=True, cwd=project)
    images = project / "opensfm" / "undistorted" / "images"
    if not images.is_dir():
        raise ValueError("ODM did not create opensfm/undistorted/images; orthorectification cannot continue.")
    names = sorted(p.name for p in images.iterdir() if p.is_file())
    if not names:
        raise ValueError("ODM produced no undistorted images; orthorectification cannot continue.")
    (project / "img_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print("Running orthorectification...", flush=True)
    subprocess.run(orthorectify_command(project, docker), check=True, cwd=project)
    print("Process complete!", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-folder", default=str(Path.cwd()), help="Folder containing images/.")
    args = parser.parse_args()
    try:
        run_project(args.project_folder)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[ODM] {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
