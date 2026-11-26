# matchplant_ui · Shared file and folder pickers

[← Back to MatchPlant](../README.md)

This folder provides the **Browse...** buttons used by the desktop tools (modules [1](../1_gcp_finder/README.md), [3](../3_min_img_finder/README.md), [4](../4_bbox_drawer/README.md), [5](../5_img_splitter/README.md), and [9](../9_spatial_stats_extractor/README.md)) and by the [dashboard](../dashboard/README.md). You do not run it directly. Keep it at the top of the MatchPlant folder, beside the module folders.

## How the pickers behave

- **Folder fields** open your system's standard folder picker and accept folders only. On Windows, files inside folders may be hidden in this dialog; that is normal.
- **File fields** show all file types, then check that you picked the required type (for example CSV or TIF).
- **Save As...** chooses an output file name; the file does not need to exist yet.
- **Cancel** keeps the previous value.
- In the dashboard, **Browse…** chooses a location on the computer running the dashboard; nothing is uploaded.

## What each tool asks for

| Module | Field | Select |
| --- | --- | --- |
| GCP Finder | Image Folder | Folder |
| GCP Finder | GCP File | Existing CSV file |
| Minimum Image Finder | Orthomosaic | Existing TIF/TIFF file |
| Minimum Image Finder | Orthorectified / Undistorted Image Folder | Two separate folders |
| BBox Drawer | Image Folder | Folder |
| Image Splitter | Image / Annotation / Output Folder | Three separate folders; annotations are a folder of JSON files |
| Spatial Stats Extractor | CSV File | Existing CSV file |
| Spatial Stats Extractor | Raster File | Existing TIF/TIFF/IMG file |
| Spatial Stats Extractor | Save Output File | CSV file name (may be new) |
| Dashboard | Browse… buttons | A folder or an existing file; each form checks the values further |

## For developers

From the repository root:

```bash
python -m unittest discover -s matchplant_ui/tests
```
