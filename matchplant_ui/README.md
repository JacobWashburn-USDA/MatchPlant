# Shared file and folder pickers

Used by the Windows and macOS GUI variants of modules 1, 3, 4, 5 and 9, and by the dashboard Browse endpoints. Keep this folder at the repository root beside the module folders.

Each folder row has one Browse button opening the standard system folder picker. Windows uses Tk's native directory dialog; macOS uses the native folder-only Open panel. There is no custom folder browsing window. Windows may hide individual files in its folder-only dialog; folder contents are therefore not guaranteed to be visible. Cancellation returns no path and keeps the module's previous input unchanged.

File pickers show all file types. A required-type check rejects an unsupported selection before returning a path to the module. Cancelling leaves the module's previous input unchanged. Output file selection supports a new filename and retains CSV defaults and overwrite protection.

| Module | Input/output | Required selection |
| --- | --- | --- |
| GCP Finder | Images | Folder |
| GCP Finder | GCP coordinates | Existing CSV file |
| Minimum Image Finder | Orthomosaic | Existing TIF/TIFF file |
| Minimum Image Finder | Rectified and undistorted images | Two separate folders |
| BBox Drawer | Images | Folder |
| Image Splitter | Images, annotations, output | Three separate folders; annotations are a folder of JSON files |
| Spatial Stats Extractor | Bounds | Existing CSV file |
| Spatial Stats Extractor | Raster | Existing TIF/TIFF/IMG file |
| Spatial Stats Extractor | Output | CSV filename; file may not exist yet |
| Dashboard | Folder/file Browse controls | Directory or existing file, respectively; module forms perform further validation |

Run policy and wiring checks with `python -m unittest discover -s matchplant_ui/tests` from the repository root. Native macOS execution still needs testing on a Mac; Windows cannot verify AppKit panel appearance or desktop permissions.

Button labels are consistent: **Browse...** chooses an input file or a folder; **Save As...** chooses an output filename. Buttons that actually write results retain their Save/Export action names.
