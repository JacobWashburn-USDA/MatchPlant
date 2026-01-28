# Output Format Reference

Quick reference for `projected_boxes.csv` format (Module 9 compatible).

## CSV Format

**File**: `projected_boxes.csv`  
**Headers**: None  
**Columns**: 13  
**Delimiter**: Comma  

## Column Structure

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| 1 | float | Affine: a (pixel width X) | 0.003 |
| 2 | float | Affine: b (rotation/skew) | 0.0 |
| 3 | float | Affine: c (X offset in meters) | 562340.234 |
| 4 | float | Affine: d (rotation/skew) | 0.0 |
| 5 | float | Affine: e (pixel height Y) | -0.003 |
| 6 | float | Affine: f (Y offset in meters) | 4512890.567 |
| 7-9 | float | Placeholders (always 0.0) | 0.0, 0.0, 0.0 |
| 10 | string | Dimensions: "(0,0),(w,h)" | "(0, 0), (85, 120)" |
| 11 | string | Filename | DJI_0123_box_1.tif |
| 12 | int | Detection ID | 1 |
| 13 | float | Confidence score | 0.92 |

## Example Row

```csv
0.003,0.0,562340.234,0.0,-0.003,4512890.567,0.0,0.0,0.0,"(0, 0), (85, 120)",DJI_0123_box_1.tif,1,0.92
```

## Quick Validation

```python
import pandas as pd

# Load CSV (no headers)
df = pd.read_csv('projected_boxes.csv', header=None)

# Check format
assert len(df.columns) == 13, f"Expected 13 columns, got {len(df.columns)}"
assert isinstance(df.iloc[0, 9], str), "Column 9 must be string"
assert df.iloc[0, 9].startswith("(0, 0)"), "Column 9 must start with (0, 0)"

print("Format valid")
```

## Usage in Module 9

1. Open Module 9 GUI
2. Browse → Select `projected_boxes.csv` as "CSV Bounds File"
3. Browse → Select your raster (CHM, NDVI, etc.)
4. Click "Update Preview" to verify alignment
5. Select statistics and calculate

Module 9 reads:
- Columns 1-6: Affine transform for georeferencing
- Column 10: Box dimensions (width, height)
- Column 11: Box identifier

---

**See README.md for complete documentation.**