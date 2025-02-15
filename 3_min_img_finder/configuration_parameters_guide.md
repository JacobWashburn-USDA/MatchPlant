# Configuration Parameters Guide

The Minimum Image Finder uses several key parameters to optimize the selection of drone images. Understanding these parameters is crucial for achieving the best results for your specific use case.

## Table of Contents
- [**Flight Line Width**](#flight-line-width)
- [**Horizontal Overlap Parameters**](#horizontal-overlap-parameters)
- [**Vertical Overlap Parameters**](#vertical-overlap-parameters)
- [**Uncovered Area Threshold**](#uncovered-area-threshold)

## Flight Line Width

- **Purpose**: Controls the spacing between selected images in a sequence
- **How it works**: 
  - Creates bins of specified width percentage relative to image width
  - Higher values result in wider spacing between selected images
  - Lower values create tighter spacing
- **Recommended range**: 10-20%
- **Tips**:
  - Start with 15% as a baseline
  - Decrease if gaps appear in coverage
  - Increase if too many redundant images are selected
- **Impact**: Directly affects the number of images selected and processing time

## Horizontal Overlap Parameters

- ### Minimum Overlap (%)
  - **Purpose**: Ensures sufficient overlap between adjacent images in the same flight line
  - **How it works**:
    - Defines the minimum required overlap between consecutive images
    - Prevents gaps in coverage along flight lines
    - Essential for maintaining data continuity
  - **Recommended range**: 1-5%
  - **Tips**: 
    - Lower values might create gaps
    - Higher values ensure better coverage but increase dataset size
- ### Maximum Overlap (%)
  - **Purpose**: Limits excessive overlap between adjacent images
  - **How it works**:
    - Sets an upper limit for acceptable overlap
    - Helps eliminate redundant images
    - Balances coverage with efficiency
  - **Recommended range**: 10-20%
  - **Tips**:
    - Too low might cause gaps
    - Too high leads to unnecessary redundancy
    - Typically set 3-4 times higher than minimum overlap

## Vertical Overlap Parameters

- ### Minimum Vertical Overlap (%)
  - **Purpose**: Ensures sufficient overlap between adjacent flight lines
  - **How it works**:
    - Controls minimum overlap between parallel flight lines
    - Prevents gaps between rows of images
    - Critical for complete area coverage
  - **Recommended range**: 1-5%
  - **Tips**:
    - Similar considerations as horizontal minimum overlap
    - Adjust based on terrain complexity
- ### Maximum Vertical Overlap (%)
  - **Purpose**: Prevents excessive overlap between flight lines
  - **How it works**:
    - Limits overlap between parallel flight lines
    - Reduces redundancy in cross-track direction
    - Helps optimize dataset size
  - **Recommended range**: 10-20%
  - **Tips**:
    - Balance with terrain variations
    - Higher values needed for complex topography

## Uncovered Area Threshold

- **Purpose**: Defines the acceptable percentage of gaps in coverage
- **How it works**:
  - Sets maximum allowable uncovered area percentage
  - Affects when the algorithm stops adding new images
  - Lower values ensure more complete coverage
- **Recommended range**: 5-15%
- **Tips**:
  - Start with 10% and adjust based on results
  - Lower values create larger datasets
  - Higher values might miss important areas
