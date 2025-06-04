# MatchPlant

An Open-Source Pipeline for UAV (Unmanned Aerial Vehicle)-Based Object Detection and Data Extraction

## Authors

- Worasit Sangjan
- Piyush Pandey
- Norman B. Best
- Jacob D. Washburn
  
*USDA-ARS, Plant Genetics Research Unit, Columbia, MO, United States*

## Table of Contents
- [MatchPlant](#matchplant)
  - [Authors](#authors)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Repository Structure](#repository-structure)
  - [Pipeline Workflow](#pipeline-workflow)
  - [Requirements](#requirements)
  - [Getting Started](#getting-started)
  - [Download Dataset](#download-dataset)
  - [Citation](#citation)
  - [Contact](#contact)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

## Overview

MatchPlant is an open-source pipeline designed for the automated detection of individual objects using UAV-derived imagery. The pipeline combines interactive tools for preparing UAV imagery with automated deep-learning methods and data extraction capabilities. For the case study, it is used for individual maize detection. By leveraging the Faster R-CNN object detection model trained on high-resolution, undistorted UAV images. MatchPlant also provides utilization modules to incorporate the detected bounding boxes and extract digital plant traits from an orthomosaic.

## Key Features

- **Modular Design**: Flexible framework adaptable to various agricultural applications
- **Automated Object Detection**: Uses Faster R-CNN for reliable individual plant detection
- **High-Precision Processing**: Trains on high-resolution, undistorted UAV images to avoid orthomosaic artifacts
- **Geospatial Transformation**: Accurately projects detected plant coordinates onto orthomosaic
- **User-Friendly Tools**: User-friendly GUI tools for data preparation and manual annotation
- **Data Extraction & Analysis**: Enables spatial analysis and phenotypic trait extraction

## Repository Structure

This repository is organized into the following modules, each containing specific components of the pipeline:

1. **Data Preprocessing**: Prepare and optimize UAV imagery
   - GPS data embedding tool
   - GCP file creation tool
   - Orthomosaic generation using OpenDroneMap

2. **Data Preparation**: Label and organize training, validation, and testing data
   - Optimal UAV image dataset creation tool
   - Interactive labeling tool
   - Data tiling and splitting tool

3. **Model Development**: Train and test detection models, including the transfer learning model
   - Faster R-CNN training and validation
   - Model testing
   - Transfer learning utility

4. **Utilization**: Project location and create a layer to extract object features
   - Detection projection tool
   - Shapefile generation tool

Each module contains its detailed README with specific installation instructions and usage guidelines.

## Pipeline Workflow

![img](https://github.com/JacobWashburn-USDA/Ortho_to_image/blob/main/images/img.png)

Figure 1: Diagram of the MathPlant modular open-source pipeline

###### 1-GPS: Global Position System, 2-UAV: Unmanned Aerial Vehicle, 3-GCP: Ground Control Point, 4-GUI: Graphical User Interface, 5-COCO: Common Objects in Context, 6-YOLO: You Only Look Once, 7-Faster R-CNN: Faster Region-based Convolutional Neural Network

## Requirements

- Python 3.9+
- OpenDroneMap (ODM)
- Additional requirements listed in module-specific documentation

## Getting Started

To begin using MatchPlant:

1. Choose the appropriate module for your task
2. Follow the module-specific installation instructions
3. Refer to the module README for detailed usage guidelines

## Download Dataset

To use the MatchPlant pipeline with our prepared dataset:

**Download from Zenodo**
   ```bash
   # Using wget
   wget [ZENODO_LINK]
   
   # Using curl
   curl -O [ZENODO_LINK]
   ```

The dataset contains:
- UAV images: UAV-captured plant images
- Annotation file: COCO format bounding boxes
- Pre-trained model: Faster R-CNN model: Download to use in [6-2_obj_det_trans_learner](https://github.com/JacobWashburn-USDA/MatchPlant/tree/main/6-2_obj_det_trans_learner)
  
After downloading the dataset (UAV images and Annotation file), please start with the module [5_img_splitter](https://github.com/JacobWashburn-USDA/Ortho_to_image/tree/main/5_img_splitter) to use our pipeline.

## Citation

If you use MatchPlant in your research, please cite:

```
```
For the dataset, please cite:

```
```

## Contact

For questions and collaboration opportunities, please contact:

**Jacob D. Washburn**; Email: jacob.washburn@usda.gov

## Acknowledgments

This research was supported in part by an appointment to the Agricultural Research Service (ARS) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Department of Agriculture (USDA). ORISE is managed by ORAU under DOE contract number DE-SC0014664. Funding was provided by the United States Department of Agriculture, Agricultural Research Service, and SCINet Postdoctoral Fellows Program. All opinions expressed in this publication are the author’s and do not necessarily reflect the policies and views of USDA, DOE, or ORAU/ORISE.

## License

This project is licensed under the MIT License. For details, see the [LICENSE](LICENSE) file.
