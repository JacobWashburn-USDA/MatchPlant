# MatchPlant

An Open-Source Pipeline for UAV (unmanned aerial vehicle)-Based Object Detection and Data Extraction

## Authors

- Worasit Sangjan
- Piyush Pandey
- Norman B. Best
- Jacob D. Washburn

*USDA-ARS, Plant Genetics Research Unit, Columbia, MO, United States*

## Overview

MatchPlant is an open-source pipeline designed for automated detection of individual objects using UAV-derived imagery. The pipeline combines interactive tools for UAV imagery preparation with automated deep learning methods and data extraction capabilities. In this case, the pipeline is used for individual maize detection, by leveraging the Faster R-CNN object detection model trained on high-resolution undistorted UAV images, MatchPlant effectively removes common artifacts from the training dataset while ensuring accurate spatial analysis and trait extraction.

## Key Features

- **Modular Design**: Flexible framework adaptable to various agricultural applications
- **Automated Object Detection**: Uses Faster R-CNN for reliable individual plant detection
- **High-Precision Processing**: Trains on high-resolution undistorted UAV images to avoid orthophoto artifacts
- **Geospatial Transformation**: Accurately projects detected plant coordinates onto orthophotos
- **User-Friendly Tools**: User-friendly GUI tools for data preparation and manual annotation
- **Data Extraction & Analysis**: Enables spatial analysis and phenotypic trait extraction

## Repository Structure

This repository is organized into the following branches, each containing specific components of the pipeline:

1. **Data Preprocessing**: Prepare and optimize UAV imagery
   - GPS data embedding tool
   - GCP file creation tool
   - Orthophoto generation using OpenDroneMap

2. **Data Preparation**: Label and organize training, validation, and testing data
   - Optimal UAV image dataset creation tool
   - Interactive labeling tool
   - Data tiling and splitting tool

3. **Model Development**: Train and test detection models, including the transfer learning model
   - Faster R-CNN training and validation
   - Model testing
   - Transfer learning utility

4. **Utilization**: Project location and creat a layer to getobject features
   - Detection projection tool
   - Shapefile generation tool

Each branch contains its own detailed README with specific installation instructions and usage guidelines.

## Pipeline Workflow

![Pipeline Diagram](media/image1.png)

Figure 1: Diagram of the MathPlant-the modular open-source pipeline

## Getting Started

To begin using MatchPlant:

1. Choose the appropriate branch for your task
2. Follow the branch-specific installation instructions
3. Refer to the branch README for detailed usage guidelines

## Usage

[Usage instructions to be added]

## Requirements

- Python 3.8+
- OpenDroneMap (ODM)
- Additional requirements listed in branch-specific documentationly

## Download Dataset

To use the MatchPlant pipeline with our prepared dataset:

**Download from Zenodo**
   ```bash
   # Using wget
   wget [ZENODO_DOWNLOAD_LINK]
   
   # Using curl
   curl -O [ZENODO_DOWNLOAD_LINK]
   ```

The dataset contains:
- UAV images: UAV-captured plant images
- Annotation file: COCO format bounding boxes
- Pre-trained model: Faster R-CNN model [To use transfet learning module]

For detailed usage instructions, start with  `Data tiling and splitting` branch README.

## Citation

If you use MatchPlant in your research, please cite:

```bibtex
@article{sangjan2025matchplant,
  title={MatchPlant: An Open-Source Pipeline for UAV-Based Single-Plant Detection and Data Extraction},
  author={Sangjan, Worasit and Pandey, Piyush and Best, Norman B and Washburn, Jacob D},
  year={2025}
}
```

For the dataset, please cite:

```bibtex
@dataset{[DATASET_CITATION_HERE],
  title={MatchPlant Dataset: UAV-Based Plant Detection Training Data},
  author={Sangjan, Worasit and Pandey, Piyush and Best, Norman B and Washburn, Jacob D},
  year={2025},
  publisher={Zenodo},
  doi={[YOUR_DOI_HERE]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and collaboration opportunities, please contact:

**Jacob Washburn** | Email: jacob.washburn@usda.gov

## Acknowledgments

This research was supported in part by an appointment to the Agricultural Research Service (ARS) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Department of Agriculture (USDA). ORISE is managed by ORAU under DOE contract number DE-SC0014664. Funding was provided by the United States Department of Agriculture, Agricultural Research Service, and SCINet Postdoctoral Fellows Program. All opinions expressed in this publication are the author’s and do not necessarily reflect the policies and views of USDA, DOE, or ORAU/ORISE.