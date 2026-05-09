# Developed: Worasit Sangjan
# Date: 19 January 2025

#!/bin/bash

# Set project paths (update these to match your directory structure)
PROJECT_FOLDER=$(pwd)
IMAGES_FOLDER="$PROJECT_FOLDER/images"
GCP_FILE="$IMAGES_FOLDER/gcp_list.txt"
IMG_LIST_FILE="$PROJECT_FOLDER/img_list.txt"

# Docker image name
ODM_IMAGE="opendronemap/odm"

# Step 1: Check for gcp_list.txt
if [ -f "$GCP_FILE" ]; then
    echo "Found gcp_list.txt. It will be included in the ODM process."
    USE_GCP=1
else
    echo "No gcp_list.txt found. Proceeding without GCP."
    USE_GCP=0
fi

# Step 2: Run ODM
echo "Running ODM..."
DOCKER_CMD="docker run -ti --rm \
    -v $IMAGES_FOLDER:/code/images \
    -v $PROJECT_FOLDER/odm_orthophoto:/code/odm_orthophoto \
    -v $PROJECT_FOLDER/odm_texturing:/code/odm_texturing \
    -v $PROJECT_FOLDER/opensfm:/code/opensfm \
    -v $PROJECT_FOLDER/odm_dem:/code/odm_dem \
    -v $PROJECT_FOLDER/odm_meshing:/code/odm_meshing \
    -v $PROJECT_FOLDER/odm_filterpoints:/code/odm_filterpoints \
    -v $PROJECT_FOLDER/odm_report:/code/odm_report \
    -v $PROJECT_FOLDER/odm_georeferencing:/code/odm_georeferencing \
    $ODM_IMAGE \
    --orthophoto-resolution=0.5 --dsm"

if [ "$USE_GCP" -eq 1 ]; then
    DOCKER_CMD="$DOCKER_CMD --gcp images/gcp_list.txt"
fi

# Execute ODM command
eval $DOCKER_CMD

# Step 3: Generate img_list.txt
echo "Generating img_list.txt..."
UNDISTORTED_IMAGES_FOLDER="$PROJECT_FOLDER/opensfm/undistorted/images"
ls $UNDISTORTED_IMAGES_FOLDER > $IMG_LIST_FILE

# Step 4: Run Orthorectification Process
echo "Running Orthorectification Process..."
docker run -ti --rm \
    -v $PROJECT_FOLDER:/datasets \
    --entrypoint /code/contrib/orthorectify/run.sh \
    $ODM_IMAGE \
    /datasets --image-list img_list.txt

echo "Process complete!"