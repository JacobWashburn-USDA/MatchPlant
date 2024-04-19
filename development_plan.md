# Part 1: From orthomosaic coordinates to image coordinates
The matching of orthomosaic geocoordinates to image coordinates is useful for obtaining data from the raw images before artifacts are introduced by the process of orthomosaic generation.

These are the steps involved in this process:
1. Create orthomosaic from raw images using Opendronemap

2. Use orthorectify.py available in the Opendronemap tool list in the Github repository. This step produces an orthrectified of each image, where every image coordinate has its own geoocoordinates. Once we have this, the conversion of the polygon from orthomosaic to raw images becomes a search peroblem. From all of the orthorectified images, we want to first find the images which contain the specifid polygon. Then we want to find the optimum image. The optimum image is probably the one that contains the polygon towards the center of the image. This is the best-case scenario. The polygon can in fact be spread across multiple images. This case is addressed below.

2. Use the script match_ortho_coordinates.py. The arguments for this script are coordinates of the polygon [x1, y1, x2, y2]. [Note: should we make this a rectangle or a polygon with any number of points? Rectangle is simpler in case the polygon is spread across multiple images.] The script will use a modified version of orthorectify.py and return a set of raw images that contain the polygon. This can be one or more images depending on how big the specified polygon is. The output includes a list of images as well as a mask for each image such that the region outside the polygon is masked out. If an image is completely inside the orthomosaic polygon, nothing will need to be masked out. This information could also be returned as coordinates specifying the location of the polygon corners and walls.


# Part 2: From image coordinates to orthomosaic coordinates

Converting image coordinates to orthomosaic coordinates is important for tasks such as object detection where it is usually easier to train a model and detect objsects in raw images. The detected objects can then be located in the orthomosaic in order to assign them spatial identity, for example, a detected plant can be assigned a certain spatial location in the field.

This process will involve the following steps:

1. Create orthomosaic from raw images using Opendronemap

2. Use orthorectify.py available in the Opendronemap tool list in the Github repository

3. Use match_image_coordinates.py. This script will take as input a raw image and polygon coordinates (it could be image coordinates or geocoordinates, if image coordinates are provided, they will simply be converted to geocoordinates using the orthorectified version of the specific image). This script will first mask the non-polygon part of the image (assign 0value to the pixels). Then it will use a modified version of orthorectify.py to get the location of this polygon in the orthomosaic.
