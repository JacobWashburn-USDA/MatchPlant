# Ortho_to_image
Repository for the development of tools and packages for easily matching geo coordinates between orthophotos (orthomosaics) and the images they are derived from, and extracting data from linked orthos and images based on user provided bounding polygons. 

**Overall process**

1. Install ODM software using a Docker in the computer/laptop
2. Prepare a raw dataset (images from drone)
    1. Create a ground control point list file: gcp_list.txt*
    2. Exicute Python code
    3. Combine the gcp_list.txt and the raw dataset and place both files in the same folder

\*ODM can still be run without a gcp file; however, the authors recommend collecting this data and using it to improve the accuracy of the targeted objects on the othomosaic image.

Figure: A diagram of arranging the folders and data for ODM

3. Run ODM software to create orthophoto (othomosaic images) and other files
     1. Open Docker software
     2. Open Windows PowerShell software
          1. The working directory on Windows PowerShell must be the project folder that contains the images folder
          2. Type or copy the command and paste on the Windows PowerShell to run ODM:
             docker run -ti --rm -v "$(pwd)/images:/code/images" -v "$(pwd)/odm_orthophoto:/code/odm_orthophoto" -v "$(pwd)/odm_texturing:/code/odm_texturing" -v "$(pwd)/opensfm:/code/opensfm" -v "$(pwd)/odm_dem:/code/odm_dem" -v "$(pwd)/odm_meshing:/code/odm_meshing" -v "$(pwd)/odm_filterpoints:/code/odm_filterpoints" -v "$(pwd)/odm_report:/code/odm_report" -v "$(pwd)/odm_georeferencing:/code/odm_georeferencing" opendronemap/odm --dem-resolution=0.5 --orthophoto-resolution=0.5 –dsm --gcp images/gcp_list.txt

\*In case there is no “gcp_list.txt”, please exclude the “--gcp images/gcp_list.txt” sentence in the command above.

4. Create a file that lists the image name to use in the orthorectification process: img_list.txt
    1. Open Windows PowerShell software
    2. The working directory on Windows PowerShell must be the project folder that contains the images folder
    3. Type or copy the command and paste on the Windows PowerShell
        1. Get-ChildItem -Path $pwd/opensfm/undistorted/images -Name | Out-file img_list.txt -Encoding ascii
5. Run ODM software to create orthorectified images from the orthorectification process
    1. Open Docker software
    2. Open Windows PowerShell software
    3. Type or copy the command and paste on the Windows PowerShell to run ODM
        1. docker run -ti –rm -v path_to_the_work_folder:/datasets --entrypoint /code/contrib/orthorectify/run.sh opendronemap/odm /datasets/project --image-list img_list.txt
6. From the process above, orthophoto and orthorectified images are used to find the optimal image coverage using Python code – find_optimal_coverage_image.jpynb
