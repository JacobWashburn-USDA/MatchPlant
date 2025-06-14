# Developed: Worasit Sangjan
# Date: 19 January 2025

# Set project paths (update these to match your directory structure)
$ProjectFolder = Get-Location
$ImagesFolder = Join-Path $ProjectFolder "images"
$GcpFile = Join-Path $ImagesFolder "gcp_list.txt"
$ImgListFile = Join-Path $ProjectFolder "img_list.txt"

# Docker image name
$OdmImage = "opendronemap/odm"

# Step 1: Check for gcp_list.txt
if (Test-Path $GcpFile) {
    Write-Host "Found gcp_list.txt. It will be included in the ODM process."
    $UseGcp = $true
} else {
    Write-Host "No gcp_list.txt found. Proceeding without GCP."
    $UseGcp = $false
}

# Step 2: Run ODM 
Write-Host "Running ODM..."
$DockerCmd = @"
docker run -ti --rm `
    -v "$ImagesFolder:/code/images" `
    -v "$ProjectFolder/odm_orthophoto:/code/odm_orthophoto" `
    -v "$ProjectFolder/odm_texturing:/code/odm_texturing" `
    -v "$ProjectFolder/opensfm:/code/opensfm" `
    -v "$ProjectFolder/odm_dem:/code/odm_dem" `
    -v "$ProjectFolder/odm_meshing:/code/odm_meshing" `
    -v "$ProjectFolder/odm_filterpoints:/code/odm_filterpoints" `
    -v "$ProjectFolder/odm_report:/code/odm_report" `
    -v "$ProjectFolder/odm_georeferencing:/code/odm_georeferencing" `
    $OdmImage --orthophoto-resolution=0.5 --dsm
"@

if ($UseGcp) {
    $DockerCmd += " --gcp images/gcp_list.txt"
}

# Execute ODM command
Invoke-Expression $DockerCmd

# Step 3: Generate img_list.txt
Write-Host "Generating img_list.txt..."
$UndistortedImagesFolder = Join-Path $ProjectFolder "opensfm/undistorted/images"
Get-ChildItem -Path $UndistortedImagesFolder -Name | Out-File -FilePath $ImgListFile -Encoding ascii

# Step 4: Run Orthorectification Process
Write-Host "Running Orthorectification Process..."
$DockerCmdOrthorectify = @"
docker run -ti --rm `
    -v "$ProjectFolder:/datasets" `
    --entrypoint /code/contrib/orthorectify/run.sh `
    $OdmImage `
    /datasets --image-list img_list.txt
"@

Invoke-Expression $DockerCmdOrthorectify

Write-Host "Process complete!"
