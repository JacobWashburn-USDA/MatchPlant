import sys, tempfile, json, subprocess, csv, os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import cv2
repo=Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='matchplant-projection-') as directory:
    root=Path(directory)
    (root/'odm_dem').mkdir()
    (root/'opensfm/undistorted/images').mkdir(parents=True)
    (root/'odm_georeferencing').mkdir()
    with rasterio.open(root/'odm_dem/dsm.tif','w',driver='GTiff',height=5,width=5,count=1,dtype='float32',crs='EPSG:32615',transform=from_origin(-2.5,2.5,1,1)) as raster:
        raster.write(np.zeros((1,5,5),dtype='float32'))
    (root/'odm_georeferencing/coords.txt').write_text('EPSG:32615\n0 0\n')
    cv2.imwrite(str(root/'opensfm/undistorted/images/DJI_plant.JPG.tif'),np.full((10,10,3),120,dtype=np.uint8))
    reconstruction=[dict(cameras={'camera':{'focal':1.0}},shots={'DJI_plant.JPG.tif':dict(camera='camera',translation=[0,0,10],rotation=[0,0,0])})]
    (root/'opensfm/undistorted/reconstruction.json').write_text(json.dumps(reconstruction))
    predictions=dict(images=[dict(id=1,file_name='DJI_plant.JPG.tif')],annotations=[dict(image_id=1,category_id=1,bbox=[4,4,2,2],score=.9)])
    (root/'predictions.json').write_text(json.dumps(predictions))
    outputs=[]
    for workers in (1,2,8):
        result=subprocess.run([sys.executable,str(repo/'8_img_to_ortho/project_boxes.py'),'--dataset-path',str(root),'--predictions-json',str(root/'predictions.json'),'--output-dir',str(root/f'workers_{workers}'),'--num-threads',str(workers)],cwd=root,env={**os.environ,'PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,timeout=30)
        if result.returncode:
            print(result.stdout,result.stderr)
            raise AssertionError(f'Failed with {workers} workers')
        output=root/f'workers_{workers}/projected_boxes.csv'
        assert output.is_file(),result.stdout
        outputs.append(output.read_text())
    assert len(set(outputs))==1,outputs
    result=subprocess.run([sys.executable,str(repo/'8_img_to_ortho/project_boxes.py'),'--dataset-path',str(root/'missing')],cwd=root,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0
    print('PASS: synthetic raster projection produced identical CSV with 1, 2 and 8 workers (including empty partitions); invalid input failed nonzero.')
