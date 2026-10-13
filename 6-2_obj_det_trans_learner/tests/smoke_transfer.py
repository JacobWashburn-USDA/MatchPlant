import sys, json, copy
from pathlib import Path
module = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(module))
import torch, yaml
from PIL import Image
import transfer_train as transfer
torch.set_num_threads(2)
import tempfile
scratch = tempfile.TemporaryDirectory(prefix='matchplant-transfer-')
root = Path(scratch.name)
images = root / 'images'
images.mkdir(exist_ok=True)
Image.new('RGB', (128,128), (120,160,80)).save(images / 'sample.jpg')
annotation = dict(info={}, licenses=[], images=[dict(id=1,file_name='sample.jpg',width=128,height=128)], categories=[dict(id=1,name='plant')], annotations=[dict(id=1,image_id=1,category_id=1,bbox=[20,20,40,50],area=2000,iscrowd=0)])
(root / 'annotations.json').write_text(json.dumps(annotation))
config = yaml.safe_load((module / 'transfer_config.yaml').read_text())
config['model'].update(min_size=128,max_size=128,rpn_pre_nms_top_n_train=30,rpn_post_nms_top_n_train=15,rpn_pre_nms_top_n_test=30,rpn_post_nms_top_n_test=15,box_detections_per_img=10)
model = transfer.build_transfer_model(config['model'], {'rpn.head.cls_logits.weight': torch.zeros(12,1,1,1)})
torch.save({'model_state_dict':model.state_dict()},root / 'pretrained.pt')
del model
config['transfer']['pretrained_model_path']='pretrained.pt'
config['transfer']['training'].update(epochs=1,batch_size=1,initial_epoch=0)
config['data'].update(train_dir='images',val_dir='images',train_annotations='annotations.json',val_annotations='annotations.json')
config['training']={'checkpoint_dir':'output','validation_dir':'validation','save_freq':1}
path=root / 'config.yaml'
path.write_text(yaml.safe_dump(config))
validated=transfer.load_config(path)
invalid=copy.deepcopy(config)
invalid['transfer'].update(freeze_backbone=True,freeze_rpn=True,freeze_roi_heads=True)
(root/'invalid.yaml').write_text(yaml.safe_dump(invalid))
try:
    transfer.load_config(root/'invalid.yaml')
except ValueError as exc:
    assert 'Unfreeze' in str(exc)
else:
    raise AssertionError('all frozen accepted')
transfer.run(validated)
assert (root/'output/best_model_transfer.pt').is_file()
before = torch.load(root/'pretrained.pt', weights_only=True)['model_state_dict']
after = torch.load(root/'output/best_model_transfer.pt', weights_only=True)['model_state_dict']
assert all(torch.equal(value,after[key]) for key,value in before.items() if key.startswith('backbone.'))
assert any(not torch.equal(value,after[key]) for key,value in before.items() if key.startswith('roi_heads.'))
del before, after
assert list((root/'validation').glob('*.json'))
print('PASS: real CPU epoch, COCO evaluation, checkpoint output, path validation, frozen-layer guard')

scratch.cleanup()
