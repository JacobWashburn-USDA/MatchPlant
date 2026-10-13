import sys, tempfile, json, subprocess, copy
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'6-1_obj_det_trainer'))
sys.path.insert(0,str(repo/'6-2_obj_det_trans_learner'))
import torch, yaml
from PIL import Image
import train, transfer_train
torch.set_num_threads(2)
with tempfile.TemporaryDirectory(prefix='matchplant-handoff-') as directory:
    root=Path(directory)
    images=root/'images'; images.mkdir()
    Image.new('RGB',(128,128),(80,140,90)).save(images/'plant.jpg')
    labels=dict(info={},licenses=[],images=[dict(id=1,file_name='plant.jpg',width=128,height=128)],categories=[dict(id=1,name='plant')],annotations=[dict(id=1,image_id=1,category_id=1,bbox=[20,20,40,50],area=2000,iscrowd=0)])
    annotation=root/'labels.json'; annotation.write_text(json.dumps(labels))
    dataset=train.MaizeDatasetCOCO(images,annotation,transforms=train.get_transform(True))
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,collate_fn=train.collate_fn)
    model=train.build_model(2,pretrained=False)
    assert model.rpn.head.cls_logits.weight.shape[0]==3
    model.transform.min_size=(128,); model.transform.max_size=128
    model.rpn._pre_nms_top_n={'training':30,'testing':30}
    model.rpn._post_nms_top_n={'training':15,'testing':15}
    optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    loss=train.train_one_epoch(model,optimizer,loader,torch.device('cpu'),0,use_amp=False)
    assert loss>0
    torch.save({'model_state_dict':model.state_dict()},root/'base.pt')
    del model, optimizer
    config=yaml.safe_load((repo/'6-2_obj_det_trans_learner/transfer_config.yaml').read_text())
    config['transfer']['pretrained_model_path']='base.pt'
    config['transfer']['training'].update(epochs=1,batch_size=1,initial_epoch=0)
    config['data'].update(train_dir='images',val_dir='images',train_annotations='labels.json',val_annotations='labels.json')
    config['model'].update(min_size=128,max_size=128,rpn_pre_nms_top_n_train=30,rpn_pre_nms_top_n_test=30,rpn_post_nms_top_n_train=15,rpn_post_nms_top_n_test=15)
    config['training'].update(checkpoint_dir='transfer',validation_dir='validation')
    path=root/'config.yaml'; path.write_text(yaml.safe_dump(config))
    transfer_train.run(transfer_train.load_config(path))
    checkpoint=root/'transfer/best_model_transfer.pt'
    completed=subprocess.run([sys.executable,str(repo/'7_obj_det_tester/test.py'),'--checkpoint',str(checkpoint),'--data-dir',str(images),'--annotation-file',str(annotation),'--results-dir',str(root/'new/nested/results')],text=True,capture_output=True,env={**__import__('os').environ,'OMP_NUM_THREADS':'2','PYTHONIOENCODING':'utf-8'},timeout=90)
    if completed.returncode:
        print(completed.stdout);print(completed.stderr)
        raise RuntimeError('Module 7 CLI failed')
    assert (root/'new/nested/results/coco_predictions.json').is_file()
    assert (root/'new/nested/results/confusion_matrix.png').is_file()
    print('PASS: real base-training batch -> default 3-anchor checkpoint -> real transfer epoch -> Module 7 CLI, metrics, prediction JSON and plots')
