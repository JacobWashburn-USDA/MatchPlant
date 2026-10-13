import copy
from pathlib import Path
import sys
import tempfile
import unittest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transfer_train
from transfer_utils import load_pretrained_model


class TransferChecks(unittest.TestCase):
    def test_paths_are_relative_to_config_and_invalid_settings_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'labels.json').write_text('{}')
            (root / 'weights.pt').write_bytes(b'placeholder')
            config = dict(
                transfer=dict(pretrained_model_path='weights.pt', freeze_backbone=True,
                              freeze_rpn=False, freeze_roi_heads=False,
                              learning_rates=dict(backbone=.0001,rpn=.001,roi_heads=.001),
                              training=dict(epochs=1,batch_size=1,initial_epoch=0)),
                model=dict(num_classes=2),
                data=dict(train_dir='images',val_dir='images',train_annotations='labels.json',val_annotations='labels.json'))
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            valid = transfer_train.load_config(path)
            self.assertEqual(valid['data']['train_dir'], str(root / 'images'))
            self.assertEqual(valid['training']['checkpoint_dir'], str(root / 'checkpoints'))
            for change in ('frozen','nan','missing','epoch'):
                broken = copy.deepcopy(config)
                if change == 'frozen':
                    broken['transfer'].update(freeze_rpn=True,freeze_roi_heads=True)
                elif change == 'nan':
                    broken['transfer']['learning_rates']['rpn'] = float('nan')
                elif change == 'missing':
                    broken['data']['train_dir']='absent'
                else:
                    broken['transfer']['training']['initial_epoch']=1
                path.write_text(yaml.safe_dump(broken))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    transfer_train.load_config(path)

    def test_incompatible_checkpoint_is_rejected(self):
        model = torch.nn.Linear(2,2)
        with self.assertRaisesRegex(ValueError,'architecture/classes'):
            load_pretrained_model(model, '', checkpoint={'weight':torch.zeros(3,2),'bias':torch.zeros(3)})
        with self.assertRaisesRegex(ValueError,'model keys'):
            load_pretrained_model(model, '', checkpoint={'weight':torch.zeros(2,2)})

    def test_training_exception_propagates(self):
        class BrokenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=torch.nn.Linear(1,1)
            def forward(self, images, targets):
                raise RuntimeError('intentional training failure')
        model=BrokenModel()
        optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        loader=[([torch.zeros(1)], [{'labels':torch.tensor([1])}])]
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            transfer_train.train_epoch(model,optimizer,loader,torch.device('cpu'),{'freeze_backbone':True})


if __name__ == '__main__':
    unittest.main()
