import ast
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dependency_check
import input_validation
import process_manager
from app import app, build_command, resolve_script
from modules import MODULES, MODULES_BY_ID, module_dir, requirements_path


class DashboardChecks(unittest.TestCase):
    def test_all_scripts_and_form_flags_match(self):
        for module in MODULES:
            for variant in module['script']:
                path=resolve_script(module,variant)
                self.assertTrue(path.is_file(),str(path))
                tree=ast.parse(path.read_text(encoding='utf-8-sig'))
                flags={arg.value for node in ast.walk(tree) if isinstance(node,ast.Call)
                       and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
                       for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)}
                for field in module['fields']:
                    if field.get('flag'):
                        self.assertIn(field['flag'],flags,(module['id'],field['name']))

    def test_module_pages_and_numeric_controls(self):
        client=app.test_client()
        for module in MODULES:
            with patch('app.dependency_check.missing_packages',return_value=[]),patch('app.docker_check.docker_status',return_value='ready'):
                page=client.get('/module/'+module['id'])
                self.assertEqual(page.status_code,200,module['id'])
                self.assertIn(b'run-status',page.data)
        page=client.get('/module/6-1_obj_det_trainer').data.decode()
        self.assertIn('type="number" id="epochs-input"',page)
        self.assertEqual(page.count('id="no_amp-input"'),1)

    def test_path_validation_inputs_outputs_and_dataset_relative_dem(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'project/odm_dem').mkdir(parents=True)
            (root/'project/odm_dem/dsm.tif').write_bytes(b'test')
            (root/'predictions.json').write_text('{}')
            module=MODULES_BY_ID['8_img_to_ortho']
            form={'dataset_path':'project','predictions_json':'predictions.json'}
            values=input_validation.validated_form(module,form,root)
            self.assertEqual(values['dem_filename'],str(root/'project/odm_dem/dsm.tif'))
            self.assertEqual(values['output_dir'],str(root/'project/orthorectified2'))
            for change in ({'num_threads':'0'},{'num_threads':'abc'},{'dataset_path':'missing'}):
                with self.assertRaises(ValueError):
                    input_validation.validated_form(module,{**form,**change},root)

    def test_dependency_versions_and_markers(self):
        data={'packages':{'torch':'2.6.0+cpu','Pillow':'9.0.0'},'environment':{'python_version':'3.11','sys_platform':'win32'}}
        with patch('dependency_check.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(data))):
            missing=dependency_check.missing_packages('python','torch==2.6.0\nPillow>=10\npyobjc; sys_platform == "darwin"')
        self.assertEqual(missing,['Pillow>=10'])
        with patch('dependency_check.subprocess.run',side_effect=OSError('missing')):
            self.assertEqual(dependency_check.missing_packages('missing','torch==2.6.0'),['torch==2.6.0'])

    def test_docker_python_uses_selected_interpreter(self):
        command=build_command(MODULES_BY_ID['2_odm_runner'],'chosen-python',Path('run_ODM_process.py'),{})
        self.assertEqual(command[0],'chosen-python')

    def test_logs_replay_for_multiple_viewers(self):
        run_id=process_manager.start('test-replay',[sys.executable,'-c','print("first");print("second")'],Path.cwd())
        run=process_manager.get(run_id)
        first=list(process_manager.output(run))
        second=list(process_manager.output(run))
        self.assertEqual(first,second)
        self.assertEqual(first,[(1,'first'),(2,'second')])
        self.assertEqual(list(process_manager.output(run,after=1)),[(2,'second')])
        response=app.test_client().get('/api/stream/'+run_id)
        self.assertIn(b'id: 1\ndata: first',response.data)
        self.assertIn(b'event: done',response.data)

    def test_duplicate_launch_does_not_mutate_or_start(self):
        active=SimpleNamespace(module_id='6-2_obj_det_trans_learner',done=False)
        with patch('app.process_manager.list_runs',return_value=[active]),patch('app.transfer_form.save') as save:
            response=app.test_client().post('/module/6-2_obj_det_trans_learner/run',data={})
            self.assertEqual(response.status_code,409)
            save.assert_not_called()

    def test_picker_failure_and_cancellation(self):
        client=app.test_client()
        with patch('app.platform.system',return_value='Windows'),patch('app.subprocess.run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='failed')):
            self.assertEqual(client.post('/api/browse-file').status_code,500)
        with patch('app.platform.system',return_value='Darwin'),patch('app.subprocess.run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='User canceled. (-128)')):
            self.assertEqual(client.post('/api/browse-folder').json,{'path':''})


if __name__ == '__main__':
    unittest.main()
