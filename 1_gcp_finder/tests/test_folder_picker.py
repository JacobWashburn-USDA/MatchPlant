import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from matchplant_ui import dialogs

class NativeFolderTests(unittest.TestCase):
    def test_windows_uses_native_folder_picker_and_returns_a_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(dialogs.sys, 'platform', 'win32'), patch('tkinter.filedialog.askdirectory', return_value=temporary) as native:
                self.assertEqual(dialogs.askdirectory(initialdir=temporary), str(Path(temporary).resolve()))
                self.assertTrue(native.call_args.kwargs['mustexist'])
                self.assertEqual(native.call_args.kwargs['initialdir'], temporary)

    def test_cancel_returns_no_path_and_file_response_is_rejected(self):
        with patch.object(dialogs.sys, 'platform', 'win32'), patch('tkinter.filedialog.askdirectory', return_value=''):
            self.assertIsNone(dialogs.askdirectory())
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'image.jpg'
            file.write_text('fixture')
            with patch.object(dialogs.sys, 'platform', 'win32'), patch('tkinter.filedialog.askdirectory', return_value=str(file)):
                with self.assertRaises(ValueError):
                    dialogs.askdirectory()

    def test_each_variant_has_one_image_folder_button(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ('gcp_finder_win.py', 'gcp_finder_mac.py'):
            tree = ast.parse((root / filename).read_text(encoding='utf-8'))
            section = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_setup_path_section')
            labels = [n.value for n in ast.walk(section) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            self.assertEqual(labels.count('Browse...'), 2)
            self.assertNotIn('Folder...', labels)

if __name__ == '__main__':
    unittest.main()
