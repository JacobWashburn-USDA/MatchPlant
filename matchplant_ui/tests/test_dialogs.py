import ast
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from matchplant_ui import dialogs


class PickerPolicyTests(unittest.TestCase):

    def test_required_type_is_enforced_even_with_all_files_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv = root / "bounds.CSV"
            csv.write_text("fixture")
            raster = root / "ortho.tiff"
            raster.write_text("fixture")
            policy = [("CSV", "*.csv"), ("All files", "*.*")]
            self.assertEqual(dialogs.valid_file(csv, policy), str(csv.resolve()))
            with self.assertRaises(ValueError):
                dialogs.valid_file(raster, policy)
            with self.assertRaises(ValueError):
                dialogs.valid_file(root)
            self.assertEqual(dialogs.valid_file(raster, [("Raster", "*.tif *.tiff *.img")]), str(raster.resolve()))

    def test_output_file_can_be_new_and_default_extension_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new output"
            policy = [("CSV", "*.csv"), ("All", "*.*")]
            self.assertEqual(dialogs.valid_file(output, policy, save=True, defaultextension=".csv"), str(output.with_suffix(".csv").resolve()))
            with self.assertRaises(ValueError):
                dialogs.valid_file(output.with_suffix(".jpg"), policy, save=True)
            with self.assertRaises(ValueError):
                dialogs.valid_file(output / "missing" / "out.csv", policy, save=True)

    def test_wrong_file_is_rejected_then_retry_accepts_correct_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "image.jpg"
            correct = root / "input.csv"
            wrong.write_text("fixture")
            correct.write_text("fixture")
            with patch.object(dialogs.sys, "platform", "darwin"), patch.object(dialogs, "_mac_panel", side_effect=[str(wrong), str(correct)]), patch.object(dialogs, "_notice") as notice:
                self.assertEqual(dialogs.askopenfilename(filetypes=[("CSV", "*.csv")]), str(correct.resolve()))
                notice.assert_called_once()

    def test_cancellation_returns_no_path(self):
        with patch.object(dialogs.sys, "platform", "darwin"), patch.object(dialogs, "_mac_panel", return_value=None):
            self.assertIsNone(dialogs.askdirectory())
            self.assertIsNone(dialogs.askopenfilename(filetypes=[("CSV", "*.csv")]))
            self.assertIsNone(dialogs.asksaveasfilename(defaultextension=".csv"))

    def test_mac_folder_response_cannot_be_a_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "image.jpg"
            file.write_text("fixture")
            with patch.object(dialogs.sys, "platform", "darwin"), patch.object(dialogs, "_mac_panel", return_value=str(file)):
                with self.assertRaises(ValueError):
                    dialogs.askdirectory()

    def test_mac_panel_uses_folder_only_flags_and_safe_arguments(self):
        reply = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(dialogs.subprocess, "run", return_value=reply) as run:
            self.assertIsNone(dialogs._mac_panel("folder", 'Images "flight 1"', "/custom path"))
            args = run.call_args.args[0]
            self.assertEqual(args[-4:], ["folder", 'Images "flight 1"', "/custom path", ""])
            self.assertIn("panel.canChooseFiles = !folder", args[4])
            self.assertIn("panel.canChooseDirectories = folder", args[4])

    def test_windows_file_dialog_shows_all_files_but_rejects_wrong_type(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "image.jpg"
            right = root / "gcp.csv"
            wrong.write_text("fixture")
            right.write_text("fixture")
            with patch.object(dialogs.sys, "platform", "win32"), patch("tkinter.filedialog.askopenfilename", side_effect=[str(wrong), str(right)]) as native, patch.object(dialogs, "_notice") as notice:
                result = dialogs.askopenfilename(parent=Mock(), filetypes=[("CSV", "*.csv")])
                self.assertEqual(result, str(right.resolve()))
                self.assertEqual(native.call_args.kwargs["filetypes"], [("All files", "*")])
                notice.assert_called_once()


class ModuleWiringTests(unittest.TestCase):
    def test_all_ten_variants_use_shared_pickers_without_duplicate_folder_buttons(self):
        repo = Path(__file__).resolve().parents[2]
        scripts = [p for directory in ("1_gcp_finder", "3_min_img_finder", "4_bbox_drawer", "5_img_splitter", "9_spatial_stats_extractor") for p in (repo / directory).glob("*.py") if p.name.endswith(("_win.py", "_mac.py"))]
        self.assertEqual(len(scripts), 10)
        for script in scripts:
            with self.subTest(script=script.name):
                source = script.read_text(encoding="utf-8")
                ast.parse(source)
                self.assertIn("matchplant_ui", source)
                self.assertNotIn("'Folder...'", source)
                self.assertNotIn("files are not shown", source)
                self.assertNotIn("Choose any file in the image folder", source)

    def test_minimum_image_finder_and_splitter_keep_distinct_input_targets(self):
        repo = Path(__file__).resolve().parents[2]
        for relative, choices in (
            ("3_min_img_finder/min_img_finder_win.py", {"ortho": "ortho_input", "img": "img_input", "undist": "undist_input"}),
            ("5_img_splitter/img_splitter_win.py", {"img": "img_input", "ann": "ann_input", "out": "out_input"}),
        ):
            tree = ast.parse((repo / relative).read_text(encoding="utf-8"))
            callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "browse_file")
            namespace = {"filedialog": SimpleNamespace(askdirectory=Mock(return_value="chosen folder"), askopenfilename=Mock(return_value="chosen file"))}
            exec(compile(ast.Module(body=[callback], type_ignores=[]), relative, "exec"), namespace)
            for choice, target in choices.items():
                inputs = {name: Mock(text="") for name in choices.values()}
                window = SimpleNamespace(**inputs, fig=SimpleNamespace(canvas=Mock()))
                namespace["browse_file"](window, choice)
                inputs[target].set_val.assert_called_once()
                for name, value in inputs.items():
                    if name != target:
                        value.set_val.assert_not_called()


if __name__ == "__main__":
    unittest.main()
