"""Regression checks for the user-facing launchers and internal app layout."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as dashboard
import settings
import venv_manager
from modules import MODULES, module_dir


class LayoutTests(unittest.TestCase):
    def test_existing_settings_and_environments_keep_their_locations(self):
        dashboard_root = Path(__file__).resolve().parents[2]
        self.assertEqual(settings.INSTANCE_DIR, dashboard_root / "instance")
        self.assertEqual(venv_manager.VENVS_DIR, dashboard_root / "instance" / "venvs")
        saved = {"python": {"test-module": "existing-environment/python"}}
        with patch.object(settings, "_load", return_value=saved):
            self.assertEqual(settings.get_python("test-module"), "existing-environment/python")

    def test_repo_discovery_and_pages_do_not_depend_on_working_directory(self):
        dashboard_root = Path(__file__).resolve().parents[2]
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                with patch.object(settings, "_load", return_value={}):
                    self.assertEqual(settings.get_repo_root(), dashboard_root.parent)
                    self.assertTrue(settings.is_valid_repo_root(settings.get_repo_root()))
                    with dashboard.app.test_client() as client:
                        for route in ("/", "/guide", "/settings", "/static/style.css", "/static/app.js", "/static/logo.png"):
                            with self.subTest(route=route):
                                with client.get(route) as response:
                                    self.assertEqual(response.status_code, 200)
                        for module in MODULES:
                            with self.subTest(module=module["id"]):
                                self.assertTrue(module_dir(module).is_dir())
                                with client.get("/module/" + module["id"]) as response:
                                    self.assertEqual(response.status_code, 200)
            finally:
                os.chdir(original_cwd)

    def test_launchers_reference_existing_helpers(self):
        root = Path(__file__).resolve().parents[2]
        helpers = {
            "1_Setup_Windows.bat": "setup_windows.bat",
            "2_Start_Windows.bat": "start_windows.bat",
            "1_Setup_macOS.command": "setup.sh",
            "2_Start_macOS.command": "app.py",
        }
        for launcher, helper in helpers.items():
            with self.subTest(launcher=launcher):
                self.assertTrue((root / "_internal" / helper).is_file())
                self.assertIn(helper, (root / launcher).read_text())
        for launcher in ("1_Setup_macOS.command", "2_Start_macOS.command"):
            self.assertNotIn(b"\r\n", (root / launcher).read_bytes())


if __name__ == "__main__":
    unittest.main()
