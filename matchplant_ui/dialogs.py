"""Native system pickers with separate folder, input-file and save-file rules."""
from fnmatch import fnmatchcase
from pathlib import Path
import re
import subprocess
import sys


def accepted_patterns(filetypes):
    patterns = []
    for _label, specification in filetypes or ():
        for pattern in (specification if isinstance(specification, (tuple, list)) else re.split(r"[\s;]+", specification)):
            if pattern and pattern not in ("*", "*.*"):
                patterns.append(pattern.casefold())
    return patterns


def valid_file(path, filetypes=None, save=False, defaultextension=None):
    candidate = Path(path).expanduser()
    if save and defaultextension and not candidate.suffix:
        candidate = candidate.with_suffix(defaultextension)
    patterns = accepted_patterns(filetypes)
    if patterns and not any(fnmatchcase(candidate.name.casefold(), pattern) for pattern in patterns):
        raise ValueError("Choose a file of the required type: " + ", ".join(patterns))
    if save:
        if candidate.is_dir() or not candidate.parent.is_dir():
            raise ValueError("Choose a file name inside an existing folder.")
    elif not candidate.is_file():
        raise ValueError("Choose an existing file, not a folder.")
    return str(candidate.resolve())


def _mac_panel(kind, title, initialdir="", initialfile=""):
    script = r'''ObjC.import('Cocoa');
function run(argv) {
    var app = $.NSApplication.sharedApplication;
    app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);
    var save = argv[0] === 'save';
    var folder = argv[0] === 'folder';
    var panel = save ? $.NSSavePanel.savePanel : $.NSOpenPanel.openPanel;
    panel.title = argv[1];
    if (save) {
        panel.canCreateDirectories = true;
        panel.nameFieldStringValue = argv[3] || 'output.csv';
    } else {
        panel.canChooseFiles = !folder;
        panel.canChooseDirectories = folder;
        panel.allowsMultipleSelection = false;
    }
    if (folder) {
        panel.prompt = 'Select Folder';
        panel.message = 'Select a folder, not an individual file.';
    }
    if (argv[2]) panel.directoryURL = $.NSURL.fileURLWithPath(argv[2]);
    app.activateIgnoringOtherApps(true);
    return panel.runModal == $.NSModalResponseOK ? ObjC.unwrap(panel.URL.path) : '';
}'''
    result = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", script, kind, title, str(initialdir or ""), str(initialfile or "")],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError("Could not open picker: " + result.stderr.strip())
    return result.stdout.strip() or None


def _notice(message, parent=None):
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-l", "JavaScript", "-e", r'''ObjC.import('Cocoa');
function run(argv) { var app=$.NSApplication.sharedApplication; app.setActivationPolicy($.NSApplicationActivationPolicyAccessory); app.activateIgnoringOtherApps(true); var alert=$.NSAlert.alloc.init; alert.messageText='Choose the required file type'; alert.informativeText=argv[0]; alert.runModal; }''', message], check=True)
    else:
        from tkinter import messagebox
        messagebox.showerror("Choose the required file type", message, parent=parent)


def askdirectory(title="Select folder", initialdir=None, parent=None, **_options):
    if sys.platform == "darwin":
        result = _mac_panel("folder", title, initialdir)
    else:
        from tkinter import filedialog
        arguments = dict(_options, title=title, mustexist=True)
        if initialdir and Path(initialdir).is_dir():
            arguments["initialdir"] = str(initialdir)
        if parent is not None:
            arguments["parent"] = parent
        result = filedialog.askdirectory(**arguments)
    if not result:
        return None
    if not Path(result).is_dir():
        raise ValueError("The picker must return a folder, not a file.")
    return str(Path(result).resolve())


def _pick_file(save=False, title="Select file", filetypes=None, parent=None,
               initialdir=None, initialfile=None, defaultextension=None, **options):
    if sys.platform != "darwin" and parent is None:
        import tkinter as tk
        owner = tk.Tk()
        owner.withdraw()
        try:
            return _pick_file(save, title, filetypes, owner, initialdir, initialfile, defaultextension, **options)
        finally:
            owner.destroy()
    while True:
        if sys.platform == "darwin":
            selected = _mac_panel("save" if save else "file", title, initialdir, initialfile)
        else:
            from tkinter import filedialog
            picker = filedialog.asksaveasfilename if save else filedialog.askopenfilename
            arguments = dict(options, title=title, filetypes=[("All files", "*")])
            for key, value in (("parent", parent), ("initialdir", initialdir), ("initialfile", initialfile)):
                if value is not None:
                    arguments[key] = value
            if save and defaultextension:
                arguments["defaultextension"] = defaultextension
            selected = picker(**arguments)
        if not selected:
            return None
        try:
            result = valid_file(selected, filetypes, save, defaultextension)
        except ValueError as error:
            _notice(str(error), parent)
            initialdir = str(Path(selected).parent)
            continue
        # A default extension can change the path after a native save dialog.
        # Confirm that exact destination as well before allowing an overwrite.
        if save and Path(result).exists() and Path(selected).resolve() != Path(result):
            if sys.platform == "darwin":
                script = 'on run argv\ndisplay dialog ("Replace existing file: " & item 1 of argv & "?") buttons {"Cancel", "Replace"} default button "Cancel"\nend run'
                confirmation = subprocess.run(["osascript", "-e", script, result], capture_output=True, text=True)
                if confirmation.returncode:
                    return None
            else:
                from tkinter import messagebox
                if not messagebox.askyesno("Replace existing file?", result, parent=parent):
                    return None
        return result


def askopenfilename(title="Select file", filetypes=None, **kwargs):
    return _pick_file(save=False, title=title, filetypes=filetypes, **kwargs)


def asksaveasfilename(title="Save As", filetypes=None, defaultextension=None, **kwargs):
    return _pick_file(save=True, title=title, filetypes=filetypes, defaultextension=defaultextension, **kwargs)


class FileDialogHandler:
    ask_directory = staticmethod(askdirectory)
    ask_open_filename = staticmethod(askopenfilename)
    ask_save_filename = staticmethod(asksaveasfilename)
