document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".folder-browse-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const status = button.closest(".field").querySelector(".folder-picker-status");
      button.disabled = true;
      const picker = button.dataset.picker || "folder";
      status.textContent = `Choose a ${picker} in the picker window.`;
      try {
        const response = await fetch(`/api/browse-${picker}`, { method: "POST" });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Picker failed.");
        if (data.path) {
          const input = document.getElementById(button.dataset.input);
          input.value = data.path;
          input.dispatchEvent(new Event("input", { bubbles: true }));
          status.textContent = `${picker === "file" ? "File" : "Folder"} selected.`;
        } else {
          status.textContent = "Selection cancelled. You can also paste a path.";
        }
      } catch (error) {
        status.textContent = "Could not open the picker. Paste the path instead.";
      } finally {
        button.disabled = false;
      }
    });
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const browseBtn = document.getElementById("browse-btn");
  if (browseBtn) {
    browseBtn.addEventListener("click", async () => {
      browseBtn.disabled = true;
      browseBtn.textContent = "Waiting for folder picker...";
      try {
        const resp = await fetch("/api/browse-folder", { method: "POST" });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || "Could not open the picker.");
        if (data.path) document.getElementById("repo-path-input").value = data.path;
      } catch (error) {
        let status = document.getElementById("repo-picker-status");
        if (!status) {
          status = document.createElement("p");
          status.id = "repo-picker-status";
          status.setAttribute("role", "status");
          browseBtn.after(status);
        }
        status.textContent = "Could not open the picker. Paste the folder path instead.";
      } finally {
        browseBtn.disabled = false;
        browseBtn.textContent = "Browse...";
      }
    });
  }
});

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("run-form");
  if (!form) return;

  const logBox = document.getElementById("log");
  const status = document.getElementById("run-status");
  const saveBtn = document.getElementById("save-transfer-btn");
  const stopBtn = document.getElementById("stop-btn");
  const runBtn = form.querySelector("button[type=submit]");
  const installBtn = document.getElementById("install-btn");
  const missingHint = document.getElementById("missing-hint");
  const moduleId = form.dataset.module;

  const MISSING_MODULE_RE = /ModuleNotFoundError|No module named/i;
  const EXIT_CODE_RE = /process exited with code (-?\d+)/;
  let activeSource = null;
  let activeRunId = null;
  let lastExitCode = null;

  function appendLine(text) {
    logBox.textContent += (logBox.textContent ? "\n" : "") + text;
    logBox.scrollTop = logBox.scrollHeight;
    if (missingHint && MISSING_MODULE_RE.test(text)) {
      missingHint.classList.remove("hidden");
    }
    const exitMatch = EXIT_CODE_RE.exec(text);
    if (exitMatch) {
      lastExitCode = exitMatch[1];
    }
  }

  function setBusy(busy) {
    runBtn.disabled = busy;
    if (saveBtn) saveBtn.disabled = busy;
    if (installBtn) installBtn.disabled = busy;
    stopBtn.disabled = !busy;
  }

  function streamRun(runId, isInstall) {
    activeRunId = runId;
    lastExitCode = null;
    if (activeSource) activeSource.close();
    activeSource = new EventSource(`/api/stream/${runId}`);
    activeSource.onmessage = (e) => appendLine(e.data);
    activeSource.addEventListener("done", () => {
      activeSource.close();
      setBusy(false);
      activeRunId = null;
      status.textContent = lastExitCode === "0" ? "Completed successfully." : `Stopped or failed (exit ${lastExitCode}). Check the log below.`;
      if (isInstall && lastExitCode === "0") {
        appendLine("[dashboard] Requirements installed. Refreshing this page to confirm...");
        setTimeout(() => location.reload(), 1200);
      }
    });
    activeSource.onerror = () => {
      status.textContent = "Log connection interrupted. Reconnecting; the process may still be running.";
    };
  }

  async function launch(url, isInstall) {
    logBox.textContent = "";
    if (missingHint) missingHint.classList.add("hidden");
    setBusy(true);
    status.textContent = isInstall ? "Installing requirements…" : "Starting…";
    try {
      const resp = await fetch(url, { method: "POST", body: new FormData(form) });
      const data = await resp.json();
      if (!resp.ok || data.error) throw new Error(data.error || "Could not start this step.");
      status.textContent = isInstall ? "Installing requirements…" : "Running. Follow the output below.";
      streamRun(data.run_id, isInstall);
    } catch (error) {
      appendLine(`[dashboard] ${error.message}`);
      status.textContent = "Could not start. Check the message below and try again.";
      setBusy(false);
    }
  }

  form.addEventListener("submit", (evt) => {
    evt.preventDefault();
    launch(`/module/${moduleId}/run`, false);
  });

  if (installBtn) {
    installBtn.addEventListener("click", () => launch(`/module/${moduleId}/install`, true));
  }

  stopBtn.addEventListener("click", async () => {
    if (!activeRunId) return;
    stopBtn.disabled = true;
    try {
      const response = await fetch(`/api/stop/${activeRunId}`, { method: "POST" });
      const data = await response.json();
      if (!response.ok || !data.stopped) throw new Error("The process could not be stopped.");
      status.textContent = "Stopping…";
    } catch (error) {
      appendLine(`[dashboard] ${error.message}`);
      status.textContent = "Could not stop. Try again or close the desktop app window.";
      stopBtn.disabled = false;
    }
  });
  document.querySelectorAll(".view-run-btn").forEach((button) => {
    button.addEventListener("click", () => {
      logBox.textContent = "";
      setBusy(true);
      status.textContent = "Loading this run's output…";
      streamRun(button.dataset.run, false);
    });
  });
});

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".field-help").forEach((help) => {
    const button = help.querySelector("button");
    const popup = help.querySelector(".field-help-popup");
    let pinned = false;
    const show = (visible) => {
      popup.hidden = !visible;
      if (visible) {
        const anchor = button.getBoundingClientRect();
        popup.style.left = `${Math.max(16, Math.min(anchor.left, innerWidth - popup.offsetWidth - 16))}px`;
        popup.style.top = `${Math.max(16, Math.min(anchor.bottom + 6, innerHeight - popup.offsetHeight - 16))}px`;
      }
      button.setAttribute("aria-expanded", String(visible));
    };
    window.addEventListener("scroll", () => { if (!popup.hidden) show(true); }, { passive: true });
    window.addEventListener("resize", () => { if (!popup.hidden) show(true); });
    help.addEventListener("mouseenter", () => show(true));
    help.addEventListener("mouseleave", () => { if (!pinned && document.activeElement !== button) show(false); });
    button.addEventListener("focus", () => show(true));
    button.addEventListener("blur", () => { if (!pinned) show(false); });
    button.addEventListener("click", () => { pinned = !pinned; show(pinned); });
    document.addEventListener("click", (event) => {
      if (!help.contains(event.target)) { pinned = false; show(false); }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") { pinned = false; show(false); }
    });
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("save-transfer-btn");
  if (!button) return;
  const form = document.getElementById("run-form");
  const status = document.getElementById("transfer-save-status");
  button.addEventListener("click", async () => {
    if (!form.reportValidity()) return;
    button.disabled = true;
    try {
      const response = await fetch("/module/6-2_obj_det_trans_learner/config/form", {method: "POST", body: new FormData(form)});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not save settings.");
      status.textContent = data.message;
      // Keep the advanced editor in sync with the saved YAML.
      location.reload();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
});
