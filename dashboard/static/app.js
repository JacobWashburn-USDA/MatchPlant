document.addEventListener("DOMContentLoaded", () => {
  const browseBtn = document.getElementById("browse-btn");
  if (browseBtn) {
    browseBtn.addEventListener("click", async () => {
      browseBtn.disabled = true;
      browseBtn.textContent = "Waiting for folder picker...";
      try {
        const resp = await fetch("/api/browse-folder", { method: "POST" });
        const data = await resp.json();
        if (data.path) {
          document.getElementById("repo-path-input").value = data.path;
        }
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
      if (isInstall && lastExitCode === "0") {
        appendLine("[dashboard] Requirements installed. Refreshing this page to confirm...");
        setTimeout(() => location.reload(), 1200);
      }
    });
    activeSource.onerror = () => {
      activeSource.close();
      setBusy(false);
    };
  }

  async function launch(url, isInstall) {
    logBox.textContent = "";
    if (missingHint) missingHint.classList.add("hidden");
    setBusy(true);
    const resp = await fetch(url, { method: "POST", body: new FormData(form) });
    const data = await resp.json();
    if (data.error) {
      appendLine(`[dashboard] ${data.error}`);
      setBusy(false);
      return;
    }
    streamRun(data.run_id, isInstall);
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
    await fetch(`/api/stop/${activeRunId}`, { method: "POST" });
  });
});
