#!/usr/bin/env bash
# Install the latest KaggleUI release (wheel includes the prebuilt web UI).
#   !curl -fsSL https://raw.githubusercontent.com/UltraFEmotes/KaggleUI/main/install.sh | bash
set -euo pipefail
REPO="UltraFEmotes/KaggleUI"
URL=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
  | python3 -c "import json,sys; a=[x['browser_download_url'] for x in json.load(sys.stdin)['assets'] if x['name'].endswith('.whl')]; print(a[0])")
WHEEL="${URL##*/}"
NEW_VERSION=$(echo "$WHEEL" | cut -d- -f2)
echo "Installing ${WHEEL}"
pip install -q --disable-pip-version-check "${URL}"

# If an older KaggleUI server is still running in this session, stop it. Otherwise the notebook's
# already-imported (old) kaggleui.launch() would keep reusing it while it serves the new web UI.
python3 - "$NEW_VERSION" <<'EOF'
import json, os, signal, sys, time

new = sys.argv[1]
home = os.environ.get("KAGGLEUI_HOME", "/tmp/kaggleui")
path = os.path.join(home, "state.json")
try:
    st = json.load(open(path))
except Exception:
    sys.exit(0)
pid = st.get("pid")

def alive(p):
    try:
        os.kill(p, 0)
    except OSError:
        return False
    try:  # a stopped server can linger as a zombie until the notebook reaps it
        with open(f"/proc/{p}/stat") as f:
            return f.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except OSError:
        return True

if not pid or not alive(pid) or st.get("version") == new:
    sys.exit(0)
print(f"Stopping the running KaggleUI ({st.get('version') or 'older version'}) so {new} starts fresh. "
      "Relaunch your models afterwards.")
os.kill(pid, signal.SIGTERM)
for _ in range(40):
    if not alive(pid):
        break
    time.sleep(0.5)
else:
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        os.kill(pid, signal.SIGKILL)
os.replace(path, path + ".old")
EOF
echo "Done. Next: import kaggleui; kaggleui.launch()"
