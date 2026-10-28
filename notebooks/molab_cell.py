# Kindle on molab: paste this into one cell (no leading spaces) and run it.
# ROLE = "head" in the session that runs the model and serves the API,
# ROLE = "helper" in every extra session that lends its GPU to the head's pool.
ROLE = "head"
REPO = "OWNER/Kindle"  # your GitHub repo (public, or see README for a private one)
REF = "main"           # branch or tag to install
UPDATE = False         # True reinstalls from GitHub; restart the kernel afterwards so the new code loads

import importlib, os, subprocess, sys
os.environ["KINDLE_REPO"] = REPO
if UPDATE or subprocess.run([sys.executable, "-c", "import kindle"], capture_output=True).returncode != 0:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", f"kindle @ git+https://github.com/{REPO}@{REF}"], check=True)
importlib.invalidate_caches()
import kindle.molab
kindle.molab.start(ROLE)
