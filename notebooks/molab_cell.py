# Kindle on molab: paste this into one cell (no leading spaces) and run it.
# ROLE = "head" in the session that runs the model and serves the API,
# ROLE = "helper" in every extra session that lends its GPU to the head's pool.
ROLE = "head"
REPO = "Arroi/Kindle"  # GitHub repo to install from (must be public for this install line)
REF = "main"           # branch or tag to install
UPDATE = False         # True: fetch the latest Kindle from GitHub (restart the kernel first if Kindle ran in it)

import importlib, os, subprocess, sys
os.environ["KINDLE_REPO"] = REPO
SOURCE = f"kindle @ git+https://github.com/{REPO}@{REF}"
if subprocess.run([sys.executable, "-c", "import kindle"], capture_output=True).returncode != 0:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", SOURCE], check=True)
elif UPDATE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", SOURCE], check=True)
importlib.invalidate_caches()
import kindle.molab
kindle.molab.start(ROLE)
