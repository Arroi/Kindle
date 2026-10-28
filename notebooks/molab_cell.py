# Kindle on molab: paste this into one cell (no leading spaces) and run it.
# ROLE = "head" in the session that runs the model and serves the API,
# ROLE = "helper" in every extra session that lends its GPU to the head's pool.
ROLE = "head"
REPO = "Arroi/Kindle"  # GitHub repo to install from (must be public for this install line)
REF = "main"           # branch or tag to install
UPDATE = False         # True: fetch the latest Kindle from GitHub

import importlib, importlib.metadata, os, subprocess, sys
os.environ["KINDLE_REPO"] = REPO
_source = f"kindle @ git+https://github.com/{REPO}@{REF}"
try:
    importlib.metadata.version("kindle")  # the installed package, not just any folder named "kindle"
    _installed = True
except importlib.metadata.PackageNotFoundError:
    _installed = False
if UPDATE or not _installed:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *(["--force-reinstall", "--no-deps"] if _installed else []), _source], check=True)
for _m in [m for m in sys.modules if m == "kindle" or m.startswith("kindle.")]:
    del sys.modules[_m]  # forget anything imported earlier (an old version, or a folder named kindle)
importlib.invalidate_caches()
import kindle.molab
kindle.molab.start(ROLE)
