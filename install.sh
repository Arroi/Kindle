#!/usr/bin/env bash
# Install the latest KaggleUI release (wheel includes the prebuilt web UI).
#   !curl -fsSL https://raw.githubusercontent.com/UltraFEmotes/KaggleUI/main/install.sh | bash
set -euo pipefail
REPO="UltraFEmotes/KaggleUI"
URL=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
  | python3 -c "import json,sys; a=[x['browser_download_url'] for x in json.load(sys.stdin)['assets'] if x['name'].endswith('.whl')]; print(a[0])")
echo "Installing ${URL##*/}"
pip install -q --disable-pip-version-check "${URL}"
echo "Done. Next: import kaggleui; kaggleui.launch()"
