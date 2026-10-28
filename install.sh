#!/usr/bin/env bash
# Install Kindle: the latest release wheel (prebuilt web UI) when there is one, otherwise the
# current main branch built from source (needs Node.js 20+ for the web UI).
#   !curl -fsSL https://raw.githubusercontent.com/Arroi/Kindle/main/install.sh | bash
set -euo pipefail
REPO="${KINDLE_REPO:-Arroi/Kindle}"
REF="${KINDLE_REF:-}"                     # a branch or tag to build from source instead of the latest release
URL=""
if [ -z "$REF" ]; then
  URL=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
    | python3 -c "import json,sys; a=[x['browser_download_url'] for x in json.load(sys.stdin).get('assets',[]) if x['name'].endswith('.whl')]; print(a[0] if a else '')" \
    2>/dev/null || true)
fi

if [ -n "$URL" ]; then
  WHEEL="${URL##*/}"
  NEW_VERSION=$(echo "$WHEEL" | cut -d- -f2)
  echo "Installing ${WHEEL}"
  pip install -q --disable-pip-version-check "${URL}"
else
  SRC=/tmp/Kindle-src
  echo "Building ${REPO}${REF:+ @ $REF} from source (a minute or two)"
  rm -rf "$SRC"
  git clone -q --depth 1 ${REF:+--branch "$REF"} "https://github.com/${REPO}" "$SRC"
  (cd "$SRC/web" && npx -y pnpm@10 install --frozen-lockfile --silent && npx -y pnpm@10 build >/dev/null)
  pip install -q --disable-pip-version-check "$SRC"
  NEW_VERSION=$(python3 -c "import re; print(re.search(r'^version = \"(.+)\"', open('$SRC/pyproject.toml').read(), re.M).group(1))")
fi

# If an older Kindle server is still running in this session, stop the app but retain its tunnel.
# Otherwise the notebook's
# already-imported (old) kindle.launch() would keep reusing it while it serves the new web UI.
python3 - "$NEW_VERSION" <<'EOF'
import sys

from kindle.config import Paths
from kindle.launcher import _read_state, _running, stop

new = sys.argv[1]
paths = Paths()
st = _read_state(paths)
if _running(st) and st.get("version") != new:
    print(f"Stopping the running Kindle ({st.get('version') or 'older version'}) so {new} starts fresh. "
          "Relaunch your models afterwards.")
    stop(str(paths.home), _keep_tunnel=True)
EOF
echo "Done. Next: import kindle; kindle.launch()"
