#!/usr/bin/env bash
# Regenerate stationeers_logic.json from the game.
#
# Device logic tables only exist at runtime (each prefab answers CanLogicRead/CanLogicWrite), so
# this downloads the Stationeers dedicated server (Steam app 600760, ~5 GB, anonymous login),
# installs BepInEx into it, and runs it headless with the SiccLogicDump.cs plugin, which writes the
# JSON and quits. Afterwards run `python scripts/build_types.py` to regenerate sicc/devices.py.
#
# Usage: scripts/game_data/dump.sh WORKDIR
#   WORKDIR holds the server, BepInEx and steamcmd state; it is reused by later runs.
#   Needs steamcmd, mcs (mono), curl, unzip, sha256sum. With nix:
#     NIXPKGS_ALLOW_UNFREE=1 nix shell --impure nixpkgs#steamcmd nixpkgs#mono -c scripts/game_data/dump.sh WORKDIR

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
work=${1:?usage: dump.sh WORKDIR}
mkdir -p "$work"
work=$(cd "$work" && pwd)
server=$work/server
managed=$server/rocketstation_DedicatedServer_Data/Managed
out=$here/stationeers_logic.json

bepinex_version=5.4.23.3
bepinex_sha256=b9ef28b37676f18277cfd8dded01f675e934f207c30f65f2a4bb93c7e41abdbb

# steamcmd and the server keep state under $HOME; keep it inside WORKDIR
export HOME=$work/home
mkdir -p "$HOME"

# +app_info_update 1: without it, a fresh steamcmd fails with "Missing configuration"
steamcmd +force_install_dir "$server" +login anonymous +app_info_update 1 +app_update 600760 validate +quit

bepinex_zip=$work/BepInEx_linux_x64_$bepinex_version.zip
if [ ! -f "$bepinex_zip" ]; then
    curl -fsSL -o "$bepinex_zip" \
        "https://github.com/BepInEx/BepInEx/releases/download/v$bepinex_version/BepInEx_linux_x64_$bepinex_version.zip"
fi
echo "$bepinex_sha256  $bepinex_zip" | sha256sum -c -
unzip -oq "$bepinex_zip" -d "$server"
chmod +x "$server/run_bepinex.sh"

refs=()
for dll in "$managed"/*.dll; do
    refs+=("-r:$dll")
done
mkdir -p "$server/BepInEx/plugins"
# CS1685: the game ships both mscorlib and netstandard facades defining the same types
mcs -target:library -nostdlib -nowarn:1685 \
    -out:"$server/BepInEx/plugins/SiccLogicDump.dll" \
    "${refs[@]}" -r:"$server/BepInEx/core/BepInEx.dll" \
    "$here/SiccLogicDump.cs"

rm -f "$work/server.log" "$out.tmp"
status=0
(
    cd "$server"
    SICC_DUMP_OUT="$out.tmp" timeout 900 ./run_bepinex.sh ./rocketstation_DedicatedServer.x86_64 \
        -batchmode -nographics -logFile "$work/server.log"
) || status=$?
grep -h 'SiccLogicDump' "$work/server.log" || true
if [ "$status" -ne 0 ]; then
    echo "dump.sh: server exited with status $status; see $work/server.log and $server/BepInEx/LogOutput.log" >&2
    exit "$status"
fi
mv "$out.tmp" "$out"
echo "dump.sh: wrote $out"
