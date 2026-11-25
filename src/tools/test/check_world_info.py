# World-info parsing check (no simulator needed)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check that world names and motor topics are derived from the SDF, not hardcoded.

Guards `quadfc.plant.gzsim.read_world_info`, which is what lets `--world` point
at a new SDF without the plant silently talking to the old world's
`/world/<name>/set_pose` service and topics.

This runs without Gazebo and on any OS: it only parses XML. The end-to-end
failure modes it protects against (wrong world name, wrong model name) do need
a running simulator, see README "Gazebo 接口明细".

    python3 tools/test/check_world_info.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # .../src

from quadfc.plant.gzsim import read_world_info

ROOT = Path(__file__).resolve().parent.parent.parent  # .../src
WORLD = ROOT / "worlds" / "x3_quadcopter.sdf"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


print(f"parsing {WORLD}")
world_name, robot_ns, command_sub = read_world_info(WORLD)
check("world name", world_name == "quadcopter", repr(world_name))
check("robot namespace", robot_ns == "X3", repr(robot_ns))
check("command subtopic", command_sub == "gazebo/command/motor_speed", repr(command_sub))

# A renamed world must be picked up, otherwise --world is a lie.
text = WORLD.read_text(encoding="utf-8")
renamed = text.replace('<world name="quadcopter">', '<world name="warehouse">')
check("rename is actually applied", renamed != text)
with tempfile.TemporaryDirectory() as tmp:
    alt = Path(tmp) / "warehouse.sdf"
    alt.write_text(renamed, encoding="utf-8")
    check("renamed world name", read_world_info(alt)[0] == "warehouse")

    # Missing file -> all None, so the caller can decide whether that is fatal.
    check("missing file is not an exception", read_world_info(Path(tmp) / "nope.sdf") == (None, None, None))

    broken = Path(tmp) / "broken.sdf"
    broken.write_text("<sdf><world", encoding="utf-8")
    check("malformed XML is not an exception", read_world_info(broken) == (None, None, None))

# Fallback path: nothing to parse means the plant must keep working unchanged.
defaults = read_world_info(Path("/definitely/not/here.sdf"))
check("unparseable yields no names", defaults == (None, None, None))

print(f"\n{'all checks passed' if not failures else 'failed: ' + ', '.join(failures)}")
raise SystemExit(1 if failures else 0)
