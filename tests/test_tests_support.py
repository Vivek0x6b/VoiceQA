import subprocess
import sys

from tests_support import child_processes


def test_child_processes_lists_a_child_with_its_full_command_line():
    # The marker comes right after the interpreter path and "-c". An
    # 80-column ps line (what ps gives without a terminal) holds at most
    # 68 characters of command, so with CI's interpreter paths (56 to 90
    # characters) it could not show these 27 characters of the marker.
    marker = "marker_" + "x" * 60
    code = f"{marker} = 1; import time; time.sleep(30)"
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        listed = [c for c in child_processes() if c.startswith(f"{proc.pid} ")]
        assert len(listed) == 1
        if sys.platform != "win32":
            assert marker[:27] in listed[0]
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert not [c for c in child_processes() if c.startswith(f"{proc.pid} ")]
