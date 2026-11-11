import subprocess
import sys

from tests_support import child_processes


def test_child_processes_lists_a_child_with_its_full_command_line():
    # Long enough that an 80-column ps line would cut the marker off.
    marker = "marker_" + "x" * 100
    code = f"import time; {marker} = 1; time.sleep(30)"
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        listed = [c for c in child_processes() if c.startswith(f"{proc.pid} ")]
        assert len(listed) == 1
        if sys.platform != "win32":
            assert marker[:100] in listed[0]
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert not [c for c in child_processes() if c.startswith(f"{proc.pid} ")]
