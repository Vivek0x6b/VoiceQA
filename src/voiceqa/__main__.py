"""Lets `python -m voiceqa` run the CLI without relying on the console script
being on PATH. Used by the end-to-end test to start `voiceqa agent` and
`voiceqa call` as real subprocesses with sys.executable.
"""

from voiceqa.cli.main import app

if __name__ == "__main__":
    app()
