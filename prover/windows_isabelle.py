
import subprocess
import os

ISABELLE_HOME = r"C:\Users\user\Desktop\Isabelle2025-2"
ISABELLE_BASH = ISABELLE_HOME + r"\contrib\cygwin\bin\bash.exe"

def start_isabelle_server_windows(name="isabelle", log_file=None):
    cmd = [
        ISABELLE_BASH,
        "--login",
        "-c",
        f"export PATH=/usr/bin:$PATH && isabelle server -n {name}"
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ISABELLE_HOME,
    )
    for _ in range(20):
        line = proc.stdout.readline().strip()
        if line.startswith("server"):
            return line, proc
    raise RuntimeError("Failed to start Isabelle server")
