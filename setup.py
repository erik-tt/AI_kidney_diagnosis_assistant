import subprocess
import os

env_name = "AI-diagnostic"
path = os.path.abspath("")

try: 
    subprocess.run(
        ["conda", "env", "config", "vars", "set", f"PYTHONPATH={path}", "-n", env_name],
        check=True,
    )
    print(f"PYTHONPATH set to: {path}")
except subprocess.CalledProcessError as e:
    print(f"Failed to set PYTHONPATH: {e}")