"""Create the local development/notebook environment: python3 setup_env.py."""

import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or later is required.")

    project = Path(__file__).resolve().parent
    environment = project / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    if not environment.exists():
        print("Creating .venv...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    elif not python.is_file():
        parser.error(".venv exists but has no Python interpreter; check that folder before retrying.")
    else:
        print("Using existing .venv...", flush=True)

    print("Installing My TinyTorch and development/notebook dependencies...", flush=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "-e", f"{project}[dev,notebook]"],
        check=True,
    )
    # Register in the user Jupyter location so other Jupyter environments can discover it.
    # Rerunning setup updates this name to the current checkout and interpreter.
    subprocess.run(
        [str(python), "-m", "ipykernel", "install", "--user",
         "--name", "my-tiny-torch", "--display-name", "My Tiny Torch (.venv)"],
        check=True,
    )

    print("\nSetup complete.")
    print("Open examples/my_tiny_torch_playground.ipynb in VS Code.")
    print("In the notebook kernel picker choose Select Another Kernel, then")
    print("Jupyter Kernel -> My Tiny Torch (.venv). Select it once if prompted.")
    print("If VS Code was already open, run Developer: Reload Window to refresh discovery.")
    print("The kernel uses this interpreter:")
    print(python)
    print("Rerun setup after moving the checkout or recreating .venv.")
    print("Restart the notebook kernel after editing imported library code.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("\nSetup did not finish. Fix the error above and rerun this script.", file=sys.stderr)
        sys.exit(error.returncode)
