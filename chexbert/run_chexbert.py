import subprocess
from local_config import CHEXBERT_ENV_PATH, CHEXBERT_PATH


def run_chexbert_labeler(reports_path, output_path):
    # Path to a Python interpreter that runs any Python script
    # under the virtualenv /path/to/virtualenv/
    python_bin = CHEXBERT_ENV_PATH

    # Path to the script that must run under the virtualenv
    script_file = "label"

    print("Starting to extract Chexbert labels...")

    process = subprocess.run([python_bin, "-m", script_file, "-c", "checkpoint/chexbert.pth", "-d", reports_path, "-o", output_path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=CHEXBERT_PATH)

    # Print the output
    print(process.stdout.decode())

    # Print the errors if there are any
    if process.stderr:
        print("Errors:")
        print(process.stderr.decode())

    print("Finished extracting Chexbert labels.")
