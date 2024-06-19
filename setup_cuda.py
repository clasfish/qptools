import os
import qptools
import sysconfig
import subprocess

prefix = os.path.dirname(qptools.__file__)
target = "cudacore" + sysconfig.get_config_var("EXT_SUFFIX")

try:
    subprocess.run(["make", "all", f"TARGET={target}", f"PREFIX={prefix}"], check=True, text=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print("Command '{}' returned non-zero exit status {}. Output: {}".format(e.cmd, e.returncode, e.stdout))
    print("Error: {}".format(e.stderr))
    exit(1)