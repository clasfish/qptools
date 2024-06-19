import os
import qptools
import sysconfig
import subprocess

prefix = os.path.dirname(qptools.__file__)
target = "cudacore" + sysconfig.get_config_var("EXT_SUFFIX")
subprocess.check_call(["make", "all", f"TARGET={target}", f"PREFIX={prefix}"])
