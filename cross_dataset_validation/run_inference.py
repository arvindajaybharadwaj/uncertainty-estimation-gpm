"""Run GPM/final_inference.py unchanged, after making its imports work on any OS.

All arguments are passed straight through. See final_inference.py for them.
run_checkpoint.py calls this, so you normally do not run it yourself.
"""
import runpy
import sys

from common import FINAL_INFERENCE, use_gpm

if __name__ == '__main__':
    use_gpm()
    sys.argv = [str(FINAL_INFERENCE)] + sys.argv[1:]
    runpy.run_path(str(FINAL_INFERENCE), run_name='__main__')
