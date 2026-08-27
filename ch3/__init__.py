import os
import sys

# Ensure the repo root is on sys.path when `ch3` is imported as a package.
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
