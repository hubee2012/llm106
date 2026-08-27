"""SFT dataset re-export so `python step10_sft.py` from ch4/ can import dataset_sft."""
import os
import sys

# When this file is loaded as a sibling (`from dataset_sft import *`), ch4/ is
# on sys.path but the repo root is not. Insert it before importing ch2.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ch2.dataset_sft import SFTDataset

__all__ = ["SFTDataset"]
