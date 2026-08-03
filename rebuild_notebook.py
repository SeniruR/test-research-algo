"""
Run this directly: python rebuild_notebook.py
Rebuilds the Colab notebook from scratch with all current source files.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from scripts.generate_colab_bootstrap import main
main()
