"""
Wrapper for implicit thermal conduction
"""
# Import the existing conduction implementation
import sys
from pathlib import Path

# Add parent directory to path to import existing conduction module
parent = Path(__file__).parent.parent
sys.path.insert(0, str(parent))

from implicit_thermal_conduction import implicit_thermal_conduction

__all__ = ["implicit_thermal_conduction"]
