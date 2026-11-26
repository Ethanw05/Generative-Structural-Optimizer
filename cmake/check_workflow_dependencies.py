"""Probe the configured interpreter using the same dependency paths as the workflows."""
from importlib import import_module
from pathlib import Path
import sys

bundled = Path(__file__).resolve().parents[1] / "build" / "sunbeam33"
sys.path[:0] = [str(bundled / name) for name in ("meshlib-runtime", "deps")]

if sys.version_info < (3, 10):
    sys.exit("The workflows require Python 3.10 or newer.")
for module in ("numpy", "scipy.sparse", "manifold3d", "trimesh", "networkx", "shapely",
               "vtkmodules.vtkFiltersCore", "PIL.Image", "matplotlib", "pyvista", "pytetwild", "tetgen",
               "meshlib.mrmeshpy", "meshlib.mrmeshnumpy", "pymeshlab"):
    try:
        import_module(module)
    except (ImportError, OSError) as error:
        sys.exit(f"Cannot load {module} in {sys.executable}: {error}")
