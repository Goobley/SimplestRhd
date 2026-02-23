"""
I/O routines for serializing and deserializing state dictionaries to/from netCDF files.
Uses xarray for efficient netCDF handling and a variable registry for extensibility.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


# ============================================================================
# Variable Registry
# ============================================================================
# Define all variables that can be serialized and their dimensions.
# Variables not in this registry cannot be serialized (will raise an exception)

VARIABLE_REGISTRY = {
    # Conserved variables (4 equations: rho, mom, E, spec_e_ion)
    "Q": {
        "dims": ["eq", "x"],
        "description": "Conserved variables (density, momentum, energy, spec_e_ion)",
    },
    # Primitive variables (computed during time stepping)
    "W": {
        "dims": ["eq", "x"],
        "description": "Primitive variables (density, velocity, pressure, spec_e_ion)",
    },
    # Tracer populations (species-dependent)
    "tracers": {
        "dims": ["species", "x"],
        "description": "Tracer populations (electron density + atomic populations)",
    },
    # Ionization fraction profile
    "y": {
        "dims": ["x"],
        "description": "Ionization fraction (ne/nh_tot)",
    },
    # Spatial grid
    "xcc": {
        "dims": ["x"],
        "description": "Cell center coordinates",
    },
}

# Scalar fields that should be stored as attributes, not DataArrays
# Each field can be a string (description, defaults to float type) or a dict with 'description' and 'type'
SCALAR_FIELDS = {
    "dx": {"description": "Grid spacing", "type": float},
    "gamma": {"description": "Adiabatic index", "type": float},
    "time": {"description": "Current simulation time", "type": float},
    "snap_num": {"description": "Snapshot counter for filename generation", "type": int},
}


# ============================================================================
# Utility Functions
# ============================================================================

def _get_field_info(field_name: str) -> Dict[str, Any]:
    """Get registry information for a field.

    Args:
        field_name: Name of the field

    Returns:
        Registry entry for the field

    Raises:
        ValueError: If field is not in registry
    """
    if field_name not in VARIABLE_REGISTRY and field_name not in SCALAR_FIELDS:
        raise ValueError(
            f"Field '{field_name}' is not in the variable registry. "
            f"Available array fields: {list(VARIABLE_REGISTRY.keys())}. "
            f"Available scalar fields: {list(SCALAR_FIELDS.keys())}. "
            f"Register new fields in VARIABLE_REGISTRY or SCALAR_FIELDS."
        )

    if field_name in SCALAR_FIELDS:
        return {"type": "scalar"}

    return VARIABLE_REGISTRY[field_name]


def _validate_array_field(field_name: str, data: np.ndarray, ref_x_size: int) -> None:
    """Validate that an array field matches registry specifications.

    Args:
        field_name: Name of the field
        data: Array data to validate
        ref_x_size: Expected size of x dimension (from xcc)

    Raises:
        ValueError: If field dimensions don't match registry
    """
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Field '{field_name}' is not a numpy array")

    if data.ndim not in [1, 2]:
        raise ValueError(
            f"Field '{field_name}' has {data.ndim} dimensions, "
            f"but only 1D or 2D arrays are supported"
        )

    info = VARIABLE_REGISTRY[field_name]
    expected_ndim = len(info["dims"])

    if data.ndim != expected_ndim:
        raise ValueError(
            f"Field '{field_name}' has {data.ndim} dimensions "
            f"but registry expects {expected_ndim}"
        )

    # Check that one of the dims is 'x' and validate its size
    if "x" in info["dims"]:
        x_axis_idx = info["dims"].index("x")
        if data.shape[x_axis_idx] != ref_x_size:
            raise ValueError(
                f"Field '{field_name}' has size {data.shape[x_axis_idx]} on x axis, "
                f"but xcc has size {ref_x_size}"
            )


def _get_serializable_fields(state: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Extract serializable array and scalar fields from state.

    Args:
        state: State dictionary

    Returns:
        Tuple of (array_fields, scalar_fields)

    Raises:
        ValueError: If a field exists but is not in registry
    """
    array_fields = {}
    scalar_fields = {}

    ref_x_size = state["xcc"].shape[0]

    # Collect scalar fields
    for field_name in SCALAR_FIELDS:
        if field_name in state:
            value = state[field_name]
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(
                    f"Scalar field '{field_name}' has type {type(value)}, "
                    f"expected numeric type"
                )
            # Get field metadata (handle both old string format and new dict format)
            field_meta = SCALAR_FIELDS[field_name]
            if isinstance(field_meta, dict):
                field_type = field_meta.get("type", float)
            else:
                field_type = float

            # Convert to specified type
            scalar_fields[field_name] = field_type(value)

    # Collect array fields
    for field_name in VARIABLE_REGISTRY:
        if field_name in state:
            data = state[field_name]
            _validate_array_field(field_name, data, ref_x_size)
            array_fields[field_name] = data

    return array_fields, scalar_fields


# ============================================================================
# Serialization Functions
# ============================================================================

def state_to_xarray(state: Dict[str, Any]) -> xr.Dataset:
    """Convert state dictionary to xarray Dataset.

    All array fields in the registry are converted to DataArrays with appropriate
    dimensions. Scalar fields are stored as attributes. Fields not in the registry
    raise an exception.

    Args:
        state: State dictionary to serialize

    Returns:
        xarray.Dataset with all serializable fields

    Raises:
        ValueError: If a field is not in the registry or has wrong dimensions
    """
    array_fields, scalar_fields = _get_serializable_fields(state)

    # Create coordinate variables
    coords = {}
    if "xcc" in array_fields:
        coords["x"] = ("x", array_fields["xcc"])

    # Create data variables from array fields
    data_vars = {}
    for field_name, data in array_fields.items():
        if field_name == "xcc":
            continue  # Already handled as coordinate

        info = VARIABLE_REGISTRY[field_name]
        dims = tuple(info["dims"])

        # Handle 1D vs 2D fields
        if data.ndim == 1:
            # 1D array (e.g., "y")
            data_vars[field_name] = (dims, data)
        else:
            # 2D array (e.g., "Q", "W", "tracers")
            # Need to create coordinates for non-spatial dimensions
            field_dims = info["dims"]

            for i, (dim_name) in enumerate(field_dims):
                if dim_name != "x" and dim_name not in coords:
                    # Create coordinate for this dimension
                    coords[dim_name] = (dim_name, np.arange(data.shape[i]))

            data_vars[field_name] = (dims, data)

    # Create dataset with attributes for scalars
    ds = xr.Dataset(data_vars, coords=coords, attrs=scalar_fields)

    return ds


def xarray_to_state(ds: xr.Dataset, state_template: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convert xarray Dataset back to state dictionary.

    Restores all array fields from DataArrays and scalar fields from attributes.
    Non-serializable fields must be provided via state_template.

    Args:
        ds: xarray Dataset from state_to_xarray()
        state_template: Optional template dict with non-serializable fields
                       (sources, unsplit_sources, bc_modes, etc.)

    Returns:
        State dictionary ready for simulation

    Raises:
        ValueError: If required fields are missing
    """
    state = {}

    # Extract coordinate (xcc)
    if "x" in ds.coords:
        state["xcc"] = ds.coords["x"].values
    else:
        raise ValueError("Dataset missing 'x' coordinate (xcc)")

    # Extract scalar fields from attributes
    for field_name in SCALAR_FIELDS:
        if field_name in ds.attrs:
            value = ds.attrs[field_name]
            # Get field metadata to restore correct type (handle both old string format and new dict format)
            field_meta = SCALAR_FIELDS[field_name]
            if isinstance(field_meta, dict):
                field_type = field_meta.get("type", float)
            else:
                field_type = float
            # Restore with correct type
            state[field_name] = field_type(value)

    # Extract array fields from data variables
    for field_name in VARIABLE_REGISTRY:
        if field_name in ds.data_vars:
            state[field_name] = ds[field_name].values

    # Merge template fields (non-serializable)
    if state_template is not None:
        for key, value in state_template.items():
            if key not in state:
                state[key] = value

    return state


# ============================================================================
# Snapshot Management
# ============================================================================

def save_snapshot(state: Dict[str, Any], output_dir: str = ".") -> str:
    """Save a snapshot of the state to a netCDF file.

    Generates filename using snapshot counter: snap_NNNNN.nc
    Increments snap_num in the state dictionary after saving.

    Args:
        state: State dictionary to save
        output_dir: Directory to save snapshot in (default: current directory)

    Returns:
        Path to the saved snapshot file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get current snap_num (default to 0 if not set)
    snap_num = state.get("snap_num", 0)

    # Generate filename
    filepath = output_dir / f"snap_{snap_num:05d}.nc"

    # Save snapshot
    ds = state_to_xarray(state)
    ds.to_netcdf(str(filepath))

    # Increment counter in state dict
    state["snap_num"] = snap_num + 1

    return str(filepath)


def load_snapshot(
        filepath: str,
        state_template: Dict[str, Any] = None,
        decrement_snap_num: bool = False,
) -> Dict[str, Any]:
    """Load a snapshot from a netCDF file.

    Args:
        filepath: Path to netCDF snapshot file
        state_template: Optional template with non-serializable fields
        decrement_snap_num: decrements the snapshot number by 1 for restarting. Default: False

    Returns:
        Reconstructed state dictionary
    """
    ds = xr.open_dataset(filepath)
    try:
        state = xarray_to_state(ds, state_template=state_template)
        if decrement_snap_num:
            state["snap_num"] -= 1
    finally:
        ds.close()

    return state


def get_snapshot_time(filepath: str) -> float:
    """Extract the simulation time from a snapshot file.

    Args:
        filepath: Path to snapshot file

    Returns:
        Simulation time stored in the snapshot
    """
    ds = xr.open_dataset(filepath)
    try:
        if "time" in ds.attrs:
            return float(ds.attrs["time"])
        else:
            raise ValueError(f"Snapshot {filepath} has no 'time' attribute")
    finally:
        ds.close()


def get_latest_snapshot_name(output_dir: str) -> Optional[str]:
    """Find the most recent snapshot in a directory.

    Snapshots are expected to be named snap_NNNNN.nc with numeric counter.

    Args:
        output_dir: Directory containing snapshots

    Returns:
        Path to latest snapshot (highest number), or None if no snapshots found
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        return None

    # Find all snapshot files matching pattern snap_NNNNN.nc
    snapshot_files = sorted(output_dir.glob("snap_*.nc"))

    if not snapshot_files:
        return None

    # Return the last file (highest number)
    return str(snapshot_files[-1])


def load_latest_snapshot(
        output_dir: str,
        state_template: Dict[str, Any] = None,
        decrement_snap_num: bool = False,
) -> Dict[str, Any]:
    """Load the latest snapshot and return state.

    Args:
        output_dir: Directory containing snapshots
        state_template: Optional template with non-serializable fields
        decrement_snap_num: decrements the snapshot number by 1 for restarting. Default: False

    Returns:
        Reconstructed state dictionary (includes snap_num for continuation)

    Raises:
        FileNotFoundError: If no snapshots found
    """
    filepath = get_latest_snapshot_name(output_dir)

    if filepath is None:
        raise FileNotFoundError(f"No snapshots found in {output_dir}")

    state = load_snapshot(filepath, state_template=state_template, decrement_snap_num=decrement_snap_num)

    return state


def register_variable(field_name: str, dims: list, description: str = "") -> None:
    """Register a new variable that can be serialized.

    This allows physics modules to register custom fields that will be
    automatically serialized/deserialized.

    Args:
        field_name: Name of the field (e.g., "custom_field")
        dims: List of dimension names for xarray (must include "x" for spatial)
              (e.g., ["eq", "x"] or ["species", "x"])
        description: Human-readable description of the field

    Raises:
        ValueError: If registration is invalid
    """
    if field_name in VARIABLE_REGISTRY:
        raise ValueError(f"Field '{field_name}' is already registered")

    if "x" not in dims:
        raise ValueError(
            f"Field '{field_name}' must have 'x' as one of its dims. "
            f"Got dims: {dims}"
        )

    VARIABLE_REGISTRY[field_name] = {
        "dims": dims,
        "description": description,
    }


def register_scalar(field_name: str, description: str = "", scalar_type=float) -> None:
    """Register a new scalar field that can be serialized.

    Args:
        field_name: Name of the scalar field
        description: Human-readable description
        scalar_type: Python type for the scalar (float or int). Default is float.

    Raises:
        ValueError: If field is already registered
    """
    if field_name in SCALAR_FIELDS:
        raise ValueError(f"Scalar field '{field_name}' is already registered")

    SCALAR_FIELDS[field_name] = {
        "description": description,
        "type": scalar_type,
    }
