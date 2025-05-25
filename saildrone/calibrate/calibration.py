import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Union
from .saildrone import calibrate


def apply_calibration(echodata, cal_file, calibrator: Optional[Union[str, Path]] = None):
    """
    Apply calibration to an Echopype Dataset using an external calibrator.

    Parameters
    ----------
    echodata
        An Echopype dataset (from ep.open_raw or converted in memory).
    cal_file : str or Path
    calibrator : str or Path, optional
        Either
          - a filesystem path to a .py file that defines a function
            `def calibrate(echodata, cal_file): ...`
          - a module import path (e.g. "calibrate.saildrone" or ".calibrate.saildrone")
        If None, defaults to the built-in module ".saildrone" (relative import).

    Returns
    -------
    The return value of `calibrate(echodata, cal_file)` (usually the same `echodata` mutated in place).
    """
    if calibrator is None:
        calibrate_fn = calibrate  # Default to the built-in saildrone calibrator
    else:
        # If it's a path on disk, load from file
        path = Path(calibrator)
        if path.is_file():
            spec = importlib.util.spec_from_file_location(path.stem, str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
        else:
            # Otherwise treat as importable module name
            module = importlib.import_module(str(calibrator))

        try:
            calibrate_fn = getattr(module, "calibrate")
        except AttributeError:
            raise AttributeError(
                f"Calibrator module '{module.__name__}' has no `calibrate(echodata, cal_file)`"
            )

    return calibrate_fn(echodata, cal_file)


