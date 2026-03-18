"""
Shim: resolves the numeric-prefix ModuleNotFoundError on 01_feature_engineering.py.
Python module names cannot start with a digit.
"""
import importlib.util, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_fe_impl", os.path.join(_HERE, "01_feature_engineering.py")
)
_fe_impl = importlib.util.module_from_spec(_spec)
sys.modules["_fe_impl"] = _fe_impl
_spec.loader.exec_module(_fe_impl)

calculate_features   = _fe_impl.calculate_features
PASSTHROUGH_FEATURES = _fe_impl.PASSTHROUGH_FEATURES
SCALE_FEATURES       = _fe_impl.SCALE_FEATURES
SessionConfig        = _fe_impl.SessionConfig
