import importlib.util, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_gp_impl", os.path.join(_HERE, "03_gp_engine.py")
)
_gp_impl = importlib.util.module_from_spec(_spec)
sys.modules["_gp_impl"] = _gp_impl
_spec.loader.exec_module(_gp_impl)

train_gp_model        = _gp_impl.train_gp_model
extract_elite_programs = _gp_impl.extract_elite_programs
hash_formula          = _gp_impl.hash_formula
