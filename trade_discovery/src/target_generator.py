import importlib.util, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_tg_impl", os.path.join(_HERE, "02_target_generator.py")
)
_tg_impl = importlib.util.module_from_spec(_spec)
sys.modules["_tg_impl"] = _tg_impl
_spec.loader.exec_module(_tg_impl)

generate_tbm_targets = _tg_impl.generate_tbm_targets
