import importlib.util, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_vbt_impl", os.path.join(_HERE, "04_vectorbt_evaluator.py")
)
_vbt_impl = importlib.util.module_from_spec(_spec)
sys.modules["_vbt_impl"] = _vbt_impl
_spec.loader.exec_module(_vbt_impl)

evaluate_formula_with_vectorbt = _vbt_impl.evaluate_formula_with_vectorbt
