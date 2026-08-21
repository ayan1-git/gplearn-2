"""
gplearn_prior_patch.py
======================
Idempotent source patch that adds real feature-prior support to gplearn.

WHY THIS EXISTS
---------------
gp_engine._apply_feature_proba() stamps a normalised probability vector on
the estimator as `_feature_proba`. Stock gplearn (<=0.4.3) has no such
mechanism, so that attribute was silently ignored and the entire GEL
feature-prior feedback loop was a no-op.

This module patches the installed gplearn source on disk so that:

1. `_Program.build_program()` / `point_mutation()` route every variable-
   terminal draw through `_Program._sample_feature_idx()`, which samples
   from the program's `feature_proba_` distribution (inverse-CDF) when
   available and falls back to uniform otherwise. The constant-vs-variable
   split of the original `randint(n_features + 1)` behaviour is preserved
   (important: SymbolicRegressor defaults to const_range=(-1, 1), so this
   branch is the hot path).
2. `genetic._parallel_evolve` receives `_feature_proba` through the fit
   params dict, attaches it to every newly constructed program AND sets it
   as the `_Program._class_feature_proba_` fallback — so generation-0
   programs (which are built inside `_Program.__init__` before any
   per-instance attachment can happen) honour the prior too.
3. The prior survives joblib/loky worker processes because the patch lives
   in the gplearn source file on disk; workers re-import gplearn fresh.

Version history (state-based detection, sequential in-place upgrades):
  v1  instance-attr prior, attached after construction (gen-0 uniform) — bug
  v2  + class-level fallback, but the elif gate still short-circuited gen-0
  v3  unconditional routing through the sampler — but the const_range branch
      (the DEFAULT path for SymbolicRegressor!) was still untouched
  v4  const_range branch also honours the prior, preserving the original
      1/(n+1) constant-vs-variable split

Fails soft: if anything goes wrong (read-only site-packages, unexpected
gplearn version) it logs a warning and the pipeline continues with uniform
sampling.
"""
import importlib
import logging
import os

logger = logging.getLogger(__name__)

_MARKER = "FEATURE-PRIOR PATCH"

# ── _program.py ──────────────────────────────────────────────────────────────

_PROGRAM_HELPER_ANCHOR = "    def build_program(self, random_state):"

_HELPER_LATEST = '''    # ── FEATURE-PRIOR PATCH (trade_discovery/project_core) v4 ───────────
    def _sample_feature_idx(self, random_state):
        """Sample a variable-terminal index using this program's
        `feature_proba_` distribution when present (instance or class-level
        fallback), else uniformly."""
        probs = getattr(self, 'feature_proba_', None)
        if probs is None:
            probs = getattr(type(self), '_class_feature_proba_', None)
        if probs is None:
            return random_state.randint(self.n_features)
        return int(random_state.choice(self.n_features, p=probs))

    def build_program(self, random_state):'''

# v1 helper body (upgrade path only)
_HELPER_V1_BODY = """        probs = getattr(self, 'feature_proba_', None)
        if probs is None:
            return random_state.randint(self.n_features)
        return int(random_state.choice(self.n_features, p=probs))"""

_HELPER_V2_BODY = """        probs = getattr(self, 'feature_proba_', None)
        if probs is None:
            probs = getattr(type(self), '_class_feature_proba_', None)
        if probs is None:
            return random_state.randint(self.n_features)
        return int(random_state.choice(self.n_features, p=probs))"""

# Original (unpatched) terminal-draw block; occurs exactly twice.
_TERMINAL_BLOCK_ORIG = """                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
"""

# v2 terminal block (elif gate short-circuited gen-0 — bug; upgrade path only)
_TERMINAL_BLOCK_V2 = """                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                elif getattr(self, 'feature_proba_', None) is not None:
                    # FEATURE-PRIOR PATCH: draw variable terminals from the
                    # estimator's feature prior instead of uniform.
                    terminal = self._sample_feature_idx(random_state)
                else:
                    terminal = random_state.randint(self.n_features)
"""

# v3 terminal block (const_range branch untouched — bug; upgrade path only)
_TERMINAL_BLOCK_V3 = """                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    # FEATURE-PRIOR PATCH v3: routes through the feature prior
                    # (instance or class-level) when present, else uniform.
                    terminal = self._sample_feature_idx(random_state)
"""

# v4 (latest): both branches honour the prior; the constant-vs-variable
# split mirrors the original randint(n_features + 1) distribution.
_TERMINAL_BLOCK_V4 = """                if self.const_range is not None:
                    # FEATURE-PRIOR PATCH v4: honour the prior for variable
                    # terminals while preserving the original 1/(n+1)
                    # constant-vs-variable split.
                    _fp = getattr(self, 'feature_proba_', None)
                    if _fp is None:
                        _fp = getattr(type(self), '_class_feature_proba_', None)
                    if _fp is None:
                        terminal = random_state.randint(self.n_features + 1)
                    elif (random_state.uniform()
                          * (self.n_features + 1) < 1.0):
                        terminal = self.n_features   # constant sentinel
                    else:
                        terminal = self._sample_feature_idx(random_state)
                else:
                    # FEATURE-PRIOR PATCH: routes through the feature prior
                    # (instance or class-level) when present, else uniform.
                    terminal = self._sample_feature_idx(random_state)
"""

# ── genetic.py ───────────────────────────────────────────────────────────────

_GENETIC_PARAMS_ANCHOR = (
    "        params['method_probs'] = self._method_probs\n"
)
_GENETIC_PARAMS_INSERT = (
    "        params['method_probs'] = self._method_probs\n"
    "\n"
    "        # FEATURE-PRIOR PATCH: forward the estimator's feature prior to\n"
    "        # worker processes so _parallel_evolve can attach it to programs.\n"
    "        params['_feature_proba'] = getattr(self, '_feature_proba', None)\n"
)

_EVOLVE_UNPACK_ANCHOR = "    feature_names = params['feature_names']\n"

_EVOLVE_UNPACK_V2 = (
    "    feature_names = params['feature_names']\n"
    "    # FEATURE-PRIOR PATCH: unpack the feature prior (may be None) and\n"
    "    # install it as the class-level fallback so generation-0 programs\n"
    "    # (built inside _Program.__init__ before per-instance attachment)\n"
    "    # also honour the prior.\n"
    "    _feature_proba_ = params.get('_feature_proba', None)\n"
    "    _Program._class_feature_proba_ = _feature_proba_\n"
)

_EVOLVE_ATTACH_ANCHOR = "        program.parents = genome\n"
_EVOLVE_ATTACH_INSERT = (
    "        program.parents = genome\n"
    "\n"
    "        # FEATURE-PRIOR PATCH: make the prior visible to terminal\n"
    "        # sampling in build_program / point_mutation / subtree_mutation.\n"
    "        program.feature_proba_ = _feature_proba_\n"
)


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_atomic(path, src):
    tmp_path = path + ".prior_patch.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(src)
    os.replace(tmp_path, path)


def _apply_ops(path, src, ops):
    for old, new, expected in ops:
        count = src.count(old)
        if count != expected:
            raise RuntimeError(
                f"anchor mismatch in {os.path.basename(path)}: expected "
                f"{expected} occurrence(s), found {count} — gplearn version "
                f"not supported."
            )
        src = src.replace(old, new)
    _write_atomic(path, src)


def ensure_gplearn_feature_prior_patch() -> bool:
    """
    Patch the installed gplearn on disk (idempotent, self-upgrading) and
    reload its modules in this process so the patch takes effect immediately.

    Returns True if the prior mechanism is available afterwards.
    """
    try:
        import gplearn
        from gplearn import _program, genetic  # noqa: F401
    except ImportError:
        logger.warning("gplearn not installed — feature prior unavailable.")
        return False

    pkg_dir = os.path.dirname(gplearn.__file__)
    program_py = os.path.join(pkg_dir, "_program.py")
    genetic_py = os.path.join(pkg_dir, "genetic.py")

    try:
        prog_src = _read(program_py)
        gen_src = _read(genetic_py)

        marker_present = _MARKER in prog_src or _MARKER in gen_src

        if not marker_present:
            # ── Fresh install on unpatched gplearn (v4) ─────────────────────
            _apply_ops(program_py, prog_src, [
                (_PROGRAM_HELPER_ANCHOR, _HELPER_LATEST,       1),
                (_TERMINAL_BLOCK_ORIG,   _TERMINAL_BLOCK_V4,   2),
            ])
            _apply_ops(genetic_py, gen_src, [
                (_GENETIC_PARAMS_ANCHOR, _GENETIC_PARAMS_INSERT, 1),
                (_EVOLVE_UNPACK_ANCHOR,  _EVOLVE_UNPACK_V2,      1),
                (_EVOLVE_ATTACH_ANCHOR,  _EVOLVE_ATTACH_INSERT,  1),
            ])
            logger.info("gplearn feature-prior patch v4 written to %s", pkg_dir)
        else:
            # ── Sequential in-place upgrade of older patch versions ────────
            changed = False

            # v1 → latest helper body (adds class-level fallback)
            if _HELPER_V1_BODY in prog_src:
                _apply_ops(program_py, _read(program_py), [
                    (_HELPER_V1_BODY, _HELPER_V2_BODY, 1),
                ])
                changed = True

            # v1 → v2 unpack (adds class-level fallback assignment)
            if ("_Program._class_feature_proba_" not in gen_src
                    and "_feature_proba_ = params.get('_feature_proba', None)\n" in gen_src):
                _apply_ops(genetic_py, _read(genetic_py), [
                    ("    # FEATURE-PRIOR PATCH: unpack the feature prior (may be None).\n"
                     "    _feature_proba_ = params.get('_feature_proba', None)\n",
                     _EVOLVE_UNPACK_V2, 1),
                ])
                changed = True

            # v2/v3 → v4 terminal block (const_range branch now prior-aware)
            cur_prog = _read(program_py)
            if _TERMINAL_BLOCK_V2 in cur_prog:
                _apply_ops(program_py, cur_prog, [
                    (_TERMINAL_BLOCK_V2, _TERMINAL_BLOCK_V4, 2),
                ])
                changed = True
            elif _TERMINAL_BLOCK_V3 in cur_prog:
                _apply_ops(program_py, cur_prog, [
                    (_TERMINAL_BLOCK_V3, _TERMINAL_BLOCK_V4, 2),
                ])
                changed = True

            if changed:
                logger.info("gplearn feature-prior patch upgraded in place.")
    except Exception as exc:
        logger.warning(
            "Could not apply gplearn feature-prior patch (%s). The feature "
            "prior will remain a no-op (uniform terminal sampling).", exc
        )
        return False

    # Reload so the current process picks up the patched code. Order matters:
    # genetic imports _program, so reload the leaf first.
    try:
        importlib.reload(_program)
        importlib.reload(genetic)
    except Exception as exc:
        logger.warning("gplearn reload failed (%s) — restart Python once to "
                       "activate the feature prior.", exc)
        return False

    return True
