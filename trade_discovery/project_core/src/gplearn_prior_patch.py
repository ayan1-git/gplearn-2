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

1. `_Program.build_program()` / `point_mutation()` sample *variable*
   terminals from `program.feature_proba_` (inverse-CDF) instead of
   uniformly, when the attribute is present.
2. `genetic._parallel_evolve` receives `_feature_proba` through the fit
   params dict and attaches it to every newly constructed program, so the
   prior survives joblib/loky worker processes (workers re-import gplearn
   from disk, which is why the patch must live in the source file rather
   than be monkeypatched in-memory).

Known limitation: generation-0 cold-start programs are built inside
`_Program.__init__` before the prior can be attached, so gen-0 init is
uniform. Every later generation (including all warm-start phases and all
seeded runs) uses the prior. This is acceptable — the prior matters most
once evolution is underway.

The patch is marker-guarded ("FEATURE-PRIOR PATCH"), verified by exact
string anchors, applied atomically, and fails soft: if anything goes wrong
(read-only site-packages, unexpected gplearn version) it logs a warning and
the pipeline continues with the previous behaviour (uniform sampling).
"""
import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

_MARKER = "FEATURE-PRIOR PATCH"

# ── _program.py patches ──────────────────────────────────────────────────────

# Insert a prior-aware terminal sampler right before build_program.
_PROGRAM_HELPER_ANCHOR = "    def build_program(self, random_state):"
_PROGRAM_HELPER_INSERT = '''    # ── FEATURE-PRIOR PATCH (trade_discovery/project_core) ──────────────
    def _sample_feature_idx(self, random_state):
        """Sample a variable-terminal index using this program's
        `feature_proba_` distribution when present, else uniformly."""
        probs = getattr(self, 'feature_proba_', None)
        if probs is None:
            return random_state.randint(self.n_features)
        return int(random_state.choice(self.n_features, p=probs))

    def build_program(self, random_state):'''

# Route variable-terminal draws through the sampler (occurs exactly twice:
# once in build_program, once in point_mutation).
_TERMINAL_BLOCK_OLD = """                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
"""
_TERMINAL_BLOCK_NEW = """                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                elif getattr(self, 'feature_proba_', None) is not None:
                    # FEATURE-PRIOR PATCH: draw variable terminals from the
                    # estimator's feature prior instead of uniform.
                    terminal = self._sample_feature_idx(random_state)
                else:
                    terminal = random_state.randint(self.n_features)
"""

# ── genetic.py patches ───────────────────────────────────────────────────────

# Plumb the prior into the per-worker params dict.
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

# Unpack it inside _parallel_evolve.
_EVOLVE_UNPACK_ANCHOR = "    feature_names = params['feature_names']\n"
_EVOLVE_UNPACK_INSERT = (
    "    feature_names = params['feature_names']\n"
    "    # FEATURE-PRIOR PATCH: unpack the feature prior (may be None).\n"
    "    _feature_proba_ = params.get('_feature_proba', None)\n"
)

# Attach it to every freshly constructed program.
_EVOLVE_ATTACH_ANCHOR = "        program.parents = genome\n"
_EVOLVE_ATTACH_INSERT = (
    "        program.parents = genome\n"
    "\n"
    "        # FEATURE-PRIOR PATCH: make the prior visible to terminal\n"
    "        # sampling in build_program / point_mutation / subtree_mutation.\n"
    "        program.feature_proba_ = _feature_proba_\n"
)


def _patch_file(path: str, ops: list) -> bool:
    """Apply (text_old -> text_new, expected_count) ops to one file.

    Returns True if the file already carries the marker or was patched
    successfully; raises on anchor mismatch.
    """
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    if _MARKER in src:
        return True  # already patched

    for old, new, expected in ops:
        count = src.count(old)
        if count != expected:
            raise RuntimeError(
                f"anchor mismatch in {os.path.basename(path)}: "
                f"expected {expected} occurrence(s), found {count} — "
                f"gplearn version not supported."
            )
        src = src.replace(old, new)

    tmp_path = path + ".prior_patch.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(src)
    os.replace(tmp_path, path)
    return True


def ensure_gplearn_feature_prior_patch() -> bool:
    """
    Patch the installed gplearn on disk (idempotent) and reload its modules
    in this process so the patch takes effect immediately.

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

    already = _MARKER in open(program_py, encoding="utf-8").read()
    try:
        changed = _patch_file(program_py, [
            (_PROGRAM_HELPER_ANCHOR, _PROGRAM_HELPER_INSERT, 1),
            (_TERMINAL_BLOCK_OLD,   _TERMINAL_BLOCK_NEW,     2),
        ])
        changed |= _patch_file(genetic_py, [
            (_GENETIC_PARAMS_ANCHOR, _GENETIC_PARAMS_INSERT, 1),
            (_EVOLVE_UNPACK_ANCHOR,  _EVOLVE_UNPACK_INSERT,  1),
            (_EVOLVE_ATTACH_ANCHOR,  _EVOLVE_ATTACH_INSERT,  1),
        ])
    except Exception as exc:
        logger.warning(
            "Could not apply gplearn feature-prior patch (%s). The feature "
            "prior will remain a no-op (uniform terminal sampling).", exc
        )
        return False

    if changed and not already:
        logger.info("gplearn feature-prior patch written to %s", pkg_dir)

    # Reload so the current process picks up the patched code. Order matters:
    # genetic imports _program, so reload the leaf first.
    try:
        importlib.reload(_program)
        importlib.reload(genetic)
    except Exception as exc:
        logger.warning("gplearn reload failed (%s) — restart Python once to "
                       "activate the feature prior.", exc)
        return False

    # Re-export the refreshed classes for anyone that imported us first.
    sys.modules["gplearn.genetic"] = genetic
    sys.modules["gplearn._program"] = _program
    return True
