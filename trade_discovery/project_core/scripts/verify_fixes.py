"""
verify_fixes.py — self-contained verification of the project_core fixes.

Run anywhere the pipeline runs (local venv or Kaggle):
    python scripts/verify_fixes.py

Prints PASS/FAIL per check and exits non-zero if anything fails.
Covers: #1 feature prior (functional), #3 imports, #5 feature tokenization,
#6 no seed cycling (static), #7 config wiring, #8 BOTH_TP exclusion,
#9 keep-pool-on-crash (static), #11 float64 raw (static), #12 nested
cancellation detector, #13 pinned requirements.
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_CORE = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_CORE))
for p in (PROJECT_CORE, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(PROJECT_CORE)

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail or ""))
        print(f"PASS | {name}" + (f" | {detail}" if detail else ""))
    except Exception as exc:
        RESULTS.append((False, name, f"{exc}"))
        print(f"FAIL | {name} | {exc}")
        traceback.print_exc()


# ── 1. gplearn feature-prior patch: functional test ─────────────────────────
def test_feature_prior():
    import numpy as np
    from src.gplearn_prior_patch import ensure_gplearn_feature_prior_patch
    assert ensure_gplearn_feature_prior_patch(), "patch did not activate"

    # idempotency + self-upgrade safety
    assert ensure_gplearn_feature_prior_patch(), "second call failed"

    from src.gp_engine import SymbolicRegressor
    import src.gp_engine as ge
    assert ge.FEATURE_PRIOR_ACTIVE, "gp_engine reports prior inactive"

    rng = np.random.RandomState(7)
    n_feat = 6
    X = rng.randn(300, n_feat).astype(np.float32)
    y = X[:, 0].astype(np.float32)

    def terminal_shares(proba):
        est = SymbolicRegressor(
            population_size=200, generations=1,      # gen-0 only → build_program path
            function_set=['add', 'sub', 'mul', 'div'],
            init_depth=(2, 4), random_state=1, verbose=0, n_jobs=1,
            feature_names=[f"f{i}" for i in range(n_feat)])
        if proba is not None:
            est._feature_proba = np.asarray(proba, dtype=float)
        est.fit(X, y)
        counts = np.zeros(n_feat)
        for prog in est._programs[-1]:
            if prog is None:
                continue
            for node in prog.program:
                if isinstance(node, (int, np.integer)):
                    counts[node] += 1
        return counts / max(counts.sum(), 1)

    uniform = terminal_shares(None)
    skewed = terminal_shares([0.90] + [0.02] * (n_feat - 1))
    assert skewed[0] > 0.6, (
        f"prior had no effect: f0 share under prior = {skewed[0]:.3f} "
        f"(uniform baseline {uniform[0]:.3f})")
    return f"f0 share uniform={uniform[0]:.3f} → prior={skewed[0]:.3f}"


# ── 2. main_pipeline import + path bootstrap (#3) ────────────────────────────
def test_imports():
    # Simulate `python scripts/main_pipeline.py` from project_core: only the
    # script dir + project_core on sys.path; main_pipeline must bootstrap the
    # repo root itself for trade_discovery.* imports.
    import importlib, importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mp_verify", os.path.join(HERE, "main_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        # Keep site-packages etc., but strip every repo path so the test only
        # passes if main_pipeline bootstraps the repo root itself (#3).
        repo_paths = {HERE, PROJECT_CORE, REPO_ROOT}
        sys.path[:] = [HERE, PROJECT_CORE] + [
            p for p in saved if p and os.path.realpath(p) not in
            {os.path.realpath(x) for x in repo_paths}]
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    assert hasattr(mod, "gel_loop")
    return "main_pipeline imported with minimal sys.path"


# ── 3. tokenized _features_used (#5) ─────────────────────────────────────────
def test_features_used():
    import importlib, importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mp_verify2", os.path.join(HERE, "main_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        repo_paths = {HERE, PROJECT_CORE, REPO_ROOT}
        sys.path[:] = [HERE, PROJECT_CORE] + [
            p for p in saved if p and os.path.realpath(p) not in
            {os.path.realpath(x) for x in repo_paths}]
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved

    got = mod._features_used("mul(talib_adxr, talib_rocr100)")
    assert set(got) == {"talib_adxr", "talib_rocr100"}, got  # no phantom adx/roc/rocr
    got = mod._features_used("sub(talib_adx, feat_icp)")
    assert set(got) == {"talib_adx", "feat_icp"}, got
    got = mod._features_used("add(feat_icp, feat_icp)")
    assert set(got) == {"feat_icp"}, got
    return "no substring collisions"


# ── 4. nested trivial-cancellation detector (#12) ────────────────────────────
def test_cancellation():
    import importlib, importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mp_verify3", os.path.join(HERE, "main_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        repo_paths = {HERE, PROJECT_CORE, REPO_ROOT}
        sys.path[:] = [HERE, PROJECT_CORE] + [
            p for p in saved if p and os.path.realpath(p) not in
            {os.path.realpath(x) for x in repo_paths}]
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved

    f = mod._find_trivial_cancellation
    assert f("sub(add(a,b), add(a,b))") is True, "nested sub(X,X) missed"
    assert f("sub(a, b)") is False
    assert f("add(x, neg(x))") is True, "add(X,neg(X)) missed"
    assert f("add(x, neg(sub(a,b)))") is False
    assert f("mul(sub(add(p,q), add(p,q)), z)") is True, "deeply nested missed"
    assert f("sub(mul(a,b), mul(a,b))") is True
    assert f("add(talib_rsi, talib_rsi)") is False, "add(X,X) is not cancellation"
    return "nested cases detected correctly"


# ── 5. BOTH_TP exclusion (#8) ────────────────────────────────────────────────
def test_both_tp():
    import numpy as np
    import pandas as pd
    from src.target_generator import generate_tbm_targets

    n, atr_p = 60, 3
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="30min")
    # Flat market with high=101/low=99 → TR=2 → Wilder ATR=2
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    open_ = np.full(n, 100.0)
    # Bar 30: range 8 ≥ 2*tp_dist(=4) → both TPs; 8 < 2*sl_dist(=12) → not wide
    high[30], low[30] = 104.0, 96.0

    df_raw = pd.DataFrame({"open": open_, "high": high, "low": low,
                           "close": close}, index=idx)
    df_feat = df_raw[["close"]].copy()  # alignment only

    def run(exclude):
        feats, y, meta = generate_tbm_targets(
            df_raw, df_feat, max_hold=5, atr_period=atr_p,
            tp_mult=1.0, sl_mult=3.0,          # tp_dist=2, sl_dist=6
            drop_both_sl=True, exclude_both_tp=exclude,
            return_metadata=True)
        return feats, y, meta

    _, y_keep, meta_keep = run(False)
    assert (meta_keep["event_type"] == "BOTH_TP").any(), "BOTH_TP never generated"
    assert (y_keep == 1.0).any(), "legacy BOTH_TP→+1 label missing"
    _, y_drop, _ = run(True)
    assert not (y_drop == 1.0).any(), "BOTH_TP rows not excluded"
    assert len(y_drop) < len(y_keep), "excluded rows were not dropped"
    return f"kept={len(y_keep)} rows, excluded={len(y_drop)} rows"


# ── 6. config wiring (#7, #8) ────────────────────────────────────────────────
def test_config():
    import src.config as cfg
    assert cfg.DROP_BOTH_TP is True
    for k in ("OOS_MAX_DRAWDOWN_HIGH_SHARPE", "HIGH_SHARPE_THRESHOLD"):
        assert hasattr(cfg, k), f"missing {k}"
    for k in ("SEED_SOFT_THRESHOLD_SHARPE", "SEED_DECAY_FRACTION",
              "ROTATION_FEATURE", "GP_RESTARTS", "IMBALANCE_LAMBDA",
              "ACTIVITY_FLOOR", "ACTIVITY_LAMBDA", "MIN_LONG", "MIN_SHORT"):
        assert not hasattr(cfg, k), f"dead key {k} still present"
    return "dead keys removed, gate keys present"


# ── 7. static checks: injection cycling, crash-pool wipe, float32 (#6,#9,#11)
def test_static():
    src = open(os.path.join(PROJECT_CORE, "src", "gp_engine.py")).read()
    # The fix removed the modulo-cycling injection; the comment explaining it
    # may remain, so match the actual code pattern, not prose.
    assert "seed_programs[seed_idx %" not in src, "seed cycling still present (#6)"
    assert "Retaining existing elite pool" in open(
        os.path.join(HERE, "main_pipeline.py")).read(), "#9 not applied"
    mp = open(os.path.join(HERE, "main_pipeline.py")).read()
    assert "astype(np.float32)\n    return df_raw" not in mp, "#11 not applied"
    req = open(os.path.join(HERE, "requirements.txt")).read()
    assert "gplearn==0.4.3" in req, "#13 gplearn not pinned"
    return "injection/crash/float64/pins verified"


if __name__ == "__main__":
    check("#1  feature prior functional", test_feature_prior)
    check("#3  path bootstrap / imports", test_imports)
    check("#5  tokenized _features_used", test_features_used)
    check("#12 nested cancellation guard", test_cancellation)
    check("#8  BOTH_TP exclusion", test_both_tp)
    check("#7  config wiring", test_config)
    check("#6/9/11/13 static checks", test_static)

    failed = [r for r in RESULTS if not r[0]]
    print("\n" + "=" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED:", ", ".join(r[1] for r in failed))
        sys.exit(1)
    print("ALL CHECKS PASSED ✓")
