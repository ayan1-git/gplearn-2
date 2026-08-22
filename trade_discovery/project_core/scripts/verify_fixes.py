"""
verify_fixes.py — self-contained verification of the project_core fixes.

Run anywhere the pipeline runs (local venv or Kaggle):
    python scripts/verify_fixes.py

Prints PASS/FAIL per check and exits non-zero if anything fails.
Covers: #1 native DEAP feature prior (functional), #3 imports, #5 feature
tokenization, #6 no seed cycling (static), #7 config wiring, #8 BOTH_TP
exclusion, #9 keep-pool-on-crash (static), #11 float64 raw (static),
#12 nested cancellation detector, #13 pinned requirements, plus a DEAP
engine smoke run (formula strings, predict, elite seeding, gplearn purge).
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


# ── 1. native DEAP feature prior: functional test ────────────────────────────
def test_feature_prior():
    import random as pyrandom
    import numpy as np
    from deap import gp as dgp
    import src.gp_engine as ge

    n_feat = 6
    names = [f"f{i}" for i in range(n_feat)]
    pset = ge.build_pset(n_feat, names)
    feat_terms = [pset.mapping[n] for n in names]
    known = set(names)

    def terminal_shares(proba):
        pyrandom.seed(11)
        np.random.seed(11)
        factory = ge.make_terminal_factory(pset, feat_terms, proba)
        counts = np.zeros(n_feat)
        total = 0
        for _ in range(400):
            tree = ge.generate_tree(pset, factory, 2, 4)
            for node in tree:
                if isinstance(node, dgp.Primitive):
                    continue
                v = getattr(node, "value", None)
                if isinstance(v, str) and v in known:
                    counts[names.index(v)] += 1
                    total += 1
        return counts / max(total, 1)

    uniform = terminal_shares(None)
    skewed = terminal_shares([0.90] + [0.02] * (n_feat - 1))
    assert skewed[0] > 0.60, (
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
def _load_main_pipeline(module_id):
    import importlib, importlib.util
    spec = importlib.util.spec_from_file_location(
        module_id, os.path.join(HERE, "main_pipeline.py"))
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
    return mod


def test_features_used():
    mod = _load_main_pipeline("_mp_verify2")
    got = mod._features_used("mul(talib_adxr, talib_rocr100)")
    assert set(got) == {"talib_adxr", "talib_rocr100"}, got  # no phantom adx/roc/rocr
    got = mod._features_used("sub(talib_adx, feat_icp)")
    assert set(got) == {"talib_adx", "feat_icp"}, got
    got = mod._features_used("add(feat_icp, feat_icp)")
    assert set(got) == {"feat_icp"}, got
    return "no substring collisions"


# ── 4. nested trivial-cancellation detector (#12) ────────────────────────────
def test_cancellation():
    mod = _load_main_pipeline("_mp_verify3")
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


# ── 6. config wiring (#7, #8, GP_N_JOBS) ─────────────────────────────────────
def test_config():
    import src.config as cfg
    assert cfg.DROP_BOTH_TP is True
    for k in ("OOS_MAX_DRAWDOWN_HIGH_SHARPE", "HIGH_SHARPE_THRESHOLD"):
        assert hasattr(cfg, k), f"missing {k}"
    assert hasattr(cfg, "GP_N_JOBS"), "missing GP_N_JOBS (DEAP parallelism)"
    for k in ("GP_MIN_NODES", "GP_MIN_FEATURES",
              "GP_SHALLOW_PENALTY_PER_FEATURE", "GP_SHALLOW_PENALTY_PER_NODE"):
        assert hasattr(cfg, k), f"missing anti-collapse key {k}"
    for k in ("SEED_SOFT_THRESHOLD_SHARPE", "SEED_DECAY_FRACTION",
              "ROTATION_FEATURE", "GP_RESTARTS", "IMBALANCE_LAMBDA",
              "ACTIVITY_FLOOR", "ACTIVITY_LAMBDA", "MIN_LONG", "MIN_SHORT"):
        assert not hasattr(cfg, k), f"dead key {k} still present"
    return "dead keys removed, gate keys + GP_N_JOBS present"


# ── 7. static checks: injection cycling, crash-pool wipe, float64, pins ─────
def test_static():
    eng_src = open(os.path.join(PROJECT_CORE, "src", "gp_engine.py")).read()
    # The fix removed the modulo-cycling injection; match code pattern, not prose.
    assert "seed_programs[seed_idx %" not in eng_src, "seed cycling still present (#6)"
    assert "import gplearn" not in eng_src \
        and "from gplearn" not in eng_src, "gplearn import still present"
    assert not os.path.exists(os.path.join(
        PROJECT_CORE, "src", "gplearn_prior_patch.py")), \
        "src/gplearn_prior_patch.py should be deleted"

    mp = open(os.path.join(HERE, "main_pipeline.py")).read()
    assert "Retaining existing elite pool" in mp, "#9 not applied"
    assert "gp._program" not in mp, "main_pipeline still touches gplearn internals"
    assert "astype(np.float32)\n    return df_raw" not in mp, "#11 not applied"

    req = open(os.path.join(HERE, "requirements.txt")).read()
    assert "deap==" in req, "#13 deap not pinned"
    assert "gplearn==" not in req, "stale gplearn pin still present"
    return "injection/crash/gplearn-purge/pins verified"


# ── 8. DEAP engine smoke run: formula strings, predict, elite seeding ───────
def test_deap_smoke():
    import re
    import numpy as np
    import pandas as pd
    import src.gp_engine as ge

    patch = {
        "POPULATION_SIZE": 150,
        "GENERATIONS": 5,
        "PHASE1_GENS": 2,
        "PHASE2_GENS": 2,
        "PHASE3_GENS": 1,
        "INIT_DEPTH_MIN": 2,
        "INIT_DEPTH_MAX": 4,
    }
    saved = {k: getattr(ge, k) for k in patch}
    for k, v in patch.items():
        setattr(ge, k, v)
    try:
        rng = np.random.RandomState(3)
        cols = [f"f{i}" for i in range(6)]
        X = pd.DataFrame(rng.randn(500, 6).astype(np.float32), columns=cols)
        y = pd.Series(np.where(X["f0"] > 0, 1.0, -1.0).astype(np.float32))

        res = ge.train_gp_model(X, y, fold=7)
        assert res is not None, "smoke run returned None (bloat guard?)"
        assert res.best_length > 0

        # Anti-collapse floor: the best program must clear the shallow-formula
        # penalty regime (≥ GP_MIN_FEATURES features, ≥ GP_MIN_NODES nodes).
        assert res.best_length >= ge.GP_MIN_NODES, \
            f"best program too short ({res.best_length} < {ge.GP_MIN_NODES})"
        used_feats = ge._features_used(res.best_program)
        assert len(used_feats) >= ge.GP_MIN_FEATURES, \
            f"best program too shallow: {len(used_feats)} features"

        formula = res.best_formula
        if res.best_length > 1:
            # Lisp-style string with named primitives (bare single terminals
            # are a legal convergence outcome and have no primitive tokens)
            assert re.search(
                r"\b(add|sub|mul|div|max|min|abs|neg|gt|lt|and|or|if_then)\(",
                formula), f"no legacy primitive tokens in: {formula[:80]}"
        else:
            assert formula in cols, f"unexpected bare terminal: {formula}"
        idents = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula))
        used_feats = idents & set(cols)
        assert used_feats, f"no feature terminals printed by name: {formula[:120]}"
        # Ephemeral constants must stringify as bare numbers
        bare_consts = re.findall(r"[^\w(,]\s*(-?\d+\.\d+)", formula)
        assert all(abs(float(c)) <= 1.0 for c in bare_consts), \
            f"ephemeral constants render oddly: {bare_consts}"

        preds = res.predict(X.values)
        assert preds.shape == (len(X),), f"predict shape {preds.shape}"
        assert bool(np.all(np.isfinite(preds))), "non-finite predictions"

        # Seeded warm-start run must accept extracted elites
        elite = ge.extract_elite_programs(res, top_n=20)
        assert len(elite) > 0, "extract_elite_programs returned empty"
        res2 = ge.train_gp_model(
            X, y, seed_programs=elite, fold=8,
            feature_proba=np.full(6, 1.0 / 6))
        assert res2 is not None, "seeded smoke run returned None"
    finally:
        for k, v in saved.items():
            setattr(ge, k, v)
    return ("cold+seeded runs OK | "
            f"formula={formula[:48]}... | feats={sorted(used_feats)}")


if __name__ == "__main__":
    check("#1  native DEAP feature prior", test_feature_prior)
    check("#3   path bootstrap / imports", test_imports)
    check("#5   tokenized _features_used", test_features_used)
    check("#12  nested cancellation guard", test_cancellation)
    check("#8   BOTH_TP exclusion", test_both_tp)
    check("#7   config wiring", test_config)
    check("#6/9/11/13 static checks", test_static)
    check("#DEAP engine smoke run", test_deap_smoke)

    failed = [r for r in RESULTS if not r[0]]
    print("\n" + "=" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED:", ", ".join(r[1] for r in failed))
        sys.exit(1)
    print("ALL CHECKS PASSED ✓")
