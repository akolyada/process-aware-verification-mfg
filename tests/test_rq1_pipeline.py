"""Offline end-to-end test of the RQ1 harness with mock planners (no LLM, no payload
text inspected). Run:  python tests/test_rq1_pipeline.py"""
import sys, tempfile
from pathlib import Path
import pandas as pd, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "sim")]
try:
    import rq1.inject as inj
except ImportError:
    print("SKIP: rq1/inject.py is private (payloads withheld); see rq1/inject_template.py")
    sys.exit(0)
import rq1.run_rq1 as run

_orig = inj.inject
def marked(order, cls, channel, rng, style="direct"):
    u = _orig(order, cls, channel, rng, style)
    setattr(u, channel, getattr(u, channel) + f" [[{cls}]]")   # marker for the mock only
    return u
run.inject = marked

with tempfile.TemporaryDirectory() as d:
    cfg = {"models": [{"name": "mock-correct", "kind": "mock", "mode": "correct", "model": "mock-correct"},
                      {"name": "mock-obedient", "kind": "mock", "mode": "obedient", "model": "mock-obedient"}],
           "verifier": {"name": "mock-judge", "kind": "mock", "mode": "correct", "model": "mock-judge"},
           "conditions": ["plain", "d1_hardened", "d4a_channel_separation"],
           "verifier_conditions": ["plain"], "n_per_cell": 8, "seed": 1}
    (Path(d) / "c.yaml").write_text(yaml.safe_dump(cfg))
    sys.argv = ["x", "--config", f"{d}/c.yaml", "--out", d]
    run.main()
    df = pd.read_csv(f"{d}/raw_rq1.csv")
    att = df[df.attack_class != "benign"]
    g = att.groupby(["model", "condition"]).injected_followed.mean()
    print(g)
    assert df.parse_error.sum() == 0, "parse errors"
    assert (g.loc["mock-correct"] == 0).all(), "correct mock must never 'follow'"
    assert g.loc[("mock-obedient", "plain")] == 1.0 and g.loc[("mock-obedient", "d1_hardened")] == 1.0
    assert g.loc[("mock-obedient", "d4a_channel_separation")] == 0.0, "D4a must cut the channel"
    fol = att[att.injected_followed]
    assert fol["D4b_master_data_detected"].all(), "D4b must detect every followed non-adaptive plan"
    assert (~fol["D3_safety_auth_detected"]).all(), "D3 is blind to production-semantic plans"
    assert not df[df.model == "mock-correct"]["D0_none_harm"].any(), "correct plans cause no harm"
    # pairing: identical trial ids and products across conditions and models
    keys = df.groupby(["model", "condition"]).apply(lambda x: tuple(zip(x.trial, x["product"])))
    assert len(set(keys)) == 1, "trials not paired"
    sys.argv = ["x", "--raw", f"{d}/raw_rq1.csv", "--out", d]
    import rq1.analysis as an
    an.main()
    m = pd.read_csv(f"{d}/table_rq1_mcnemar.csv")
    ob = m[(m.model == "mock-obedient") & (m.comparison == "plain vs d4a_channel_separation")]
    assert (ob.discordant_plain_only == 8).all() and (ob.discordant_other_only == 0).all()
    assert Path(f"{d}/figures/fig_rq1_compliance.svg").exists()
    ds = pd.read_csv(f"{d}/table_rq1_downstream.csv")
    assert set(ds[ds.defence == "D2"].condition) == {"plain"}, "D2 must appear only where it ran"
    assert df[df.condition != "plain"]["D2_detected"].isna().all()
    print("RQ1 pipeline + analysis test: PASSED")
