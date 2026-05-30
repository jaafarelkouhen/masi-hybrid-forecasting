"""Smoke tests for the package CLI."""

import os
import subprocess
import sys


def _run_cli(project_root, *args):
    env = os.environ.copy()
    src = str(project_root / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
    return subprocess.run(
        [sys.executable, "-m", "masi_hybrid_forecasting.pipeline", *args],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_cli_help(project_root):
    res = _run_cli(project_root, "--help")
    assert res.returncode == 0
    assert "predict" in res.stdout
    assert "backtest" in res.stdout


def test_cli_predict_smoke(project_root, require_outputs):
    res = _run_cli(project_root, "predict")
    assert res.returncode == 0, res.stderr
    assert "Stats prédictions" in res.stdout
    assert "948 jours TEST" in res.stderr or "948 jours TEST" in res.stdout


def test_cli_backtest_hmm_gate_smoke(project_root, require_outputs):
    res = _run_cli(project_root, "backtest", "--strategy", "hmm_gate", "--cost-bps", "5")
    assert res.returncode == 0, res.stderr
    assert "BACKTEST HMM_GATE" in res.stdout
    assert "Sharpe ann." in res.stdout
