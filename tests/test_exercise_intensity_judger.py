import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "exercise_intensity_judger.py"


def run_judger(text):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
        check=True,
    )
    return result.stdout


def test_reports_ratio_and_medium_advice():
    output = run_judger("20\n130\n")

    assert "最大心率约为 194.0 次/分钟" in output
    assert "运动强度约为 67.0%" in output
    assert "属于中等强度运动" in output
    assert "继续保持" in output


def test_warns_when_input_is_unreasonable():
    output = run_judger("18\n-5\n")

    assert "输入的数据不太合理" in output
