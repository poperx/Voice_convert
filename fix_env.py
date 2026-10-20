"""Point the virtual environment at the Python shipped in this folder."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = ROOT / "python"
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
WHEELS = ROOT / "wheels"
REQ = ROOT / "requirements.txt"
MODEL = ROOT / "models" / "UVR-MDX-NET-Voc_FT.onnx"
BASE_PY = HOME / "python.exe"
RUNTIME_DLLS = ("python312.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(HOME) + os.pathsep + str(HOME / "DLLs") + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def _write_cfg() -> None:
    VENV.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"home = {HOME}",
            "include-system-site-packages = false",
            "version = 3.12.10",
            f"executable = {BASE_PY}",
            f"command = {BASE_PY} -m venv {VENV}",
            "",
        ]
    )
    (VENV / "pyvenv.cfg").write_text(text, encoding="utf-8")


def _copy_runtime_dlls() -> None:
    scripts = VENV / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_DLLS:
        src = HOME / name
        dest = scripts / name
        if src.exists() and (not dest.exists() or dest.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dest)


def _imports_ok() -> bool:
    if not VENV_PY.exists():
        return False
    code = "import av, numpy, scipy, onnxruntime, parselmouth, tkinter"
    result = subprocess.run([str(VENV_PY), "-c", code], capture_output=True, env=_env())
    if result.returncode != 0 and os.environ.get("VOICE_ENV_DEBUG") == "1":
        sys.stderr.buffer.write(result.stderr)
    return result.returncode == 0


def main() -> int:
    if not BASE_PY.exists():
        print("缺少内置 Python：python\\python.exe")
        return 1
    if not VENV_PY.exists():
        print("正在创建运行环境…")
        subprocess.run([str(BASE_PY), "-m", "venv", str(VENV)], check=True, env=_env())
    _write_cfg()
    _copy_runtime_dlls()
    if not MODEL.exists() or MODEL.stat().st_size < 60_000_000:
        print("缺少人声分离模型 models\\UVR-MDX-NET-Voc_FT.onnx")
        return 1
    if not WHEELS.exists():
        print("缺少离线组件目录 wheels。")
        return 1
    if not _imports_ok():
        print("正在从本地组件安装环境…")
        subprocess.run(
            [
                str(VENV_PY),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                f"--find-links={WHEELS}",
                "-r",
                str(REQ),
            ],
            check=True,
            env=_env(),
        )
        if not _imports_ok():
            print("环境安装后仍无法导入依赖。")
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"环境准备失败（退出码 {exc.returncode}）。")
        raise SystemExit(exc.returncode) from exc
