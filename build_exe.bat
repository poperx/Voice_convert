@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%~dp0python;%~dp0python\DLLs;%PATH%"
"%~dp0python\python.exe" "%~dp0fix_env.py"
if errorlevel 1 exit /b 1
echo 正在准备打包工具…
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-index --find-links=wheels pyinstaller
if errorlevel 1 exit /b 1
echo 正在打包，可能需要几分钟…
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --onefile --noupx --name VoiceAdjust --manifest "%CD%\app.manifest" --distpath dist --workpath build --specpath build --add-data "%CD%\models\UVR-MDX-NET-Voc_FT.onnx;models" --collect-all onnxruntime --collect-all av --hidden-import parselmouth --hidden-import scipy.signal --hidden-import scipy.special --exclude-module matplotlib --exclude-module pytest --exclude-module IPython voice_gui.py
if errorlevel 1 exit /b 1
powershell -NoProfile -Command "$name = ([string]([char]0x4EBA)+[char]0x58F0+[char]0x8C03+[char]0x6574+'.exe'); $dest = Join-Path 'dist' $name; if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Force }; Rename-Item -LiteralPath 'dist\VoiceAdjust.exe' -NewName $name"
if errorlevel 1 exit /b 1
echo 完成：dist\
