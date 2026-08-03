@echo off
echo Rebuilding Colab notebook...
cd /d "%~dp0"
python rebuild_notebook.py
if %ERRORLEVEL% NEQ 0 (
    echo FAILED - see error above
    pause
) else (
    echo SUCCESS - notebook rebuilt
    pause
)
