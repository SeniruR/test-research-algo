@echo off
cd /d "%~dp0"
echo Step 1: Restore notebook from git...
git checkout HEAD -- notebooks/haptic_groundtruth_colab.ipynb
if errorlevel 1 (
    echo Git restore failed. Trying to continue anyway...
)
echo Step 2: Regenerate bootstrap cell from current source...
python scripts\generate_colab_bootstrap.py
if errorlevel 1 (
    echo ERROR: Bootstrap generation failed.
    pause
    exit /b 1
)
echo.
echo SUCCESS - notebook regenerated.
pause
