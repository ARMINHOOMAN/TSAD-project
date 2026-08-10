@echo off
REM ---------------------------------------------------------------
REM Runs the diffusion-TSAD experiments on the GPU.
REM Uses the gen_AI conda environment by absolute path, so it works
REM regardless of PATH, conda activation, or VS Code settings.
REM
REM   RUN.bat              -> quick smoke test (~1 min)
REM   RUN.bat smd          -> SMD machine-1-1 benchmark
REM   RUN.bat synthetic    -> full synthetic run (20 epochs)
REM ---------------------------------------------------------------

set PY=C:\Users\armin\miniconda3\envs\gen_AI\python.exe
cd /d "%~dp0code"

if "%1"=="smd" (
    "%PY%" -u run_experiments.py --dataset smd --entity machine-1-1 --epochs 10 --test-stride 5 --out ../results_gpu
) else if "%1"=="synthetic" (
    "%PY%" -u run_experiments.py --epochs 20 --out ../results_gpu
) else (
    "%PY%" -u run_experiments.py --quick --out ../results_gpu
)

echo.
echo ================= FINISHED =================
pause
