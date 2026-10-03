@echo off
rem View the 200 Hz walking policy with the same plant flags as the scorer.
rem Usage: run_view_walking_decim5.cmd [policy.zip | none] [extra flags, e.g. --loop]
rem "none" shows the control (fixed law, zero residual).
cd /d "%~dp0.."
set PYTHONPATH=%CD%
set OMP_NUM_THREADS=1
set POLICY=%~1
if "%POLICY%"=="" set POLICY=models\walking_decim5_20260916.zip
"%CD%\.venv\Scripts\python.exe" -u tools\view_policy.py ^
  --policy %POLICY% ^
  --assisted --exo-action --gait-drive ^
  --gait-params models\gait_walk_base.json ^
  --free-ankle --stabilize-foot ^
  --joint-cap 100 --ankle-cap 100 ^
  --exo-residual-scale 20 20 20 20 20 20 ^
  --episode-seconds 12 --settle-decisions 120 ^
  --walk-progress-weight 8 --walk-target-speed 0.30 ^
  --human-effort-weight 20 --survival-bonus 1 ^
  --exo-decimation 5 --offload 0.6 ^
  --seed0 40000 --episodes 3 %2 %3 %4 %5 %6
