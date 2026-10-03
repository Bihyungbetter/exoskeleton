@echo off
rem Score the 200 Hz walking policy against the offload 0.6 control, 48 held-out
rem seeds from 40000. Plant flags must match training (--exo-decimation 5, --free-ankle).
rem Usage: score_walking_decim5.cmd [policy.zip] [extra flags], e.g. "" --seeds 2
cd /d "%~dp0.."
if not exist results mkdir results
set PYTHONPATH=%CD%
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set POLICY=%~1
if "%POLICY%"=="" set POLICY=models\walking_decim5_20260916.zip

"%CD%\.venv\Scripts\python.exe" -u tools\evaluate_walking.py ^
  --assisted --exo-action --gait-drive ^
  --gait-params models\gait_walk_base.json ^
  --free-ankle --stabilize-foot ^
  --joint-cap 100 --ankle-cap 100 ^
  --exo-residual-scale 20 20 20 20 20 20 ^
  --episode-seconds 12 --settle-decisions 120 ^
  --walk-progress-weight 8 --walk-target-speed 0.30 ^
  --human-effort-weight 20 --survival-bonus 1 ^
  --exo-decimation 5 ^
  --seeds 48 --seed0 40000 ^
  --arms offload:0.6 model:%POLICY%:0.6 ^
  --out results\walking_decim5_direct_48seeds.json %2 %3 %4 %5 %6
