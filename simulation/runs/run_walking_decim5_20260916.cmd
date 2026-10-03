@echo off
rem Train the walking exo policy on the 200 Hz exo clock (idealized direct plant).
rem The watchdog resumes from the newest checkpoint after a resource kill.
rem Score and view with the same plant flags, in particular --exo-decimation 5
rem and --free-ankle (see score_walking_decim5.cmd, run_view_walking_decim5.cmd).
rem Keep the laptop lid open; the watchdog only blocks idle sleep.
cd /d "%~dp0.."
if not exist logs mkdir logs
set PYTHONPATH=%CD%
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
"%CD%\.venv\Scripts\python.exe" -u tools\train_watchdog.py ^
  --checkpoint-dir models\walking_decim5_20260916_checkpoints ^
  --target-steps 2000000 ^
  --log-prefix logs\walking_decim5_20260916 ^
  --max-restarts 12 ^
  -- ^
  --envs 6 --n-steps 256 --batch-size 256 --vec subproc ^
  --arena-memory-mb 16 ^
  --exo-decimation 5 ^
  --assisted --offload 0.6 ^
  --joint-cap 100 --ankle-cap 100 ^
  --free-ankle --stabilize-foot ^
  --gait-drive --gait-params models\gait_walk_base.json ^
  --exo-action --exo-residual-scale 20 20 20 20 20 20 ^
  --push-force-range 0 0 --push-bodies torso --push-axes 0 ^
  --walk-progress-weight 8.0 --walk-target-speed 0.30 ^
  --human-effort-weight 20.0 --gamma 0.997 ^
  --survival-bonus 1.0 ^
  --settle-decisions 120 --episode-seconds 12 ^
  --ent-coef 0.005 --learning-rate 3e-4 --target-kl 0.15 ^
  --output models\walking_decim5_20260916 ^
  --eval-episodes 0 --device cpu ^
  >> logs\walking_decim5_20260916_watchdog.log 2>&1
