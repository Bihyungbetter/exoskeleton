# exo-sim

MuJoCo simulation of a person wearing a hip/knee exoskeleton (waist, bilateral
hip abduction and flexion, knee), with a PPO policy that drives the exo motors
while the person walks. The exo model comes from our SolidWorks assembly and
the human model is MyoSuite's MyoLeg from MyoAssist.

## Layout

| path | what |
|---|---|
| `controllers/human_baseline.py` | human side: PD hold on the 8 leg joints, per-joint torque caps, trunk (root) hold |
| `controllers/gait.py` | fixed periodic gait for the human's hips and knees |
| `controllers/exo_assistance.py` | exo side: the only code that writes the 6 motor commands; applies rated torque, slew and cuff limits |
| `controllers/safety.py`, `measurements.py`, `wholebody_allocator.py`, `torso_balance.py`, `balance_margin.py` | sensing, safety supervisor, torque allocation |
| `envs/stepping_env.py` | Gymnasium env (plant + human + exo + reward). Walking uses `exo_action=True, gait_drive=True` |
| `envs/plant_setup.py` | drops the feet onto the floor and lines the exo up with the body at reset |
| `scripts/train_stepping_ppo.py` | PPO trainer (Stable-Baselines3), normally run through `tools/train_watchdog.py` |
| `tools/evaluate_walking.py` | scores policies against the control on held-out seeds |
| `tools/view_policy.py` | plays a policy in the MuJoCo viewer |
| `models/human_exo_direct_ak80abd.xml` | plant used for all results below (exo torque applied straight to the human joints, abduction motors sized as AK80-64) |
| `models/human_exo_direct.xml` | same direct plant with the original AK70-10 abduction motors; `SteppingEnv`'s default |
| `models/human_exo.xml` | the real attachment: exo coupled to the body through five compliant cuffs |
| `models/meshes/` | exo meshes from the CAD (visual and collision) |
| `models/walking_decim5_20260916.zip` | trained policy |
| `models/gait_walk_base.json`, `models/exo_transmission_calibration*.json` | gait parameters, exo transmission calibration |
| `runs/` | train, score and view scripts with matching flags |

## Running it

```powershell
runs\run_view_walking_decim5.cmd            # watch the policy (pass "none" for the control)
runs\score_walking_decim5.cmd               # policy vs control on 48 held-out seeds
runs\run_walking_decim5_20260916.cmd        # retrain; ~2 h on 6 workers, overwrites the policy file
```

Setup: Python 3.11, `pip install -r requirements.txt`, and a clone of
[MyoAssist](https://github.com/neumovelab/myoassist) at `third_party/myoassist`.
Only its MyoLeg model files (`myosuite/simhive/myo_sim`) are used, so it doesn't
need to be pip-installed. The model XMLs have absolute paths in them, but
`tools/model_paths.py` rewrites those relative to wherever the repo lives, so a
fresh clone works once MyoAssist is in place.

## Results so far

48 held-out seeds (40000 to 40047), 12 s episodes, exo control running at 200 Hz:

| | falls | distance | exo share of hip+knee torque |
|---|---|---|---|
| control (fixed assist law, offload 0.6) | 32/48 | 1.82 m | 29% |
| trained policy | 0/48 | 3.72 m | 53% |

Most of the gain is in not falling. Human hip/knee torque only drops about 5%
compared with the control, and under the policy at least one motor is pinned at
its 48 N·m limit about half the time.

## Limitations

These numbers don't transfer to a real person yet, for a few reasons.

The trunk is kept upright by a 100 N·m PD torque applied directly to the root
joint, with no reaction on the rest of the body (about 43 N·m on average under
the policy). Without it the model falls over during the settle phase. Relative
comparisons between arms are fine since every arm gets the same hold, but
absolute survival rates aren't meaningful.

The human is torque driven. The MyoLeg model has 80 muscles but none of them
are used, and the human doesn't adapt to the exo: it's a fixed gait on fixed
PD gains. The ankle is held at neutral by a PD spring, so there's no push-off,
and the exo has no ankle motor.

Everything above is on the direct-drive plant. On the cuff model
(`human_exo.xml`) every seed falls: the cuffs only pass about 6 N·m in total, so
fixing the attachment is a mechanical design problem before it's a control one.
