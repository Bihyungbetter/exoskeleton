"""Simple periodic gait for the human side.

Params:

    step_period_s     step duration
    swing_hip         step length (swing-leg hip flexion amplitude)
    swing_knee        step height (swing-leg knee flexion, i.e. clearance)
    weight_shift      step width (frontal-plane weight transfer amplitude)
    shift_lead        phase lead of the weight shift ahead of the swing
    stance_knee       stance-leg crouch

Don't skip the frontal plane - the CoM has to get over the stance foot before
the other foot can lift. `weight_shift` drives the anti-symmetric hip adduction
pair, which is what moves the CoM laterally; the symmetric pair barely does.

Output is a 6-entry normalized action in the same layout as
stepping_env.ACTION_JOINTS, so the gait can drive the plant the same way a
policy does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Action layout: r = (hip_flex, hip_add, knee) at 0,1,2 and l at 3,4,5.
LEG_ACTION = {"r": (0, 1, 2), "l": (3, 4, 5)}


@dataclass(frozen=True)
class GaitParams:
    # defaults are just a starting guess, they don't walk
    step_period_s: float = 1.2
    swing_hip: float = 0.45
    swing_knee: float = 0.55
    weight_shift: float = 0.55
    shift_lead: float = 0.25
    stance_knee: float = 0.10

    @staticmethod
    def from_vector(v) -> "GaitParams":
        return GaitParams(*(float(x) for x in v))


class GaitController:
    """phase goes 0->1 over one step. Stance/swing swap every step so one
    stride = two phases."""

    def __init__(self, params: GaitParams | None = None) -> None:
        self.params = params or GaitParams()
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.swing = "r"            # leg currently swinging
        self.phase = 0.0

    @property
    def stance(self) -> str:
        return "l" if self.swing == "r" else "r"

    def advance(self, dt: float) -> bool:
        """Advance the clock. Returns True if this call forced a timeout swap.

        Legs normally swap in `on_touchdown`. Until then `phase` wraps rather
        than clamping at 1, because every swing term is zero at ph=1 and a clamp
        would zero the action while waiting. A swap is forced at 1.5x the
        nominal period so a missed touchdown can't stall the gait.
        """
        self.t += dt
        self.phase = (self.t / self.params.step_period_s) % 1.0
        if self.t >= 1.5 * self.params.step_period_s:
            self._swap()
            return True
        return False

    def on_touchdown(self) -> None:
        self._swap()

    def _swap(self) -> None:
        self.t = 0.0
        self.swing = "l" if self.swing == "r" else "r"
        self.phase = 0.0

    def action(self) -> np.ndarray:
        p = self.params
        a = np.zeros(6)
        ph = self.phase

        # Frontal plane: anti-symmetric hip adduction leading the swing.
        # Positive on the stance hip, negative on the swing hip moves the CoM
        # toward the stance foot. Hip adduction is mirrored, so no side factor.
        # This jumps at touchdown when the roles swap; phase-continuous
        # versions walked worse, so it stays a plain sine.
        shift = np.sin(2.0 * np.pi * (ph + p.shift_lead))
        s_hf, s_ha, s_kn = LEG_ACTION[self.stance]
        w_hf, w_ha, w_kn = LEG_ACTION[self.swing]
        a[s_ha] = p.weight_shift * max(0.0, shift)
        a[w_ha] = -p.weight_shift * max(0.0, shift)

        # Sagittal: swing hip is a half sine; the knee's is front-loaded for
        # early clearance and a straight leg at landing. Both are 0 at ph=0, 1.
        a[w_hf] = p.swing_hip * np.sin(np.pi * ph)
        a[w_kn] = p.swing_knee * np.sin(np.pi * min(1.0, 1.35 * ph))

        # Constant stance-knee flexion so the knee never locks (a locked stance
        # knee turns the step into a vault).
        a[s_kn] = p.stance_knee
        return np.clip(a, -1.0, 1.0)
