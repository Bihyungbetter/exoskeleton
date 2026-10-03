"""Plant setup done once when the env is built (feet on floor, exo alignment,
ankle lock)."""
from __future__ import annotations

import mujoco
import numpy as np

from controllers.measurements import FOOT_BODY_NAMES


def lowest_point(model, data, g: int) -> float:
    # exact for spheres/capsules, bounding sphere for everything else
    pos = data.geom_xpos[g]
    t = model.geom_type[g]
    s = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(pos[2] - s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        R = data.geom_xmat[g].reshape(3, 3)
        half_axis = R[:, 2] * s[1]
        return float(min(pos[2] - half_axis[2], pos[2] + half_axis[2]) - s[0])
    return float(pos[2] - model.geom_rbound[g])


def drop_onto_floor(model, data) -> None:
    # Put the lowest foot geom right at z=0. If the feet start floating or
    # penetrating, the first contact impulse knocks the model over.
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_BODY_NAMES]
    foot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in body_ids
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if root < 0 or not foot_geoms:
        return
    mujoco.mj_forward(model, data)
    lowest = min(lowest_point(model, data, g) for g in foot_geoms)
    data.qpos[model.jnt_qposadr[root] + 2] -= lowest
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


CUFF_SITE_PAIRS = (
    ("cuff_waist_site", "exo_cuff_waist_site"),
    ("cuff_thigh_r_site", "exo_cuff_thigh_r_site"),
    ("cuff_thigh_l_site", "exo_cuff_thigh_l_site"),
    ("cuff_shank_r_site", "exo_cuff_shank_r_site"),
    ("cuff_shank_l_site", "exo_cuff_shank_l_site"),
)


def align_exo_to_human(model, data) -> float:
    """Move the exo (on its exo_free joint) so the cuff sites line up with
    the human's. Returns how far it moved (m).

    In the CAD all five exo cuff sites are 10.133 mm above the human ones. The
    compliant cuffs can soak that up but rigid welds snap it shut in one step
    and the impulse trips the cuff limit.
    """
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "exo_free")
    if joint < 0:
        return 0.0
    mujoco.mj_forward(model, data)
    offsets = []
    for human_site, exo_site in CUFF_SITE_PAIRS:
        ih = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, human_site)
        ie = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, exo_site)
        if ih < 0 or ie < 0:
            return 0.0
        offsets.append(data.site_xpos[ih] - data.site_xpos[ie])
    correction = np.mean(offsets, axis=0)
    adr = model.jnt_qposadr[joint]
    data.qpos[adr:adr + 3] += correction
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(correction))


def lock_ankles(model: mujoco.MjModel) -> None:
    for name in ("ankle_angle_r", "ankle_angle_l"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"missing ankle joint: {name}")
        model.jnt_limited[jid] = 1
        model.jnt_range[jid] = (-1e-6, 1e-6)
        model.jnt_stiffness[jid] = 5000.0
        model.dof_damping[model.jnt_dofadr[jid]] = 100.0
