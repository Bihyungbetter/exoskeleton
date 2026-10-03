from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.fastmath import norm


FOOT_BODY_NAMES = ("calcn_r", "talus_r", "toes_r", "calcn_l", "talus_l", "toes_l")


@dataclass(frozen=True)
class StandingMeasurement:
    time: float
    root_orientation: np.ndarray
    foot_contact_count: int
    com_position: np.ndarray
    com_velocity: np.ndarray
    # Active foot-floor contact positions (world). Their convex hull is the
    # base of support.
    contact_points: np.ndarray
    right_contacts: int
    left_contacts: int
    # Per-foot GRF magnitude (N), so a step can check the swing foot unloaded.
    right_grf_n: float
    left_grf_n: float


class StandingMeasurements:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_geom < 0:
            raise ValueError("standing measurements require a geom named 'floor'")
        foot_bodies = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODY_NAMES
        }
        self.foot_geoms = {
            geom_id for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] in foot_bodies
        }
        if not self.foot_geoms:
            raise ValueError("no human foot collision geoms found")
        # Body id -> "r"/"l" from the body-name suffix.
        self.body_side = {}
        for name in FOOT_BODY_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.body_side[bid] = name[-1]
        self.root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qadr = model.jnt_qposadr[self.root_joint]
        # Precompute geom -> (body, side) so the per-contact loop stays cheap.
        self.foot_geom_info = {
            geom: (int(model.geom_bodyid[geom]),
                   self.body_side.get(int(model.geom_bodyid[geom]), "?"))
            for geom in self.foot_geoms
        }

    def read(self, data: mujoco.MjData) -> StandingMeasurement:
        # mj_step does not fill subtree_linvel; without this it reads zero.
        mujoco.mj_subtreeVel(self.model, data)
        contact_bodies: set[int] = set()
        points: list[np.ndarray] = []
        sides: list[str] = []
        grf = {"r": np.zeros(3), "l": np.zeros(3)}
        force_contact = np.zeros(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self.floor_geom:
                foot_geom = geom2
            elif geom2 == self.floor_geom:
                foot_geom = geom1
            else:
                continue
            foot_info = self.foot_geom_info.get(foot_geom)
            if foot_info is None:
                continue
            # Contact force comes from mj_contactForce, not cfrc_ext.
            mujoco.mj_contactForce(self.model, data, contact_id, force_contact)
            frame = contact.frame.reshape(3, 3)
            # Frame axes are stored as rows, so the transpose maps to world.
            contact_world = frame.T @ force_contact[:3]
            body, side = foot_info
            contact_bodies.add(body)
            points.append(np.asarray(contact.pos).copy())
            sides.append(side)
            if side in grf:
                grf[side] += contact_world

        root_qadr = self.root_qadr
        root_quat = data.qpos[root_qadr + 3:root_qadr + 7].copy()
        return StandingMeasurement(
            time=float(data.time),
            root_orientation=root_quat,
            foot_contact_count=len(contact_bodies),
            com_position=data.subtree_com[0].copy(),
            com_velocity=data.subtree_linvel[0].copy(),
            contact_points=(np.asarray(points) if points
                            else np.zeros((0, 3))),
            right_contacts=sides.count("r"),
            left_contacts=sides.count("l"),
            right_grf_n=norm(grf["r"]),
            left_grf_n=norm(grf["l"]),
        )
