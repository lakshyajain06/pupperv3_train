import jax
from jax import numpy as jp
from brax.base import Motion, Transform
from brax import base, math
import numpy as np

EPS = 1e-6
# ------------ reward functions----------------
def reward_lin_vel_z(xd: Motion) -> jax.Array:
    # Penalize z axis base linear velocity
    return jp.clip(jp.square(xd.vel[0, 2]), -1000.0, 1000.0)


def reward_ang_vel_xy(xd: Motion) -> jax.Array:
    # Penalize xy axes base angular velocity
    return jp.clip(jp.sum(jp.square(xd.ang[0, :2])), -1000.0, 1000.0)

def reward_anchor_pose_drift(
    command: jax.Array,
    joint_angles: jax.Array, 
    default_pose: jax.Array,
) -> jax.Array:
    """
    Penalizes deviation from the default pose ONLY for legs 
    that are NOT currently being commanded.
    """
    # command[3] is the foot_idx (0, 1, 2, or 3)
    active_foot_idx = command[3].astype(jp.int32)
    
    # Calculate squared error for all 12 joints
    #
    errors = jp.square(joint_angles - default_pose)
    
    # Reshape errors to (4 legs, 3 joints per leg) to make masking easy
    errors_per_leg = errors.reshape(4, 3)
    
    # Create a mask: 1.0 for inactive legs, 0.0 for the active leg
    # We use jp.arange(4) to compare against the active_foot_idx
    mask = (jp.arange(4) != active_foot_idx).astype(jp.float32)
    
    # Apply mask (broadcasting the 4-element mask across the 3 joints per leg)
    # This zeros out the penalty for the leg that is supposed to move
    masked_errors = errors_per_leg * mask[:, None]
    
    return -jp.sum(masked_errors)

def reward_tracking_orientation(
    desired_world_z_in_body_frame: jax.Array, x: Transform, tracking_sigma: float
) -> jax.Array:
    # Tracking of desired body orientation
    world_z = jp.array([0.0, 0.0, 1.0])
    world_z_in_body_frame = math.rotate(world_z, math.quat_inv(x.rot[0]))
    error = jp.sum(jp.square(world_z_in_body_frame - desired_world_z_in_body_frame))
    return jp.clip(jp.exp(-error / (tracking_sigma + EPS)), -1000.0, 1000.0)


def reward_orientation(x: Transform) -> jax.Array:
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.clip(jp.sum(jp.square(rot_up[:2])), -1000.0, 1000.0)


def reward_torques(torques: jax.Array) -> jax.Array:
    # Penalize torques
    # This has a sparifying effect
    # return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))
    # Use regular sum-squares like in LeggedGym
    return jp.clip(jp.sum(jp.square(torques)), -1000.0, 1000.0)


def reward_joint_acceleration(
    joint_vel: jax.Array, last_joint_vel: jax.Array, dt: float
) -> jax.Array:
    return jp.clip(jp.sum(jp.square((joint_vel - last_joint_vel) / (dt + EPS))), -1000.0, 1000.0)


def reward_mechanical_work(torques: jax.Array, velocities: jax.Array) -> jax.Array:
    # Penalize mechanical work
    return jp.clip(jp.sum(jp.abs(torques * velocities)), -1000.0, 1000.0)


def reward_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Penalize changes in actions
    return jp.clip(jp.sum(jp.square(act - last_act)), -1000.0, 1000.0)


# def reward_tracking_lin_vel(
#     commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
# ) -> jax.Array:
#     # Tracking of linear velocity commands (xy axes)
#     local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
#     lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
#     lin_vel_reward = jp.exp(-lin_vel_error / (tracking_sigma + EPS))
#     return jp.clip(lin_vel_reward, -1000.0, 1000.0)


# def reward_tracking_ang_vel(
#     commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
# ) -> jax.Array:
#     # Tracking of angular velocity commands (yaw)
#     base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
#     ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
#     return jp.clip(jp.exp(-ang_vel_error / (tracking_sigma + EPS)), -1000.0, 1000.0)

def reward_tracking_foot_lin_pos(
    commands: jax.Array,
    target_world_pos: jax.Array,
    left_foot_pos: jax.Array,
    right_foot_pos: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    """
    Rewards the robot based on the distance between the reaching foot 
    and the target world pose.
    """
    # 1. Identify which leg is supposed to be moving
    # 1 is Left, 0 is Right
    leg_idx = commands[3].astype(jp.int32) 

    # 2. Select the reaching foot's current position using jp.where
    reaching_foot_pos = jp.where(leg_idx == 1, left_foot_pos, right_foot_pos)

    # 3. Calculate Euclidean distance to the pre-computed world target
    distance = jp.linalg.norm(reaching_foot_pos - target_world_pos)

    # 4. Standard Gaussian reward
    return jp.exp(-jp.square(distance) / jp.square(tracking_sigma))

def reward_stand(commands, contact):
    # Use jp.int32 for JAX-compatible indexing
    selected_leg_idx = commands[3].astype(jp.int32) 
    
    # 1. Get contact forces for all 4 feet at once
    # Result: a 4-element array of booleans
    # contact_forces = pipeline_state.contact.force[feet_site_id] 
    # is_contact = contact_forces > 0.1
    
    # 2. Create a "Moving Mask" (e.g., [0, 1, 0, 0] if leg 1 is selected)
    # jp.arange(4) generates [0, 1, 2, 3]
    moving_mask = (jp.arange(4) == selected_leg_idx)
    
    # 3. Calculate rewards for all legs simultaneously
    # If moving leg is in contact: -2.0. If stance leg is in contact: +0.5.
    rewards = jp.where(
        moving_mask, 
        -1.0 * contact,   # Penalty for the manipulator leg
        1.0 * contact     # Reward for the three stance legs
    )
    
    # 4. Sum them up into a single float
    return jp.sum(rewards)

# def reward_feet_air_time(
#     air_time: jax.Array,
#     first_contact: jax.Array,
#     commands: jax.Array,
#     minimum_airtime: float = 0.1,
# ) -> jax.Array:
#     # Reward air time.
#     rew_air_time = jp.sum((air_time - minimum_airtime) * first_contact)
#     rew_air_time *= math.normalize(commands[:3])[1] > 0.05  # no reward for zero command
#     return jp.clip(rew_air_time, -1000.0, 1000.0)


def reward_abduction_angle(
    joint_angles: jax.Array, desired_abduction_angles: jax.Array = jp.zeros(4)
):
    # Penalize abduction angle
    return jp.clip(jp.sum(jp.square(joint_angles[1::3] - desired_abduction_angles)), -1000.0, 1000.0)


# def reward_stand_still(
#     commands: jax.Array,
#     joint_angles: jax.Array,
#     default_pose: jax.Array,
#     command_threshold: float,
# ) -> jax.Array:
#     """
#     Penalize motion at zero commands
#     Args:
#         commands: robot velocity commands
#         joint_angles: joint angles
#         default_pose: default pose
#         command_threshold: if norm of commands is less than this, return non-zero penalty
#     """

#     # Penalize motion at zero commands
#     return jp.clip(
#         jp.sum(jp.abs(joint_angles - default_pose)) * (
#             math.normalize(commands[:3])[1] < command_threshold
#         ),
#         -1000.0,
#         1000.0
#     )


def reward_foot_slip(
    pipeline_state: base.State,
    contact_filt: jax.Array,
    feet_site_id: np.array,
    lower_leg_body_id: np.array,
) -> jax.Array:
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.site_xpos[feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.xpos[lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.clip(
        jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1))),
        -1000.0,
        1000.0
    )


def reward_termination(done: jax.Array, step: jax.Array, step_threshold: int) -> jax.Array:
    return done & (step < step_threshold)


def reward_geom_collision(pipeline_state: base.State, geom_ids: np.array) -> jax.Array:
    contact = jp.array(0.0)
    for id in geom_ids:
        contact += jp.sum(
            ((pipeline_state.contact.geom1 == id) | (pipeline_state.contact.geom2 == id))
            * (pipeline_state.contact.dist < 0.0)
        )
    return jp.clip(contact, -1000.0, 1000.0)
