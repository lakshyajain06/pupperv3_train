"""
Standalone evaluation script for PupperV3 foot-reaching policy.

Uses native MuJoCo for simulation and rendering, with the trained JAX
policy for action inference. No noise or latency is applied (clean eval).

Usage A – Colab (right after training, make_inference_fn already in memory):

    from pupperv3_mjx.eval_tracking import run_eval
    results = run_eval(
        model_xml_path=CONFIG.simulation.model_path,
        make_inference_fn=make_inference_fn,
        params=params,
        ...
    )

Usage B – Mac / standalone (load from saved checkpoint):

    from pupperv3_mjx.eval_tracking import load_policy_from_checkpoint, run_eval
    make_inference_fn, params = load_policy_from_checkpoint(
        checkpoint_dir="path/to/checkpoint_dir",
        model_xml_path="description/pupper_v3.xml",
    )
    results = run_eval(
        model_xml_path="description/pupper_v3.xml",
        make_inference_fn=make_inference_fn,
        params=params,
    )
"""

import mujoco
import numpy as np
import jax
from jax import numpy as jp
from typing import List, Optional, Tuple
import time
import functools


# ── Checkpoint loader (for standalone Mac usage) ─────────────────────────────

def load_policy_from_checkpoint(
    checkpoint_dir: str,
    model_xml_path: str,
    hidden_layer_sizes: Tuple = (256, 128, 128, 128),
    activation: str = "elu",
    env_kwargs: Optional[dict] = None,
):
    """Reconstruct the PPO network and load saved params from an orbax checkpoint.

    Args:
        checkpoint_dir: Path to the orbax checkpoint directory (e.g. ``output_run/40304640``).
        model_xml_path: Path to the MuJoCo XML used for training.
        hidden_layer_sizes: Must match the training CONFIG.
        activation: Must match the training CONFIG.
        env_kwargs: Extra kwargs forwarded to the environment constructor.

    Returns:
        (make_inference_fn, params) ready to pass to ``run_eval``.
    """
    from brax import envs
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.io import model as brax_model
    from pupperv3_mjx import utils as pup_utils

    activation_fn = pup_utils.activation_fn_map(activation)

    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs.setdefault("path", model_xml_path)

    env = envs.get_environment("PupperV3Env", **env_kwargs)

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=hidden_layer_sizes,
        activation=activation_fn,
    )

    ppo_network = make_networks_factory(
        observation_size=env.observation_size,
        action_size=env.action_size,
    )

    make_inference_fn = ppo_network.make_inference_fn

    from orbax import checkpoint as ocp
    from pathlib import Path
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    params = orbax_checkpointer.restore(str(Path(checkpoint_dir).resolve()))

    return make_inference_fn, params


# ── Quaternion helpers (numpy, matching MuJoCo/Brax [w,x,y,z] convention) ─────

def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by quaternion *q* = [w, x, y, z]."""
    xyz = q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[0] * t + np.cross(xyz, t)


# ── Command sampling ─────────────────────────────────────────────────────────

def sample_commands(
    bounding_box: List,
    n_commands: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample *n_commands* random foot-target commands from the bounding box.

    Returns array of shape (n_commands, 4) where columns are [x, y, z, foot_idx].
    foot_idx is always 1.0 (left front foot).
    """
    x_range, y_abs_range, z_range = bounding_box
    foot_idx = 1.0  # left front foot
    y_sign = foot_idx * 2.0 - 1.0  # +1 for left

    xs = rng.uniform(x_range[0], x_range[1], size=n_commands)
    ys = rng.uniform(y_abs_range[0], y_abs_range[1], size=n_commands) * y_sign
    zs = rng.uniform(z_range[0], z_range[1], size=n_commands)
    idxs = np.full(n_commands, foot_idx)

    return np.stack([xs, ys, zs, idxs], axis=-1)


# ── Observation builder (deterministic – no noise / latency) ─────────────────

def _compute_obs(
    d: mujoco.MjData,
    command: np.ndarray,
    desired_world_z: np.ndarray,
    default_pose: np.ndarray,
    last_action: np.ndarray,
    obs_history: np.ndarray,
) -> np.ndarray:
    """Build a 37-dim observation matching the Brax env, then stack into history."""
    torso_quat = d.qpos[3:7].copy()
    inv_rot = _quat_inv(torso_quat)

    # Angular velocity in body frame
    local_ang_vel = _quat_rotate(d.qvel[3:6].copy(), inv_rot)

    # Gravity direction in body frame (normalised)
    gravity_body = _quat_rotate(np.array([0.0, 0.0, -1.0]), inv_rot)
    gravity_body /= np.linalg.norm(gravity_body) + 1e-6

    # Motor angles relative to default pose
    motor_angles = d.qpos[7:19].copy() - default_pose

    obs = np.concatenate([
        local_ang_vel,        # 3
        gravity_body,         # 3
        command,              # 4
        desired_world_z,      # 3
        motor_angles,         # 12
        last_action,          # 12
    ])  # total = 37

    # Newest observation at the front
    new_history = np.roll(obs_history, obs.shape[0])
    new_history[: obs.shape[0]] = obs
    return new_history


# ── Tracking-error computation ───────────────────────────────────────────────

def _compute_error(
    d: mujoco.MjData,
    command: np.ndarray,
    foot_site_ids: List[int],
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (dist_error, error_xyz, target_world, actual_world)."""
    target_local = command[:3]
    leg_idx = int(command[3])
    anchor_idx = 1 - leg_idx

    anchor_pos = d.site_xpos[foot_site_ids[anchor_idx]].copy()
    torso_quat = d.qpos[3:7].copy()
    target_world = anchor_pos + _quat_rotate(target_local, torso_quat)

    actual_pos = d.site_xpos[foot_site_ids[leg_idx]].copy()
    error_xyz = target_world - actual_pos
    dist = np.linalg.norm(error_xyz)
    return dist, error_xyz, target_world, actual_pos


# ── Main evaluation routine ──────────────────────────────────────────────────

def run_eval(
    model_xml_path: str,
    make_inference_fn,
    params,
    bounding_box: List = [(-0.05, 0.15), (0.15, 0.3), (-0.05, 0.1)],
    action_scale: float = 0.75,
    default_pose=np.array(
        [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52]
    ),
    position_control_kp: float = 5.5,
    dof_damping: float = 0.25,
    observation_history: int = 20,
    observation_dim: int = 37,
    physics_dt: float = 0.004,
    env_dt: float = 0.02,
    n_episodes: int = 50,
    steps_per_episode: int = 300,
    settle_steps: int = 100,
    render: bool = True,
    render_episodes: int = 5,
    render_width: int = 640,
    render_height: int = 480,
    camera_name: str = "tracking_cam",
    seed: int = 42,
):
    """Evaluate a trained policy on random foot-target commands.

    Args:
        model_xml_path: Path to the MuJoCo XML used for training.
        make_inference_fn: ``make_inference_fn`` returned by ``ppo.train``.
        params: Policy params returned by ``ppo.train``.
        bounding_box: [[x_lo, x_hi], [y_abs_lo, y_abs_hi], [z_lo, z_hi]].
        action_scale / default_pose / …: must match training CONFIG.
        n_episodes: How many random commands to evaluate.
        steps_per_episode: Env steps per command (at 50 Hz → 6 s).
        settle_steps: Discard the first N steps when computing steady-state error.
        render: Whether to render video for a subset of episodes.
        render_episodes: How many episodes to render (first N).
        seed: Random seed.

    Returns:
        dict with aggregate statistics.
    """
    default_pose = np.asarray(default_pose, dtype=np.float64)
    n_frames = int(env_dt / physics_dt)

    # ── Load MuJoCo model & apply same actuator overrides as training ────
    m = mujoco.MjModel.from_xml_path(model_xml_path)
    m.opt.timestep = physics_dt
    m.actuator_gainprm[:, 0] = position_control_kp
    m.actuator_biasprm[:, 1] = -position_control_kp
    m.actuator_biasprm[:, 2] = -dof_damping

    # Write default pose into the home keyframe
    home_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        m.key_qpos[home_id, 7:19] = default_pose

    d = mujoco.MjData(m)

    # Resolve foot site IDs
    foot_names = [
        "leg_front_r_3_foot_site",
        "leg_front_l_3_foot_site",
        "leg_back_r_3_foot_site",
        "leg_back_l_3_foot_site",
    ]
    foot_site_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in foot_names
    ]

    # Joint limits (from model)
    lowers = m.jnt_range[1:, 0].copy()
    uppers = m.jnt_range[1:, 1].copy()

    # ── Build JAX inference function ─────────────────────────────────────
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    # ── Sample commands ──────────────────────────────────────────────────
    np_rng = np.random.default_rng(seed)
    commands = sample_commands(bounding_box, n_episodes, np_rng)

    # ── Storage ──────────────────────────────────────────────────────────
    all_errors = []        # per-episode list of per-step distances
    steady_errors = []     # per-episode steady-state mean distance
    steady_errors_xyz = [] # per-episode steady-state mean [ex, ey, ez]
    frames = []            # rendered frames (across all rendered episodes)

    # Optional renderer
    renderer = None
    if render:
        renderer = mujoco.Renderer(m, width=render_width, height=render_height)

    desired_world_z = np.array([0.0, 0.0, 1.0])

    print(f"Running {n_episodes} eval episodes, {steps_per_episode} steps each …")
    t0 = time.time()

    for ep in range(n_episodes):
        cmd = commands[ep]

        # Reset to home keyframe
        mujoco.mj_resetDataKeyframe(m, d, home_id)
        mujoco.mj_forward(m, d)

        obs_history = np.zeros(observation_history * observation_dim)
        last_action = np.zeros(12)
        jax_rng = jax.random.PRNGKey(seed + ep)

        ep_dists = []
        ep_errors_xyz = []

        should_render = render and ep < render_episodes

        for step in range(steps_per_episode):
            # Build observation
            obs_history = _compute_obs(
                d, cmd, desired_world_z, default_pose, last_action, obs_history
            )

            # Query the policy (JAX)
            jax_rng, act_rng = jax.random.split(jax_rng)
            obs_jax = jp.array(obs_history)
            action_jax, _ = jit_inference_fn(obs_jax, act_rng)
            action = np.array(action_jax)
            last_action = action.copy()

            # Apply action → motor targets (PD position control)
            motor_targets = default_pose + action * action_scale
            motor_targets = np.clip(motor_targets, lowers, uppers)
            d.ctrl[:] = motor_targets

            # Step physics n_frames times
            for _ in range(n_frames):
                mujoco.mj_step(m, d)

            # Measure tracking error
            dist, err_xyz, target_world, actual_pos = _compute_error(
                d, cmd, foot_site_ids
            )
            ep_dists.append(dist)
            ep_errors_xyz.append(np.abs(err_xyz))

            # Render
            if should_render:
                renderer.update_scene(d, camera=camera_name)
                frame = renderer.render()
                frames.append(frame)

        all_errors.append(ep_dists)

        # Steady-state = after settle_steps
        ss_dists = ep_dists[settle_steps:]
        ss_xyz = ep_errors_xyz[settle_steps:]
        steady_errors.append(np.mean(ss_dists))
        steady_errors_xyz.append(np.mean(ss_xyz, axis=0))

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  Episode {ep+1:3d}/{n_episodes}  "
                f"cmd=[{cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f}]  "
                f"steady-state error = {np.mean(ss_dists)*100:.2f} cm"
            )

    elapsed = time.time() - t0

    # ── Aggregate stats ──────────────────────────────────────────────────
    steady_errors = np.array(steady_errors)
    steady_errors_xyz = np.array(steady_errors_xyz)

    results = {
        "mean_error_m": float(np.mean(steady_errors)),
        "mean_error_cm": float(np.mean(steady_errors) * 100),
        "std_error_cm": float(np.std(steady_errors) * 100),
        "median_error_cm": float(np.median(steady_errors) * 100),
        "mean_error_x_cm": float(np.mean(steady_errors_xyz[:, 0]) * 100),
        "mean_error_y_cm": float(np.mean(steady_errors_xyz[:, 1]) * 100),
        "mean_error_z_cm": float(np.mean(steady_errors_xyz[:, 2]) * 100),
        "pct_within_1cm": float(np.mean(steady_errors < 0.01) * 100),
        "pct_within_2cm": float(np.mean(steady_errors < 0.02) * 100),
        "pct_within_5cm": float(np.mean(steady_errors < 0.05) * 100),
        "n_episodes": n_episodes,
        "steps_per_episode": steps_per_episode,
        "settle_steps": settle_steps,
        "elapsed_s": elapsed,
    }

    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    print(f"  Episodes evaluated:        {n_episodes}")
    print(f"  Steps / episode:           {steps_per_episode}")
    print(f"  Settle steps (discarded):  {settle_steps}")
    print(f"  Wall time:                 {elapsed:.1f} s")
    print("-" * 60)
    print(f"  Mean steady-state error:   {results['mean_error_cm']:.2f} ± {results['std_error_cm']:.2f} cm")
    print(f"  Median steady-state error: {results['median_error_cm']:.2f} cm")
    print(f"  Mean error X / Y / Z:      {results['mean_error_x_cm']:.2f} / {results['mean_error_y_cm']:.2f} / {results['mean_error_z_cm']:.2f} cm")
    print(f"  Within 1 cm:               {results['pct_within_1cm']:.1f} %")
    print(f"  Within 2 cm:               {results['pct_within_2cm']:.1f} %")
    print(f"  Within 5 cm:               {results['pct_within_5cm']:.1f} %")
    print("=" * 60)

    # ── Save video ───────────────────────────────────────────────────────
    if render and frames:
        fps = int(1.0 / env_dt)
        results["frames"] = frames
        try:
            import mediapy as media
            video_path = "eval_rollouts.mp4"
            media.write_video(video_path, frames, fps=fps)
            print(f"\nSaved video to {video_path} ({len(frames)} frames at {fps} fps)")
            # Also try to display inline (works in notebooks, no-op otherwise)
            try:
                media.show_video(frames, fps=fps, title="Eval rollouts")
            except Exception:
                pass
        except Exception as e:
            print(f"\nCould not save video: {e}")
            print("Frames array is available in results['frames'].")

    if renderer is not None:
        renderer.close()

    results["all_errors"] = all_errors
    results["commands"] = commands
    return results


# ── CLI entry point (for running on Mac) ─────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate PupperV3 foot-reaching policy")
    parser.add_argument("--model_xml", type=str, required=True,
                        help="Path to MuJoCo XML (e.g. description/pupper_v3.xml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to orbax checkpoint dir (e.g. output_run/40304640)")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--settle", type=int, default=100)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--render_episodes", type=int, default=3)
    args = parser.parse_args()

    print("Loading policy from checkpoint …")
    make_inf, par = load_policy_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        model_xml_path=args.model_xml,
    )

    run_eval(
        model_xml_path=args.model_xml,
        make_inference_fn=make_inf,
        params=par,
        n_episodes=args.n_episodes,
        steps_per_episode=args.steps,
        settle_steps=args.settle,
        render=not args.no_render,
        render_episodes=args.render_episodes,
    )

