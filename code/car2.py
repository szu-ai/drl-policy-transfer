#!/usr/bin/env python3

"""Training-only implementation of uncertainty-calibrated ego-relational
policy transfer for safety-aware autonomous driving in CARLA 0.9.15.

Code-to-paper mapping (T-PAMI manuscript, Secs. III-IV):
  Sec. III-B  MDP / route / corridor / TTC : CarlaReliableTransferEnv (Eqs. 1-4)
  Sec. III-C  Ego-relational state         : GraphAttention, CompactStateEncoder (Eqs. 5-7)
  Sec. III-D  Dense multi-objective reward : _compute_reward_components,
                                             build_reward_from_info (Eqs. 8-9)
  Sec. III-E  Uncertainty-gated exploration: compute_sigma_dec / compute_sigma_bar /
                                             entropy_coefficient / compute_actor_loss (Eqs. 10-12)
  Sec. III-F  Influence-consistent transfer: compute_transfer_loss,
                                             maml_style_initialize (Eqs. 13-15)
Deviation register (every departure from the manuscript is switchable and
defaults to the paper-described behavior unless noted):

  D1 Eq. (6) attention score. As printed, eta_i = -||dp_i||^2 / (sigma_i^2 + eps)
     INCREASES with edge variance, so it up-weights unreliable entities, which
     contradicts the text below Eq. (6) and Table I ("prioritizes nearby reliable
     entities"). Default attention_score_form="reliability" implements the stated
     behavior, eta_i = -lambda_d ||dp_i||^2 - lambda_sigma sigma_i^2. Setting
     attention_score_form="paper_ratio" reproduces the printed formula verbatim.
     The manuscript equation should be corrected to match whichever form is used.

  D2 MAML (Eq. 14) is first-order: the inner loop updates fast weights in
     place, so second derivatives are omitted. The exact second-order
     meta-gradient is NOT implemented. Report Eq. (14) as a first-order MAML
     approximation in Sec. III-F; Table VI already shows no established gain
     from the meta-initialization, so nothing in the results depends on it.

  D3 Training curriculum (scripted route-guidance blended into early actions)
     is NOT part of the paper's method and now defaults to OFF. Enable with
     --enable-training-curriculum; if used, it must be disclosed in Sec. IV-A
     and applied to baseline arms as well, or Fig. 7 is confounded.

  D4 Progress term (Sec. III-D) defaults to the primary arc-length form
     rp = tanh(ds/tau_s); rp_arc_weight < 1 blends in the stated velocity
     surrogate tanh(vhat/tau_v).

  D5 Rollout sigma_bar uses a cheap proxy and is recomputed with the real critic
     ensemble before the transition is stored, so replay rewards match Eq. (9).

  D6 Paper-ablation safeguard: positive safety/uncertainty reward is gated by
     measured forward route progress and a small open-road idle cost is used.
     Without this shared correction, a stopped lane-centred vehicle receives a
     positive return and the experiment measures reward exploitation instead of
     component contribution.  Report this protocol detail with every arm.

Protocol constants not stated in the manuscript and required for reproduction:
route target 500 m (accepted 490-510 m), success at route_success_pct route
completion, max_episode_steps control steps per episode, and training traffic
density set on the command line.

Ablation runner:
  --ablation selects exactly one matched component removal.  Every run is
  isolated under <out-dir>/ablations/<ablation>/ so checkpoints cannot overwrite
  the full model or another ablation.  Source-training ablations are evaluated
  later with the frozen actor on target towns; evaluation is intentionally not
  implemented in this training-only file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "Gymnasium is required. Install it in this environment with: "
        "python -m pip install gymnasium"
    ) from exc

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# CARLA import (0.9.15 robust)
# ======================================================
def _candidate_carla_roots() -> List[str]:
    env_root = os.environ.get("CARLA_ROOT", "").strip()
    candidates = [
        env_root,
        os.path.expanduser("~/carla"),
        os.path.expanduser("~/CARLA_0.9.15"),
        os.path.expanduser("~/CARLA_0.9.15/LinuxNoEditor"),
        os.path.expanduser("~/CARLA_0.9.14"),
        os.path.expanduser("~/CARLA_0.9.14/LinuxNoEditor"),
        os.path.expanduser("~/CARLA_0.9.13"),
        os.path.expanduser("~/CARLA_0.9.13/LinuxNoEditor"),
    ]
    return [p for p in candidates if p and os.path.exists(p)]

def _setup_carla_pythonapi() -> None:
    """
    Import CARLA without letting helper-source folders shadow a working install.

    Order:
    1) If `import carla` already works, keep it.
    2) Otherwise, try CARLA wheel/egg from PythonAPI/carla/dist.
    3) Do NOT add helper folders here; add them later after carla imports.
    """
    try:
        import carla  # noqa: F401
        return
    except Exception:
        pass

    pymaj, pymin = sys.version_info.major, sys.version_info.minor
    searched_roots: List[str] = []
    searched_dist_dirs: List[str] = []
    tried_pkgs: List[str] = []

    for root in _candidate_carla_roots():
        searched_roots.append(root)

        pythonapi = os.path.join(root, "PythonAPI")
        carla_pkg_dir = os.path.join(pythonapi, "carla")
        dist_dir = os.path.join(carla_pkg_dir, "dist")

        if not os.path.isdir(dist_dir):
            continue

        searched_dist_dirs.append(dist_dir)

        pkg_patterns = [
            os.path.join(dist_dir, f"carla-*-cp{pymaj}{pymin}-*.whl"),
            os.path.join(dist_dir, f"carla-*-py{pymaj}.{pymin}-linux-x86_64.egg"),
            os.path.join(dist_dir, "carla-*.whl"),
            os.path.join(dist_dir, "carla-*.egg"),
        ]

        matches: List[str] = []
        for pat in pkg_patterns:
            matches.extend(glob.glob(pat))

        for pkg in sorted(set(matches), reverse=True):
            tried_pkgs.append(pkg)
            if pkg not in sys.path:
                sys.path.insert(0, pkg)
            try:
                import carla  # noqa: F401
                return
            except Exception:
                try:
                    sys.path.remove(pkg)
                except ValueError:
                    pass

    raise ImportError(
        "Could not import CARLA PythonAPI.\n"
        f"Python version: {pymaj}.{pymin}\n"
        f"Searched CARLA roots: {searched_roots}\n"
        f"Searched dist dirs: {searched_dist_dirs}\n"
        f"Tried packages: {tried_pkgs}\n\n"
        "Your previous run indicates CARLA was importable before this patch, so the usual cause here is\n"
        "that helper-source paths were added too early and shadowed the working CARLA package.\n\n"
        "Fix options:\n"
        "1) Keep this function exactly as shown\n"
        "2) Make sure your working environment can do: python3 -c 'import carla; print(carla)'\n"
        "3) If needed, export CARLA_ROOT to the real CARLA install root"
    )

def _setup_carla_helper_paths() -> None:
    """
    Add helper-module paths only AFTER `import carla` succeeds.

    Needed for:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    """
    for root in _candidate_carla_roots():
        pythonapi = os.path.join(root, "PythonAPI")
        carla_pkg_dir = os.path.join(pythonapi, "carla")

        # `agents` lives under PythonAPI/carla/agents
        if os.path.isdir(carla_pkg_dir) and carla_pkg_dir not in sys.path:
            sys.path.insert(0, carla_pkg_dir)

        # Optional helper path; safe after `carla` is already imported
        if os.path.isdir(pythonapi) and pythonapi not in sys.path:
            sys.path.append(pythonapi)

_setup_carla_pythonapi()

import carla  # noqa: E402

_setup_carla_helper_paths()

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except Exception as e1:
    try:
        from carla.agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    except Exception as e2:
        print(f"[WARN] GlobalRoutePlanner import failed: {e1}")
        print(f"[WARN] Fallback GlobalRoutePlanner import failed: {e2}")
        GlobalRoutePlanner = None
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


# ======================================================
# Config
# ======================================================
@dataclass
class Config:
    fps: int = 20
    client_timeout: float = 120.0
    tm_port: int = 8000
    seed: int = 42
    debug_mode: bool = True
    debug_step_freq: int = 20
    follow_ego_view: bool = True
    no_rendering_mode: bool = False

    car_name: str = "vehicle.tesla.model3"
    fixed_color: str = "255,0,0"

    fixed_weather: str = "night_rain_fog"
    target_weather: str = "mixed"
    # Route assistance is opt-in during learning. The default training path
    # always uses the policy command directly.
    use_safety_shield: bool = False
    mode_name: str = "train"

    # Matched ablation configuration.  Only one ablation is selected per run.
    # These values are saved inside every checkpoint through cfg.__dict__.
    ablation_name: str = "full"
    use_uncertainty_attention: bool = True
    use_critic_ensemble: bool = True
    use_entropy_gate: bool = True
    use_dense_reward: bool = True
    use_transfer_alignment: bool = True
    use_maml_initialization: bool = True
    # Protocol safeguards shared by the full model and every ablation.  These
    # are not ablated: they prevent action/reward and episode-boundary bugs
    # from becoming hidden experimental variables.
    replay_applied_action: bool = True
    timeout_is_terminal: bool = True
    progress_gate_positive_reward: bool = True
    progress_gate_distance_m: float = 0.05
    open_road_idle_penalty: float = 0.05
    open_road_idle_delta_s_m: float = 0.01
    goal_terminal_reward: float = 25.0
    collision_terminal_reward: float = -25.0
    offroute_terminal_reward: float = -15.0
    stuck_terminal_reward: float = -10.0
    timeout_terminal_reward: float = -10.0
    save_best_training_checkpoint: bool = False
    progress_every_steps: int = 1000
    event_step_reward: float = -0.001

    # D3: practical off-policy bootstrap that is NOT described in the paper.
    # Defaults to OFF so a default run matches the manuscript's method exactly.
    # Enable with --enable-training-curriculum, and if enabled, disclose it in
    # Sec. IV-A and apply the same bootstrap to the baseline arms.
    use_training_curriculum: bool = False
    curriculum_steps: int = 60000
    curriculum_initial_blend: float = 0.95
    curriculum_final_blend: float = 0.0
    curriculum_action_noise: float = 0.06
    curriculum_traffic_steps: int = 100000

    npc_min: int = 0
    npc_max: int = 2
    # Pedestrian NPCs (paper Sec. III-C / Table I: pedestrians are first-class
    # relational entities; Fig. 2 / Fig. 12 pedestrian-crossing scenarios).
    walker_min: int = 2
    walker_max: int = 6
    train_walker_min: int = 8
    train_walker_max: int = 12
    walker_cross_factor: float = 0.6      # fraction of walkers allowed to cross roads
    walker_spawn_radius_m: float = 80.0   # bias walker spawns near the planned route
    npc_spawn_radius_m: float = 120.0     # bias NPC vehicle spawns near the planned route
    # Keep the loaded CARLA world alive between env instances and episodes:
    # resets then only destroy/respawn the ego, NPC vehicles, walkers, and
    # sensors instead of triggering a full client.load_world() map reload.
    reuse_loaded_world: bool = True
    world_settle_wait_s: float = 5.0
    max_entity_obs: int = 10
    entity_max_dist: float = 60.0

    # A route endpoint is selected automatically unless a fixed spawn-point
    # goal is explicitly supplied. Every accepted route is trimmed to about 500 m.
    use_fixed_destination: bool = False
    fixed_goal_index: int = -1
    target_goal_index: int = -1
    # Paper Sec. III-B: reference route R = {r_j} along lane centerlines.
    # This implementation uses the requested ~500 m experimental protocol.
    min_route_length_m: float = 480.0          # hard floor on arc-length
    # Euclidean prefilter only: routes wind, so the straight-line spawn-to-goal
    # distance must stay well below the 500 m arc-length target or long routes
    # become unreachable in small towns. The arc window below governs length.
    candidate_goal_min_dist_m: float = 50.0
    candidate_goal_max_tries: int = 160
    route_success_pct: float = 98.0
    strict_goal_route: bool = True
    allow_fallback_route: bool = True
    route_target_length_m: float = 500.0       # target arc-length
    route_target_tolerance_m: float = 10.0     # accepted 490--510 m
    route_soft_min_length_m: float = 490.0
    route_soft_max_length_m: float = 510.0
    prefer_auto_length_route: bool = True
    enforce_route_length_for_fixed_goal: bool = True   # fixed-length route protocol
    max_reset_start_progress_pct: float = 10.0
    max_reset_start_dL_m: float = 3.5

    # D1 (see module docstring): "reliability" implements the behavior stated
    # in the text and Table I; "paper_ratio" reproduces Eq. (6) as printed.
    attention_score_form: str = "reliability"
    attention_distance_weight: float = 4.0
    attention_uncertainty_weight: float = 2.0
    attention_ratio_eps: float = 1e-6
    # D2: the MAML meta-gradient (Eq. 14) is a first-order approximation.
    # Kept as an explicit constant so the deviation is visible in the config.
    maml_first_order: bool = True

    sem_dim: int = 3
    eps_min: float = 1.5
    eps_max: float = 4.0

    tau_d: float = 1.6
    tau_p: float = 12.0
    # Paper Sec. III-D: rp = tanh(Δs/τs). Smaller τs = more signal at small steps.
    # At 18 km/h and 20 Hz, Δs≈0.25 m. τs=0.5 gives a strong immediate
    # progress signal while retaining the exact tanh form in Sec. III-D.
    tau_s: float = 0.5
    tau_v: float = 5.0   # paper Sec. III-D velocity-surrogate scale (v̂_t / τv)
    k_l: float = 0.8
    # Eq. (4) constraint penalties. phi_L is the corridor excess beyond
    # epsilon(mu_A); lambda_phi_L scales it, and the term is capped so a single
    # bad step cannot dominate the return.
    lambda_phi_L: float = 0.35
    phi_L_cap: float = 4.0
    # Explicit wrong-lane / opposite-lane components of phi_L. Being in the
    # neighbouring lane is a lane-discipline failure, not merely a large d_L.
    lambda_wrong_lane: float = 0.25
    lambda_opposite_lane: float = 0.60
    k_p: float = 0.6
    k_r: float = 0.7
    # Coefficients operate on SI jerk and steering-rate values. Values are
    # kept small enough that the comfort term does not swamp route progress.
    k_j: float = 0.002
    k_delta: float = 0.02

    # Paper Eq. (8): r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u, Σw = 1
    w_s: float = 0.45
    w_p: float = 0.30
    w_c: float = 0.15
    w_u: float = 0.10
    # Paper Sec. III-D uses either arc progress or a velocity surrogate.
    # The default is the stated arc-length form, not a mixture.
    rp_arc_weight: float = 1.0
    tl_near_dist: float = 22.0
    tl_stop_dist: float = 14.0
    # Scan range for lights the ego is approaching but has not yet entered the
    # trigger volume of. vehicle.get_traffic_light() alone gives roughly 10-15 m
    # of warning, which is shorter than the braking distance at 25 km/h.
    tl_detect_dist: float = 45.0
    # Comfortable deceleration used to derive the distance at which the red-light
    # penalty must already be active: d_brake = v^2 / (2a) + tl_stop_margin.
    tl_brake_decel: float = 3.0
    tl_stop_margin: float = 2.5
    # Perception/actuation lag folded into the onset distance so the penalty and
    # the assist both appear before braking becomes impossible.
    tl_reaction_s: float = 0.6
    # How far past the stop line a red light is still tracked, so that crossing
    # on red is detected and scored as full noncompliance (rho_t = 1).
    tl_cross_track_m: float = 8.0
    # Bounded red-light braking in the default (no-shield) path. Off by default:
    # enabling it raises the reported intervention rate, which Sec. IV-F requires
    # to be read alongside DS.
    red_light_assist: bool = False
    min_ttc_s: float = 1.8
    terminate_on_collision: bool = True
    terminate_on_offroad: bool = True

    n_critics: int = 5
    calib_temperature: float = 1.0
    action_dim: int = 3
    gamma: float = 0.99
    target_tau: float = 5e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    # Learned SAC temperature is combined with the paper's fixed
    # uncertainty-gated entropy term.
    init_alpha: float = 0.2
    # Paper Fig. 9 traces a decaying SAC temperature alongside sigma_bar, so
    # the learned temperature must be active. With it off, the effective
    # entropy coefficient is pinned at beta0 (1 - sigma_bar) and never anneals,
    # which keeps the actor near-random and produces steady lateral drift.
    use_learned_alpha: bool = True
    beta0: float = 1.0
    # lambda_ent scales L_ent in Eq. (15); the manuscript does not fix its
    # value. The effective SAC temperature is
    # lambda_ent * beta0 * (1 - sigma_bar) + alpha. At sigma_bar ~ 0.37 and
    # beta0 = 1.0, lambda_ent = 1.0 gives
    # ~0.63, several times a workable temperature for a 3-D action space, so
    # exploration noise dominates steering. 0.2 keeps the gate's dynamics while
    # putting the temperature in the usual SAC range.
    lambda_ent: float = 0.2
    lambda_transfer: float = 1.0
    lambda_alpha_align: float = 1.0
    lambda_u_align: float = 1.0
    replay_size: int = 200000
    batch_size: int = 512
    maml_inner_lr: float = 1e-4
    maml_inner_steps: int = 1
    maml_meta_step_size: float = 1.0
    # Keep the uncertainty mapping fixed while replay is populated; changing
    # it would silently change the reward definition for newer transitions.
    # Dense-traffic training protocol: ~20 NPC vehicles and ~10 pedestrians
    # around the route (CLI flags override these defaults).
    train_npc_min: int = 14
    train_npc_max: int = 22
    adapt_episodes: int = 100

    obs_miss_base: float = 0.08
    obs_pos_noise_m: float = 0.35
    obs_vel_noise_ms: float = 0.45
    tl_obs_noise_m: float = 1.25

    route_point_spacing_m: float = 2.0
    lookahead_base_m: float = 6.0
    lookahead_speed_gain: float = 0.35
    route_steer_kp: float = 2.6
    route_cte_kp: float = 1.95
    route_speed_kp: float = 0.13
    target_speed_kmh: float = 18.0
    min_target_speed_kmh: float = 5.5
    curve_speed_penalty_kmh: float = 11.0
    max_safe_steer_at_speed: float = 0.24
    hard_speed_cap_kmh: float = 21.0
    route_override_dL_m: float = 0.95
    route_hard_dL_m: float = 1.80
    route_override_heading_rad: float = 0.22
    route_hard_heading_rad: float = 0.38

    min_throttle_deadzone: float = 0.05
    min_brake_deadzone: float = 0.05
    low_speed_steer_limit: float = 0.18
    max_throttle_step: float = 0.06
    max_brake_step: float = 0.08
    max_steer_step: float = 0.08

    stuck_speed_kmh: float = 1.0
    stuck_steps_threshold: int = 18
    release_duration_steps: int = 25
    release_throttle: float = 0.42
    stuck_terminate_steps: int = 250

    front_vehicle_max_dist: float = 25.0
    front_vehicle_block_dist: float = 7.5
    front_vehicle_soft_block_dist: float = 12.0
    front_vehicle_soft_block_speed_kmh: float = 1.0

    goal_reach_dist_m: float = 10.0
    goal_reach_remaining_s_m: float = 8.0

    max_route_deviation_m: float = 5.5
    offroute_grace_steps: int = 22
    offroute_heading_gate_rad: float = 0.35
    strong_offroute_factor: float = 2.0

    route_search_back: int = 2
    route_search_ahead: int = 20

    # Paper-faithful dense reward uses no terminal bonus/penalty terms.
    # These are kept as zeroed compatibility fields for older checkpoints/configs.
    goal_bonus: float = 0.0
    collision_penalty: float = 0.0
    offroute_penalty: float = 0.0
    timeout_penalty: float = 0.0

    blocked_timeout_extension_steps: int = 600
    blocked_timeout_speed_kmh: float = 1.0
    near_goal_timeout_extension_steps: int = 1000   # extra steps when within near_goal_remaining_s_m
    near_goal_remaining_s_m: float = 30.0            # "near goal" zone starts at 30 m remaining
    coast_speed_band_kmh: float = 1.5
    soft_brake_speed_excess_kmh: float = 4.0
    brake_release_threshold: float = 0.12
    center_deadband_m: float = 0.06
    center_push_gain: float = 0.55
    center_push_hard_gain: float = 0.85
    center_push_start_m: float = 0.45
    route_soft_dL_m: float = 0.85
    route_hard_dL_m_tight: float = 1.45
    # Optional bounded stabilization (Table I). Nominal policy actions
    # are untouched; assistance activates only near route/safety constraints.
    steer_guidance_blend: float = 0.35
    policy_route_blend_base: float = 0.0
    policy_route_blend_bad: float = 0.35
    policy_route_blend_hard: float = 0.85
    open_road_brake_soft_limit: float = 0.06
    open_road_brake_curve_limit: float = 0.14
    caution_ttc_s: float = 2.8
    hard_ttc_s: float = 1.8

    spectator_distance_m: float = 8.0
    spectator_height_m: float = 4.0
    spectator_pitch_deg: float = -15.0

    enable_collision_sensor: bool = True
    enable_lane_invasion_sensor: bool = False
    cleanup_sleep_s: float = 0.10
    tick_retry_count: int = 3
    tick_retry_sleep_s: float = 0.50
    post_spawn_settle_ticks: int = 2
    warmup_reset_ticks: int = 3
    rebuild_env_on_reset_failure: bool = True
    destroy_stale_owned_actors_on_reset: bool = True
    destroy_stale_spawn_blockers: bool = True
    spawn_blocker_radius_m: float = 2.5
    spawn_retry_lift_m: float = 0.35

    # 500 m at the nominal 18 km/h requires about 2000 steps at 20 Hz.
    # A 6000-step budget permits traffic-light waits and recovery without
    # allowing stalled episodes to dominate the replay buffer.
    max_episode_steps: int = 6000
    out_dir: str = "./culrt_carla_0915_aligned"

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def edge_dim(self) -> int:
        return 2 + 2 + self.sem_dim + 2 + 1

    @property
    def max_entities(self) -> int:
        return self.max_entity_obs

    @property
    def scalar_dim(self) -> int:
        return 13

    @property
    def model_dir(self) -> str:
        return os.path.join(self.out_dir, "models")

CFG = Config()

ABLATION_CHOICES: Tuple[str, ...] = (
    "full",
    "no_uncertainty_attention",
    "no_critic_ensemble",
    "no_entropy_gate",
    "event_reward",
    "no_transfer_alignment",
    "no_maml",
)


# ======================================================
# Utilities
# ======================================================
def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vec3_length(v: carla.Vector3D) -> float:
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def distance2d(a: carla.Location, b: carla.Location) -> float:
    dx = float(a.x - b.x)
    dy = float(a.y - b.y)
    return float(math.sqrt(dx * dx + dy * dy))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def point_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float, float]:
    ab = b - a
    ab2 = float(np.dot(ab, ab) + 1e-9)
    t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
    proj = a + t * ab
    dist = float(np.linalg.norm(p - proj))
    seg_len = float(np.linalg.norm(ab))
    return proj, dist, seg_len * t


def apply_weather(world: carla.World, mode: str) -> None:
    if mode == "night_rain_fog":
        weather = carla.WeatherParameters(
            cloudiness=90.0,
            precipitation=90.0,
            precipitation_deposits=85.0,
            wind_intensity=30.0,
            sun_altitude_angle=-25.0,
            fog_density=40.0,
            fog_distance=8.0,
            wetness=100.0,
        )
    elif mode == "mixed":
        weather = random.choice(
            [
                carla.WeatherParameters(
                    cloudiness=80.0,
                    precipitation=60.0,
                    precipitation_deposits=55.0,
                    wind_intensity=20.0,
                    sun_altitude_angle=-20.0,
                    fog_density=20.0,
                    fog_distance=20.0,
                    wetness=70.0,
                ),
                carla.WeatherParameters(
                    cloudiness=50.0,
                    precipitation=20.0,
                    precipitation_deposits=20.0,
                    wind_intensity=10.0,
                    sun_altitude_angle=10.0,
                    fog_density=5.0,
                    fog_distance=60.0,
                    wetness=20.0,
                ),
                carla.WeatherParameters(
                    cloudiness=95.0,
                    precipitation=80.0,
                    precipitation_deposits=75.0,
                    wind_intensity=35.0,
                    sun_altitude_angle=-10.0,
                    fog_density=25.0,
                    fog_distance=15.0,
                    wetness=90.0,
                ),
            ]
        )
    else:
        weather = carla.WeatherParameters.Default
    world.set_weather(weather)


def get_fog_norm(world: carla.World) -> float:
    try:
        return float(np.clip(float(world.get_weather().fog_density) / 100.0, 0.0, 1.0))
    except Exception:
        return 0.0


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_module_state_compat(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load weights while tolerating scalar-state expansion."""
    model_state = module.state_dict()
    patched: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            patched[key] = value
            continue
        if key.endswith('mlp_scal.0.weight') and value.ndim == 2 and target.ndim == 2 and value.shape[0] == target.shape[0]:
            new_weight = target.clone()
            cols = min(value.shape[1], target.shape[1])
            new_weight[:, :cols] = value[:, :cols]
            patched[key] = new_weight
    module.load_state_dict(patched, strict=False)


def resolve_existing_path(path_text: str) -> str:
    path_text = str(path_text).strip()
    if not path_text:
        return path_text
    if os.path.isabs(path_text):
        return path_text

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(path_text),
        os.path.abspath(os.path.join(script_dir, path_text)),
        os.path.abspath(os.path.join(os.path.dirname(script_dir), path_text)),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def resolve_output_dir(path_text: str) -> str:
    path_text = str(path_text).strip()
    if not path_text:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(script_dir, "culrt_carla_0915_aligned"))

    if os.path.isabs(path_text):
        return path_text

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prefer output relative to the script location.
    script_relative = os.path.abspath(os.path.join(script_dir, path_text))
    cwd_relative = os.path.abspath(path_text)

    if os.path.exists(script_relative):
        return script_relative
    if os.path.exists(cwd_relative):
        return cwd_relative

    return script_relative



# ======================================================
# Environment
# ======================================================
class CarlaReliableTransferEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2200,
        town_name: str = "Town02",
        fixed_spawn_index: int = 0,
        fixed_goal_index: Optional[int] = None,
        weather_mode: str = "night_rain_fog",
        cfg: Config = CFG,
    ):
        super().__init__()
        self.cfg = cfg
        self.host = host
        self.port = port
        self.requested_town_name = town_name
        self.fixed_spawn_index = fixed_spawn_index
        self.fixed_goal_index = fixed_goal_index if fixed_goal_index is not None else cfg.fixed_goal_index
        self.weather_mode = weather_mode
        # Actor ownership must be unique per environment instance.  Generic
        # roles such as "hero" and "autopilot" can cause one client to delete
        # another client's actors during cleanup.
        self.owner_tag = f"culrt_{os.getpid()}_{id(self):x}".lower()
        self.ego_role = f"{self.owner_tag}_ego"
        self.npc_role = f"{self.owner_tag}_npc"
        self.walker_role = f"{self.owner_tag}_walker"

        self.client = carla.Client(host, port)
        self.client.set_timeout(cfg.client_timeout)
        self.world: carla.World = self._load_world_with_fallback(town_name)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.original_settings = self.world.get_settings()
        self._apply_sync_settings()

        self.tm = self.client.get_trafficmanager(cfg.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(cfg.seed)

        self.spawn_points = self.map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in the loaded CARLA map.")

        self.vehicle: Optional[carla.Vehicle] = None
        self.npcs: List[carla.Vehicle] = []
        self.walkers: List[carla.Actor] = []
        self.walker_controllers: List[carla.Actor] = []
        self.sensor_list: List[carla.Actor] = []
        self.collision_events: List[object] = []
        self.lane_invasion_events: List[object] = []
        
        self._teardown_in_progress = False
        self._episode_live = False
        self._terminal_reason = ""

        self.episode_steps = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_steer = 0.0
        self.prev_acc = np.zeros(2, dtype=np.float32)
        self.prev_loc: Optional[carla.Location] = None
        self.distance_driven_m = 0.0

        self.stuck_steps = 0
        self.release_steps_left = 0
        self.offroute_steps = 0
        self.prev_route_s: Optional[float] = None
        self.safety_interventions = 0
        self.blocked_steps_credit = 0

        self.route_xy: List[Tuple[float, float]] = []
        self.route_wps: List[carla.Waypoint] = []
        self.route_cumdist: List[float] = []
        self.goal_transform: Optional[carla.Transform] = None
        self.goal_spawn_transform: Optional[carla.Transform] = None
        self.current_goal_index: Optional[int] = None
        self.current_spawn_index: Optional[int] = None
        self.grp = None
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "scalars": spaces.Box(low=-10.0, high=10.0, shape=(cfg.scalar_dim,), dtype=np.float32),
                "edges": spaces.Box(low=-10.0, high=10.0, shape=(cfg.max_entities, cfg.edge_dim), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(cfg.max_entities,), dtype=np.float32),
            }
        )

        self._safe_tick(label="env_init")
        if self.cfg.debug_mode:
            print(
                f"[OK] Connected to CARLA | host={self.host} port={self.port} "
                f"town={self.town_name} sync=True dt={self.cfg.dt:.3f}s weather={self.weather_mode} "
                f"fixed_goal_index={self.fixed_goal_index}"
            )

    def _available_map_names(self) -> List[str]:
        try:
            return list(self.client.get_available_maps())
        except BaseException:
            return []

    @staticmethod
    def _basename_map(map_path: str) -> str:
        if not map_path:
            return map_path
        return map_path.split("/")[-1].split(".")[0]

    def _resolve_town_name(self, requested: str) -> str:
        available = self._available_map_names()
        available_basenames = {self._basename_map(x): x for x in available}
        preferred: List[str] = []
        if requested:
            preferred.append(requested)
            if requested.endswith("_Opt"):
                preferred.append(requested.replace("_Opt", ""))
            else:
                preferred.append(f"{requested}_Opt")
        preferred.extend(["Town02", "Town03", "Town10HD_Opt", "Town10HD"])
        for name in preferred:
            if name in available_basenames:
                return available_basenames[name]
        return available[0] if available else requested

    def _load_world_with_fallback(self, requested_town: str) -> carla.World:
        resolved = self._resolve_town_name(requested_town)
        # Reuse the already-loaded world when it matches the requested town.
        # client.load_world() performs a full map reload (a "total reset" of
        # the simulator); reusing the world means episode resets only
        # destroy/respawn the ego, NPC vehicles, pedestrians, and sensors.
        if bool(getattr(self.cfg, "reuse_loaded_world", True)):
            try:
                current = self.client.get_world()
                current_name = self._basename_map(current.get_map().name)
                if current_name == self._basename_map(resolved):
                    self.town_name = current_name
                    if self.cfg.debug_mode:
                        print(f"[OK] Reusing loaded CARLA world: {current_name} (no map reload)")
                    return current
            except BaseException:
                pass
        try:
            world = self.client.load_world(resolved, reset_settings=False)
            self.town_name = self._basename_map(resolved)
            return world
        except Exception as e:
            print(f"[WARN] Failed to load {resolved}: {e}")
            fallback = self._resolve_town_name("Town02")
            world = self.client.load_world(fallback, reset_settings=False)
            self.town_name = self._basename_map(fallback)
            return world

    def _apply_sync_settings(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.cfg.dt
        settings.no_rendering_mode = bool(self.cfg.no_rendering_mode)
        self.world.apply_settings(settings)

    def _restore_world_settings(self) -> None:
        try:
            self.tm.set_synchronous_mode(bool(self.original_settings.synchronous_mode))
        except BaseException:
            pass
        try:
            # Restore the exact settings that existed before this environment
            # changed them; forcing asynchronous mode corrupts nested/restarted
            # training environments that share a server.
            self.world.apply_settings(self.original_settings)
        except BaseException:
            pass

    def _make_spectator_transform(self, vehicle_tf: carla.Transform) -> carla.Transform:
        yaw = math.radians(vehicle_tf.rotation.yaw)
        cam_loc = carla.Location(
            x=vehicle_tf.location.x - self.cfg.spectator_distance_m * math.cos(yaw),
            y=vehicle_tf.location.y - self.cfg.spectator_distance_m * math.sin(yaw),
            z=vehicle_tf.location.z + self.cfg.spectator_height_m,
        )
        cam_rot = carla.Rotation(
            pitch=self.cfg.spectator_pitch_deg,
            yaw=vehicle_tf.rotation.yaw,
            roll=0.0,
        )
        return carla.Transform(cam_loc, cam_rot)

    def _snap_spectator_to_ego(self) -> None:
        if self.vehicle is None:
            return
        try:
            spectator = self.world.get_spectator()
            spectator.set_transform(self._make_spectator_transform(self.vehicle.get_transform()))
        except BaseException:
            pass

    def _update_spectator(self) -> None:
        if self.vehicle is None or not self.cfg.follow_ego_view:
            return
        self._snap_spectator_to_ego()

    def _safe_tick(self, label: str = "tick", raise_on_fail: bool = True) -> bool:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.tick_retry_count + 1):
            try:
                self.world.tick()
                return True
            except BaseException as e:
                last_err = Exception(str(e))
                if self.cfg.debug_mode:
                    print(f"[WARN] {label} failed attempt {attempt}/{self.cfg.tick_retry_count}: {e}")
                time.sleep(self.cfg.tick_retry_sleep_s)
        if raise_on_fail and last_err is not None:
            raise RuntimeError(f"CARLA world tick failed during {label}: {last_err}")
        return False

    def _destroy_actor(self, actor: carla.Actor) -> None:
        if actor is None:
            return
        try:
            if getattr(actor, "is_alive", False):
                actor.destroy()
        except BaseException:
            pass

    def _stop_sensor_only(self, actor: carla.Actor) -> None:
        if actor is None:
            return
        try:
            if getattr(actor, "is_alive", False) and bool(getattr(actor, "is_listening", True)):
                actor.stop()
        except BaseException:
            pass

    def _batch_destroy_actors(self, actors: List[carla.Actor]) -> None:
        """
        Safe destroy for synchronous mode:
        - detach TM first for vehicles
        - destroy one-by-one
        - tick between small groups
        """
        alive: List[carla.Actor] = [a for a in actors if self._actor_is_alive(a)]
        if not alive:
            return

        for actor in alive:
            try:
                if "vehicle." in getattr(actor, "type_id", ""):
                    self._safe_set_autopilot(actor, False)
            except BaseException:
                pass

        self._drain_world_ticks(1, "batch_detach_tm", sleep_s=0.02)
        # Destroy through server-side commands, filtered against a live actor
        # snapshot. Per-actor destroy() on a cached handle is what produced the
        # "unable to destroy actor: not found" lines during cleanup.
        self._batch_destroy_ids(alive, label="residual_actors")


    def _role_name_of_actor(self, actor: carla.Actor) -> str:
        try:
            return str(actor.attributes.get("role_name", "")).strip().lower()
        except BaseException:
            return ""

    def _spawn_transform_variants(self, tf: carla.Transform) -> List[carla.Transform]:
        base = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z),
            carla.Rotation(tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
        )
        dz = float(max(self.cfg.spawn_retry_lift_m, 0.05))
        variants = [base]
        for mul in (1.0, 2.0):
            variants.append(
                carla.Transform(
                    carla.Location(base.location.x, base.location.y, base.location.z + dz * mul),
                    carla.Rotation(base.rotation.pitch, base.rotation.yaw, base.rotation.roll),
                )
            )
        return variants

    def _destroy_residual_owned_actors(self) -> int:
        # Flush pending destroys so the actor snapshot below is not stale;
        # on the reused world a stale snapshot after _cleanup_episode_actors
        # produces "already dead" / "not found" server warnings.
        self._safe_tick(label='pre_residual_sweep', raise_on_fail=False)
        actors_to_destroy: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, 'filter'):
                iterable = list(actors.filter('vehicle.*'))
            else:
                iterable = [a for a in actors if 'vehicle.' in getattr(a, 'type_id', '')]
            for actor in iterable:
                if actor is None:
                    continue
                try:
                    if self.vehicle is not None and actor.id == self.vehicle.id:
                        continue
                except BaseException:
                    pass
                role = self._role_name_of_actor(actor)
                if role in {self.ego_role, self.npc_role}:
                    actors_to_destroy.append(actor)
            # Residual pedestrians and their AI controllers left on the reused
            # world by a previous episode or env instance.
            if hasattr(actors, 'filter'):
                controller_iter = list(actors.filter('controller.ai.walker'))
                walker_iter = list(actors.filter('walker.pedestrian.*'))
            else:
                controller_iter = [a for a in actors if 'controller.ai.walker' in getattr(a, 'type_id', '')]
                walker_iter = [a for a in actors if 'walker.pedestrian' in getattr(a, 'type_id', '')]
            for ctrl in controller_iter:
                if ctrl is None or not self._actor_is_alive(ctrl):
                    continue
                try:
                    parent = ctrl.parent
                except BaseException:
                    parent = None
                if parent is None or self._role_name_of_actor(parent) != self.walker_role:
                    continue
                try:
                    ctrl.stop()
                except BaseException:
                    pass
                actors_to_destroy.append(ctrl)
            for wk in walker_iter:
                if wk is None or not self._actor_is_alive(wk):
                    continue
                if self._role_name_of_actor(wk) == self.walker_role:
                    actors_to_destroy.append(wk)
        except BaseException:
            return 0
        if not actors_to_destroy:
            return 0
        self._batch_destroy_actors(actors_to_destroy)
        time.sleep(self.cfg.cleanup_sleep_s)
        self._safe_tick(label='destroy_residual_owned_actors', raise_on_fail=False)
        return len(actors_to_destroy)

    def _clear_spawn_blockers(self, tf: carla.Transform, radius: Optional[float] = None) -> int:
        radius = float(self.cfg.spawn_blocker_radius_m if radius is None else radius)
        actors_to_destroy: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, 'filter'):
                iterable = list(actors.filter('vehicle.*'))
            else:
                iterable = [a for a in actors if 'vehicle.' in getattr(a, 'type_id', '')]
            for actor in iterable:
                if actor is None:
                    continue
                try:
                    if self.vehicle is not None and actor.id == self.vehicle.id:
                        continue
                except BaseException:
                    pass
                try:
                    if distance2d(actor.get_location(), tf.location) <= radius:
                        actors_to_destroy.append(actor)
                except BaseException:
                    continue
        except BaseException:
            return 0
        if not actors_to_destroy:
            return 0
        self._batch_destroy_actors(actors_to_destroy)
        time.sleep(self.cfg.cleanup_sleep_s)
        self._safe_tick(label='clear_spawn_blockers', raise_on_fail=False)
        return len(actors_to_destroy)

    def _cleanup_episode_actors(self) -> None:
        """
        Robust synchronous cleanup.

        Order:
        1. Mark teardown active, stop sensors, detach TM.
        2. Drain queued callbacks.
        3. Destroy sensors.
        4. Destroy ego.
        5. Destroy NPCs after TM detach.
        6. Final settle ticks.
        """
        self._begin_episode_teardown(reason="cleanup")

        sensors = list(self.sensor_list)
        self.sensor_list.clear()

        ego = self.vehicle
        self.vehicle = None

        npcs = list(self.npcs)
        self.npcs.clear()

        walker_controllers = list(self.walker_controllers)
        self.walker_controllers.clear()
        walkers = list(self.walkers)
        self.walkers.clear()

        # Extra drain after stop() because goal often arrives at nonzero speed.
        self._drain_world_ticks(3, "cleanup_drain_pre", sleep_s=0.03)

        # Destroy sensors directly. The collision/lane-invasion callbacks are
        # already gated by _teardown_in_progress/_episode_live, and destroy()
        # unsubscribes the active stream itself; calling stop() first makes
        # the later destroy emit CARLA's "attempting to unsubscribe ...
        # wasn't listening" warning, so the explicit stop is skipped.
        self._batch_destroy_ids(sensors, label="sensors")

        self.collision_events.clear()
        self.lane_invasion_events.clear()
        self._drain_world_ticks(2, "cleanup_post_sensor_destroy", sleep_s=0.03)

        # Destroy ego next.
        if self._actor_is_alive(ego):
            self._safe_set_autopilot(ego, False)
            try:
                ego.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False,
                    )
                )
            except BaseException:
                pass

        self._batch_destroy_ids([ego], label="ego_vehicle")
        self._drain_world_ticks(2, "cleanup_post_ego", sleep_s=0.03)

        # Detach TM from NPCs before destroying them.
        alive_npcs = [npc for npc in npcs if self._actor_is_alive(npc)]
        for npc in alive_npcs:
            self._safe_set_autopilot(npc, False)

        self._drain_world_ticks(2, "cleanup_post_tm_detach", sleep_s=0.03)

        # Destroy NPC vehicles through the batch-command path so ids the
        # server has already collected are skipped without error output.
        self._batch_destroy_ids(alive_npcs, label="npc_vehicles")

        # Stop walker AI controllers, then destroy controllers and walkers
        # through the batch-command path. Ids the server has already collected
        # (e.g. a walker killed mid-episode, which auto-removes its attached
        # controller) are filtered out against a live snapshot first.
        live_ids = self._live_actor_ids()
        for ctrl in walker_controllers:
            try:
                if int(ctrl.id) not in live_ids:
                    continue
                ctrl.stop()
            except BaseException:
                pass
        self._drain_world_ticks(1, "cleanup_post_walker_stop", sleep_s=0.02)
        self._batch_destroy_ids(walker_controllers, label="walker_controllers")
        self._batch_destroy_ids(walkers, label="walkers")

        self._drain_world_ticks(3, "cleanup_final", sleep_s=0.03)

        try:
            time.sleep(max(self.cfg.cleanup_sleep_s, 0.05))
        except BaseException:
            pass

        self._terminal_reason = ""
        self._teardown_in_progress = False

    def _get_global_planner(self):
        if GlobalRoutePlanner is None:
            return None
        if self.grp is None:
            self.grp = GlobalRoutePlanner(self.map, self.cfg.route_point_spacing_m)
        return self.grp
        
    def _rebuild_local_route_from_current_pose(self) -> bool:
        if self.vehicle is None:
            return False
        cur_wp = self.map.get_waypoint(
            self.vehicle.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if cur_wp is None:
            return False
        self.route_progress_idx = 0
        target_with_margin = self.cfg.route_target_length_m + max(
            2.0 * self.cfg.route_point_spacing_m,
            self.cfg.route_target_tolerance_m,
        )
        route = self._extend_waypoint_chain(
            cur_wp, target_with_margin, ds=self.cfg.route_point_spacing_m
        )
        try:
            self._install_route(route, goal_index=None)
        except RuntimeError:
            return False
        return True

    def _route_length_from_wps(self, wps: Sequence[carla.Waypoint]) -> float:
        if len(wps) < 2:
            return 0.0
        s = 0.0
        for i in range(len(wps) - 1):
            s += distance2d(wps[i].transform.location, wps[i + 1].transform.location)
        return float(s)

    def _route_length_ok(self, route_len: float, strict: bool = False) -> bool:
        route_len = float(route_len)
        if strict:
            return abs(route_len - self.cfg.route_target_length_m) <= self.cfg.route_target_tolerance_m
        return self.cfg.route_soft_min_length_m <= route_len <= self.cfg.route_soft_max_length_m

    def _route_length_score(self, route_len: float) -> float:
        return -abs(float(route_len) - float(self.cfg.route_target_length_m))

    def _dedupe_waypoints(self, wps: Sequence[carla.Waypoint], min_sep: float = 0.75) -> List[carla.Waypoint]:
        out: List[carla.Waypoint] = []
        prev: Optional[carla.Waypoint] = None
        for wp in wps:
            if prev is None or distance2d(prev.transform.location, wp.transform.location) > min_sep:
                out.append(wp)
                prev = wp
        return out

    def _compute_route_cumdist(self) -> None:
        self.route_cumdist = []
        s = 0.0
        for i, wp in enumerate(self.route_wps):
            if i == 0:
                self.route_cumdist.append(0.0)
            else:
                s += distance2d(self.route_wps[i - 1].transform.location, wp.transform.location)
                self.route_cumdist.append(float(s))
        self.route_total_len_m = float(self.route_cumdist[-1]) if self.route_cumdist else 0.0

    def _route_projection_global(self, loc: carla.Location) -> Tuple[float, float, int]:
        if len(self.route_xy) < 2:
            return 0.0, 0.0, 0
        p = np.array([loc.x, loc.y], dtype=np.float32)
        best_d = 1e9
        best_s = 0.0
        best_idx = 0
        for i in range(len(self.route_xy) - 1):
            a = np.array(self.route_xy[i], dtype=np.float32)
            b = np.array(self.route_xy[i + 1], dtype=np.float32)
            _, d, local_s = point_segment_projection(p, a, b)
            if d < best_d:
                best_d = d
                best_s = self.route_cumdist[i] + local_s
                best_idx = i
        return float(best_s), float(best_d), int(best_idx)


    def _route_projection_start_window(self, loc: carla.Location, window_frac: float = 0.15) -> Tuple[float, float, int]:
        """Project onto the first window_frac of the route only.

        At episode start the ego is at the route origin by construction. A
        lane-follow long route can loop back near its own start (city blocks),
        so an unrestricted global projection may match a late segment and
        report a large spurious start progress. Restricting the search to the
        opening window makes the reset alignment check reflect the ego's
        actual position on the route.
        """
        if len(self.route_xy) < 2:
            return 0.0, 0.0, 0
        n_seg = len(self.route_xy) - 1
        hi = int(np.clip(int(math.ceil(window_frac * n_seg)), 1, n_seg))
        p = np.array([loc.x, loc.y], dtype=np.float32)
        best_d = 1e9
        best_s = 0.0
        best_idx = 0
        for i in range(hi):
            a = np.array(self.route_xy[i], dtype=np.float32)
            b = np.array(self.route_xy[i + 1], dtype=np.float32)
            _, d, local_s = point_segment_projection(p, a, b)
            if d < best_d:
                best_d = d
                best_s = self.route_cumdist[i] + local_s
                best_idx = i
        return float(best_s), float(best_d), int(best_idx)

    def _route_projection_monotonic(self, loc: carla.Location) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        """Project the ego onto the route.

        Returns (arc_s, cross_track, segment_index, projection_point, tangent).

        cross_track is pure geometry: the distance to the nearest admissible
        segment in the local window. It is never rate limited, because it is a
        distance rather than a rate; limiting it was what froze the projection
        and produced phantom off-route terminations.

        arc_s is monotonic and rate limited to what the vehicle can physically
        cover in one control step, so a route that loops back near itself cannot
        teleport progress. When the geometric match lies beyond the budget, arc_s
        slews toward it rather than rejecting it, so it always catches up.
        """
        if len(self.route_xy) < 2:
            return 0.0, 0.0, 0, np.zeros(2, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)
        p = np.array([loc.x, loc.y], dtype=np.float32)
        start_idx = max(0, self.route_progress_idx - self.cfg.route_search_back)
        end_idx = min(len(self.route_xy) - 2, self.route_progress_idx + self.cfg.route_search_ahead)
        ego_forward = np.array([1.0, 0.0], dtype=np.float32)
        speed_ms = 0.0
        if self.vehicle is not None:
            tf = self.vehicle.get_transform()
            yaw = math.radians(tf.rotation.yaw)
            ego_forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            speed_ms = vec3_length(self.vehicle.get_velocity())
        prev_s = float(self.prev_route_s if self.prev_route_s is not None else 0.0)

        best_score = 1e18
        best_d = 1e9
        cand_s_best = prev_s
        best_idx = start_idx
        best_proj = np.zeros(2, dtype=np.float32)
        best_dir = np.array([1.0, 0.0], dtype=np.float32)
        fallback_d = 1e9
        fallback = None

        for i in range(start_idx, end_idx + 1):
            a = np.array(self.route_xy[i], dtype=np.float32)
            b = np.array(self.route_xy[i + 1], dtype=np.float32)
            proj, d, local_s = point_segment_projection(p, a, b)
            seg = b - a
            seg_norm = float(np.linalg.norm(seg))
            if seg_norm < 1e-6:
                continue
            seg_dir = seg / seg_norm
            cand_s = float(self.route_cumdist[i] + local_s)
            # Nearest segment regardless of heading, used if the alignment gate
            # rejects everything (e.g. mid-recovery while pointing off-route).
            if d < fallback_d:
                fallback_d = d
                fallback = (d, cand_s, i, proj, seg_dir)
            align = float(np.dot(ego_forward, seg_dir))
            if align < -0.35 and d > 0.75:
                continue
            # Prefer close, forward-aligned segments. A mild penalty on large
            # forward offsets discourages latching onto a later pass of the same
            # street, without ever rejecting a candidate outright.
            score = float(d + 1.5 * max(0.0, 1.0 - align) + 0.02 * max(cand_s - prev_s, 0.0))
            if score < best_score:
                best_score = score
                best_d = d
                cand_s_best = cand_s
                best_idx = i
                best_proj = proj
                best_dir = seg_dir

        if best_score >= 1e17:
            if fallback is None:
                return prev_s, 0.0, self.route_progress_idx, p, ego_forward
            best_d, cand_s_best, best_idx, best_proj, best_dir = fallback

        # Rate limit the arc coordinate only. Forward budget is twice the
        # distance covered in one step plus slack; a small backward allowance
        # absorbs projection noise without letting progress unwind.
        fwd_budget = max(0.35, 2.0 * float(speed_ms) * float(self.cfg.dt) + 0.30)
        back_budget = 0.50
        best_s = float(np.clip(cand_s_best, prev_s - back_budget, prev_s + fwd_budget))
        best_s = float(np.clip(best_s, 0.0, max(self.route_total_len_m, 1e-6)))

        # Anchor the search window to the accepted arc position, not to the raw
        # geometric index, so a distant false match cannot drag the window along.
        idx_for_s = int(np.clip(np.searchsorted(self.route_cumdist, best_s, side="right") - 1,
                                0, len(self.route_xy) - 2))
        self.route_progress_idx = max(self.route_progress_idx, idx_for_s)
        return float(best_s), float(best_d), int(best_idx), best_proj, best_dir

    def _route_projection(self, loc: carla.Location) -> Tuple[float, float]:
        s_arc, dL, _, _, _ = self._route_projection_monotonic(loc)
        return float(s_arc), float(dL)

    def _route_reference_state(self, loc: carla.Location, lookahead_m: float) -> Dict[str, object]:
        s_arc, dL, _, _, _ = self._route_projection_monotonic(loc)
        remaining_s = max(0.0, self.route_total_len_m - s_arc)

        if len(self.route_wps) == 0:
            return {
                "s_arc": 0.0,
                "dL": 0.0,
                "remaining_s": 0.0,
                "ref_idx": 0,
                "ref_wp": None,
                "signed_lane_err": 0.0,
                "heading_err": 0.0,
                "lane_width": 3.5,
                "wrong_lane": False,
                "opposite_lane": False,
            }

        ref_s = min(s_arc + max(2.0, 0.5 * lookahead_m), self.route_total_len_m)
        ref_idx = int(np.searchsorted(np.asarray(self.route_cumdist, dtype=np.float32), ref_s, side="left"))
        ref_idx = int(np.clip(ref_idx, 0, len(self.route_wps) - 1))
        ref_wp = self.route_wps[ref_idx]
        ref_tf = ref_wp.transform
        ref_loc = ref_tf.location

        route_yaw = math.radians(ref_tf.rotation.yaw)
        ego_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        route_right = np.array([-math.sin(route_yaw), math.cos(route_yaw)], dtype=np.float32)
        rel = np.array([loc.x - ref_loc.x, loc.y - ref_loc.y], dtype=np.float32)
        signed_lane_err = float(np.dot(rel, route_right))
        heading_err = wrap_pi(route_yaw - ego_yaw)
        lane_width = max(float(getattr(ref_wp, "lane_width", 3.5)), 3.5)

        ego_wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        wrong_lane = False
        opposite_lane = False
        if ego_wp is not None and not ref_wp.is_junction:
            same_road = ego_wp.road_id == ref_wp.road_id and ego_wp.section_id == ref_wp.section_id
            same_lane = same_road and (ego_wp.lane_id == ref_wp.lane_id)
            if same_road and ego_wp.lane_id != 0 and ref_wp.lane_id != 0:
                opposite_lane = np.sign(ego_wp.lane_id) != np.sign(ref_wp.lane_id)
            wrong_lane = opposite_lane or ((not same_lane) and (abs(signed_lane_err) > 0.55 * lane_width))

        return {
            "s_arc": float(s_arc),
            "dL": float(dL),
            "remaining_s": float(remaining_s),
            "ref_idx": int(ref_idx),
            "ref_wp": ref_wp,
            "signed_lane_err": float(signed_lane_err),
            "heading_err": float(heading_err),
            "lane_width": float(lane_width),
            "wrong_lane": bool(wrong_lane),
            "opposite_lane": bool(opposite_lane),
        }

    def _waypoint_visit_key(self, wp: carla.Waypoint) -> Tuple[int, int, int, int]:
        return (
            int(getattr(wp, "road_id", 0)),
            int(getattr(wp, "section_id", 0)),
            int(getattr(wp, "lane_id", 0)),
            int(round(float(getattr(wp, "s", 0.0)) / max(self.cfg.route_point_spacing_m, 1.0))),
        )

    def _select_route_successor(
        self,
        cur: carla.Waypoint,
        candidates: Sequence[carla.Waypoint],
        visit_count: Dict[Tuple[int, int, int, int], int],
        recent: Sequence[carla.Waypoint],
    ) -> Optional[carla.Waypoint]:
        """Choose a continuous, mostly forward, non-repeating lane successor.

        CARLA returns multiple successors at junctions. Selecting only the same
        road/lane can create short loops and ambiguous route projection. This
        score prefers forward continuity while strongly discouraging revisits.
        """
        if not candidates:
            return None
        cur_yaw = math.radians(cur.transform.rotation.yaw)
        cur_dir = np.array([math.cos(cur_yaw), math.sin(cur_yaw)], dtype=np.float32)
        recent_locs = [wp.transform.location for wp in recent[-30:]]
        scored: List[Tuple[float, carla.Waypoint]] = []
        for cand in candidates:
            loc0 = cur.transform.location
            loc1 = cand.transform.location
            delta = np.array([loc1.x - loc0.x, loc1.y - loc0.y], dtype=np.float32)
            d = float(np.linalg.norm(delta))
            if d < 0.20:
                continue
            tangent = delta / max(d, 1e-6)
            forward = float(np.dot(cur_dir, tangent))
            key = self._waypoint_visit_key(cand)
            revisits = int(visit_count.get(key, 0))
            near_recent = 0
            for rloc in recent_locs[:-3]:
                if distance2d(loc1, rloc) < 3.0:
                    near_recent += 1
            score = 8.0 * forward
            score += 1.5 if cand.lane_id == cur.lane_id else 0.0
            score += 0.5 if cand.road_id == cur.road_id else 0.0
            score -= 12.0 * revisits
            score -= 4.0 * near_recent
            # Seeded jitter prevents every training episode from taking the
            # identical junction branch while preserving deterministic seeds.
            score += random.uniform(-0.15, 0.15)
            scored.append((score, cand))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _extend_waypoint_chain(
        self,
        start_wp: carla.Waypoint,
        target_length_m: float,
        ds: Optional[float] = None,
        prefix: Optional[Sequence[carla.Waypoint]] = None,
    ) -> List[carla.Waypoint]:
        ds = float(self.cfg.route_point_spacing_m if ds is None else ds)
        chain = list(prefix or [])
        if not chain:
            chain = [start_wp]
        cur = chain[-1]
        visit_count: Dict[Tuple[int, int, int, int], int] = {}
        for wp in chain:
            key = self._waypoint_visit_key(wp)
            visit_count[key] = visit_count.get(key, 0) + 1
        length = self._route_length_from_wps(chain)
        max_points = int(max(100, math.ceil((target_length_m + 80.0) / max(ds, 0.5))))
        while length < target_length_m and len(chain) < max_points:
            try:
                candidates = list(cur.next(ds))
            except BaseException:
                candidates = []
            nxt = self._select_route_successor(cur, candidates, visit_count, chain)
            if nxt is None:
                break
            seg = distance2d(cur.transform.location, nxt.transform.location)
            if seg < 0.20:
                break
            chain.append(nxt)
            cur = nxt
            length += seg
            key = self._waypoint_visit_key(nxt)
            visit_count[key] = visit_count.get(key, 0) + 1
        return self._dedupe_waypoints(chain, min_sep=0.50)

    def _trim_route_to_target(
        self,
        wps: Sequence[carla.Waypoint],
        target_m: Optional[float] = None,
    ) -> List[carla.Waypoint]:
        """Trim a continuous waypoint route to the waypoint nearest target_m."""
        route = self._dedupe_waypoints(wps, min_sep=0.50)
        if len(route) < 2:
            return route
        target = float(self.cfg.route_target_length_m if target_m is None else target_m)
        cum = [0.0]
        for i in range(1, len(route)):
            cum.append(cum[-1] + distance2d(route[i - 1].transform.location, route[i].transform.location))
        idx = int(np.argmin(np.abs(np.asarray(cum, dtype=np.float32) - target)))
        idx = max(1, idx)
        return route[: idx + 1]

    def _install_route(self, wps: Sequence[carla.Waypoint], goal_index: Optional[int] = None) -> None:
        route = self._trim_route_to_target(wps)
        if len(route) < 2:
            raise RuntimeError("Route contains fewer than two waypoints.")
        route_len = self._route_length_from_wps(route)
        if not self._route_length_ok(route_len, strict=True):
            raise RuntimeError(
                f"Could not construct a {self.cfg.route_target_length_m:.0f} m route: "
                f"obtained {route_len:.1f} m (allowed ±{self.cfg.route_target_tolerance_m:.1f} m)."
            )
        self.route_wps = list(route)
        self.route_xy = [(float(wp.transform.location.x), float(wp.transform.location.y)) for wp in route]
        self._compute_route_cumdist()
        goal_wp = route[-1]
        goal_loc = goal_wp.transform.location
        self.goal_transform = carla.Transform(
            carla.Location(goal_loc.x, goal_loc.y, goal_loc.z + 0.5),
            goal_wp.transform.rotation,
        )
        self.current_goal_index = goal_index
        self.goal_spawn_transform = self.spawn_points[goal_index] if goal_index is not None else None

    def _build_route_polyline_local(self, start_wp: carla.Waypoint, ds: float = 2.0) -> carla.Waypoint:
        target_with_margin = self.cfg.route_target_length_m + max(2.0 * ds, self.cfg.route_target_tolerance_m)
        route = self._extend_waypoint_chain(start_wp, target_with_margin, ds=ds)
        self.route_wps = self._trim_route_to_target(route)
        self.route_xy = [(float(wp.transform.location.x), float(wp.transform.location.y)) for wp in self.route_wps]
        return self.route_wps[-1] if self.route_wps else start_wp

    def _lane_follow_extension(self, start_wp: carla.Waypoint, extra_m: float, ds: float = 2.0) -> List[carla.Waypoint]:
        if extra_m <= 0.0:
            return []
        route = self._extend_waypoint_chain(start_wp, extra_m, ds=ds)
        return route[1:] if len(route) > 1 else []

    def _project_drive_wp(self, loc: carla.Location) -> Optional[carla.Waypoint]:
        try:
            return self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        except BaseException:
            return None

    def _snap_spawn_to_driving_lane(self, tf: carla.Transform) -> carla.Transform:
        wp = self._project_drive_wp(tf.location)
        if wp is None:
            return carla.Transform(
                carla.Location(tf.location.x, tf.location.y, tf.location.z),
                carla.Rotation(tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
            )
        lane_tf = wp.transform
        z = max(float(tf.location.z), float(lane_tf.location.z)) + max(0.15, float(self.cfg.spawn_retry_lift_m))
        return carla.Transform(
            carla.Location(float(lane_tf.location.x), float(lane_tf.location.y), z),
            carla.Rotation(float(lane_tf.rotation.pitch), float(lane_tf.rotation.yaw), float(lane_tf.rotation.roll)),
        )

    def _build_route_from_goal_index(self, start_tf: carla.Transform, start_wp: carla.Waypoint, goal_index: int) -> bool:
        grp = self._get_global_planner()
        if grp is None or goal_index < 0 or goal_index >= len(self.spawn_points):
            return False
        start_wp_proj = self._project_drive_wp(start_tf.location) or start_wp
        goal_wp_proj = self._project_drive_wp(self.spawn_points[goal_index].location)
        if goal_wp_proj is None:
            return False
        try:
            trace = grp.trace_route(
                start_wp_proj.transform.location,
                goal_wp_proj.transform.location,
            )
        except BaseException:
            return False
        route = self._dedupe_waypoints([wp for wp, _ in trace if wp is not None], min_sep=0.50)
        route_len = self._route_length_from_wps(route)
        # A fixed destination remains a fixed destination: do not truncate or
        # extend its route. It is accepted only when already close to 500 m.
        if len(route) < 2 or not self._route_length_ok(route_len, strict=True):
            return False
        self._install_route(route, goal_index=goal_index)
        return True

    def _find_best_auto_route_from_spawn(
        self,
        start_tf: carla.Transform,
        start_wp: carla.Waypoint,
    ) -> Tuple[List[carla.Waypoint], Optional[int], float]:
        start_loc = start_wp.transform.location
        start_yaw = math.radians(start_tf.rotation.yaw)
        start_dir = np.array([math.cos(start_yaw), math.sin(start_yaw)], dtype=np.float32)
        grp = self._get_global_planner()
        if grp is None:
            return [], None, 0.0

        candidate_ids = list(range(len(self.spawn_points)))
        random.shuffle(candidate_ids)
        max_tries = min(len(candidate_ids), max(self.cfg.candidate_goal_max_tries, len(candidate_ids)))

        best_soft_route: List[carla.Waypoint] = []
        best_soft_goal_idx: Optional[int] = None
        best_soft_score = -1e18

        best_relaxed_route: List[carla.Waypoint] = []
        best_relaxed_goal_idx: Optional[int] = None
        best_relaxed_score = -1e18

        best_any_route: List[carla.Waypoint] = []
        best_any_goal_idx: Optional[int] = None
        best_any_score = -1e18

        # Why candidates get rejected — reported once per reset in debug mode
        # so a persistent lane-follow fallback is diagnosable instead of silent.
        rej = {"no_wp": 0, "euclid": 0, "trace_fail": 0, "trace_short": 0,
               "too_short": 0, "detour": 0, "align": 0, "kept": 0}

        for idx in candidate_ids[:max_tries]:
            if self.current_spawn_index is not None and idx == int(self.current_spawn_index):
                continue
            goal_sp = self.spawn_points[idx]
            goal_wp_proj = self._project_drive_wp(goal_sp.location)
            if goal_wp_proj is None:
                rej["no_wp"] += 1
                continue
            goal_loc_req = goal_wp_proj.transform.location
            euclid = distance2d(start_loc, goal_loc_req)
            # Euclidean prefilter only. Town10HD spans roughly 200 x 210 m, so
            # no pair of spawn points is 500 m apart in a straight line and the
            # planner's shortest paths top out near the map diagonal. The floor
            # is therefore a small fraction of the arc target; the route is
            # brought up to length by lane-follow extension in _build_route.
            if euclid < min(self.cfg.candidate_goal_min_dist_m, 0.12 * self.cfg.route_target_length_m):
                rej["euclid"] += 1
                continue
            try:
                trace = grp.trace_route(start_loc, goal_loc_req)
            except BaseException:
                rej["trace_fail"] += 1
                continue
            if len(trace) < 2:
                rej["trace_short"] += 1
                continue
            route = [wp for wp, _ in trace if wp is not None]
            route = self._dedupe_waypoints(route, min_sep=0.75)
            if len(route) < 2:
                rej["trace_short"] += 1
                continue
            route_len = self._route_length_from_wps(route)
            # Planner shortest paths in small towns are often shorter than the
            # 500 m target. Keep them as "any"-tier candidates; _build_route
            # extends the winner with lane-follow continuation to target length.
            if route_len < 0.55 * self.cfg.min_route_length_m:
                rej["too_short"] += 1
                continue

            detour_ratio = route_len / max(euclid, 1e-3)
            # Long fixed-length routes legitimately wind (arc >> Euclid). In a
            # compact town a 500 m route over a 100 m displacement is ratio 5,
            # so this cap must stay generous.
            if detour_ratio > 7.0:
                rej["detour"] += 1
                continue

            k = min(3, len(route) - 1)
            p0 = np.array([route[0].transform.location.x, route[0].transform.location.y], dtype=np.float32)
            pk = np.array([route[k].transform.location.x, route[k].transform.location.y], dtype=np.float32)
            tangent = pk - p0
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-6:
                continue
            tangent = tangent / tangent_norm
            align = float(np.dot(start_dir, tangent))
            if align < -0.35:
                rej["align"] += 1
                continue
            rej["kept"] += 1

            # Planner routes are extended to the target length afterwards, so
            # score for shape (forward alignment, useful length, sane detour)
            # and only mildly penalise falling short of the target.
            len_error = max(0.0, self.cfg.route_target_length_m - route_len)
            score = (
                16.0 * align
                + 0.10 * route_len
                - 0.02 * len_error
                - 3.0 * max(detour_ratio - 3.0, 0.0)
            )
            any_score = 12.0 * align + 0.15 * route_len + 1.20 * euclid - 4.0 * max(detour_ratio - 2.8, 0.0)

            if self._route_length_ok(route_len, strict=False):
                if score > best_soft_score:
                    best_soft_score = score
                    best_soft_route = route
                    best_soft_goal_idx = idx
            elif route_len >= self.cfg.min_route_length_m and route_len <= max(
                self.cfg.route_target_length_m + max(180.0, 0.35 * self.cfg.route_target_length_m),
                self.cfg.route_soft_max_length_m + 120.0,
            ):
                relaxed_score = score - 0.010 * max(0.0, len_error - self.cfg.route_target_tolerance_m)
                if relaxed_score > best_relaxed_score:
                    best_relaxed_score = relaxed_score
                    best_relaxed_route = route
                    best_relaxed_goal_idx = idx

            if any_score > best_any_score:
                best_any_score = any_score
                best_any_route = route
                best_any_goal_idx = idx

        if self.cfg.debug_mode and rej["kept"] == 0:
            print(
                f"[INFO] auto-route: no candidate kept from spawn {self.current_spawn_index} "
                f"(tried={min(max_tries, len(candidate_ids))} no_wp={rej['no_wp']} euclid={rej['euclid']} "
                f"trace_fail={rej['trace_fail']} trace_short={rej['trace_short']} "
                f"too_short={rej['too_short']} detour={rej['detour']} align={rej['align']})"
            )

        if len(best_soft_route) >= 2:
            return best_soft_route, best_soft_goal_idx, self._route_length_from_wps(best_soft_route)
        if len(best_relaxed_route) >= 2:
            return best_relaxed_route, best_relaxed_goal_idx, self._route_length_from_wps(best_relaxed_route)
        if len(best_any_route) >= 2:
            return best_any_route, best_any_goal_idx, self._route_length_from_wps(best_any_route)
        return [], None, 0.0

    def _build_route(self, start_tf: carla.Transform, start_wp: carla.Waypoint, goal_index: Optional[int] = None) -> None:
        self.route_xy = []
        self.route_wps = []
        self.route_cumdist = []
        self.goal_transform = None
        self.goal_spawn_transform = None
        self.current_goal_index = None
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        chosen_goal = goal_index
        if chosen_goal is None and self.cfg.use_fixed_destination:
            chosen_goal = self.fixed_goal_index
        if chosen_goal is not None and int(chosen_goal) < 0:
            chosen_goal = None

        if chosen_goal is not None:
            if self._build_route_from_goal_index(start_tf, start_wp, int(chosen_goal)):
                return
            if self.cfg.strict_goal_route and not self.cfg.allow_fallback_route:
                raise RuntimeError(
                    f"Goal index {chosen_goal} does not define a route within "
                    f"{self.cfg.route_target_length_m:.0f}±{self.cfg.route_target_tolerance_m:.0f} m. "
                    "Use goal index -1 for automatic 500 m route generation."
                )

        # Prefer a planner route with useful global structure, then continue
        # along connected lane-center waypoints until the route exceeds 500 m.
        best_route, _, best_len = self._find_best_auto_route_from_spawn(start_tf, start_wp)
        target_with_margin = self.cfg.route_target_length_m + max(
            2.0 * self.cfg.route_point_spacing_m,
            self.cfg.route_target_tolerance_m,
        )
        if len(best_route) >= 2:
            route = list(best_route)
            if best_len < target_with_margin:
                route = self._extend_waypoint_chain(
                    route[-1], target_with_margin, ds=self.cfg.route_point_spacing_m,
                    prefix=route,
                )
            try:
                self._install_route(route, goal_index=None)
                return
            except RuntimeError as exc:
                if self.cfg.debug_mode:
                    print(f"[WARN] planner-based 500 m route rejected: {exc}")

        # Guaranteed fallback: construct a continuous lane-follow route from
        # the actual ego lane and trim it to the waypoint nearest 500 m.
        local_route = self._extend_waypoint_chain(
            start_wp, target_with_margin, ds=self.cfg.route_point_spacing_m
        )
        self._install_route(local_route, goal_index=None)
        if self.cfg.debug_mode:
            print(
                f"[INFO] installed automatic route: {self.route_total_len_m:.1f} m "
                f"from spawn {self.current_spawn_index} in {self.town_name}"
            )

    def _route_completion_pct(self, s_arc: Optional[float] = None) -> float:
        if self.route_total_len_m <= 1e-6:
            return 0.0
        if s_arc is None and self.vehicle is not None:
            s_arc, _ = self._route_projection(self.vehicle.get_location())
        s_arc = 0.0 if s_arc is None else float(s_arc)
        return float(np.clip(100.0 * s_arc / max(self.route_total_len_m, 1e-6), 0.0, 100.0))

    def _curvature_ahead(self, wp: Optional[carla.Waypoint], ds: float = 4.0) -> float:
        if wp is None:
            return 0.0
        nxt = wp.next(ds)
        if not nxt:
            return 0.0
        nxt2 = nxt[0].next(ds)
        if not nxt2:
            return 0.0
        p0 = np.array([wp.transform.location.x, wp.transform.location.y], dtype=np.float32)
        p1 = np.array([nxt[0].transform.location.x, nxt[0].transform.location.y], dtype=np.float32)
        p2 = np.array([nxt2[0].transform.location.x, nxt2[0].transform.location.y], dtype=np.float32)
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        if float(a * b * c) < 1e-6:
            return 0.0
        area2 = abs(np.cross(p1 - p0, p2 - p0))
        return float(2.0 * area2 / (a * b * c))

    def _lane_heading_offset(self, wp_onlane: Optional[carla.Waypoint]) -> float:
        if wp_onlane is None or self.vehicle is None:
            return 0.0
        yaw_lane = math.radians(wp_onlane.transform.rotation.yaw)
        yaw_veh = math.radians(self.vehicle.get_transform().rotation.yaw)
        return float(np.clip(wrap_pi(yaw_veh - yaw_lane) / (math.pi / 2.0), -1.0, 1.0))

    @staticmethod
    def _tl_state_name(state: object) -> str:
        try:
            return str(state).split(".")[-1]
        except BaseException:
            return "Unknown"

    def _stop_line_longitudinal(self, light: carla.Actor, loc: carla.Location,
                                lane_fwd: np.ndarray) -> float:
        """Signed along-lane distance from the ego to a light's stop line.

        Positive means the stop line is ahead, negative means the ego has already
        crossed it. Returns 1e9 when no stop waypoint of this light belongs to
        the ego's travel direction.
        """
        best = 1e9
        try:
            stop_wps = light.get_stop_waypoints() or []
        except BaseException:
            return 1e9
        for swp in stop_wps:
            try:
                vec = swp.transform.location - loc
                vec2 = np.array([vec.x, vec.y], dtype=np.float32)
                forward = float(np.dot(vec2, lane_fwd))
                lateral = abs(float(np.linalg.norm(vec2 - forward * lane_fwd)))
                # Same travel direction as the ego lane, and laterally on our lane.
                yaw_s = math.radians(swp.transform.rotation.yaw)
                s_fwd = np.array([math.cos(yaw_s), math.sin(yaw_s)], dtype=np.float32)
                if float(np.dot(s_fwd, lane_fwd)) < 0.60:
                    continue
                if lateral > 3.0:
                    continue
                if forward < -float(self.cfg.tl_cross_track_m):
                    continue
                if abs(forward) < abs(best):
                    best = forward
            except BaseException:
                continue
        return float(best)

    def _get_active_traffic_light_info(self) -> Tuple[str, float]:
        """Return (state, signed along-lane distance to the stop line).

        The distance is negative for a few metres after the ego crosses the line
        so that running a red light remains observable to the reward. 1e9 means
        no relevant light.
        """
        if self.vehicle is None:
            return "None", 1e9
        loc = self.vehicle.get_location()
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_onlane is None:
            return "None", 1e9
        yaw_lane = math.radians(wp_onlane.transform.rotation.yaw)
        lane_fwd = np.array([math.cos(yaw_lane), math.sin(yaw_lane)], dtype=np.float32)

        candidates: List[carla.Actor] = []
        ego_light = None
        try:
            ego_light = self.vehicle.get_traffic_light()
        except BaseException:
            ego_light = None
        if ego_light is not None:
            candidates.append(ego_light)
        # Look further ahead than the trigger volume so braking can begin in time.
        try:
            actors = self.world.get_actors()
            iterable = actors.filter("traffic.traffic_light*") if hasattr(actors, "filter") else                 [a for a in actors if "traffic_light" in getattr(a, "type_id", "")]
            for tl in iterable:
                if tl is None or (ego_light is not None and int(tl.id) == int(ego_light.id)):
                    continue
                if distance2d(tl.get_location(), loc) > float(self.cfg.tl_detect_dist) + 25.0:
                    continue
                candidates.append(tl)
        except BaseException:
            pass

        best_state = "None"
        best_dist = 1e9
        for tl in candidates:
            d = self._stop_line_longitudinal(tl, loc, lane_fwd)
            if d > 1e8 or d > float(self.cfg.tl_detect_dist):
                continue
            if abs(d) < abs(best_dist):
                try:
                    best_state = self._tl_state_name(tl.get_state())
                except BaseException:
                    best_state = "Unknown"
                best_dist = d
        if best_dist > 1e8:
            return "None", 1e9
        return best_state, float(best_dist)

    def _red_brake_distance(self) -> float:
        """Distance at which a red light must already be influencing control."""
        v = 0.0
        if self.vehicle is not None:
            try:
                v = float(vec3_length(self.vehicle.get_velocity()))
            except BaseException:
                v = 0.0
        a = max(float(self.cfg.tl_brake_decel), 0.5)
        return float(v * v / (2.0 * a) + v * float(self.cfg.tl_reaction_s) + float(self.cfg.tl_stop_margin))

    def _world_actors(self) -> Iterable[carla.Actor]:
        try:
            return self.world.get_actors()
        except BaseException:
            return []

    def _nearby_walkers(self, ego_loc: carla.Location) -> List[carla.Actor]:
        walkers: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, "filter"):
                iterable = actors.filter("walker.pedestrian.*")
            else:
                iterable = [a for a in actors if "walker.pedestrian" in getattr(a, "type_id", "")]
            for actor in iterable:
                if distance2d(actor.get_location(), ego_loc) <= self.cfg.entity_max_dist:
                    walkers.append(actor)
        except Exception:
            pass
        return walkers

    def _get_front_vehicle_info(self, max_dist: Optional[float] = None) -> Tuple[Optional[float], float, Optional[carla.Vehicle]]:
        if self.vehicle is None:
            return None, 0.0, None

        if max_dist is None:
            max_dist = self.cfg.front_vehicle_max_dist

        ego_loc = self.vehicle.get_location()
        ego_speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        lookahead_m = max(self.cfg.lookahead_base_m + 0.25 * max(ego_speed_kmh, 0.0), 5.0)

        rs = self._route_reference_state(ego_loc, lookahead_m=lookahead_m)
        ref_wp = rs.get("ref_wp", None)
        if ref_wp is None:
            return None, 0.0, None

        ref_loc = ref_wp.transform.location
        route_yaw = math.radians(ref_wp.transform.rotation.yaw)
        route_fwd = np.array([math.cos(route_yaw), math.sin(route_yaw)], dtype=np.float32)
        route_right = np.array([-math.sin(route_yaw), math.cos(route_yaw)], dtype=np.float32)
        lane_width = max(float(getattr(ref_wp, "lane_width", 3.5)), 3.5)

        best_metric = 1e9
        best_dist: Optional[float] = None
        best_speed_kmh = 0.0
        best_actor: Optional[carla.Vehicle] = None

        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            try:
                nloc = npc.get_location()
                rel_ego = np.array([nloc.x - ego_loc.x, nloc.y - ego_loc.y], dtype=np.float32)
                rel_ref = np.array([nloc.x - ref_loc.x, nloc.y - ref_loc.y], dtype=np.float32)
                forward_dist = float(np.dot(rel_ego, route_fwd))
                lateral_dist = abs(float(np.dot(rel_ref, route_right)))

                if forward_dist <= 0.5 or forward_dist > float(max_dist):
                    continue
                if lateral_dist > max(1.40, 0.45 * lane_width):
                    continue

                nwp = self.map.get_waypoint(nloc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if nwp is None:
                    continue
                npc_yaw = math.radians(nwp.transform.rotation.yaw)
                yaw_err = abs(wrap_pi(npc_yaw - route_yaw))
                same_road_family = (nwp.road_id == ref_wp.road_id) or bool(ref_wp.is_junction) or bool(nwp.is_junction)
                same_direction = yaw_err < 0.60
                if not same_road_family or not same_direction:
                    continue

                metric = forward_dist + 1.75 * lateral_dist
                if metric < best_metric:
                    best_metric = metric
                    best_dist = forward_dist
                    best_actor = npc
                    best_speed_kmh = 3.6 * vec3_length(npc.get_velocity())
            except Exception:
                continue

        # Pedestrians ahead in the ego corridor block progress exactly like a
        # lead vehicle. They are matched on corridor geometry only (no lane /
        # heading test) because walkers cross rather than follow the lane.
        for walker in self._nearby_walkers(ego_loc):
            if walker is None:
                continue
            try:
                wloc = walker.get_location()
                rel_ego = np.array([wloc.x - ego_loc.x, wloc.y - ego_loc.y], dtype=np.float32)
                rel_ref = np.array([wloc.x - ref_loc.x, wloc.y - ref_loc.y], dtype=np.float32)
                forward_dist = float(np.dot(rel_ego, route_fwd))
                lateral_dist = abs(float(np.dot(rel_ref, route_right)))
                if forward_dist <= 0.5 or forward_dist > float(max_dist):
                    continue
                if lateral_dist > max(1.20, 0.40 * lane_width):
                    continue
                metric = forward_dist + 1.75 * lateral_dist
                if metric < best_metric:
                    best_metric = metric
                    best_dist = forward_dist
                    best_actor = walker
                    best_speed_kmh = 3.6 * vec3_length(walker.get_velocity())
            except Exception:
                continue

        return best_dist, float(best_speed_kmh), best_actor

    def _get_min_vehicle_ttc(self, max_dist: float = 25.0) -> Tuple[float, float]:
        if self.vehicle is None:
            return 999.0, 1e9
        ego_loc = self.vehicle.get_location()
        ego_vel = self.vehicle.get_velocity()
        ego_v = np.array([ego_vel.x, ego_vel.y], dtype=np.float32)
        min_ttc = 999.0
        min_dist = 1e9
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            try:
                nloc = npc.get_location()
                dist = distance2d(ego_loc, nloc)
                if dist < 0.5 or dist > max_dist:
                    continue
                min_dist = min(min_dist, dist)
                nvel = npc.get_velocity()
                rel_p = np.array([nloc.x - ego_loc.x, nloc.y - ego_loc.y], dtype=np.float32)
                rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_v
                closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
                if closing > 1e-3:
                    min_ttc = min(min_ttc, float(dist / closing))
            except Exception:
                continue
        return float(min_ttc), float(min_dist)

    def _observation_dropout(self, dist: float, fog_norm: float, kind: str = "vehicle") -> bool:
        dist_n = float(np.clip(dist / max(self.cfg.entity_max_dist, 1e-3), 0.0, 1.0))
        miss = self.cfg.obs_miss_base + 0.22 * dist_n + 0.22 * fog_norm
        if kind == "walker":
            miss += 0.08
        if kind == "traffic_light":
            miss += 0.05
        miss = float(np.clip(miss, 0.02, 0.65))
        return random.random() < miss

    def _noisy_relative_measurement(
        self,
        delta_world: np.ndarray,
        rel_v_world: np.ndarray,
        dist: float,
        fog_norm: float,
        ego_forward: np.ndarray,
        ego_left: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        dist_n = float(np.clip(dist / max(self.cfg.entity_max_dist, 1e-3), 0.0, 1.0))
        pos_std = self.cfg.obs_pos_noise_m * (0.45 + 0.80 * dist_n + 0.90 * fog_norm)
        vel_std = self.cfg.obs_vel_noise_ms * (0.35 + 0.75 * dist_n + 0.80 * fog_norm)
        noisy_delta = delta_world + np.random.normal(0.0, pos_std, size=2).astype(np.float32)
        noisy_rel_v = rel_v_world + np.random.normal(0.0, vel_std, size=2).astype(np.float32)
        rel_x = float(np.dot(noisy_delta, ego_forward) / self.cfg.entity_max_dist)
        rel_y = float(np.dot(noisy_delta, ego_left) / self.cfg.entity_max_dist)
        rel_vx = float(np.dot(noisy_rel_v, ego_forward) / 15.0)
        rel_vy = float(np.dot(noisy_rel_v, ego_left) / 10.0)
        sigma2 = float(np.clip(0.05 + 0.22 * dist_n + 0.28 * fog_norm + 0.08 * pos_std + 0.04 * vel_std, 0.05, 0.95))
        return rel_x, rel_y, rel_vx, rel_vy, sigma2

    def _collect_entity_features(self) -> Tuple[np.ndarray, np.ndarray, float]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        curv = self._curvature_ahead(wp_onlane)
        lane_curv = float(np.tanh(10.0 * curv))
        lane_head = float(self._lane_heading_offset(wp_onlane))
        ego_tf = self.vehicle.get_transform()
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
        ego_left = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float32)
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)
        fog_norm = get_fog_norm(self.world)

        feats: List[Tuple[float, np.ndarray]] = []
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            nloc = npc.get_location()
            delta = np.array([nloc.x - loc.x, nloc.y - loc.y], dtype=np.float32)
            dist = float(np.linalg.norm(delta))
            if dist < 1.5 or dist > self.cfg.entity_max_dist:
                continue
            if self._observation_dropout(dist, fog_norm, kind="vehicle"):
                continue
            nvel = npc.get_velocity()
            rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_vel
            rel_x, rel_y, rel_vx, rel_vy, sigma2 = self._noisy_relative_measurement(delta, rel_v, dist, fog_norm, ego_forward, ego_left)
            c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            kappa = np.array([lane_curv, lane_head], dtype=np.float32)
            e = np.concatenate([np.array([rel_x, rel_y, rel_vx, rel_vy], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
            feats.append((dist, e))

        for walker in self._nearby_walkers(loc):
            try:
                wloc = walker.get_location()
                delta = np.array([wloc.x - loc.x, wloc.y - loc.y], dtype=np.float32)
                dist = float(np.linalg.norm(delta))
                if dist < 1.5 or dist > self.cfg.entity_max_dist:
                    continue
                if self._observation_dropout(dist, fog_norm, kind="walker"):
                    continue
                wvel = walker.get_velocity()
                rel_v = np.array([wvel.x, wvel.y], dtype=np.float32) - ego_vel
                rel_x, rel_y, rel_vx, rel_vy, sigma2 = self._noisy_relative_measurement(delta, rel_v, dist, fog_norm, ego_forward, ego_left)
                sigma2 = float(np.clip(sigma2 + 0.06, 0.05, 0.98))
                c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                kappa = np.array([lane_curv, lane_head], dtype=np.float32)
                e = np.concatenate([np.array([rel_x, rel_y, rel_vx, rel_vy], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
                feats.append((dist, e))
            except Exception:
                continue

        tl_state, tl_dist = self._get_active_traffic_light_info()
        # Include a light once it is within braking range, not just the fixed
        # near distance, so the relational state can see it early enough to act.
        tl_edge_range = max(float(self.cfg.tl_near_dist), self._red_brake_distance() + 6.0)
        if 0.0 <= tl_dist < tl_edge_range and (not self._observation_dropout(tl_dist, fog_norm, kind="traffic_light")):
            noisy_tl_dist = float(max(0.0, tl_dist + np.random.normal(0.0, self.cfg.tl_obs_noise_m * (0.35 + fog_norm))))
            rel_x = float(np.clip(noisy_tl_dist / self.cfg.entity_max_dist, 0.0, 1.0))
            sigma2 = float(np.clip(0.08 + 0.24 * (noisy_tl_dist / max(self.cfg.tl_near_dist, 1e-3)) + 0.25 * fog_norm, 0.05, 0.90))
            c = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            kappa = np.array([lane_curv, lane_head], dtype=np.float32)
            e = np.concatenate([np.array([rel_x, 0.0, 0.0, 0.0], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
            feats.append((noisy_tl_dist, e))

        # Top-K relational aggregation (Sec. III / Table I): keep the K = M
        # nearest entities within 60 m for the compact single-actor state.
        feats.sort(key=lambda x: x[0])
        feats = feats[: self.cfg.max_entities]

        edges = np.zeros((self.cfg.max_entities, self.cfg.edge_dim), dtype=np.float32)
        mask = np.zeros((self.cfg.max_entities,), dtype=np.float32)
        if len(feats) == 0:
            mask[0] = 1.0
            edges[0, -1] = 0.8
            nearest = self.cfg.entity_max_dist
        else:
            for i, (_, e) in enumerate(feats):
                edges[i, :] = e
                mask[i] = 1.0
            nearest = float(feats[0][0])
        return edges, mask, nearest

    def _route_observation_features(self) -> Dict[str, float]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        ego_speed_ms = vec3_length(vel)
        ego_speed_kmh = 3.6 * ego_speed_ms
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_curv = float(np.tanh(10.0 * self._curvature_ahead(wp_onlane)))
        lane_head = float(self._lane_heading_offset(wp_onlane))

        lookahead_m = max(self.cfg.lookahead_base_m + self.cfg.lookahead_speed_gain * max(ego_speed_kmh, 0.0), 4.0)
        rs = self._route_reference_state(loc, lookahead_m=lookahead_m)
        lane_width = max(float(rs.get('lane_width', 3.5)), 3.5)
        signed_lane_err = float(rs.get('signed_lane_err', 0.0))
        heading_err = float(rs.get('heading_err', 0.0))
        ref_wp = rs.get('ref_wp', None)

        ego_tf = self.vehicle.get_transform()
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
        ego_left = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float32)
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)

        look_x = 0.0
        look_y = 0.0
        v_route = 0.0
        if ref_wp is not None:
            ref_loc = ref_wp.transform.location
            delta = np.array([ref_loc.x - loc.x, ref_loc.y - loc.y], dtype=np.float32)
            look_x = float(np.clip(np.dot(delta, ego_forward) / max(lookahead_m, 1.0), -2.0, 2.0))
            look_y = float(np.clip(np.dot(delta, ego_left) / max(1.5 * lane_width, 1.0), -2.0, 2.0))
            route_yaw = math.radians(ref_wp.transform.rotation.yaw)
            route_fwd = np.array([math.cos(route_yaw), math.sin(route_yaw)], dtype=np.float32)
            v_route = float(np.clip(np.dot(ego_vel, route_fwd) / 10.0, -1.5, 2.0))

        fog_norm = get_fog_norm(self.world)
        density_den = self.cfg.train_npc_max if str(self.cfg.mode_name).lower() in ("train", "adapt", "policy") else self.cfg.npc_max
        density_norm = float(np.clip(len(self.npcs) / max(density_den, 1), 0.0, 1.0))
        curv_norm = float(np.clip(abs(lane_curv), 0.0, 1.0))
        # μ_A ∈ [0,1] (Sec. III-B): membership of fragile conditions (rain/fog,
        # dense traffic, sharp curvature). High μ_A means a fragile scene.
        muA = float(np.clip(0.45 * density_norm + 0.35 * fog_norm + 0.20 * curv_norm, 0.0, 1.0))

        return {
            'lane_curv': float(lane_curv),
            'lane_head': float(lane_head),
            'route_cte': float(np.clip(signed_lane_err / max(lane_width, 1.0), -2.0, 2.0)),
            'route_heading': float(np.clip(heading_err / (math.pi / 2.0), -1.0, 1.0)),
            'lookahead_x': float(look_x),
            'lookahead_y': float(look_y),
            'v_route': float(v_route),
            'wrong_lane': float(1.0 if bool(rs.get('wrong_lane', False)) else 0.0),
            'opposite_lane': float(1.0 if bool(rs.get('opposite_lane', False)) else 0.0),
            'muA': float(muA),
            'remaining_s': float(rs.get('remaining_s', 0.0)),
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed = vec3_length(vel)
        v_ego = float(np.clip(speed / 10.0, 0.0, 2.0))

        s_arc, _ = self._route_projection(loc)
        remaining_s = max(0.0, self.route_total_len_m - s_arc)
        total_route_len = max(self.route_total_len_m, 1.0)
        d_goal = float(np.clip(remaining_s / total_route_len, 0.0, 1.5))

        route_obs = self._route_observation_features()
        scalars = np.array(
            [
                v_ego,
                float(self.prev_action[0]),
                float(self.prev_action[1]),
                float(self.prev_action[2]),
                d_goal,
                float(route_obs['lane_curv']),
                float(route_obs['lane_head']),
                float(route_obs['route_cte']),
                float(route_obs['route_heading']),
                float(route_obs['lookahead_x']),
                float(route_obs['lookahead_y']),
                float(route_obs['v_route']),
                float(route_obs['muA']),
            ],
            dtype=np.float32,
        )
        edges, mask, _ = self._collect_entity_features()
        return {"scalars": scalars, "edges": edges, "mask": mask}

    def _compute_route_guidance_action(self, tl_state: str, tl_dist: float, front_vehicle_dist: Optional[float], front_vehicle_speed_kmh: float) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.vehicle is None or len(self.route_wps) < 2:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), {
                "desired_speed_kmh": 0.0,
                "route_idx": 0.0,
                "route_target_idx": 0.0,
                "route_cte": 0.0,
                "route_heading_err": 0.0,
                "route_remaining_s": 0.0,
                "route_dL": 0.0,
                "wrong_lane": 0.0,
                "opposite_lane": 0.0,
            }

        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed_kmh = 3.6 * vec3_length(vel)
        lookahead_m = max(self.cfg.lookahead_base_m + self.cfg.lookahead_speed_gain * max(speed_kmh, 0.0), 4.0)
        rs = self._route_reference_state(loc, lookahead_m=lookahead_m)
        dL = float(rs["dL"])
        remaining_s = float(rs["remaining_s"])
        ref_idx = int(rs["ref_idx"])
        ref_wp = rs["ref_wp"]
        signed_lane_err = float(rs["signed_lane_err"])
        heading_err = float(rs["heading_err"])
        lane_width = float(rs["lane_width"])
        wrong_lane = bool(rs["wrong_lane"])
        opposite_lane = bool(rs["opposite_lane"])

        curv = self._curvature_ahead(ref_wp)
        curv_n = abs(float(np.tanh(10.0 * curv)))
        desired_speed_kmh = self.cfg.target_speed_kmh - self.cfg.curve_speed_penalty_kmh * curv_n
        desired_speed_kmh = max(self.cfg.min_target_speed_kmh, desired_speed_kmh)
        desired_speed_kmh = min(desired_speed_kmh, self.cfg.hard_speed_cap_kmh)

        if remaining_s < 30.0:
            desired_speed_kmh = min(desired_speed_kmh, 12.0)
        if remaining_s < 15.0:
            desired_speed_kmh = min(desired_speed_kmh, 8.0)
        if remaining_s < 8.0:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        abs_lane_err = abs(signed_lane_err)
        if abs_lane_err > 0.22 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 13.0)
        if abs_lane_err > 0.35 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 11.0)
        if abs_lane_err > 0.50 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 9.0)
        if abs_lane_err > 0.70 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 7.0)
        if abs_lane_err > 0.95 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 5.5)

        if abs(heading_err) > 0.25:
            desired_speed_kmh = min(desired_speed_kmh, 11.0)
        if abs(heading_err) > 0.40:
            desired_speed_kmh = min(desired_speed_kmh, 8.5)
        if abs(heading_err) > 0.60:
            desired_speed_kmh = min(desired_speed_kmh, 6.5)

        if wrong_lane:
            desired_speed_kmh = min(desired_speed_kmh, 4.5)
        if opposite_lane:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        if tl_state == "Red":
            if tl_dist <= self.cfg.tl_stop_dist:
                desired_speed_kmh = 0.0
            elif tl_dist < self.cfg.tl_near_dist:
                ratio = (tl_dist - self.cfg.tl_stop_dist) / max(self.cfg.tl_near_dist - self.cfg.tl_stop_dist, 1e-3)
                desired_speed_kmh = min(desired_speed_kmh, max(0.0, ratio * self.cfg.min_target_speed_kmh))
        elif tl_state == "Yellow" and tl_dist < self.cfg.tl_stop_dist:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        if front_vehicle_dist is not None:
            if front_vehicle_dist < self.cfg.front_vehicle_block_dist:
                desired_speed_kmh = 0.0
            elif front_vehicle_dist < self.cfg.front_vehicle_soft_block_dist:
                ratio = (front_vehicle_dist - self.cfg.front_vehicle_block_dist) / max(self.cfg.front_vehicle_soft_block_dist - self.cfg.front_vehicle_block_dist, 1e-3)
                desired_speed_kmh = min(desired_speed_kmh, max(0.0, ratio * max(front_vehicle_speed_kmh, 5.0)))
            elif front_vehicle_dist < self.cfg.front_vehicle_max_dist:
                desired_speed_kmh = min(desired_speed_kmh, max(front_vehicle_speed_kmh + 3.0, 7.0))

        cte_term = math.atan2(1.45 * signed_lane_err, max(0.45 * lookahead_m, 1.0))
        steer = (self.cfg.route_steer_kp + 0.55) * heading_err - (self.cfg.route_cte_kp + 0.55) * cte_term

        center_push = 0.0
        if abs_lane_err > self.cfg.center_deadband_m:
            lane_push_mag = max(0.0, abs_lane_err - self.cfg.center_push_start_m)
            center_push = self.cfg.center_push_gain * math.tanh(lane_push_mag / 0.75) * float(np.sign(signed_lane_err))
            steer -= center_push
            if abs_lane_err > self.cfg.route_hard_dL_m_tight:
                hard_push = self.cfg.center_push_hard_gain * math.tanh((abs_lane_err - self.cfg.route_hard_dL_m_tight) / 0.60)
                steer -= hard_push * float(np.sign(signed_lane_err))

        if wrong_lane:
            steer *= 1.20
        if opposite_lane:
            steer *= 1.35
        steer = float(np.clip(steer, -1.0, 1.0))
        if speed_kmh > 10.0:
            steer = float(np.clip(steer, -self.cfg.max_safe_steer_at_speed, self.cfg.max_safe_steer_at_speed))

        speed_err = desired_speed_kmh - speed_kmh
        if desired_speed_kmh <= 0.1:
            throttle = 0.0
            brake = float(np.clip(0.35 + (max(0.0, self.cfg.tl_stop_dist - tl_dist) / max(self.cfg.tl_stop_dist, 1e-3)), 0.35, 1.0))
        elif speed_err >= 0.0:
            throttle = float(np.clip(0.12 + self.cfg.route_speed_kp * speed_err, 0.0, 0.75))
            brake = 0.0
        elif speed_err > -self.cfg.coast_speed_band_kmh:
            throttle = 0.0
            brake = 0.0
        elif speed_err > -self.cfg.soft_brake_speed_excess_kmh:
            throttle = 0.0
            brake = float(np.clip(((-speed_err) - self.cfg.coast_speed_band_kmh) / 24.0, 0.0, 0.10))
        else:
            throttle = 0.0
            brake = float(np.clip(0.10 + (((-speed_err) - self.cfg.soft_brake_speed_excess_kmh) / 10.0), 0.0, 0.42))

        meta = {
            "desired_speed_kmh": float(desired_speed_kmh),
            "route_idx": float(ref_idx),
            "route_target_idx": float(min(ref_idx + 1, len(self.route_wps) - 1)),
            "route_cte": float(signed_lane_err),
            "route_heading_err": float(heading_err),
            "route_remaining_s": float(remaining_s),
            "route_dL": float(dL),
            "wrong_lane": float(1.0 if wrong_lane else 0.0),
            "opposite_lane": float(1.0 if opposite_lane else 0.0),
            "center_push": float(center_push),
        }
        return np.array([throttle, brake, steer], dtype=np.float32), meta

    def route_guidance_action(self) -> np.ndarray:
        """Return the deterministic lane/traffic controller action.

        It is used only to bootstrap the replay buffer during the optional
        training curriculum and by the explicitly enabled safety shield.
        """
        if self.vehicle is None:
            return np.zeros(3, dtype=np.float32)
        tl_state, tl_dist = self._get_active_traffic_light_info()
        front_dist, front_speed, _ = self._get_front_vehicle_info()
        action, _ = self._compute_route_guidance_action(
            tl_state=tl_state,
            tl_dist=tl_dist,
            front_vehicle_dist=front_dist,
            front_vehicle_speed_kmh=front_speed,
        )
        return action.astype(np.float32)

    def _red_light_assist(self, a: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        """Bounded braking when a red stop line is inside braking distance.

        Disabled unless cfg.red_light_assist is set, because every activation is
        counted in the reported intervention rate (Sec. IV-F).
        """
        if not bool(getattr(self.cfg, "red_light_assist", False)):
            return a, False, ""
        try:
            tl_state, tl_dist = self._get_active_traffic_light_info()
        except BaseException:
            return a, False, ""
        if tl_state != "Red" or tl_dist > 1e8:
            return a, False, ""
        onset = max(float(self.cfg.tl_stop_dist), self._red_brake_distance())
        if tl_dist < 0.0 or tl_dist > onset:
            return a, False, ""
        a = np.asarray(a, dtype=np.float32).copy()
        urgency = float(np.clip((onset - tl_dist) / max(onset, 1e-3), 0.0, 1.0))
        a[0] = 0.0
        a[1] = max(float(a[1]), float(np.clip(0.35 + 0.55 * urgency, 0.35, 1.0)))
        return a, True, "red_light"

    def _policy_passthrough_filter(self, raw_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], bool]:
        """Apply actuator bounds only; no hidden route-controller blending."""
        assert self.vehicle is not None
        speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        tl_state, tl_dist = self._get_active_traffic_light_info()
        front_dist, front_speed, _ = self._get_front_vehicle_info()
        _, route_meta = self._compute_route_guidance_action(
            tl_state=tl_state,
            tl_dist=tl_dist,
            front_vehicle_dist=front_dist,
            front_vehicle_speed_kmh=front_speed,
        )
        a = np.asarray(raw_action, dtype=np.float32).copy()
        a[0] = float(np.clip(a[0], 0.0, 1.0))
        a[1] = float(np.clip(a[1], 0.0, 1.0))
        a[2] = float(np.clip(a[2], -1.0, 1.0))
        if a[0] > 0.0 and a[1] > 0.0:
            if a[0] >= a[1]:
                a[1] = 0.0
            else:
                a[0] = 0.0
        # Actuator slew limits are modeled as vehicle-interface dynamics, not
        # route assistance. The replay buffer stores this applied action.
        a[0] = float(np.clip(a[0], self.prev_action[0] - self.cfg.max_throttle_step, self.prev_action[0] + self.cfg.max_throttle_step))
        a[1] = float(np.clip(a[1], self.prev_action[1] - self.cfg.max_brake_step, self.prev_action[1] + self.cfg.max_brake_step))
        a[2] = float(np.clip(a[2], self.prev_action[2] - self.cfg.max_steer_step, self.prev_action[2] + self.cfg.max_steer_step))
        route_meta.update({
            "safety_tl_state": tl_state,
            "safety_tl_dist": float(tl_dist),
            "safety_front_vehicle_dist": -1.0 if front_dist is None else float(front_dist),
            "safety_front_vehicle_speed_kmh": float(front_speed),
            "safety_front_ttc": 999.0,
            "safety_min_vehicle_ttc": 999.0,
            "safety_reason": "none",
            "policy_route_blend": 0.0,
            "policy_speed_kmh": float(speed_kmh),
        })
        # Optional bounded red-light braking (cfg.red_light_assist, off by
        # default). Counted as an intervention so it stays visible in the
        # reported intervention rate.
        a, red_assist, red_reason = self._red_light_assist(a)
        if red_assist:
            a[1] = float(np.clip(a[1], self.prev_action[1] - self.cfg.max_brake_step, self.prev_action[1] + self.cfg.max_brake_step))
            route_meta["safety_reason"] = red_reason
            return a.astype(np.float32), route_meta, True
        return a.astype(np.float32), route_meta, False

    def _safety_filter(self, raw_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], bool]:
        """Optional bounded control stabilization from Table I.

        Nominal actions pass through unchanged. Interventions are activated only
        by route-constraint, TTC, red-light, stopped-lead-vehicle, overspeed, or
        prolonged-stall conditions, and every activation is reported.
        """
        assert self.vehicle is not None
        base, route_meta, _ = self._policy_passthrough_filter(raw_action)
        tl_state = str(route_meta.get("safety_tl_state", "None"))
        tl_dist = safe_float(route_meta.get("safety_tl_dist", 1e9), 1e9)
        front_dist_raw = safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0)
        front_dist = None if front_dist_raw < 0.0 else front_dist_raw
        front_speed = safe_float(route_meta.get("safety_front_vehicle_speed_kmh", 0.0))
        guidance, guidance_meta = self._compute_route_guidance_action(
            tl_state=tl_state,
            tl_dist=tl_dist,
            front_vehicle_dist=front_dist,
            front_vehicle_speed_kmh=front_speed,
        )
        route_meta.update(guidance_meta)
        speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        cte = abs(safe_float(route_meta.get("route_cte", 0.0)))
        hdg = abs(safe_float(route_meta.get("route_heading_err", 0.0)))
        wrong_lane = bool(safe_float(route_meta.get("wrong_lane", 0.0)) > 0.5)
        opposite_lane = bool(safe_float(route_meta.get("opposite_lane", 0.0)) > 0.5)
        route_bad = wrong_lane or opposite_lane or cte > self.cfg.route_override_dL_m or hdg > self.cfg.route_override_heading_rad
        route_hard = wrong_lane or opposite_lane or cte > self.cfg.route_hard_dL_m or hdg > self.cfg.route_hard_heading_rad

        ego_speed_ms = vec3_length(self.vehicle.get_velocity())
        front_ttc = 999.0
        if front_dist is not None:
            closing = max(0.0, ego_speed_ms - front_speed / 3.6)
            if closing > 1e-3:
                front_ttc = front_dist / closing
        min_ttc, min_ttc_dist = self._get_min_vehicle_ttc(max_dist=max(self.cfg.front_vehicle_max_dist, 25.0))
        red_stop = (
            tl_state == "Red"
            and tl_dist < 1e8
            and tl_dist <= max(self.cfg.tl_stop_dist + 0.5, self._red_brake_distance())
        )
        front_blocked = front_dist is not None and (
            front_dist < self.cfg.front_vehicle_block_dist
            or (front_ttc < self.cfg.hard_ttc_s and front_dist < self.cfg.front_vehicle_soft_block_dist)
        )
        open_road = (not red_stop) and (front_dist is None or front_dist > self.cfg.front_vehicle_soft_block_dist)

        a = base.copy()
        intervention = False
        reason = "none"
        blend = 0.0
        if route_hard:
            blend = float(self.cfg.policy_route_blend_hard)
            reason = "route_recovery"
        elif route_bad:
            blend = float(self.cfg.policy_route_blend_bad)
            reason = "route_stabilization"
        if blend > 0.0:
            a = (1.0 - blend) * a + blend * guidance
            intervention = True

        if min_ttc < self.cfg.caution_ttc_s and min_ttc_dist < 18.0 and not red_stop:
            a[0] = min(float(a[0]), 0.20)
            if min_ttc < self.cfg.hard_ttc_s:
                a[1] = max(float(a[1]), 0.35)
            intervention = True
            reason = "ttc"

        if speed_kmh > self.cfg.hard_speed_cap_kmh:
            a[0] = 0.0
            a[1] = max(float(a[1]), float(np.clip(0.20 + 0.08 * (speed_kmh - self.cfg.hard_speed_cap_kmh), 0.20, 1.0)))
            intervention = True
            reason = "speed_cap"

        if red_stop:
            a[0] = 0.0
            a[1] = max(float(a[1]), 0.80)
            a[2] = float(np.clip(guidance[2], -0.30, 0.30))
            intervention = True
            reason = "red_light"
        elif front_blocked:
            a[0] = 0.0
            a[1] = max(float(a[1]), 0.70)
            intervention = True
            reason = "front_vehicle"

        if speed_kmh < self.cfg.stuck_speed_kmh and open_road and not route_hard:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        if self.stuck_steps >= self.cfg.stuck_steps_threshold and open_road:
            self.release_steps_left = max(self.release_steps_left, self.cfg.release_duration_steps)
            self.stuck_steps = 0
        if self.release_steps_left > 0 and open_road:
            a = guidance.copy()
            a[0] = max(float(a[0]), self.cfg.release_throttle)
            a[1] = 0.0
            a[2] = float(np.clip(a[2], -self.cfg.low_speed_steer_limit, self.cfg.low_speed_steer_limit))
            self.release_steps_left -= 1
            intervention = True
            reason = "unstuck"

        a[0] = float(np.clip(a[0], 0.0, 1.0))
        a[1] = float(np.clip(a[1], 0.0, 1.0))
        a[2] = float(np.clip(a[2], -1.0, 1.0))
        if a[0] > 0.0 and a[1] > 0.0:
            if a[0] >= a[1]:
                a[1] = 0.0
            else:
                a[0] = 0.0
        route_meta.update({
            "safety_front_ttc": float(front_ttc),
            "safety_min_vehicle_ttc": float(min_ttc),
            "safety_reason": reason,
            "policy_route_blend": float(blend),
        })
        return a.astype(np.float32), route_meta, bool(intervention)

    def _actor_is_alive(self, actor: Optional[carla.Actor]) -> bool:
        if actor is None:
            return False
        try:
            return bool(actor.is_alive)
        except BaseException:
            return False

    def _safe_set_autopilot(self, actor: Optional[carla.Actor], enabled: bool) -> None:
        if not self._actor_is_alive(actor):
            return
        try:
            actor.set_autopilot(bool(enabled), self.tm.get_port())
            return
        except TypeError:
            pass
        except BaseException:
            pass
        try:
            actor.set_autopilot(bool(enabled))
        except BaseException:
            pass

    def _live_actor_ids(self) -> set:
        """Ids the server currently knows, from one world snapshot.

        Cheaper and fresher than per-actor get_actor() calls, and unlike the
        cached handle's is_alive it reflects server-side collection (e.g. a
        walker removed mid-episode, which also removes its attached
        controller).
        """
        try:
            return {int(a.id) for a in self.world.get_actors()}
        except BaseException:
            return set()

    def _batch_destroy_ids(self, actors: Sequence[Optional[carla.Actor]], label: str = "batch") -> int:
        """Destroy actors through client.apply_batch_sync(DestroyActor(...)).

        The batch command path is the CARLA-recommended teardown: ids absent
        from the server are reported in the response instead of raising, which
        avoids the "unable to destroy actor: not found" server error lines that
        per-actor destroy() emits from the C++ layer.
        """
        live_ids = self._live_actor_ids()
        ids: List[int] = []
        for actor in actors:
            if actor is None:
                continue
            try:
                aid = int(actor.id)
            except BaseException:
                continue
            if aid in live_ids and aid not in ids:
                ids.append(aid)
        if not ids:
            return 0
        try:
            self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in ids], True)
        except BaseException as e:
            if self.cfg.debug_mode:
                print(f"[WARN] {label} batch destroy failed ({e}); falling back to per-actor destroy")
            for actor in actors:
                live = self._refetch(actor)
                if live is not None:
                    self._safe_destroy_one(live, sleep_s=0.02)
        self._drain_world_ticks(1, f"{label}_post_batch_destroy", sleep_s=0.02)
        return len(ids)

    def _refetch(self, actor: Optional[carla.Actor]) -> Optional[carla.Actor]:
        """Re-fetch an actor by id from the current world snapshot.

        Returns None when the server no longer knows the actor. The cached
        client-side handle's is_alive can be stale in synchronous mode, so
        destroying through a fresh handle avoids CARLA's "already dead" /
        "unable to destroy actor: not found" teardown warnings.
        """
        if actor is None:
            return None
        try:
            return self.world.get_actor(int(actor.id))
        except BaseException:
            return None

    def _safe_destroy_one(self, actor: Optional[carla.Actor], sleep_s: float = 0.0) -> None:
        if not self._actor_is_alive(actor):
            return
        try:
            actor.destroy()
        except RuntimeError as e:
            # Ignore the exact error we are fixing.
            if "destroyed actor" not in str(e).lower() and self.cfg.debug_mode:
                print(f"[WARN] destroy actor runtime error: {e}")
        except BaseException as e:
            if self.cfg.debug_mode:
                print(f"[WARN] destroy actor failed: {e}")
        if sleep_s > 0.0:
            try:
                time.sleep(sleep_s)
            except BaseException:
                pass

    def _drain_world_ticks(self, count: int, label: str, sleep_s: float = 0.03) -> None:
        for i in range(max(0, int(count))):
            self._safe_tick(label=f"{label}_{i+1}", raise_on_fail=False)
            if sleep_s > 0.0:
                try:
                    time.sleep(sleep_s)
                except BaseException:
                    pass

    def _on_collision_event(self, event: object) -> None:
        if self._teardown_in_progress or (not self._episode_live):
            return
        if not self._actor_is_alive(self.vehicle):
            return
        try:
            self.collision_events.append(event)
        except BaseException:
            pass

    def _on_lane_invasion_event(self, event: object) -> None:
        if self._teardown_in_progress or (not self._episode_live):
            return
        if not self._actor_is_alive(self.vehicle):
            return
        try:
            self.lane_invasion_events.append(event)
        except BaseException:
            pass

    def _begin_episode_teardown(self, reason: str = "") -> None:
        """
        Stop new callbacks and detach Traffic Manager ownership before reset().
        This is intentionally lightweight and idempotent.
        """
        if self._teardown_in_progress:
            return

        self._teardown_in_progress = True
        self._episode_live = False
        self._terminal_reason = str(reason or self._terminal_reason or "")

        # Stop sensors first so no new callbacks are queued.
        for sensor in list(self.sensor_list):
            self._stop_sensor_only(sensor)

        # Detach TM from NPCs before any future destroy.
        for npc in list(self.npcs):
            self._safe_set_autopilot(npc, False)

        # Freeze ego control.
        if self._actor_is_alive(self.vehicle):
            self._safe_set_autopilot(self.vehicle, False)
            try:
                self.vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False,
                    )
                )
            except BaseException:
                pass

        # One drain tick here helps clear queued callbacks before reset() starts.
       # self._drain_world_ticks(1, "terminal_teardown", sleep_s=0.02)    

    def _setup_collision(self) -> None:
        if not self.cfg.enable_collision_sensor or not self._actor_is_alive(self.vehicle):
            return
        try:
            bp = self.bp_lib.find("sensor.other.collision")
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            sensor.listen(self._on_collision_event)
            self.sensor_list.append(sensor)
        except Exception as e:
            if self.cfg.debug_mode:
                print(f"[WARN] collision sensor setup failed: {e}")

    def _setup_lane_invasion(self) -> None:
        if not self.cfg.enable_lane_invasion_sensor or not self._actor_is_alive(self.vehicle):
            return
        try:
            bp = self.bp_lib.find("sensor.other.lane_invasion")
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            sensor.listen(self._on_lane_invasion_event)
            self.sensor_list.append(sensor)
        except Exception as e:
            if self.cfg.debug_mode:
                print(f"[WARN] lane invasion sensor setup failed: {e}")

    def _tick(self) -> None:
        self._safe_tick(label="step")
        self._update_spectator()

    def _ego_spawn_candidate_indices(self) -> List[int]:
        n = len(self.spawn_points)
        if n == 0:
            return [0]
        # fixed_spawn_index < 0  → fully random spawn each episode (prevents
        # overfitting to a single scenario; confirmed necessary from training log
        # where 97% collisions all happened at the same map location).
        if self.fixed_spawn_index < 0:
            indices = list(range(n))
            random.shuffle(indices)
            return indices
        # fixed_spawn_index >= 0 → deterministic base + nearby fallbacks
        offsets = [0, 1, 2, 3, 5, 8, 13]
        out: List[int] = []
        seen: set = set()
        for off in offsets:
            idx = (self.fixed_spawn_index + off) % n
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        return out

    def _route_exists_between(self, start_tf: carla.Transform, goal_index: int) -> bool:
        grp = self._get_global_planner()
        if grp is None:
            return False
        if goal_index < 0 or goal_index >= len(self.spawn_points):
            return False
        try:
            start_wp = self._project_drive_wp(start_tf.location)
            goal_wp = self._project_drive_wp(self.spawn_points[goal_index].location)
            if start_wp is None or goal_wp is None:
                return False
            start_loc = start_wp.transform.location
            goal_loc = goal_wp.transform.location
            if distance2d(start_loc, goal_loc) < self.cfg.candidate_goal_min_dist_m:
                return False
            trace = grp.trace_route(start_loc, goal_loc)
            if len(trace) < 2:
                return False
            route = [wp for wp, _ in trace if wp is not None]
            route = self._dedupe_waypoints(route, min_sep=0.75)
            if len(route) < 2:
                return False
            route_len = self._route_length_from_wps(route)
            return self._route_length_ok(route_len, strict=True)
        except BaseException:
            return False

    def _spawn_ego(self, preferred_goal_index: Optional[int] = None) -> carla.Transform:
        bp = self.bp_lib.find(self.cfg.car_name)
        if bp.has_attribute("color"):
            bp.set_attribute("color", self.cfg.fixed_color)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", self.ego_role)

        base_candidates = self._ego_spawn_candidate_indices()
        ordered_indices: List[int] = []
        seen: set[int] = set()

        def _append_idx(idx: int) -> None:
            if idx not in seen:
                ordered_indices.append(idx)
                seen.add(idx)

        for idx in base_candidates:
            _append_idx(idx)
        for idx in range(len(self.spawn_points)):
            _append_idx(idx)

        last_spawn_error: Optional[Exception] = None
        best_route_issue: Optional[str] = None

        for idx in ordered_indices:
            raw_tf = self.spawn_points[idx]
            snapped_tf = self._snap_spawn_to_driving_lane(raw_tf)
            spawn_bases: List[carla.Transform] = [snapped_tf]

            raw_offset = distance2d(raw_tf.location, snapped_tf.location)
            if raw_offset > 0.75:
                spawn_bases.append(raw_tf)
                if self.cfg.debug_mode:
                    print(
                        f"[SPAWN] spawn {idx} snapped to lane center by {raw_offset:.2f}m "
                        f"for map {self.town_name}"
                    )

            if self.cfg.destroy_stale_spawn_blockers:
                cleared = 0
                for base_tf in spawn_bases:
                    cleared += self._clear_spawn_blockers(base_tf)
                if cleared > 0 and self.cfg.debug_mode:
                    print(f"[SPAWN] cleared {cleared} blocking actor(s) near spawn {idx}")

            # Cheap pre-check: skip fixed-goal spawns that definitely cannot route.
            if preferred_goal_index is not None and int(preferred_goal_index) >= 0 and self.cfg.strict_goal_route:
                try:
                    start_wp_hint = self._project_drive_wp(snapped_tf.location)
                    if start_wp_hint is None or (not self._route_exists_between(snapped_tf, int(preferred_goal_index))):
                        best_route_issue = f"No valid fixed-goal route from spawn {idx} to goal {int(preferred_goal_index)}"
                        continue
                except Exception as e:
                    best_route_issue = f"Route precheck failed at spawn {idx}: {e}"
                    continue

            for base_tf in spawn_bases:
                for tf in self._spawn_transform_variants(base_tf):
                    vehicle = None

                    try:
                        vehicle = self.world.try_spawn_actor(bp, tf)
                    except Exception as e:
                        last_spawn_error = e
                        vehicle = None

                    if vehicle is None and self.cfg.destroy_stale_spawn_blockers:
                        cleared = self._clear_spawn_blockers(tf, radius=max(self.cfg.spawn_blocker_radius_m, 3.0))
                        if cleared > 0 and self.cfg.debug_mode:
                            print(f"[SPAWN] retry after clearing {cleared} actor(s) near spawn {idx}")
                        try:
                            vehicle = self.world.try_spawn_actor(bp, tf)
                        except Exception as e:
                            last_spawn_error = e
                            vehicle = None

                    if vehicle is None:
                        continue

                    self.vehicle = vehicle
                    try:
                        self.vehicle.set_autopilot(False)
                    except BaseException:
                        pass
                    self.current_spawn_index = idx
                    self._snap_spectator_to_ego()
                    return self.vehicle.get_transform()

            time.sleep(0.05)

        if preferred_goal_index is not None and int(preferred_goal_index) >= 0 and self.cfg.strict_goal_route and best_route_issue is not None:
            raise RuntimeError(
                f"Failed to find a route-valid spawn for goal {int(preferred_goal_index)} in map {self.town_name}. "
                f"Last issue: {best_route_issue}."
            )
        if last_spawn_error is not None:
            raise RuntimeError(f"Failed to spawn ego vehicle after trying multiple spawn points: {last_spawn_error}")
        raise RuntimeError(
            f"Failed to spawn ego vehicle after trying multiple spawn points in map {self.town_name}. "
            "Spawn locations appear occupied or blocked by residual actors."
        )


    def _spawn_npcs(self, npc_count: int) -> None:
        npc_bps = list(self.bp_lib.filter("vehicle.*"))
        if not npc_bps or self.vehicle is None or npc_count <= 0:
            return
        ego_loc = self.vehicle.get_location()
        goal_loc = self.goal_transform.location if self.goal_transform is not None else None
        # Prefer spawn points near the planned route so NPC traffic is dense
        # where the ego actually drives (Sec. IV-A dense-traffic protocol)
        # rather than scattered across the whole town; remaining slots fall
        # back to town-wide spawn points.
        anchors: List[carla.Location] = []
        if self.route_wps:
            step = max(1, len(self.route_wps) // 16)
            anchors = [self.route_wps[i].transform.location for i in range(0, len(self.route_wps), step)]
        near_route: List[carla.Transform] = []
        far_route: List[carla.Transform] = []
        for sp in self.spawn_points:
            if distance2d(sp.location, ego_loc) <= 25.0:
                continue
            if goal_loc is not None and distance2d(sp.location, goal_loc) <= 20.0:
                continue
            if anchors and min(distance2d(sp.location, a) for a in anchors) <= self.cfg.npc_spawn_radius_m:
                near_route.append(sp)
            else:
                far_route.append(sp)
        random.shuffle(near_route)
        random.shuffle(far_route)
        candidates = near_route + far_route
        for sp in candidates[: max(0, npc_count * 5)]:
            if len(self.npcs) >= npc_count:
                break
            npc_bp = random.choice(npc_bps)
            if npc_bp.has_attribute("role_name"):
                npc_bp.set_attribute("role_name", self.npc_role)
            npc = self.world.try_spawn_actor(npc_bp, sp)
            if npc is None:
                continue
            try:
                npc.set_autopilot(True, self.tm.get_port())
                self.tm.vehicle_percentage_speed_difference(npc, random.uniform(-25.0, 10.0))
                self.tm.distance_to_leading_vehicle(npc, random.uniform(2.0, 6.0))
                self.tm.auto_lane_change(npc, random.choice([True, False]))
                self.npcs.append(npc)
            except BaseException:
                self._destroy_actor(npc)

    def _spawn_walkers(self, walker_count: int) -> None:
        """Spawn pedestrian NPCs with AI controllers (Sec. III-C / Table I).

        Walkers are biased toward the planned route so they are
        control-relevant entities for the ego-relational state and for the
        pedestrian-crossing interaction scenarios (Fig. 2 / Fig. 12).
        """
        if walker_count <= 0 or self.vehicle is None:
            return
        try:
            self.world.set_pedestrians_cross_factor(
                float(np.clip(self.cfg.walker_cross_factor, 0.0, 1.0))
            )
        except BaseException:
            pass
        walker_bps = list(self.bp_lib.filter("walker.pedestrian.*"))
        try:
            controller_bp = self.bp_lib.find("controller.ai.walker")
        except BaseException:
            controller_bp = None
        if not walker_bps or controller_bp is None:
            return

        anchors: List[carla.Location] = []
        if self.route_wps:
            step = max(1, len(self.route_wps) // 12)
            anchors = [self.route_wps[i].transform.location for i in range(0, len(self.route_wps), step)]

        ego_loc = self.vehicle.get_location()
        spawned = 0
        attempts = 0
        max_attempts = max(walker_count * 8, 16)
        while spawned < walker_count and attempts < max_attempts:
            attempts += 1
            try:
                loc = self.world.get_random_location_from_navigation()
            except BaseException:
                loc = None
            if loc is None:
                continue
            if distance2d(loc, ego_loc) < 8.0:
                continue
            if anchors:
                near = min(distance2d(loc, a) for a in anchors)
                # Mostly keep walkers near the route; allow a few far ones.
                if near > self.cfg.walker_spawn_radius_m and random.random() < 0.8:
                    continue
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", self.walker_role)
            tf = carla.Transform(carla.Location(loc.x, loc.y, loc.z + 0.3))
            walker = self.world.try_spawn_actor(bp, tf)
            if walker is None:
                continue
            try:
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            except BaseException:
                controller = None
            if controller is None:
                self._destroy_actor(walker)
                continue
            self.walkers.append(walker)
            self.walker_controllers.append(controller)
            spawned += 1

        if spawned <= 0:
            return
        # One tick so controllers register before start()/go_to_location().
        self._safe_tick(label="walker_spawn_settle", raise_on_fail=False)
        for controller in self.walker_controllers:
            try:
                controller.start()
                target = self.world.get_random_location_from_navigation()
                if target is not None:
                    controller.go_to_location(target)
                controller.set_max_speed(float(random.uniform(0.8, 2.0)))
            except BaseException:
                continue

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._episode_live = False
        self._terminal_reason = ""

        self._cleanup_episode_actors()
        if self.cfg.destroy_stale_owned_actors_on_reset:
            cleared = self._destroy_residual_owned_actors()
            if cleared > 0 and self.cfg.debug_mode:
                print(f"[RESET] destroyed {cleared} residual owned actor(s) before spawning")
        apply_weather(self.world, self.weather_mode)

        opt = options or {}
        npc_requested = int(opt.get("npc_count", random.randint(self.cfg.npc_min, self.cfg.npc_max)))
        npc_cap = max(int(self.cfg.npc_max), int(self.cfg.train_npc_max))
        npc_count = int(np.clip(npc_requested, 0, npc_cap))
        mode_l = str(self.cfg.mode_name).strip().lower()
        if mode_l in ("train", "adapt", "policy"):
            walker_lo, walker_hi = int(self.cfg.train_walker_min), int(self.cfg.train_walker_max)
        else:
            walker_lo, walker_hi = int(self.cfg.walker_min), int(self.cfg.walker_max)
        walker_lo, walker_hi = min(walker_lo, walker_hi), max(walker_lo, walker_hi)
        walker_requested = int(opt.get("walker_count", random.randint(walker_lo, walker_hi)))
        walker_count = int(np.clip(walker_requested, 0, 40))
        goal_index_raw = opt.get("goal_index", self.fixed_goal_index if self.cfg.use_fixed_destination else None)
        goal_index = None if goal_index_raw is None or int(goal_index_raw) < 0 else int(goal_index_raw)

        self.episode_steps = 0
        self.prev_action[:] = 0.0
        self.prev_steer = 0.0
        self.prev_acc[:] = 0.0
        self.prev_loc = None
        self.distance_driven_m = 0.0
        self.stuck_steps = 0
        self.release_steps_left = 0
        self.offroute_steps = 0
        self.prev_route_s = None
        self.safety_interventions = 0
        self.blocked_steps_credit = 0
        self._free_stuck_steps = 0
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        requested_goal_index = goal_index
        spawn_tf = self._spawn_ego(preferred_goal_index=requested_goal_index)
        start_wp = self.map.get_waypoint(spawn_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if start_wp is None:
            raise RuntimeError("Could not obtain a driving waypoint for the ego spawn location.")

        self._build_route(spawn_tf, start_wp, goal_index=requested_goal_index)

        self._snap_spectator_to_ego()
        for _ in range(max(0, self.cfg.post_spawn_settle_ticks)):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            self._safe_tick(label="reset_settle")
            self._snap_spectator_to_ego()

        for _ in range(max(6, self.cfg.warmup_reset_ticks)):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            self._tick()

                # If we are using local fallback routing (no planner goal selected),
        # rebuild the local route after the ego has settled on the road.
        if requested_goal_index is None and self.current_goal_index is None:
            # Anchor the lane-follow route to the pose the ego actually settled
            # in. The pre-settle route starts from the spawn waypoint, which
            # can sit metres away once physics settles the vehicle onto the
            # lane, so rebuilding here guarantees start_progress ~ 0 and
            # start_dL ~ 0 without a conditional check.
            rebuilt = self._rebuild_local_route_from_current_pose()
            if not rebuilt and self.cfg.debug_mode:
                print("[WARN] lane-follow route rebuild failed; keeping pre-settle route")

        # Episode start: project within the opening window so a route that
        # loops back near its origin cannot report spurious start progress.
        s0_global, d0_global, idx0_global = self._route_projection_start_window(self.vehicle.get_location())
        progress0_pct = 100.0 * s0_global / max(self.route_total_len_m, 1e-6)

        start_to_route0 = d0_global
        if self.route_wps:
            try:
                start_to_route0 = distance2d(self.vehicle.get_location(), self.route_wps[0].transform.location)
            except BaseException:
                pass

        if progress0_pct > self.cfg.max_reset_start_progress_pct or d0_global > self.cfg.max_reset_start_dL_m:
            # For local fallback routes, do not hard-fail reset after rebuild.
            if requested_goal_index is None and self.current_goal_index is None:
                if self.cfg.debug_mode:
                    print(
                        f"[WARN] relaxing reset alignment for local fallback route: "
                        f"progress={progress0_pct:.2f}% dL={d0_global:.2f}m"
                    )
                idx0_global = 0
                s0_global = 0.0
                progress0_pct = 0.0
                d0_global = 0.0
            elif start_to_route0 <= max(self.cfg.max_reset_start_dL_m, 4.0):
                if self.cfg.debug_mode:
                    print(
                        f"[WARN] reset alignment relaxed: progress={progress0_pct:.2f}% dL={d0_global:.2f}m; "
                        "clamping start progress to route origin"
                    )
                idx0_global = 0
                s0_global = 0.0
                progress0_pct = 0.0
            else:
                raise RuntimeError(
                    f"Invalid reset alignment: progress={progress0_pct:.2f}% dL={d0_global:.2f}m "
                    f"route_len={self.route_total_len_m:.2f}m goal_idx={self.current_goal_index}"
                )


        self.route_progress_idx = idx0_global
        self.prev_route_s = s0_global

        self._spawn_npcs(npc_count)
        self._spawn_walkers(walker_count)

        self._setup_collision()
        self._setup_lane_invasion()
        self._safe_tick(label="post_sensor_attach", raise_on_fail=False)
        self._snap_spectator_to_ego()
        self.prev_loc = self.vehicle.get_location()
        self._teardown_in_progress = False
        self._episode_live = True

        obs = self._get_obs()
        info = {
            "town": self.town_name,
            "npc_count": len(self.npcs),
            "walker_count": len(self.walkers),
            "weather": self.weather_mode,
            "spawn_index": self.current_spawn_index,
            "goal_index": self.current_goal_index,
            "requested_goal_index": requested_goal_index,
            "route_total_len_m": float(self.route_total_len_m),
            "carla_server_version": getattr(self.client, "get_server_version", lambda: "unknown")(),
            "carla_client_version": getattr(self.client, "get_client_version", lambda: "unknown")(),
        }
        if self.cfg.debug_mode:
            tf = self.vehicle.get_transform()
            print(
                f"[RESET] town={info['town']} npc={info['npc_count']} walkers={info['walker_count']} "
                f"weather={info['weather']} "
                f"spawn_idx={info['spawn_index']} goal_idx={info['goal_index']} route_len={info['route_total_len_m']:.1f}m "
                f"spawn=({tf.location.x:.1f},{tf.location.y:.1f},{tf.location.z:.1f}) yaw={tf.rotation.yaw:.1f} "
                f"start_progress={progress0_pct:.2f}% start_dL={d0_global:.2f}"
            )
        return obs, info

    def _compute_reward_components(self, applied_action: np.ndarray) -> Tuple[float, float, float, Dict[str, float]]:
        """Dense multi-objective reward shaping (paper Sec. III-D, Eqs. 8–9).

        r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u    (Eq. 8)

        r_s: safety  – lane adherence + proximity + red-light   (Sec. III-D)
        r_p: progress – tanh(Δs / τ_s) or tanh(v̂_t / τ_v)      (Sec. III-D)
        r_c: comfort  – quadratic jerk + steer-rate penalty     (Sec. III-D)
        r_u: uncertainty – 1 − σ̄  (computed externally, Eq. 9)

        No terminal bonus / penalty (goal_bonus = collision_penalty = 0).
        """
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed = vec3_length(vel)
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_curv = self._curvature_ahead(wp_onlane)
        lane_curv_s = float(np.tanh(10.0 * lane_curv))
        lane_head = self._lane_heading_offset(wp_onlane)
        s_arc, dL = self._route_projection(loc)
        route_obs = self._route_observation_features()
        route_heading_err = float(route_obs.get('route_heading', 0.0)) * (math.pi / 2.0)
        route_cte_norm = float(route_obs.get('route_cte', 0.0))
        delta_s = 0.0 if self.prev_route_s is None else float(np.clip(s_arc - self.prev_route_s, -2.0, 2.0))

        fog_norm = get_fog_norm(self.world)
        density_den = self.cfg.train_npc_max if str(self.cfg.mode_name).lower() in ("train", "adapt", "policy") else self.cfg.npc_max
        density_norm = float(np.clip(len(self.npcs) / max(density_den, 1), 0.0, 1.0))
        curv_norm = float(np.clip(abs(lane_curv_s), 0.0, 1.0))
        # μ_A ∈ [0,1] (Sec. III-B): membership of fragile conditions (rain/fog,
        # dense traffic, sharp curvature). ε(μ_A) = ε_min + (ε_max − ε_min)·μ_A,
        # so the admissible corridor tolerance adapts with the scene context.
        muA = float(np.clip(0.45 * density_norm + 0.35 * fog_norm + 0.20 * curv_norm, 0.0, 1.0))
        eps_mu = self.cfg.eps_min + (self.cfg.eps_max - self.cfg.eps_min) * muA
        # Paper Eq. (8): ψ_L = tanh(d_L / τ_d). The adaptive corridor
        # epsilon(mu_A) is used for constraint/recovery logic, not inserted into
        # the printed reward equation.
        psiL = math.tanh(dL / max(self.cfg.tau_d, 1e-3))

        # Eq. (4) phi_L(d_L, mu_A): corridor excess. Zero inside the admissible
        # corridor, quadratic outside it, so the gradient survives where psi_L
        # has saturated. eps_mu was computed above from mu_A.
        corridor_excess = max(0.0, float(dL) - float(eps_mu))
        phi_L = float(np.clip((corridor_excess / max(eps_mu, 1e-3)) ** 2, 0.0, self.cfg.phi_L_cap))
        # route_obs already exposes these as 0/1 floats (see _route_observation_features).
        wrong_lane_f = float(route_obs.get("wrong_lane", 0.0))
        opposite_lane_f = float(route_obs.get("opposite_lane", 0.0))
        phi_lane_total = float(
            self.cfg.lambda_phi_L * phi_L
            + self.cfg.lambda_wrong_lane * wrong_lane_f
            + self.cfg.lambda_opposite_lane * opposite_lane_f
        )

        # Proximity and time-to-conflict τ_t (Sec. III-B)
        nearest = self.cfg.entity_max_dist
        tau_t = 999.0
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            nloc = npc.get_location()
            dist = distance2d(loc, nloc)
            if dist < nearest:
                nearest = dist
            nvel = npc.get_velocity()
            rel_p = np.array([nloc.x - loc.x, nloc.y - loc.y], dtype=np.float32)
            rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_vel
            closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
            if dist > 0.5:
                tau = dist / max(closing, 1e-3) if closing > 1e-6 else 999.0
                tau_t = min(tau_t, tau)
        for walker in self._nearby_walkers(loc):
            try:
                wloc = walker.get_location()
                dist = distance2d(loc, wloc)
                nearest = min(nearest, dist)
                wvel = walker.get_velocity()
                rel_p = np.array([wloc.x - loc.x, wloc.y - loc.y], dtype=np.float32)
                rel_v = np.array([wvel.x, wvel.y], dtype=np.float32) - ego_vel
                closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
                if dist > 0.5:
                    tau = dist / max(closing, 1e-3) if closing > 1e-6 else 999.0
                    tau_t = min(tau_t, tau)
            except BaseException:
                continue
        # ψ_P: proximity surrogate ψ_P = exp(-dist/τ_p)  (Sec. III-D)
        psiP = math.exp(-nearest / max(self.cfg.tau_p, 1e-3))

        # Red-light noncompliance ρ_t ∈ [0,1]  (Sec. III-D).
        # The onset distance follows the braking requirement, so the penalty
        # appears while stopping is still physically possible; crossing the line
        # on red saturates it at 1.
        rho = 0.0
        red_violation = False
        tl_state, tl_dist = self._get_active_traffic_light_info()
        if tl_state == "Red" and tl_dist < 1e8:
            onset = max(float(self.cfg.tl_stop_dist), self._red_brake_distance())
            if tl_dist < 0.0:
                # Stop line already crossed while red: full noncompliance.
                rho = 1.0
                red_violation = bool(speed > 0.5)
            elif tl_dist < onset:
                z1 = float(np.clip((onset - tl_dist) / max(onset, 1e-3), 0.0, 1.0))
                z2 = float(np.clip(speed / 3.0, 0.0, 1.0))
                rho = float(np.clip(z1 * z2, 0.0, 1.0))

        # Sec. III-D: r_s = 1 − κ_L ψ_L − κ_P ψ_P − κ_R ρ_t
        rs = float(np.clip(1.0 - self.cfg.k_l * psiL - self.cfg.k_p * psiP - self.cfg.k_r * rho, -2.0, 1.0))

        # Sec. III-D: use the incremental arc-length form by default.
        # rp_arc_weight remains a compatibility option for the stated velocity
        # surrogate, but the two alternatives are not mixed in the default run.
        rp_arc = float(math.tanh(delta_s / max(self.cfg.tau_s, 1e-3)))
        # Velocity projection onto route tangent v̂_t (Sec. III-D surrogate)
        v_hat = float(route_obs.get('v_route', 0.0)) * 10.0  # de-normalise
        rp_vel = float(math.tanh(v_hat / max(self.cfg.tau_v, 1e-3)))
        w_arc = float(np.clip(self.cfg.rp_arc_weight, 0.0, 1.0))
        rp = float(w_arc * rp_arc + (1.0 - w_arc) * rp_vel)

        # Sec. III-D: r_c = −κ_j j_t² − κ_δ δ̇_t²
        acc = self.vehicle.get_acceleration()
        acc2 = np.array([acc.x, acc.y], dtype=np.float32)
        jerk = float(np.linalg.norm(acc2 - self.prev_acc) / max(self.cfg.dt, 1e-3))
        steer_rate = float((applied_action[2] - self.prev_steer) / max(self.cfg.dt, 1e-3))
        lane_cross = 1.0 if len(self.lane_invasion_events) > 0 else 0.0

        rc_raw = -(self.cfg.k_j * (jerk ** 2)) - (self.cfg.k_delta * (steer_rate ** 2))
        rc = float(np.clip(rc_raw, -5.0, 0.0))

        return rs, rp, rc, {
            "dL": float(dL),
            "route_s": float(s_arc),
            "delta_s": float(delta_s),
            "muA": float(muA),
            "eps_mu": float(eps_mu),
            "psiL": float(psiL),
            "nearest_entity_dist": float(nearest),
            "psiP": float(psiP),
            "rho_red": float(rho),
            "phi_lane": float(phi_lane_total),
            "phi_L": float(phi_L),
            "wrong_lane": float(wrong_lane_f),
            "opposite_lane": float(opposite_lane_f),
            "red_light_violation": bool(red_violation),
            "jerk": float(jerk),
            "steer_rate": float(steer_rate),
            "lane_curv": float(lane_curv_s),
            "lane_head": float(lane_head),
            "route_heading_err": float(route_heading_err),
            "route_cte_norm": float(route_cte_norm),
            "lane_cross": float(lane_cross),
            "rc_raw": float(rc_raw),
            "rp_arc": float(rp_arc),
            "rp_vel": float(rp_vel),
            "time_to_conflict": float(tau_t),
            "tl_state": tl_state,
            "tl_dist": float(tl_dist) if tl_dist < 1e8 else -1.0,
        }

    def step(self, action: np.ndarray):
        if self._teardown_in_progress:
            raise RuntimeError("step() called while teardown is in progress")
        if not self._actor_is_alive(self.vehicle):
            raise RuntimeError("step() called with missing/destroyed ego actor")

        self.episode_steps += 1
        self.lane_invasion_events.clear()

        raw = np.array(action, dtype=np.float32).copy()
        raw[0] = float(np.clip(raw[0], 0.0, 1.0))
        raw[1] = float(np.clip(raw[1], 0.0, 1.0))
        raw[2] = float(np.clip(raw[2], -1.0, 1.0))
        # Keep both actions for auditability.  The training loop stores the
        # actually executed action by default, because its next state and
        # reward were generated by that action.  This is essential whenever a
        # curriculum, rate limiter, red-light assist, or safety shield changes
        # the policy command.
        command_action = raw.copy()
        if raw[0] < self.cfg.min_throttle_deadzone:
            raw[0] = 0.0
        if raw[1] < self.cfg.min_brake_deadzone:
            raw[1] = 0.0
        if raw[0] > 0.0 and raw[1] > 0.0:
            if raw[0] >= raw[1]:
                raw[1] = 0.0
            else:
                raw[0] = 0.0

        if self.cfg.use_safety_shield:
            applied, route_meta, shield_active = self._safety_filter(raw)
        else:
            applied, route_meta, shield_active = self._policy_passthrough_filter(raw)
        applied[0] = float(np.clip(applied[0], 0.0, 1.0))
        applied[1] = float(np.clip(applied[1], 0.0, 1.0))
        applied[2] = float(np.clip(applied[2], -1.0, 1.0))
        if applied[0] > 0.0 and applied[1] > 0.0:
            if applied[0] >= applied[1]:
                applied[1] = 0.0
            else:
                applied[0] = 0.0

        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=float(applied[0]),
                steer=float(applied[2]),
                brake=float(applied[1]),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
        )
        self._tick()

        cur_loc = self.vehicle.get_location()
        if self.prev_loc is not None:
            self.distance_driven_m += distance2d(cur_loc, self.prev_loc)
        self.prev_loc = cur_loc

        rs, rp, rc, comp = self._compute_reward_components(applied)
        terminated = False
        truncated = False
        reason: Optional[str] = None
        if self.cfg.terminate_on_collision and len(self.collision_events) > 0:
            terminated = True
            reason = "collision"

        dL_after = float(comp.get("dL", 0.0))
        eps_mu_now = float(comp.get("eps_mu", self.cfg.eps_max))
        route_heading_err_now = abs(float(route_meta.get("route_heading_err", 0.0)))
        offroute_thresh = max(self.cfg.max_route_deviation_m, eps_mu_now + 0.75)
        strong_offroute = dL_after > (self.cfg.strong_offroute_factor * offroute_thresh)
        offroute_bad = strong_offroute or ((dL_after > offroute_thresh) and (route_heading_err_now > self.cfg.offroute_heading_gate_rad))
        if offroute_bad:
            self.offroute_steps += 1
        else:
            self.offroute_steps = max(0, self.offroute_steps - 1)
        if self.cfg.terminate_on_offroad and self.offroute_steps >= self.cfg.offroute_grace_steps:
            terminated = True
            reason = "off_route"

        s_arc_now = float(comp.get("route_s", 0.0))
        remaining_s = max(0.0, self.route_total_len_m - s_arc_now)
        goal_euclid = 1e9
        if self.goal_transform is not None:
            goal_euclid = float(cur_loc.distance(self.goal_transform.location))
        route_completion_pct = self._route_completion_pct(s_arc_now)
        success = False
        completion_gate = route_completion_pct >= self.cfg.route_success_pct
        physical_goal_gate = (
            remaining_s <= self.cfg.goal_reach_remaining_s_m
            and goal_euclid <= self.cfg.goal_reach_dist_m
        )
        final_projection_gate = (
            route_completion_pct >= 99.5
            and goal_euclid <= 1.5 * self.cfg.goal_reach_dist_m
        )
        # Completion alone is not sufficient: the vehicle must physically reach
        # the final route endpoint, preventing premature success on looped routes.
        if (completion_gate and physical_goal_gate) or final_projection_gate:
            terminated = True
            reason = "goal"
            success = True

        blocked_wait = (
            (str(comp.get("tl_state", "None")) == "Red" and (3.6 * vec3_length(self.vehicle.get_velocity())) <= self.cfg.blocked_timeout_speed_kmh)
            or (
                safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0) > 0.0
                and safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0) < self.cfg.front_vehicle_soft_block_dist
                and (3.6 * vec3_length(self.vehicle.get_velocity())) <= self.cfg.blocked_timeout_speed_kmh
            )
        )
        if blocked_wait:
            self.blocked_steps_credit = min(self.blocked_steps_credit + 1, self.cfg.blocked_timeout_extension_steps)
        else:
            self.blocked_steps_credit = max(0, self.blocked_steps_credit - 1)

        tl_now = str(comp.get("tl_state", "None"))
        front_dist_now = safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0)
        genuinely_open = (
            tl_now not in ("Red", "Yellow")
            and (front_dist_now < 0.0 or front_dist_now > self.cfg.front_vehicle_soft_block_dist)
        )
        if (3.6 * vec3_length(self.vehicle.get_velocity())) < self.cfg.stuck_speed_kmh and genuinely_open:
            self._free_stuck_steps = getattr(self, "_free_stuck_steps", 0) + 1
        else:
            self._free_stuck_steps = 0

        if (not terminated) and self._free_stuck_steps >= self.cfg.stuck_terminate_steps:
            # Stuck is a route-task failure state, not an external collector
            # interruption.  Treat it as terminal so SAC cannot bootstrap the
            # positive value of a stationary state across an episode reset.
            terminated = True
            reason = "stuck"
            self._free_stuck_steps = 0

        near_goal_extra = self.cfg.near_goal_timeout_extension_steps if remaining_s < self.cfg.near_goal_remaining_s_m else 0
        hard_timeout_steps = self.cfg.max_episode_steps + self.blocked_steps_credit + near_goal_extra
        if (not terminated) and (self.episode_steps >= hard_timeout_steps):
            # The fixed route deadline is part of this task protocol.  It is a
            # terminal failure unless timeout_is_terminal is explicitly
            # disabled for a continuing-task experiment.
            if bool(self.cfg.timeout_is_terminal):
                terminated = True
            else:
                truncated = True
            reason = "timeout"

        self.prev_route_s = s_arc_now
        if shield_active:
            self.safety_interventions += 1
        self.prev_action = applied.copy()
        self.prev_steer = float(applied[2])
        acc = self.vehicle.get_acceleration()
        self.prev_acc = np.array([acc.x, acc.y], dtype=np.float32)

        obs = self._get_obs()
        info = {
            "rs": float(rs),
            "rp": float(rp),
            "rc": float(rc),
            "term_reason": reason or "",
            "goal_reached": bool(reason == "goal"),
            "collision": bool(reason == "collision"),
            "off_road": bool(reason == "off_route"),
            "off_route": bool(reason == "off_route"),
            "timeout": bool(reason in ("timeout", "stuck")),
            "stuck": bool(reason == "stuck"),
            "success": bool(success),
            "steps": int(self.episode_steps),
            "goal_dist": float(remaining_s),
            "goal_euclid": float(goal_euclid),
            "route_completion_pct": float(route_completion_pct),
            "route_total_len_m": float(self.route_total_len_m),
            "stuck_steps": int(self.stuck_steps),
            "offroute_steps": int(self.offroute_steps),
            "blocked_steps_credit": int(self.blocked_steps_credit),
            "blocked_wait": bool(blocked_wait),
            "shield_active": int(bool(shield_active)),
            "safety_intervention": int(bool(shield_active)),
            "intervention_rate": float(self.safety_interventions / max(self.episode_steps, 1)),
            "desired_speed_kmh": float(route_meta.get("desired_speed_kmh", 0.0)),
            "route_idx": float(route_meta.get("route_idx", 0.0)),
            "route_target_idx": float(route_meta.get("route_target_idx", 0.0)),
            "route_cte": float(route_meta.get("route_cte", 0.0)),
            "route_heading_err": float(route_meta.get("route_heading_err", 0.0)),
            "route_remaining_s": float(route_meta.get("route_remaining_s", remaining_s)),
            "route_policy_blend": float(route_meta.get("policy_route_blend", 0.0)),
            "spawn_index": self.current_spawn_index,
            "goal_index": self.current_goal_index,
            "command_action": command_action,
            "applied_action": applied.copy(),
            "replay_action": applied.copy() if self.cfg.replay_applied_action else command_action.copy(),
            "distance_driven_m": float(self.distance_driven_m),
        }
        info.update(comp)
        # ------------------------------------------------------------------
        # σ̄ proxy for env-time reward (Eq. 9: r_u = 1 − σ̄).
        # During rollout the critic ensemble is not called every step for
        # efficiency; instead we compute a lightweight proxy.
        # Proxy: σ̄ ≈ μ_A (fragile contexts → higher uncertainty; Sec. III-B).
        # The training loop overwrites this with the real ensemble-based σ̄
        # via recompute_agent_reward() so the stored replay reward is correct.
        # ------------------------------------------------------------------
        muA_val = float(np.clip(safe_float(info.get("muA", 0.5)), 0.0, 1.0))
        # Also incorporate nearest-entity distance for a richer proxy:
        # closer entities → higher uncertainty even in clear weather.
        near_norm = float(np.clip(safe_float(info.get("nearest_entity_dist", self.cfg.entity_max_dist))
                                  / max(self.cfg.entity_max_dist, 1.0), 0.0, 1.0))
        # sigma_proxy ↑ when μ_A ↑ (fog/dense/curved) or entities are very close
        sigma_proxy = float(np.clip(muA_val * 0.70 + (1.0 - near_norm) * 0.30, 0.0, 1.0))
        reward = build_reward_from_info(info, sigma_bar=sigma_proxy, cfg=self.cfg)
        info["sigma_proxy"] = float(sigma_proxy)
        info["env_reward_proxy"] = float(reward)

        if self.cfg.debug_mode and (self.episode_steps == 1 or self.episode_steps % self.cfg.debug_step_freq == 0 or terminated or truncated):
            vel = self.vehicle.get_velocity()
            speed_kmh = 3.6 * vec3_length(vel)
            print(
                f"[STEP] step={self.episode_steps:04d} speed={speed_kmh:6.2f}km/h "
                f"thr={applied[0]:.2f} brk={applied[1]:.2f} str={applied[2]:+.2f} safe={int(bool(shield_active))} "
                f"progress={route_completion_pct:.1f}% goal={remaining_s:.2f}m dL={dL_after:.2f} "
                f"ttc={safe_float(info.get('time_to_conflict', 999.0)):.2f}s tl={info.get('tl_state', 'None')} "
                f"credit={int(self.blocked_steps_credit)} reason={reason or ''}"
            )
        if terminated or truncated:
            self._begin_episode_teardown(reason or ("timeout" if truncated else "terminated"))    
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self._cleanup_episode_actors()
        except BaseException:
            pass
        try:
            self._restore_world_settings()
        except BaseException:
            pass


# ======================================================
# Networks / Agent
# ======================================================
class GraphAttention(nn.Module):
    """Reliability-aware ego-edge aggregation (paper Sec. III-C).

    The printed ratio in Eq. (6) increases attention when variance increases,
    while the surrounding text and Table I require nearby *reliable* entities
    to dominate. This implementation follows that stated behavior explicitly:

        eta_i = -lambda_d ||delta p_i||^2 - lambda_sigma sigma_i^2
        alpha_i = softmax_i(eta_i),  z_t = sum_i alpha_i W_e e_i.

    Edge variance is supplied by the observation model and is not freely
    inflated by the policy network.
    """

    def __init__(self, edge_dim: int, z_dim: int = 64, cfg: Config = CFG):
        super().__init__()
        self.we = nn.Linear(edge_dim, z_dim)
        # Retained for checkpoint compatibility with older files; not used to
        # redefine observation uncertainty.
        self.sigma_head = nn.Sequential(nn.Linear(edge_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.score_form = str(getattr(cfg, "attention_score_form", "reliability")).strip().lower()
        self.distance_weight = float(getattr(cfg, "attention_distance_weight", 4.0))
        self.uncertainty_weight = float(getattr(cfg, "attention_uncertainty_weight", 2.0))
        self.ratio_eps = float(getattr(cfg, "attention_ratio_eps", 1e-6))
        self.use_uncertainty_attention = bool(
            getattr(cfg, "use_uncertainty_attention", True)
        )

    def forward(self, edges: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma2 = edges[..., -1].clamp(min=1e-3, max=2.0)
        dp2 = edges[..., 0:2].pow(2).sum(dim=-1)
        if not self.use_uncertainty_attention:
            # Matched ablation: retain relational, distance-aware aggregation
            # but remove variance from the attention logits.  Keeping the
            # architecture unchanged makes parameter count and checkpoint
            # structure comparable with the full model.
            logits = -self.distance_weight * dp2
        elif self.score_form == "paper_ratio":
            # Eq. (6) exactly as printed. Note this raises attention for higher
            # variance edges; see deviation D1 in the module docstring.
            logits = -dp2 / (sigma2 + self.ratio_eps)
        else:
            # Behavior stated in the text below Eq. (6) and in Table I:
            # nearby AND reliable entities dominate.
            logits = -self.distance_weight * dp2 - self.uncertainty_weight * sigma2
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(mask > 0.5, logits, torch.full_like(logits, neg_inf))
        all_masked = mask.sum(dim=1, keepdim=True) < 0.5
        if all_masked.any():
            logits = logits.clone()
            logits[all_masked.squeeze(1), 0] = 0.0
        alpha = torch.softmax(logits, dim=1)
        alpha = alpha * (mask > 0.5).float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-6)
        z = torch.sum(alpha.unsqueeze(-1) * self.we(edges), dim=1)
        sigma_ale = torch.sum(alpha * sigma2, dim=1)
        return z, alpha, sigma_ale


class CompactStateEncoder(nn.Module):
    """Ego-centric relational state encoder (paper Sec. III-C, Eqs. 5–7).

    s_t = [z_t; v_ego; a^{t-1}_ego; d_goal; φ_lane; σ²_ale]   (Eq. 7)

    z_t is the uncertainty-weighted interaction embedding (Eq. 6).
    σ²_ale is appended so the actor and critic see raw aleatoric variance.
    """

    def __init__(self, cfg: Config = CFG, z_dim: int = 64, scalar_dim: Optional[int] = None, out_dim: int = 256):
        super().__init__()
        scalar_dim = int(cfg.scalar_dim if scalar_dim is None else scalar_dim)
        self.graph = GraphAttention(edge_dim=cfg.edge_dim, z_dim=z_dim, cfg=cfg)
        self.mlp_scal = nn.Sequential(
            nn.Linear(scalar_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(z_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def forward(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, alpha, sigma_ale = self.graph(edges, mask)
        # Append σ²_ale to scalars (Eq. 7: σ²_ale term in s_t)
        scal_plus = torch.cat([scalars, sigma_ale.unsqueeze(-1)], dim=-1)
        s = self.mlp_scal(scal_plus)
        h = self.fuse(torch.cat([z, s], dim=-1))
        return h, alpha, sigma_ale


class Actor(nn.Module):
    def __init__(self, enc: CompactStateEncoder, action_dim: int = 3):
        super().__init__()
        self.enc = enc
        self.mu = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.logstd = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        # New policies begin in a gentle forward/coast regime rather than the
        # pathological 0.5-throttle/0.5-brake midpoint of an uninitialized
        # squashed Gaussian. Checkpoint loading overwrites these values.
        with torch.no_grad():
            self.mu[-1].bias.copy_(torch.tensor([-0.70, -2.20, 0.0], dtype=torch.float32))
            self.logstd[-1].bias.fill_(-1.0)
        self.register_buffer("action_scale", torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32))

    def forward(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor):
        h, alpha, sigma_ale = self.enc(scalars, edges, mask)
        mu = self.mu(h)
        logstd = self.logstd(h).clamp(-5.0, 1.0)
        return mu, logstd, alpha, sigma_ale

    def _squash_action(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tanh_u = torch.tanh(u)
        action = tanh_u * self.action_scale + self.action_bias
        log_scale = torch.log(self.action_scale.clamp(min=1e-6)).sum()
        log_det_tanh = torch.log((1.0 - tanh_u.pow(2)).clamp(min=1e-6)).sum(dim=-1)
        log_abs_det = log_scale + log_det_tanh
        return action, log_abs_det

    def sample(
        self,
        scalars: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd, alpha, sigma_ale = self.forward(scalars, edges, mask)
        std = logstd.exp()
        normal = torch.distributions.Normal(mu, std)
        u = mu if deterministic else normal.rsample()
        action, log_abs_det = self._squash_action(u)
        log_pi: Optional[torch.Tensor] = None
        if with_logprob:
            log_pi = normal.log_prob(u).sum(dim=-1) - log_abs_det
        return action, log_pi, mu, logstd, alpha, sigma_ale

    def act_deterministic(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        action, _, _, _, _, _ = self.sample(scalars, edges, mask, deterministic=True, with_logprob=False)
        return action


class Critic(nn.Module):
    def __init__(self, enc: CompactStateEncoder):
        super().__init__()
        self.enc = enc
        self.q = nn.Sequential(nn.Linear(256 + 3, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h, _, _ = self.enc(scalars, edges, mask)
        return self.q(torch.cat([h, action], dim=-1)).squeeze(-1)

class ReplayBuffer:
    def __init__(self, capacity: int, cfg: Config = CFG):
        self.capacity = int(capacity)
        self.cfg = cfg
        self.ptr = 0
        self.size = 0
        self.scalars = np.zeros((self.capacity, cfg.scalar_dim), dtype=np.float32)
        self.edges = np.zeros((self.capacity, cfg.max_entities, cfg.edge_dim), dtype=np.float32)
        self.mask = np.zeros((self.capacity, cfg.max_entities), dtype=np.float32)
        self.actions = np.zeros((self.capacity, cfg.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_scalars = np.zeros((self.capacity, cfg.scalar_dim), dtype=np.float32)
        self.next_edges = np.zeros((self.capacity, cfg.max_entities, cfg.edge_dim), dtype=np.float32)
        self.next_mask = np.zeros((self.capacity, cfg.max_entities), dtype=np.float32)
        self.terminals = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        terminal: bool,
    ) -> None:
        i = self.ptr
        self.scalars[i] = obs["scalars"]
        self.edges[i] = obs["edges"]
        self.mask[i] = obs["mask"]
        self.actions[i] = action
        self.rewards[i] = float(reward)
        self.next_scalars[i] = next_obs["scalars"]
        self.next_edges[i] = next_obs["edges"]
        self.next_mask[i] = next_obs["mask"]
        # `terminal` follows the route-task semantics set by the environment.
        # Collision, off-route, goal, stuck, and (by default) the fixed route
        # deadline suppress bootstrapping.  Only a genuine external truncation
        # retains a continuation value.
        self.terminals[i] = float(terminal)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("ReplayBuffer is empty.")
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "scalars": torch.from_numpy(self.scalars[idx]).to(device).float(),
            "edges": torch.from_numpy(self.edges[idx]).to(device).float(),
            "mask": torch.from_numpy(self.mask[idx]).to(device).float(),
            "actions": torch.from_numpy(self.actions[idx]).to(device).float(),
            "rewards": torch.from_numpy(self.rewards[idx]).to(device).float(),
            "next_scalars": torch.from_numpy(self.next_scalars[idx]).to(device).float(),
            "next_edges": torch.from_numpy(self.next_edges[idx]).to(device).float(),
            "next_mask": torch.from_numpy(self.next_mask[idx]).to(device).float(),
            "terminals": torch.from_numpy(self.terminals[idx]).to(device).float(),
        }


class UncertaintyCalibrator:
    def __init__(self, temperature: float = 1.0):
        self.center = 0.5
        self.scale = 0.25
        self.min_val = 0.0
        self.max_val = 1.0
        self.temperature = float(max(temperature, 1e-3))
        self.fitted = False

    def fit_from_values(self, values: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "fitted": 0.0,
                "center": float(self.center),
                "scale": float(self.scale),
                "min_val": float(self.min_val),
                "max_val": float(self.max_val),
            }
        q_lo = float(np.clip(q_lo, 0.0, 1.0))
        q_hi = float(np.clip(q_hi, q_lo + 1e-6, 1.0))
        lo = float(np.quantile(arr, q_lo))
        hi = float(np.quantile(arr, q_hi))
        if not np.isfinite(lo):
            lo = float(np.min(arr))
        if not np.isfinite(hi):
            hi = float(np.max(arr))
        if hi <= lo:
            hi = lo + 1e-3
        arr_mm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        center = float(np.median(arr_mm))
        mad = float(np.median(np.abs(arr_mm - center)))
        scale = max(1.4826 * mad, 0.05)
        self.center = center
        self.scale = scale
        self.min_val = lo
        self.max_val = hi
        self.fitted = True
        return {
            "fitted": 1.0,
            "center": self.center,
            "scale": self.scale,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.fitted:
            x = (x - self.min_val) / max(self.max_val - self.min_val, 1e-6)
            x = x.clamp(0.0, 1.0)
        z = (x - self.center) / max(self.scale * self.temperature, 1e-6)
        return torch.sigmoid(z)

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not state:
            return
        self.center = float(state.get("center", self.center))
        self.scale = float(max(float(state.get("scale", self.scale)), 1e-6))
        self.min_val = float(state.get("min_val", self.min_val))
        self.max_val = float(max(float(state.get("max_val", self.max_val)), self.min_val + 1e-6))
        self.temperature = float(max(float(state.get("temperature", self.temperature)), 1e-6))
        self.fitted = bool(state.get("fitted", self.fitted))

    def state_dict(self) -> Dict[str, object]:
        return {
            "center": float(self.center),
            "scale": float(self.scale),
            "min_val": float(self.min_val),
            "max_val": float(self.max_val),
            "temperature": float(self.temperature),
            "fitted": bool(self.fitted),
        }


class SACAgent:
    def __init__(self, device: torch.device, cfg: Config = CFG):
        self.device = device
        self.cfg = cfg
        self.actor = Actor(CompactStateEncoder(cfg=cfg).to(device), action_dim=cfg.action_dim).to(device)
        self.critics = nn.ModuleList([Critic(CompactStateEncoder(cfg=cfg).to(device)).to(device) for _ in range(cfg.n_critics)])
        self.target_critics = nn.ModuleList([Critic(CompactStateEncoder(cfg=cfg).to(device)).to(device) for _ in range(cfg.n_critics)])
        for k in range(cfg.n_critics):
            self.target_critics[k].load_state_dict(self.critics[k].state_dict())
        self.sigma_cal = UncertaintyCalibrator(temperature=cfg.calib_temperature)

        critic_params: List[nn.Parameter] = []
        for critic in self.critics:
            critic_params.extend(list(critic.parameters()))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(critic_params, lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(math.log(max(cfg.init_alpha, 1e-6)), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        # The policy density includes the affine action scaling Jacobian, so
        # its entropy target must use the same coordinates as log_pi.
        self.target_entropy = -float(cfg.action_dim) + float(
            torch.log(self.actor.action_scale).sum().detach().cpu().item()
        )
        self.training_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def entropy_coefficient(self, sigma_bar: torch.Tensor) -> torch.Tensor:
        # Paper Eq. (11) contributes a fixed uncertainty-gated term.  Learned
        # SAC alpha is added independently so its optimizer controls the same
        # coefficient that appears in both actor and critic targets.
        if bool(getattr(self.cfg, "use_entropy_gate", True)):
            beta = self.cfg.lambda_ent * self.cfg.beta0 * (1.0 - sigma_bar)
        else:
            # Entropy-gate ablation: standard SAC temperature without any
            # uncertainty-dependent multiplier.
            beta = torch.zeros_like(sigma_bar)
        if bool(self.cfg.use_learned_alpha):
            beta = beta + self.alpha.detach()
        return beta

    def train(self) -> None:
        self.actor.train()
        self.critics.train()
        self.target_critics.train()

    @torch.no_grad()
    def act(self, obs: Dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        scalars = torch.from_numpy(obs["scalars"][None]).to(self.device).float()
        edges = torch.from_numpy(obs["edges"][None]).to(self.device).float()
        mask = torch.from_numpy(obs["mask"][None]).to(self.device).float()
        if deterministic:
            action = self.actor.act_deterministic(scalars, edges, mask)
        else:
            action, _, _, _, _, _ = self.actor.sample(scalars, edges, mask, deterministic=False, with_logprob=False)
        return action[0].detach().cpu().numpy().astype(np.float32)

    def critic_ensemble(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor, action: torch.Tensor, target: bool = False) -> torch.Tensor:
        critics = self.target_critics if target else self.critics
        qs = [critics[k](scalars, edges, mask, action) for k in range(self.cfg.n_critics)]
        return torch.stack(qs, dim=0)

    @torch.no_grad()
    def critic_ensemble_nograd(self, scalars, edges, mask, action, target: bool = False) -> torch.Tensor:
        return self.critic_ensemble(scalars, edges, mask, action, target=target)

    def compute_sigma_dec(self, sigma_ale: torch.Tensor, qs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Eq. (10): σ²_dec = σ²_ale + σ²_epi, with σ²_epi = Var_k[Q_φk(s, a)].
        sigma_epi = qs.var(dim=0, unbiased=False)
        sigma_dec = sigma_ale + sigma_epi
        return sigma_dec, sigma_epi

    def compute_sigma_bar(self, sigma_ale: torch.Tensor, qs: torch.Tensor, calibrator: Optional[UncertaintyCalibrator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_dec, sigma_epi = self.compute_sigma_dec(sigma_ale, qs)
        cal = self.sigma_cal if calibrator is None else calibrator
        sigma_bar = cal.transform_tensor(sigma_dec)
        return sigma_bar, sigma_epi

    def uncertainty_features(self, sigma_bar: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        edge_var = edges[..., -1]
        weight = mask.float()
        denom = weight.sum(dim=-1).clamp(min=1.0)
        mean_edge = (edge_var * weight).sum(dim=-1) / denom
        centered = (edge_var - mean_edge.unsqueeze(-1)) * weight
        # Sec. III-F: u = [σ̄, mean_i(σ²_i), var_i(σ²_i)] — variance, not std.
        var_edge = (centered.pow(2).sum(dim=-1) / denom).clamp(min=0.0)
        return torch.stack([sigma_bar, mean_edge, var_edge], dim=-1)

    def _actor_rl_loss_for_actor(self, actor: "Actor", batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        action, log_pi, _, _, _, sigma_ale = actor.sample(batch["scalars"], batch["edges"], batch["mask"], deterministic=False, with_logprob=True)
        # Preserve gradients from Q(s,a) to the candidate actor action.
        qs = self.critic_ensemble(batch["scalars"], batch["edges"], batch["mask"], action)
        min_q = qs.min(dim=0)[0]
        sigma_bar, _ = self.compute_sigma_bar(sigma_ale, qs.detach())
        beta = self.entropy_coefficient(sigma_bar.detach())
        return (beta * log_pi - min_q).mean()

    @staticmethod
    def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
        x = x.float()
        y = y.float()
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        zz = torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * zz
        bandwidth = 1.0
        XX = torch.zeros_like(xx)
        YY = torch.zeros_like(yy)
        XY = torch.zeros_like(zz)
        for _ in range(kernel_num):
            XX = XX + torch.exp(-dxx / bandwidth)
            YY = YY + torch.exp(-dyy / bandwidth)
            XY = XY + torch.exp(-dxy / bandwidth)
            bandwidth *= kernel_mul
        return (XX.mean() + YY.mean() - 2.0 * XY.mean()).clamp(min=0.0)

    def compute_actor_loss(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Uncertainty-gated SAC actor objective (paper Eqs. 11--12)."""
        action, log_pi, _, _, _, sigma_ale = self.actor.sample(
            scalars, edges, mask, deterministic=False, with_logprob=True
        )
        qs = self.critic_ensemble(scalars, edges, mask, action, target=False)
        min_q = qs.min(dim=0)[0]
        sigma_bar, _ = self.compute_sigma_bar(sigma_ale, qs.detach())
        beta = self.entropy_coefficient(sigma_bar.detach())
        actor_loss = (beta * log_pi - min_q).mean()
        stats = {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "log_pi": float(log_pi.detach().mean().cpu().item()),
            "sigma_bar": float(sigma_bar.detach().mean().cpu().item()),
            "beta": float(beta.detach().mean().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()) if self.cfg.use_learned_alpha else 0.0,
            "temp_eff": float(beta.detach().mean().cpu().item()),
        }
        return actor_loss, stats

    def compute_transfer_loss(
        self,
        source_agent: "SACAgent",
        scalars: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        source_scalars: Optional[torch.Tensor] = None,
        source_edges: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Influence-consistent transfer loss (paper Eq. 13).

        L_trans = KL(π_θs(·|s) ‖ π_θt(·|s)) + λ_α MMD(α_s, α_t) + λ_u ‖u_s − u_t‖²

        KL    : divergence between source and target action distributions
        MMD   : Maximum Mean Discrepancy on relational attention vectors (α, Eq. 6)
        ‖u‖²  : MSE on uncertainty moment vectors u = [σ̄, mean_i(σ²_i), var_i(σ²_i)]

        Combined with MAML initialisation (Eq. 14) in maml_style_initialize().
        Full objective (Eq. 15): L_RL + λ_ent L_ent + λ_T L_trans
        """
        s_scalars = scalars if source_scalars is None else source_scalars
        s_edges = edges if source_edges is None else source_edges
        s_mask = mask if source_mask is None else source_mask

        # KL is computed on the same target states: the frozen source policy
        # acts as a behavioral teacher, while MMD and uncertainty moments align
        # the source and target domain distributions.
        mu_t, logstd_t, alpha_t, sigma_ale_t = self.actor.forward(scalars, edges, mask)
        with torch.no_grad():
            mu_s_on_t, logstd_s_on_t, _, _ = source_agent.actor.forward(scalars, edges, mask)
            _, _, alpha_s_domain, sigma_ale_s = source_agent.actor.forward(s_scalars, s_edges, s_mask)
            action_s, _, _, _, _, _ = source_agent.actor.sample(
                s_scalars, s_edges, s_mask, deterministic=True, with_logprob=False
            )
            qs_s = source_agent.critic_ensemble_nograd(s_scalars, s_edges, s_mask, action_s)
            sigma_bar_s, _ = source_agent.compute_sigma_bar(sigma_ale_s, qs_s)
            u_s = source_agent.uncertainty_features(sigma_bar_s, s_edges, s_mask)

        dist_t = torch.distributions.Normal(mu_t, logstd_t.exp())
        dist_s = torch.distributions.Normal(mu_s_on_t, logstd_s_on_t.exp())
        action_t, _, _, _, _, _ = self.actor.sample(
            scalars, edges, mask, deterministic=True, with_logprob=False
        )
        qs_t = self.critic_ensemble(scalars, edges, mask, action_t)
        sigma_bar_t, _ = self.compute_sigma_bar(sigma_ale_t, qs_t.detach())
        u_t = self.uncertainty_features(sigma_bar_t, edges, mask)

        kl_loss = torch.distributions.kl.kl_divergence(dist_s, dist_t).sum(dim=-1).mean()
        mmd_loss = self.compute_mmd(
            alpha_s_domain.reshape(alpha_s_domain.shape[0], -1),
            alpha_t.reshape(alpha_t.shape[0], -1),
        )
        u_loss = F.mse_loss(u_t.mean(dim=0), u_s.detach().mean(dim=0))
        total = kl_loss + self.cfg.lambda_alpha_align * mmd_loss + self.cfg.lambda_u_align * u_loss
        stats = {
            "transfer_loss": float(total.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "mmd_loss": float(mmd_loss.detach().cpu().item()),
            "u_loss": float(u_loss.detach().cpu().item()),
        }
        return total, stats

    def compute_critic_uncertainty_alignment(
        self,
        source_agent: "SACAgent",
        target_batch: Dict[str, torch.Tensor],
        source_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Critic-side uncertainty moment term for the joint Eq. (15) update."""
        with torch.no_grad():
            action_s, _, _, _, _, sigma_ale_s = source_agent.actor.sample(
                source_batch["scalars"], source_batch["edges"], source_batch["mask"],
                deterministic=True, with_logprob=False,
            )
            qs_s = source_agent.critic_ensemble_nograd(
                source_batch["scalars"], source_batch["edges"], source_batch["mask"], action_s
            )
            sigma_bar_s, _ = source_agent.compute_sigma_bar(sigma_ale_s, qs_s)
            u_s = source_agent.uncertainty_features(
                sigma_bar_s, source_batch["edges"], source_batch["mask"]
            ).mean(dim=0)
            action_t, _, _, _, _, sigma_ale_t = self.actor.sample(
                target_batch["scalars"], target_batch["edges"], target_batch["mask"],
                deterministic=True, with_logprob=False,
            )
        qs_t = self.critic_ensemble(
            target_batch["scalars"], target_batch["edges"], target_batch["mask"], action_t.detach()
        )
        sigma_bar_t, _ = self.compute_sigma_bar(sigma_ale_t.detach(), qs_t)
        u_t = self.uncertainty_features(
            sigma_bar_t, target_batch["edges"], target_batch["mask"]
        ).mean(dim=0)
        return F.mse_loss(u_t, u_s.detach())

    def soft_update_targets(self) -> None:
        tau = float(self.cfg.target_tau)
        with torch.no_grad():
            for critic, target in zip(self.critics, self.target_critics):
                for p, tp in zip(critic.parameters(), target.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        source_agent: Optional["SACAgent"] = None,
        source_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        self.train()
        scalars = batch["scalars"]
        edges = batch["edges"]
        mask = batch["mask"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_scalars = batch["next_scalars"]
        next_edges = batch["next_edges"]
        next_mask = batch["next_mask"]
        terminals = batch["terminals"]

        with torch.no_grad():
            next_action, next_log_pi, _, _, _, next_sigma_ale = self.actor.sample(next_scalars, next_edges, next_mask, deterministic=False, with_logprob=True)
            next_qs = self.critic_ensemble(next_scalars, next_edges, next_mask, next_action, target=True)
            next_q_min = next_qs.min(dim=0)[0]
            next_sigma_bar, _ = self.compute_sigma_bar(next_sigma_ale, next_qs)
            beta_next = self.entropy_coefficient(next_sigma_bar)
            target_q = rewards + (1.0 - terminals) * self.cfg.gamma * (
                next_q_min - beta_next * next_log_pi
            )

        current_qs = self.critic_ensemble(scalars, edges, mask, actions, target=False)
        critic_terms: List[torch.Tensor] = []
        for k in range(self.cfg.n_critics):
            # Bootstrap resampling maintains useful ensemble diversity for
            # epistemic variance instead of training identical critics.
            boot = (torch.rand_like(target_q) < 0.80).float()
            denom = boot.sum().clamp(min=1.0)
            critic_terms.append((((current_qs[k] - target_q).pow(2)) * boot).sum() / denom)
        critic_loss = torch.stack(critic_terms).mean()
        critic_u_loss = torch.zeros((), device=self.device)
        if (
            bool(getattr(self.cfg, "use_transfer_alignment", True))
            and source_agent is not None
            and source_batch is not None
        ):
            critic_u_loss = self.compute_critic_uncertainty_alignment(source_agent, batch, source_batch)
            critic_loss = critic_loss + self.cfg.lambda_transfer * self.cfg.lambda_u_align * critic_u_loss
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critics.parameters()), 10.0)
        self.critic_opt.step()

        actor_loss, actor_stats = self.compute_actor_loss(scalars, edges, mask)
        total_actor_loss = actor_loss
        transfer_stats: Dict[str, float] = {}
        if (
            bool(getattr(self.cfg, "use_transfer_alignment", True))
            and source_agent is not None
        ):
            transfer_loss, transfer_stats = self.compute_transfer_loss(
                source_agent,
                scalars,
                edges,
                mask,
                None if source_batch is None else source_batch["scalars"],
                None if source_batch is None else source_batch["edges"],
                None if source_batch is None else source_batch["mask"],
            )
            total_actor_loss = total_actor_loss + self.cfg.lambda_transfer * transfer_loss

        self.actor_opt.zero_grad(set_to_none=True)
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        alpha_loss = torch.zeros((), device=self.device)
        if bool(self.cfg.use_learned_alpha):
            with torch.no_grad():
                _, log_pi_alpha, _, _, _, _ = self.actor.sample(
                    scalars, edges, mask, deterministic=False, with_logprob=True
                )
            alpha_loss = -(self.log_alpha * (log_pi_alpha + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=math.log(0.01), max=math.log(2.0))

        self.soft_update_targets()
        self.training_steps += 1

        stats = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "critic_u_loss": float(critic_u_loss.detach().cpu().item()),
        }
        stats.update(actor_stats)
        stats.update(transfer_stats)
        return stats

    def maml_style_initialize(self, domain_batches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """MAML initialization corresponding to paper Eq. (14).

        Each domain batch is split into support/query sets. Fast parameters take
        inner RL steps on support data and query gradients at the adapted
        parameters are accumulated onto the shared actor.

        With cfg.maml_first_order = True (default, deviation D2) second
        derivatives are omitted, which is the standard first-order
        approximation. Setting it to False keeps the inner-step graph so the
        exact meta-gradient of Eq. (14) is used, at higher memory cost.
        """
        if not bool(getattr(self.cfg, "use_maml_initialization", True)):
            return {
                "maml_batches": 0.0,
                "maml_inner_loss": 0.0,
                "maml_query_loss": 0.0,
                "maml_disabled": 1.0,
            }
        if not domain_batches:
            return {"maml_batches": 0.0, "maml_inner_loss": 0.0, "maml_query_loss": 0.0}
        base_state = {k: v.detach().clone() for k, v in self.actor.state_dict().items()}
        meta_grads = [torch.zeros_like(p) for p in self.actor.parameters()]
        inner_losses: List[float] = []
        query_losses: List[float] = []
        used = 0
        for batch in domain_batches:
            bs = int(batch["scalars"].shape[0])
            split = max(1, bs // 2)
            support = {k: v[:split] for k, v in batch.items()}
            query = {k: v[split:] for k, v in batch.items()}
            if int(query["scalars"].shape[0]) == 0:
                query = support
            fast_actor = Actor(
                CompactStateEncoder(cfg=self.cfg).to(self.device),
                action_dim=self.cfg.action_dim,
            ).to(self.device)
            fast_actor.load_state_dict(base_state)
            support_loss_val = 0.0
            for _ in range(max(1, int(self.cfg.maml_inner_steps))):
                loss = self._actor_rl_loss_for_actor(fast_actor, support)
                fast_params = tuple(fast_actor.parameters())
                grads = torch.autograd.grad(loss, fast_params, allow_unused=True)
                with torch.no_grad():
                    for param, grad in zip(fast_params, grads):
                        if grad is not None:
                            param.add_(grad, alpha=-float(self.cfg.maml_inner_lr))
                support_loss_val = float(loss.detach().cpu().item())
            query_loss = self._actor_rl_loss_for_actor(fast_actor, query)
            fast_params = tuple(fast_actor.parameters())
            qgrads = torch.autograd.grad(query_loss, fast_params, allow_unused=True)
            for acc, grad in zip(meta_grads, qgrads):
                if grad is not None:
                    acc.add_(grad.detach())
            inner_losses.append(support_loss_val)
            query_losses.append(float(query_loss.detach().cpu().item()))
            used += 1
        if used > 0:
            self.actor_opt.zero_grad(set_to_none=True)
            scale = float(self.cfg.maml_meta_step_size) / float(used)
            for param, grad in zip(self.actor.parameters(), meta_grads):
                param.grad = grad * scale
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()
        return {
            "maml_batches": float(used),
            "maml_inner_loss": float(np.mean(inner_losses)) if inner_losses else 0.0,
            "maml_query_loss": float(np.mean(query_losses)) if query_losses else 0.0,
        }

    def save(self, path: str) -> None:
        ckpt = {
            "actor": self.actor.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
            "target_critics": [c.state_dict() for c in self.target_critics],
            "sigma_cal": self.sigma_cal.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "alpha_opt": self.alpha_opt.state_dict(),
            "training_steps": int(self.training_steps),
            "cfg": self.cfg.__dict__.copy(),
        }
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        load_module_state_compat(self.actor, ckpt["actor"])
        for k, sd in enumerate(ckpt.get("critics", [])):
            if k < len(self.critics):
                load_module_state_compat(self.critics[k], sd)
        target_sds = ckpt.get("target_critics", [])
        if target_sds:
            for k, sd in enumerate(target_sds):
                if k < len(self.target_critics):
                    load_module_state_compat(self.target_critics[k], sd)
        else:
            for k in range(self.cfg.n_critics):
                self.target_critics[k].load_state_dict(self.critics[k].state_dict())
        self.sigma_cal.load_state_dict(ckpt.get("sigma_cal", {}))
        if "actor_opt" in ckpt:
            try:
                self.actor_opt.load_state_dict(ckpt["actor_opt"])
            except Exception:
                pass
        if "critic_opt" in ckpt:
            try:
                self.critic_opt.load_state_dict(ckpt["critic_opt"])
            except Exception:
                pass
        if "alpha_opt" in ckpt:
            try:
                self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
            except Exception:
                pass
        if "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(float(ckpt["log_alpha"]), device=self.device))
        saved_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        if isinstance(saved_cfg, dict):
            for key in ["beta0", "lambda_ent", "lambda_transfer", "lambda_alpha_align", "lambda_u_align", "calib_temperature"]:
                if key in saved_cfg and hasattr(self.cfg, key):
                    setattr(self.cfg, key, type(getattr(self.cfg, key))(saved_cfg[key]))
        self.training_steps = int(ckpt.get("training_steps", 0))


# ======================================================
# Reward / Metrics
# ======================================================
def build_reward_from_info(info: Dict[str, object], sigma_bar: float, cfg: Config = CFG) -> float:
    """Dense/event reward with shared task-level terminal semantics.

    r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u    (Eq. 8)
    r_u = 1 − σ̄                                     (Eq. 9)

    A progress gate is applied only to the *positive* safety/uncertainty
    portions.  Negative safety, constraint and comfort terms remain active.
    This prevents the agent from collecting positive return by remaining
    stationary on an empty, lane-centred road.  The same task-level terminal
    rewards are used by all ablations, including event_reward.
    """
    term_reason = str(info.get("term_reason", ""))
    terminal_adjustment = 0.0
    if bool(info.get("success", False)) or bool(info.get("goal_reached", False)):
        terminal_adjustment += float(cfg.goal_terminal_reward)
    elif bool(info.get("collision", False)):
        terminal_adjustment += float(cfg.collision_terminal_reward)
    elif bool(info.get("off_route", info.get("off_road", False))):
        terminal_adjustment += float(cfg.offroute_terminal_reward)
    elif term_reason == "stuck" or bool(info.get("stuck", False)):
        terminal_adjustment += float(cfg.stuck_terminal_reward)
    elif term_reason == "timeout" or bool(info.get("timeout", False)):
        terminal_adjustment += float(cfg.timeout_terminal_reward)

    if not bool(getattr(cfg, "use_dense_reward", True)):
        # Sparse/event-driven arm: remove all dense components while keeping
        # the identical terminal task definition used by the full model.
        return float(cfg.event_step_reward + terminal_adjustment)

    rs = safe_float(info.get("rs", 0.0))
    rp = safe_float(info.get("rp", 0.0))
    rc = safe_float(info.get("rc", 0.0))
    ru = 1.0 - float(np.clip(sigma_bar, 0.0, 1.0))   # Eq. (9)
    delta_s = safe_float(info.get("delta_s", 0.0))
    progress_gate = 1.0
    if bool(cfg.progress_gate_positive_reward):
        progress_gate = float(np.clip(
            max(delta_s, 0.0) / max(float(cfg.progress_gate_distance_m), 1e-6),
            0.0,
            1.0,
        ))
    rs_effective = min(rs, 0.0) + progress_gate * max(rs, 0.0)
    ru_effective = progress_gate * ru
    r_dense = float(
        cfg.w_s * rs_effective
        + cfg.w_p * rp
        + cfg.w_c * rc
        + cfg.w_u * ru_effective
    )
    # Eq. (4): the constrained objective subtracts the smooth penalty vector
    # lambda^T phi_t from the dense reward. phi_lane carries phi_L(d_L, mu_A)
    # together with the wrong-lane / opposite-lane components. Eq. (8) and its
    # weight normalisation are untouched.
    phi_lane = safe_float(info.get("phi_lane", 0.0))
    idle_penalty = 0.0
    is_terminal = bool(term_reason)
    if (
        not is_terminal
        and not bool(info.get("blocked_wait", False))
        and delta_s <= float(cfg.open_road_idle_delta_s_m)
    ):
        idle_penalty = float(cfg.open_road_idle_penalty)
    return float(r_dense - phi_lane - idle_penalty + terminal_adjustment)


@torch.no_grad()
def infer_sigma_bar(agent: SACAgent, obs: Dict[str, np.ndarray], action: np.ndarray) -> float:
    scalars = torch.from_numpy(obs["scalars"][None]).to(agent.device).float()
    edges = torch.from_numpy(obs["edges"][None]).to(agent.device).float()
    mask = torch.from_numpy(obs["mask"][None]).to(agent.device).float()
    act = torch.from_numpy(action[None]).to(agent.device).float()
    _, _, _, sigma_ale = agent.actor.forward(scalars, edges, mask)
    qs = agent.critic_ensemble_nograd(scalars, edges, mask, act)
    sigma_bar, _ = agent.compute_sigma_bar(sigma_ale.reshape(1), qs.reshape(agent.cfg.n_critics, 1))
    return float(sigma_bar.reshape(-1)[0].cpu().item())


def recompute_agent_reward(agent: SACAgent, obs: Dict[str, np.ndarray], action: np.ndarray, info: Dict[str, object], cfg: Config = CFG) -> Tuple[float, float]:
    sigma_bar = infer_sigma_bar(agent, obs, action)
    reward = build_reward_from_info(info, sigma_bar, cfg=cfg)
    return reward, sigma_bar


def robust_reset(
    env: CarlaReliableTransferEnv,
    env_builder: Callable[[], CarlaReliableTransferEnv],
    npc_count: int,
    goal_index: int,
    walker_count: Optional[int] = None,
    max_tries: int = 5,
) -> Tuple[CarlaReliableTransferEnv, Dict[str, np.ndarray], Dict[str, object]]:
    last_err: Optional[Exception] = None
    cur_env = env
    sleep_secs = [2.0, 4.0, 8.0, 12.0, 20.0]
    for attempt in range(1, max_tries + 1):
        try:
            reset_options: Dict[str, object] = {"npc_count": int(npc_count), "goal_index": int(goal_index)}
            if walker_count is not None:
                reset_options["walker_count"] = int(walker_count)
            obs, info = cur_env.reset(options=reset_options)
            return cur_env, obs, info
        except BaseException as e:
            last_err = Exception(str(e))
            print(f"[WARN] reset attempt {attempt}/{max_tries} failed: {e}")
        try:
            cur_env.close()
        except BaseException:
            pass
        if attempt >= max_tries:
            break
        wait = sleep_secs[min(attempt - 1, len(sleep_secs) - 1)]
        print(f"[WARN] waiting {wait:.0f}s before rebuild (attempt {attempt+1}/{max_tries})")
        time.sleep(wait)
        try:
            cur_env = env_builder()
        except BaseException as e2:
            print(f"[WARN] env_builder() failed: {e2}")
            last_err = Exception(str(e2))
    raise RuntimeError(f"robust_reset failed after {max_tries} attempts. Last: {last_err}")


# ======================================================
# Training / Adaptation helpers
# ======================================================
def wait_for_carla_ready(host: str, port: int, timeout_s: float = 60.0, poll_interval: float = 2.0) -> None:
    """Wait for a successful CARLA RPC, not merely an open TCP socket."""
    deadline = time.monotonic() + timeout_s
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            probe = carla.Client(host, int(port))
            probe.set_timeout(min(5.0, max(2.0, float(poll_interval) * 2.0)))
            probe.get_server_version()
            probe.get_world().get_map().name
            return
        except BaseException as exc:
            last_error = exc
        time.sleep(poll_interval)
    raise TimeoutError(
        f"CARLA at {host}:{port} did not answer RPC within {timeout_s:.1f}s; "
        f"last error: {last_error}"
    )


def make_target_cfg(base_cfg: Config) -> Config:
    """Copy the source configuration without changing the route protocol.

    Source training and target adaptation use the same arc-length target and
    tolerance. A target goal index of -1 invokes
    automatic route construction; an explicit fixed goal is accepted only
    when its planned route satisfies the same length window.
    """
    import copy
    cfg = copy.copy(base_cfg)
    cfg.strict_goal_route = False
    cfg.allow_fallback_route = True
    cfg.enforce_route_length_for_fixed_goal = True
    cfg.use_fixed_destination = int(base_cfg.target_goal_index) >= 0
    cfg.target_goal_index = int(base_cfg.target_goal_index)
    return cfg


def make_env_builder(
    host: str,
    port: int,
    town_name: str,
    spawn_index: int,
    goal_index: int,
    weather_mode: str,
    cfg: Config,
) -> Callable[[], CarlaReliableTransferEnv]:
    return lambda: CarlaReliableTransferEnv(
        host=host,
        port=port,
        town_name=town_name,
        fixed_spawn_index=spawn_index,
        fixed_goal_index=goal_index,
        weather_mode=weather_mode,
        cfg=cfg,
    )


def sample_exploration_action(cfg: Config = CFG) -> np.ndarray:
    throttle = float(np.random.uniform(0.18, 0.62))
    brake = 0.0 if np.random.rand() < 0.90 else float(np.random.uniform(0.08, 0.32))
    if brake > 0.0:
        throttle = 0.0
    steer = float(np.random.uniform(-0.35, 0.35))
    return np.array([throttle, brake, steer], dtype=np.float32)


def curriculum_progress(step_idx: int, duration: int) -> float:
    """Return curriculum completion in [0, 1]."""
    return float(np.clip(float(step_idx) / max(float(duration), 1.0), 0.0, 1.0))


def curriculum_actor_action(
    env: CarlaReliableTransferEnv,
    policy_action: np.ndarray,
    step_idx: int,
    cfg: Config,
) -> np.ndarray:
    """Blend a safe route demonstration into early training only.

    The blend decays to exactly zero, so the learned actor controls the car by
    the end of training. The resulting command is stored in replay; later
    actuator filtering remains part of the environment transition.
    """
    a = np.asarray(policy_action, dtype=np.float32).copy()
    if not bool(cfg.use_training_curriculum):
        return a
    frac = curriculum_progress(step_idx, int(cfg.curriculum_steps))
    blend = (1.0 - frac) * float(cfg.curriculum_initial_blend) + frac * float(cfg.curriculum_final_blend)
    if blend <= 1e-6:
        return a
    guide = env.route_guidance_action().astype(np.float32)
    noise_scale = float(cfg.curriculum_action_noise) * (1.0 - frac)
    noise = np.array([
        np.random.normal(0.0, 0.5 * noise_scale),
        np.random.normal(0.0, 0.25 * noise_scale),
        np.random.normal(0.0, noise_scale),
    ], dtype=np.float32)
    guide = guide + noise
    guide[0:2] = np.clip(guide[0:2], 0.0, 1.0)
    guide[2] = np.clip(guide[2], -1.0, 1.0)
    out = (1.0 - blend) * a + blend * guide
    out[0:2] = np.clip(out[0:2], 0.0, 1.0)
    out[2] = np.clip(out[2], -1.0, 1.0)
    if out[0] > 0.0 and out[1] > 0.0:
        if out[0] >= out[1]:
            out[1] = 0.0
        else:
            out[0] = 0.0
    return out.astype(np.float32)


def curriculum_population(step_idx: int, lo: int, hi: int, duration: int) -> int:
    """Ramp traffic inside the requested inclusive population bounds."""
    lo_i, hi_i = sorted((max(0, int(lo)), max(0, int(hi))))
    target = random.randint(lo_i, hi_i)
    frac = curriculum_progress(step_idx, duration)
    return int(np.clip(round(lo_i + frac * (target - lo_i)), lo_i, hi_i))


def collect_domain_batches(
    env_builder: Callable[[], CarlaReliableTransferEnv],
    agent: SACAgent,
    device: torch.device,
    batch_count: int,
    batch_size: int,
    goal_index: int,
    npc_min: int,
    npc_max: int,
    host: str = "localhost",
    port: int = 2200,
) -> List[Dict[str, torch.Tensor]]:
    if batch_count <= 0:
        return []
    env = env_builder()
    target_transitions = int(batch_count * batch_size)
    print(
        f"[INFO] MAML warmup: collecting >= {target_transitions} transitions over WHOLE "
        "episodes. An episode is always driven to its own terminal condition (goal, "
        "collision, off_route, stuck, or timeout); the target only decides whether "
        "another episode is started."
    )
    # An episode can run to max_episode_steps, so the buffer must hold at least
    # the target plus one full episode without evicting the episode in progress.
    buf = ReplayBuffer(
        capacity=max(target_transitions * 2, int(agent.cfg.max_episode_steps) + target_transitions + 1, batch_size + 1),
        cfg=agent.cfg,
    )
    try:
        obs = None
        episodes_done = 0
        episode_steps = 0
        # Hard ceiling so a pathological episode cannot stall the warmup forever.
        max_total_steps = max(target_transitions * 3, int(agent.cfg.max_episode_steps) * 3, batch_size)
        for _ in range(max_total_steps):
            if obs is None:
                # Only start a new episode while the target is still unmet.
                if len(buf) >= target_transitions:
                    break
                env, obs, _ = robust_reset(
                    env, env_builder,
                    npc_count=curriculum_population(len(buf), npc_min, npc_max, agent.cfg.curriculum_traffic_steps),
                    goal_index=goal_index,
                    walker_count=curriculum_population(
                        len(buf), agent.cfg.train_walker_min, agent.cfg.train_walker_max,
                        agent.cfg.curriculum_traffic_steps,
                    ),
                    max_tries=5,
                )
                episode_steps = 0
            raw_action = sample_exploration_action(agent.cfg)
            action = curriculum_actor_action(env, raw_action, len(buf), agent.cfg)
            try:
                next_obs, _, terminated, truncated, info = env.step(action)
            except BaseException as e:
                print(f"[WARN] collect_domain_batches step failed: {e}")
                try:
                    env.close()
                except BaseException:
                    pass
                wait_for_carla_ready(host, port, timeout_s=60.0)
                env = env_builder()
                obs = None
                continue
            replay_action = np.asarray(info.get("replay_action", info.get("applied_action", action)), dtype=np.float32)
            reward, _ = recompute_agent_reward(agent, obs, replay_action, info, cfg=agent.cfg)
            buf.add(obs, replay_action, reward, next_obs, bool(terminated))
            episode_steps += 1
            if terminated or truncated:
                episodes_done += 1
                print(
                    f"[INFO] MAML warmup episode {episodes_done} finished: steps={episode_steps} "
                    f"progress={safe_float(info.get('route_completion_pct', 0.0)):.1f}% "
                    f"reason={info.get('term_reason', '')} buffer={len(buf)}/{target_transitions}"
                )
                obs = None
            else:
                obs = next_obs
        if len(buf) < target_transitions:
            print(
                f"[WARN] MAML warmup stopped at {len(buf)}/{target_transitions} transitions "
                f"after {episodes_done} episode(s) (step ceiling reached)"
            )
        else:
            print(
                f"[INFO] MAML warmup collection complete: {len(buf)} transitions "
                f"from {episodes_done} complete episode(s)"
            )
        batches: List[Dict[str, torch.Tensor]] = []
        if len(buf) >= batch_size:
            for _ in range(batch_count):
                batches.append(buf.sample(batch_size, device))
        return batches
    finally:
        try:
            env.close()
        except BaseException:
            pass
        settle_s = float(max(getattr(agent.cfg, "world_settle_wait_s", 5.0), 1.0))
        print(f"[INFO] collect_domain_batches: waiting {settle_s:.0f} s for CARLA to settle...")
        time.sleep(settle_s)
        wait_for_carla_ready(host, port, timeout_s=60.0)
        print("[INFO] CARLA ready; proceeding to train_loop.")


TRAINING_EPISODE_FIELDS: Tuple[str, ...] = (
    "ablation",
    "seed",
    "episode",
    "environment_steps",
    "optimizer_steps",
    "episode_steps",
    "episode_reward",
    "success",
    "route_completion_pct",
    "term_reason",
    "distance_driven_m",
    "intervention_rate",
    "sigma_bar_terminal",
)


def write_training_episode(path: str, row: Dict[str, object]) -> None:
    """Append one auditable episode summary, creating the CSV header once."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(TRAINING_EPISODE_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in TRAINING_EPISODE_FIELDS})


def train_loop(
    agent: SACAgent,
    args: argparse.Namespace,
    cfg: Config,
    device: torch.device,
    env_builder: Callable[[], CarlaReliableTransferEnv],
    save_path: str,
    total_steps: int,
    goal_index: int,
    npc_min: int,
    npc_max: int,
    source_agent: Optional[SACAgent] = None,
    source_env_builder: Optional[Callable[[], CarlaReliableTransferEnv]] = None,
    source_goal_index: Optional[int] = None,
    source_npc_min: Optional[int] = None,
    source_npc_max: Optional[int] = None,
    max_episodes: Optional[int] = None,
) -> str:
    if not bool(cfg.use_training_curriculum):
        print(
            "[WARN] training curriculum is OFF: cold-start SAC in dense adverse "
            "traffic explores very slowly and early episodes will end in "
            "collision/off_route for a long time. Pass --enable-training-curriculum "
            "to bootstrap (and disclose it in Sec. IV-A if you report those results)."
        )
    print(
        f"[INFO] train_loop starting: total_steps={int(total_steps)} "
        f"max_episode_steps={int(cfg.max_episode_steps)} "
        f"npc=[{int(npc_min)},{int(npc_max)}] route_target={cfg.route_target_length_m:.0f}m"
    )
    replay = ReplayBuffer(capacity=cfg.replay_size, cfg=cfg)
    source_replay: Optional[ReplayBuffer] = None
    if source_agent is not None and source_env_builder is not None:
        source_replay = ReplayBuffer(capacity=cfg.replay_size, cfg=cfg)

    _tl_host = getattr(args, "host", "localhost")
    _tl_port = getattr(args, "port", 2200)
    env: Optional[CarlaReliableTransferEnv] = None
    for _init_attempt in range(8):
        try:
            wait_for_carla_ready(_tl_host, _tl_port, timeout_s=30.0)
            env = env_builder()
            break
        except BaseException as e:
            wait = [10, 15, 20, 25, 30, 35, 40, 45][min(_init_attempt, 7)]
            print(f"[WARN] train_loop env init attempt {_init_attempt+1}/8 failed: {e}; retrying in {wait}s")
            time.sleep(wait)
    if env is None:
        raise RuntimeError("train_loop: could not create initial environment after 8 attempts")

    source_env: Optional[CarlaReliableTransferEnv] = None
    if source_env_builder is not None and source_agent is not None:
        for _sinit in range(4):
            try:
                wait_for_carla_ready(
                    getattr(args, "host", "localhost"),
                    int(getattr(args, "source_port", 2201)),
                    timeout_s=30.0,
                )
                source_env = source_env_builder()
                break
            except BaseException as e:
                wait = [5, 8, 12, 18][min(_sinit, 3)]
                print(f"[WARN] source_env init attempt {_sinit+1}/4 failed: {e}; retrying in {wait}s")
                time.sleep(wait)
    obs: Optional[Dict[str, np.ndarray]] = None
    source_obs: Optional[Dict[str, np.ndarray]] = None
    episode = 0
    episode_reward = 0.0
    env_steps = 0
    best_success_like = -1.0
    best_path = os.path.splitext(save_path)[0] + '_best.pt'
    episode_csv_path = os.path.join(cfg.out_dir, "training_episodes.csv")
    # This function represents a fresh run.  Existing run protection happens
    # in main(); an explicit --overwrite-run starts a fresh episode log.
    if bool(getattr(args, "overwrite_run", False)) and os.path.exists(episode_csv_path):
        with open(episode_csv_path, "w", encoding="utf-8"):
            pass
    consecutive_step_failures = 0
    try:
        while True:
            if obs is None and max_episodes is not None and episode >= max_episodes:
                break
            if env_steps >= total_steps:
                break

            if obs is None:
                env, obs, _ = robust_reset(
                    env,
                    env_builder,
                    npc_count=curriculum_population(env_steps, npc_min, npc_max, cfg.curriculum_traffic_steps)
                    if cfg.use_training_curriculum else random.randint(npc_min, npc_max),
                    goal_index=goal_index,
                    walker_count=curriculum_population(
                        env_steps, cfg.train_walker_min, cfg.train_walker_max, cfg.curriculum_traffic_steps
                    ) if cfg.use_training_curriculum else random.randint(cfg.train_walker_min, cfg.train_walker_max),
                    max_tries=3,
                )
                episode += 1
                episode_reward = 0.0

            if env_steps < int(args.start_steps):
                raw_action = sample_exploration_action(cfg)
            else:
                raw_action = agent.act(obs, deterministic=False)
            action = curriculum_actor_action(env, raw_action, env_steps, cfg)

            try:
                next_obs, _, terminated, truncated, info = env.step(action)
            except BaseException as e:
                print(f"[WARN] train env.step failed: {e}; restarting episode")
                consecutive_step_failures += 1
                try:
                    env.close()
                except BaseException:
                    pass
                wait_for_carla_ready(_tl_host, _tl_port, timeout_s=60.0)
                env = env_builder()
                obs = None
                if consecutive_step_failures >= 8:
                    raise RuntimeError("training aborted after 8 consecutive CARLA step failures") from e
                continue
            consecutive_step_failures = 0
            env_steps += 1
            replay_action = np.asarray(info.get("replay_action", info.get("applied_action", action)), dtype=np.float32)
            reward, sigma_bar = recompute_agent_reward(agent, obs, replay_action, info, cfg=cfg)
            info["sigma_bar"] = float(sigma_bar)
            done = bool(terminated or truncated)
            replay.add(obs, replay_action, reward, next_obs, bool(terminated))
            obs = None if done else next_obs
            episode_reward += reward

            if source_agent is not None and source_env is not None and source_replay is not None and source_env_builder is not None:
                if source_obs is None:
                    src_lo = int(source_npc_min if source_npc_min is not None else npc_min)
                    src_hi = int(source_npc_max if source_npc_max is not None else npc_max)
                    source_env, source_obs, _ = robust_reset(
                        source_env,
                        source_env_builder,
                        npc_count=curriculum_population(env_steps, src_lo, src_hi, cfg.curriculum_traffic_steps)
                        if cfg.use_training_curriculum else random.randint(src_lo, src_hi),
                        goal_index=int(source_goal_index if source_goal_index is not None else goal_index),
                        walker_count=curriculum_population(
                            env_steps, cfg.train_walker_min, cfg.train_walker_max, cfg.curriculum_traffic_steps
                        ) if cfg.use_training_curriculum else random.randint(cfg.train_walker_min, cfg.train_walker_max),
                        max_tries=3,
                    )
                source_raw_action = source_agent.act(source_obs, deterministic=False)
                source_action = curriculum_actor_action(source_env, source_raw_action, env_steps, cfg)
                try:
                    source_next_obs, _, source_terminated, source_truncated, source_info = source_env.step(source_action)
                except BaseException as e:
                    print(f"[WARN] source env.step failed: {e}; restarting source episode")
                    try:
                        source_env.close()
                    except BaseException:
                        pass
                    wait_for_carla_ready(
                        getattr(args, "host", "localhost"),
                        int(getattr(args, "source_port", 2201)),
                        timeout_s=60.0,
                    )
                    source_env = source_env_builder()
                    source_obs = None
                else:
                    source_replay_action = np.asarray(
                        source_info.get("replay_action", source_info.get("applied_action", source_action)),
                        dtype=np.float32,
                    )
                    source_reward, source_sigma_bar = recompute_agent_reward(
                        source_agent, source_obs, source_replay_action, source_info, cfg=source_agent.cfg
                    )
                    source_info["sigma_bar"] = float(source_sigma_bar)
                    source_done = bool(source_terminated or source_truncated)
                    source_replay.add(source_obs, source_replay_action, source_reward, source_next_obs, bool(source_terminated))
                    source_obs = None if source_done else source_next_obs

            if len(replay) >= cfg.batch_size and env_steps >= int(args.update_after):
                for _ in range(int(args.updates_per_step)):
                    batch = replay.sample(cfg.batch_size, device)
                    source_batch = None
                    if source_agent is not None and source_replay is not None and len(source_replay) >= cfg.batch_size:
                        source_batch = source_replay.sample(cfg.batch_size, device)
                    stats = agent.update(batch, source_agent=source_agent if source_batch is not None else None, source_batch=source_batch)
                if cfg.debug_mode and agent.training_steps % max(1, cfg.debug_step_freq * 5) == 0:
                    print(
                        f"[TRAIN] step={agent.training_steps} critic={safe_float(stats.get('critic_loss')):.4f} "
                        f"actor={safe_float(stats.get('actor_loss')):.4f} alpha={safe_float(stats.get('alpha')):.4f} "
                        f"sigma={safe_float(stats.get('sigma_bar')):.4f} beta0={cfg.beta0:.2f} "
                        f"temp={safe_float(stats.get('temp_eff')):.4f}"
                    )

            if env_steps % int(cfg.progress_every_steps) == 0:
                print(
                    f"[PROGRESS] ablation={cfg.ablation_name} seed={cfg.seed} "
                    f"env_steps={env_steps}/{int(total_steps)} "
                    f"updates={agent.training_steps} episodes={episode} "
                    f"replay={len(replay)}"
                )

            if done:
                success_like = 1.0 if bool(info.get('success', False)) else safe_float(info.get('route_completion_pct', 0.0)) / 100.0
                if cfg.save_best_training_checkpoint and success_like > best_success_like:
                    best_success_like = success_like
                    agent.save(best_path)
                write_training_episode(
                    episode_csv_path,
                    {
                        "ablation": cfg.ablation_name,
                        "seed": cfg.seed,
                        "episode": episode,
                        "environment_steps": env_steps,
                        "optimizer_steps": agent.training_steps,
                        "episode_steps": int(safe_float(info.get("steps", 0))),
                        "episode_reward": float(episode_reward),
                        "success": int(bool(info.get("success", False))),
                        "route_completion_pct": safe_float(info.get("route_completion_pct", 0.0)),
                        "term_reason": str(info.get("term_reason", "")),
                        "distance_driven_m": safe_float(info.get("distance_driven_m", 0.0)),
                        "intervention_rate": safe_float(info.get("intervention_rate", 0.0)),
                        "sigma_bar_terminal": safe_float(info.get("sigma_bar", 0.0)),
                    },
                )
                print(
                    f"[EP] ep={episode} steps={safe_float(info.get('steps', 0)):.0f} reward={episode_reward:.2f} "
                    f"success={int(bool(info.get('success', False)))} progress={safe_float(info.get('route_completion_pct', 0.0)):.1f}% "
                    f"reason={info.get('term_reason', '')}"
                )

            if env_steps > 0 and env_steps % int(args.save_every_steps) == 0:
                agent.save(save_path)
                print(f"[OK] Saved checkpoint: {save_path}")

        agent.save(save_path)
        print(f"[OK] Final checkpoint saved: {save_path}")
        if cfg.save_best_training_checkpoint and os.path.exists(best_path):
            print(f"[OK] Best checkpoint saved : {best_path}")
        print(f"[OK] Episode log: {episode_csv_path}")
        return save_path
    finally:
        if source_env is not None:
            try:
                source_env.close()
            except BaseException:
                pass
        try:
            env.close()
        except BaseException:
            pass


def run_source_training(args: argparse.Namespace, cfg: Config, device: torch.device) -> None:
    # Use random spawn (-1) for source training unless the user explicitly fixed one.
    # Spawn diversity is critical: training log shows 97.3% collision rate when
    # spawn is fixed to index 0 — the policy overfits to one road segment and
    # hits the same NPC deterministically after alpha collapses at ~18k steps.
    train_spawn_index = args.spawn_index if args.spawn_index >= 0 else -1
    train_builder = make_env_builder(
        host=args.host,
        port=args.port,
        town_name=args.train_town,
        spawn_index=train_spawn_index,
        goal_index=cfg.fixed_goal_index,
        weather_mode=cfg.fixed_weather,
        cfg=cfg,
    )
    agent = SACAgent(device=device, cfg=cfg)
    save_path = resolve_existing_path(os.path.join(cfg.model_dir, 'source_agent.pt'))
    train_loop(
        agent=agent,
        args=args,
        cfg=cfg,
        device=device,
        env_builder=train_builder,
        save_path=save_path,
        total_steps=int(args.train_steps),
        goal_index=cfg.fixed_goal_index,
        npc_min=cfg.train_npc_min,
        npc_max=cfg.train_npc_max,
        source_agent=None,
    )


def run_target_adaptation(args: argparse.Namespace, cfg: Config, device: torch.device) -> None:
    source_ckpt = resolve_existing_path(args.source_checkpoint or args.checkpoint)
    if not os.path.exists(source_ckpt):
        raise FileNotFoundError(f'Source checkpoint not found: {source_ckpt}')
    import copy
    source_cfg = copy.copy(cfg)
    source_cfg.tm_port = int(args.source_tm_port)
    use_transfer_alignment = bool(
        getattr(cfg, "use_transfer_alignment", True)
    )
    source_agent: Optional[SACAgent] = None
    source_builder: Optional[Callable[[], CarlaReliableTransferEnv]] = None
    if use_transfer_alignment:
        source_agent = SACAgent(device=device, cfg=source_cfg)
        source_agent.load(source_ckpt)
        if int(args.source_port) == int(args.port):
            raise ValueError(
                "adapt mode with transfer alignment needs a second CARLA "
                "server for source replay; set --source-port differently "
                "from --port"
            )
        if int(args.source_tm_port) == int(args.tm_port):
            raise ValueError(
                "adapt mode with transfer alignment needs a distinct Traffic "
                "Manager; set --source-tm-port differently from --tm-port"
            )
        source_builder = make_env_builder(
            host=args.host, port=args.source_port, town_name=args.train_town,
            spawn_index=args.spawn_index, goal_index=cfg.fixed_goal_index,
            weather_mode=source_cfg.fixed_weather, cfg=source_cfg,
        )
    else:
        print(
            "[ABLATION] Transfer alignment disabled; source replay and the "
            "second CARLA server are not used."
        )
    target_cfg = make_target_cfg(cfg)
    target_agent = SACAgent(device=device, cfg=target_cfg)
    target_agent.load(source_ckpt)
    target_agent.train()
    adapt_builder = make_env_builder(
        host=args.host, port=args.port, town_name=args.target_town,
        spawn_index=args.spawn_index, goal_index=target_cfg.target_goal_index,
        weather_mode=target_cfg.target_weather, cfg=target_cfg,
    )
    if (
        bool(getattr(target_cfg, "use_maml_initialization", True))
        and int(args.adapt_maml_warmup_batches) > 0
    ):
        warm_batches = collect_domain_batches(
            adapt_builder, target_agent, device,
            batch_count=int(args.adapt_maml_warmup_batches), batch_size=target_cfg.batch_size,
            goal_index=target_cfg.target_goal_index,
            npc_min=target_cfg.npc_min, npc_max=target_cfg.npc_max,
            host=args.host, port=args.port,
        )
        if warm_batches:
            meta_stats = target_agent.maml_style_initialize(warm_batches)
            print(f"[OK] Applied target-domain MAML-style initialization: {meta_stats}")
    elif int(args.adapt_maml_warmup_batches) > 0:
        print("[ABLATION] MAML initialization disabled; warmup collection skipped.")
    save_path = resolve_existing_path(os.path.join(cfg.model_dir, 'target_agent.pt'))
    train_loop(
        agent=target_agent, args=args, cfg=target_cfg, device=device,
        env_builder=adapt_builder, save_path=save_path,
        total_steps=int(args.adapt_steps), goal_index=target_cfg.target_goal_index,
        npc_min=target_cfg.npc_min, npc_max=target_cfg.npc_max,
        source_agent=source_agent, source_env_builder=source_builder,
        source_goal_index=cfg.fixed_goal_index,
        source_npc_min=cfg.train_npc_min, source_npc_max=cfg.train_npc_max,
        max_episodes=int(args.adapt_episodes),
    )


def run_target_policy_learning(args: argparse.Namespace, cfg: Config, device: torch.device) -> None:
    target_cfg = make_target_cfg(cfg)
    policy_builder = make_env_builder(
        host=args.host, port=args.port, town_name=args.target_town,
        spawn_index=args.spawn_index, goal_index=target_cfg.target_goal_index,
        weather_mode=target_cfg.target_weather, cfg=target_cfg,
    )
    agent = SACAgent(device=device, cfg=target_cfg)
    save_path = resolve_existing_path(os.path.join(cfg.model_dir, 'target_policy_agent.pt'))
    train_loop(
        agent=agent, args=args, cfg=target_cfg, device=device,
        env_builder=policy_builder, save_path=save_path,
        total_steps=int(args.adapt_steps), goal_index=target_cfg.target_goal_index,
        npc_min=target_cfg.npc_min, npc_max=target_cfg.npc_max,
        source_agent=None, max_episodes=int(args.adapt_episodes),
    )


# ======================================================
# CLI / main
# ======================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA 0.9.15 – Ego-Relational Policy Transfer with DRL (paper-aligned, 500 m route)")
    p.add_argument("--mode", type=str, default="train", choices=["train", "adapt", "policy"],
                   help="train=source training, adapt=target-domain transfer training, "
                        "policy=target-only SAC training without transfer")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=2200)
    p.add_argument("--source-port", type=int, default=2201,
                   help="second CARLA RPC port used for source replay in adapt mode")
    p.add_argument("--tm-port", type=int, default=CFG.tm_port)
    p.add_argument("--source-tm-port", type=int, default=8001,
                   help="Traffic Manager port for the second CARLA server in adapt mode")
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--fps", type=int, default=CFG.fps)
    p.add_argument("--spawn-index", type=int, default=-1,
                   help="-1 = random valid spawn each episode (recommended for training)")
    p.add_argument("--train-town", type=str, default="Town10HD_Opt")
    p.add_argument("--target-town", type=str, default="Town02")
    p.add_argument("--source-weather", type=str, default="night_rain_fog", choices=["night_rain_fog", "mixed", "default"])
    p.add_argument("--target-weather", type=str, default="mixed", choices=["night_rain_fog", "mixed", "default"])
    p.add_argument("--train-goal-index", type=int, default=-1,
                   help="-1 = automatically construct a route in the configured length window")
    p.add_argument("--target-goal-index", type=int, default=-1,
                   help="-1 = auto-route (safe for Town02)")
    p.add_argument("--npc-min", type=int, default=CFG.npc_min)
    p.add_argument("--npc-max", type=int, default=CFG.npc_max)
    p.add_argument("--train-npc-min", type=int, default=CFG.train_npc_min)
    p.add_argument("--train-npc-max", type=int, default=CFG.train_npc_max)
    p.add_argument("--walker-min", type=int, default=CFG.walker_min)
    p.add_argument("--walker-max", type=int, default=CFG.walker_max)
    p.add_argument("--train-walker-min", type=int, default=CFG.train_walker_min)
    p.add_argument("--train-walker-max", type=int, default=CFG.train_walker_max)
    p.add_argument("--no-world-reuse", action="store_true",
                   help="force client.load_world() on every env construction "
                        "(default: reuse the loaded town and only reset actors)")
    p.add_argument("--checkpoint", type=str, default=os.path.join(CFG.model_dir, "source_agent.pt"))
    p.add_argument("--source-checkpoint", type=str, default="", help="source checkpoint for adaptation; defaults to --checkpoint")
    p.add_argument("--out-dir", type=str, default=CFG.out_dir)
    p.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=ABLATION_CHOICES,
        help=(
            "train exactly one matched component variant. Outputs are isolated "
            "under <ablation-out-root>/<ablation>/"
        ),
    )
    p.add_argument(
        "--ablation-out-root",
        type=str,
        default="",
        help=(
            "root directory for isolated ablation runs. Default: "
            "<out-dir>/ablations"
        ),
    )
    p.add_argument(
        "--describe-ablation",
        action="store_true",
        help="print the resolved ablation configuration and exit before CARLA use",
    )
    p.add_argument(
        "--overwrite-run",
        action="store_true",
        help="allow this exact ablation/seed directory to overwrite prior run artifacts",
    )
    p.add_argument("--train-steps", type=int, default=500000,   # paper: 5×10^5 steps
                   help="total environment steps for source training")
    p.add_argument("--adapt-steps", type=int, default=50000)
    p.add_argument("--adapt-episodes", type=int, default=CFG.adapt_episodes)
    p.add_argument("--start-steps", type=int, default=5000)
    p.add_argument("--update-after", type=int, default=2048)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--save-every-steps", type=int, default=10000)
    p.add_argument("--adapt-maml-warmup-batches", type=int, default=0,
                   help="optional target-domain meta-initialization batches in adapt mode")
    p.add_argument("--maml-inner-steps", type=int, default=CFG.maml_inner_steps)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-step-freq", type=int, default=CFG.debug_step_freq)
    p.add_argument("--no-follow-ego-view", action="store_true")
    p.add_argument("--no-rendering", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--use-safety-shield", action="store_true",
                   help="enable bounded low-level assistance during any mode")
    p.add_argument("--red-light-assist", action="store_true",
                   help="enable bounded red-light braking in the default no-shield path. "
                        "Every activation counts toward the reported intervention rate")
    p.add_argument(
        "--allow-training-interventions",
        action="store_true",
        help=(
            "explicitly allow safety-shield/red-light interventions during "
            "learning. Not recommended for component attribution; without "
            "this acknowledgement, training modes reject intervention flags"
        ),
    )
    p.add_argument(
        "--save-best-training-checkpoint",
        action="store_true",
        help=(
            "also save a checkpoint selected from training episodes. The "
            "paper protocol should evaluate final checkpoints or select on a "
            "separate validation set"
        ),
    )
    p.add_argument(
        "--progress-every-steps",
        type=int,
        default=CFG.progress_every_steps,
        help="print an unconditional training heartbeat at this environment-step interval",
    )
    p.add_argument("--enable-training-curriculum", action="store_true",
                   help="enable the scripted route-guidance bootstrap during early "
                        "training. NOT part of the paper's method (deviation D3); if "
                        "used, disclose it in Sec. IV-A and apply it to baselines too")
    p.add_argument("--attention-score-form", type=str, default=CFG.attention_score_form,
                   choices=["reliability", "paper_ratio"],
                   help="'reliability' (default) implements the behavior stated in the "
                        "text and Table I; 'paper_ratio' reproduces Eq. (6) verbatim")
    p.add_argument("--disable-training-curriculum", action="store_true",
                   help="disable the decaying route-demonstration and traffic curriculum")
    p.add_argument("--curriculum-steps", type=int, default=CFG.curriculum_steps,
                   help="steps over which route-action demonstration decays to zero")
    p.add_argument("--curriculum-traffic-steps", type=int, default=CFG.curriculum_traffic_steps,
                   help="steps over which NPC/pedestrian density ramps to the requested range")
    p.add_argument("--route-target-length", type=float, default=0.0,
                   help="Override route target length in metres (0 = use default 500m). "
                        "Scales all related route length thresholds and the episode "
                        "step budget proportionally.")
    return p.parse_args()


def apply_ablation(cfg: Config, ablation_name: str) -> None:
    """Apply one matched training ablation to the configuration.

    The full configuration is reset first so repeated calls are deterministic.
    no_entropy_gate retains learned SAC alpha and removes only the
    uncertainty-dependent multiplier. no_critic_ensemble uses one critic,
    which makes epistemic variance exactly zero.
    """
    name = str(ablation_name).strip().lower()
    if name not in ABLATION_CHOICES:
        raise ValueError(
            f"Unknown ablation {name!r}; expected one of {ABLATION_CHOICES}"
        )

    cfg.ablation_name = name
    cfg.use_uncertainty_attention = True
    cfg.use_critic_ensemble = True
    cfg.use_entropy_gate = True
    cfg.use_dense_reward = True
    cfg.use_transfer_alignment = True
    cfg.use_maml_initialization = True
    cfg.n_critics = 5
    cfg.use_learned_alpha = True
    cfg.lambda_transfer = 1.0

    if name == "no_uncertainty_attention":
        cfg.use_uncertainty_attention = False
    elif name == "no_critic_ensemble":
        cfg.use_critic_ensemble = False
        cfg.n_critics = 1
    elif name == "no_entropy_gate":
        cfg.use_entropy_gate = False
        # Standard SAC entropy remains active through learned alpha.
        cfg.use_learned_alpha = True
    elif name == "event_reward":
        cfg.use_dense_reward = False
    elif name == "no_transfer_alignment":
        cfg.use_transfer_alignment = False
        cfg.lambda_transfer = 0.0
    elif name == "no_maml":
        cfg.use_maml_initialization = False


def validate_ablation_mode(args: argparse.Namespace, cfg: Config) -> None:
    """Reject ablations that are inactive in the selected training mode."""
    name = str(cfg.ablation_name)
    mode = str(args.mode)
    adaptation_only = {"no_transfer_alignment", "no_maml"}
    if name in adaptation_only and mode != "adapt":
        raise ValueError(
            f"--ablation {name} is inactive in --mode {mode}. "
            "It is meaningful only in --mode adapt, which uses target-domain "
            "training and therefore is not a zero-shot evaluation."
        )
    if name == "no_maml" and int(args.adapt_maml_warmup_batches) <= 0:
        raise ValueError(
            "--ablation no_maml requires --adapt-maml-warmup-batches > 0 "
            "to define a matched full-model MAML baseline; collection will be "
            "skipped in the no_maml run."
        )
    if (
        mode in {"train", "adapt", "policy"}
        and (bool(cfg.use_safety_shield) or bool(cfg.red_light_assist))
        and not bool(getattr(args, "allow_training_interventions", False))
    ):
        raise ValueError(
            "Safety shield/red-light assist changes the executed training "
            "actions and confounds learned-component attribution. Remove "
            "--use-safety-shield and --red-light-assist for the paper ablation "
            "protocol. If an assisted-training experiment is intentional, "
            "add --allow-training-interventions and use it for every arm."
        )


def ablation_manifest(args: argparse.Namespace, cfg: Config) -> Dict[str, object]:
    return {
        "ablation": str(cfg.ablation_name),
        "mode": str(args.mode),
        "source_town": str(args.train_town),
        "target_town": str(args.target_town),
        "source_weather": str(cfg.fixed_weather),
        "target_weather": str(cfg.target_weather),
        "seed": int(cfg.seed),
        "train_steps": int(args.train_steps),
        "adapt_steps": int(args.adapt_steps),
        "n_critics": int(cfg.n_critics),
        "use_uncertainty_attention": bool(cfg.use_uncertainty_attention),
        "use_critic_ensemble": bool(cfg.use_critic_ensemble),
        "use_entropy_gate": bool(cfg.use_entropy_gate),
        "use_dense_reward": bool(cfg.use_dense_reward),
        "use_transfer_alignment": bool(cfg.use_transfer_alignment),
        "use_maml_initialization": bool(cfg.use_maml_initialization),
        "use_learned_alpha": bool(cfg.use_learned_alpha),
        "use_safety_shield": bool(cfg.use_safety_shield),
        "red_light_assist": bool(cfg.red_light_assist),
        "training_curriculum": bool(cfg.use_training_curriculum),
        "replay_action": "applied" if cfg.replay_applied_action else "command",
        "timeout_is_terminal": bool(cfg.timeout_is_terminal),
        "progress_gate_positive_reward": bool(cfg.progress_gate_positive_reward),
        "progress_gate_distance_m": float(cfg.progress_gate_distance_m),
        "open_road_idle_penalty": float(cfg.open_road_idle_penalty),
        "terminal_rewards": {
            "goal": float(cfg.goal_terminal_reward),
            "collision": float(cfg.collision_terminal_reward),
            "off_route": float(cfg.offroute_terminal_reward),
            "stuck": float(cfg.stuck_terminal_reward),
            "timeout": float(cfg.timeout_terminal_reward),
        },
        "checkpoint_for_paper": "final",
        "save_best_training_checkpoint": bool(cfg.save_best_training_checkpoint),
        "train_npc_min": int(cfg.train_npc_min),
        "train_npc_max": int(cfg.train_npc_max),
        "train_walker_min": int(cfg.train_walker_min),
        "train_walker_max": int(cfg.train_walker_max),
        "output_dir": str(cfg.out_dir),
    }


def write_ablation_manifest(args: argparse.Namespace, cfg: Config) -> str:
    path = os.path.join(cfg.out_dir, "ablation_config.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(ablation_manifest(args, cfg), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def make_runtime_cfg(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.tm_port = int(args.tm_port)
    cfg.seed = int(args.seed)
    cfg.fps = int(args.fps)
    cfg.debug_mode = bool(args.debug)
    cfg.debug_step_freq = int(args.debug_step_freq)
    cfg.mode_name = str(args.mode)
    cfg.follow_ego_view = not bool(args.no_follow_ego_view)
    cfg.no_rendering_mode = bool(args.no_rendering)
    cfg.fixed_goal_index = int(args.train_goal_index)
    cfg.target_goal_index = int(args.target_goal_index)
    cfg.use_fixed_destination = cfg.fixed_goal_index >= 0
    cfg.npc_min = int(args.npc_min)
    cfg.npc_max = int(args.npc_max)
    cfg.train_npc_min = int(args.train_npc_min)
    cfg.train_npc_max = int(args.train_npc_max)
    cfg.walker_min = int(args.walker_min)
    cfg.walker_max = int(args.walker_max)
    cfg.train_walker_min = int(args.train_walker_min)
    cfg.train_walker_max = int(args.train_walker_max)
    cfg.reuse_loaded_world = not bool(args.no_world_reuse)
    cfg.fixed_weather = str(args.source_weather)
    cfg.target_weather = str(args.target_weather)
    cfg.use_safety_shield = bool(args.use_safety_shield)
    cfg.use_training_curriculum = (
        bool(getattr(args, "enable_training_curriculum", False))
        and not bool(args.disable_training_curriculum)
    )
    cfg.attention_score_form = str(getattr(args, "attention_score_form", cfg.attention_score_form))
    cfg.red_light_assist = bool(getattr(args, "red_light_assist", False))
    cfg.save_best_training_checkpoint = bool(
        getattr(args, "save_best_training_checkpoint", False)
    )
    cfg.progress_every_steps = max(1, int(getattr(args, "progress_every_steps", 1000)))
    cfg.curriculum_steps = max(1, int(args.curriculum_steps))
    cfg.curriculum_traffic_steps = max(1, int(args.curriculum_traffic_steps))
    cfg.maml_inner_steps = int(args.maml_inner_steps)
    cfg.adapt_episodes = int(args.adapt_episodes)
    base_out_dir = resolve_output_dir(str(args.out_dir))
    ablation_root_text = str(getattr(args, "ablation_out_root", "")).strip()
    ablation_root = (
        resolve_output_dir(ablation_root_text)
        if ablation_root_text
        else os.path.join(base_out_dir, "ablations")
    )
    apply_ablation(cfg, str(args.ablation))
    # Every seed gets a distinct directory; otherwise sequential paper runs
    # silently overwrite checkpoints and manifests from earlier seeds.
    cfg.out_dir = os.path.join(ablation_root, cfg.ablation_name, f"seed_{cfg.seed}")
    ensure_dirs(cfg)

    # Route length override — scales all related thresholds proportionally.
    # The default protocol is a 500 m route; e.g. --route-target-length 800
    # produces longer trajectories for figures, 200 recovers short routes.
    _rtl = float(getattr(args, "route_target_length", 0.0))
    if _rtl > 50.0:
        _scale = _rtl / 500.0          # ratio vs default 500 m
        cfg.route_target_length_m          = _rtl
        cfg.route_target_tolerance_m       = max(5.0, 10.0 * _scale)
        cfg.route_soft_min_length_m        = _rtl - cfg.route_target_tolerance_m
        cfg.route_soft_max_length_m        = _rtl + cfg.route_target_tolerance_m
        cfg.min_route_length_m             = max(50.0, _rtl - 2.0 * cfg.route_target_tolerance_m)
        # Euclidean distance is only a planner prefilter; winding arc length is
        # governed by the strict target window above.
        cfg.candidate_goal_min_dist_m      = min(cfg.candidate_goal_min_dist_m, _rtl * 0.20)
        cfg.candidate_goal_max_tries       = max(160, int(160 * _scale))
        cfg.max_reset_start_progress_pct   = min(5.0, 5.0 / max(_scale, 1e-3))
        cfg.max_episode_steps              = max(2500, int(6000 * _scale))
        print(f"[CFG] route_target_length overridden to {_rtl:.0f} m "
              f"(soft window {cfg.route_soft_min_length_m:.0f}–{cfg.route_soft_max_length_m:.0f} m)")

    return cfg


def main() -> None:
    args = parse_args()
    cfg = make_runtime_cfg(args)
    validate_ablation_mode(args, cfg)
    existing_artifacts = (
        glob.glob(os.path.join(cfg.model_dir, "*.pt"))
        + glob.glob(os.path.join(cfg.out_dir, "training_episodes.csv"))
    )
    if (
        existing_artifacts
        and not bool(args.overwrite_run)
        and not bool(args.describe_ablation)
    ):
        raise FileExistsError(
            "This ablation/seed run already has artifacts and training does "
            "not implement exact replay-buffer resume. Choose a new --seed or "
            "pass --overwrite-run intentionally. Existing: "
            + ", ".join(existing_artifacts)
        )
    manifest_path = write_ablation_manifest(args, cfg)
    set_global_seed(cfg.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(f"Mode: {args.mode}")
    print(f"Ablation: {cfg.ablation_name}")
    print(f"Device: {device}")
    print(f"GlobalRoutePlanner available: {GlobalRoutePlanner is not None}")

    print(f"Python: {sys.version.split()[0]}")
    src_goal_txt = "auto" if int(cfg.fixed_goal_index) < 0 else str(cfg.fixed_goal_index)
    tgt_goal_txt = "auto" if int(cfg.target_goal_index) < 0 else str(cfg.target_goal_index)
    print(f"Source town: {args.train_town} | goal={src_goal_txt} | NPC=[{cfg.train_npc_min}, {cfg.train_npc_max}]")
    print(f"Target town: {args.target_town} | goal={tgt_goal_txt} | NPC=[{cfg.npc_min}, {cfg.npc_max}]")
    print(f"Weather: source={cfg.fixed_weather} target={cfg.target_weather}")
    print(f"Safety shield: {cfg.use_safety_shield}")
    print(
        f"Attention score form: {cfg.attention_score_form} | "
        f"Training curriculum: {cfg.use_training_curriculum} | "
        f"action decay={cfg.curriculum_steps} steps | traffic ramp={cfg.curriculum_traffic_steps} steps"
    )
    print(f"beta0: {cfg.beta0}")
    print(f"Follow ego view: {cfg.follow_ego_view}")
    print(f"No rendering: {cfg.no_rendering_mode}")
    print(f"Output dir: {cfg.out_dir}")
    print(f"Ablation manifest: {manifest_path}")
    print(
        "Ablation switches: "
        f"uncertainty_attention={cfg.use_uncertainty_attention} "
        f"critics={cfg.n_critics} "
        f"entropy_gate={cfg.use_entropy_gate} "
        f"dense_reward={cfg.use_dense_reward} "
        f"transfer_alignment={cfg.use_transfer_alignment} "
        f"maml={cfg.use_maml_initialization}"
    )
    print(f"Auto-selected route target length: {cfg.route_target_length_m:.0f}m ± {cfg.route_target_tolerance_m:.0f}m")
    print(f"Train steps: {args.train_steps} | Adapt steps cap: {args.adapt_steps} | Adapt episodes: {args.adapt_episodes}")

    if bool(args.describe_ablation):
        print(json.dumps(ablation_manifest(args, cfg), indent=2, sort_keys=True))
        return

    if args.mode == "train":
        run_source_training(args, cfg, device)
        return

    if args.mode == "adapt":
        run_target_adaptation(args, cfg, device)
        return

    if args.mode == "policy":
        run_target_policy_learning(args, cfg, device)
        return

if __name__ == "__main__":
    main()
