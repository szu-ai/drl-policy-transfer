#!/usr/bin/env python3
"""Paper-aligned, evaluation-only companion for ``car.py``.

Run exactly one CARLA town and one weather condition per process.  The script
loads a frozen checkpoint, executes deterministic actions, performs no gradient
updates, and writes crash-resilient episode, route, trajectory, uncertainty,
attention, and summary records.

The reported local Driving Score follows the bounded paper form

    DS_i = RC_i * IS_i,

where RC is route completion in percent and IS is the CARLA Leaderboard 2.1
reciprocal penalty over events observable in the local environment.  Because
this environment does not expose every official Leaderboard event, the output
is named ``local_*`` and must not be presented as an official CARLA score.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, get_type_hints

import numpy as np
import torch


WEATHER_CHOICES = ("auto", "default", "night_rain_fog", "mixed")
PROTOCOL_CHOICES = ("auto", "source_stress", "zero_shot", "cross_town", "extended")


def load_car_module(path_text: str):
    """Load the sibling training module without executing its main function."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        sibling = Path(__file__).resolve().parent / path
        path = sibling if sibling.exists() else Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared module not found: {path}. Put car_eval.py beside car.py "
            "or pass --car-module /absolute/path/to/car.py"
        )
    spec = importlib.util.spec_from_file_location("culrt_car_shared", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import shared module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "Config",
        "SACAgent",
        "CarlaReliableTransferEnv",
        "build_reward_from_info",
        "set_global_seed",
        "safe_float",
        "vec3_length",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(f"{path} is missing required APIs: {', '.join(missing)}")
    return module, path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate one frozen policy on one CARLA town; no training is performed."
    )
    p.add_argument("--car-module", default="car.py")
    p.add_argument("--checkpoint", default="", help="empty selects the protocol checkpoint")
    p.add_argument("--town", required=True, help="one town only, e.g. Town10HD_Opt")
    p.add_argument("--protocol", choices=PROTOCOL_CHOICES, default="auto")
    p.add_argument("--weather", choices=WEATHER_CHOICES, default="auto")
    p.add_argument("--episodes", "--episodes-per-condition", dest="episodes", type=int, default=20)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2200)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--server-timeout", type=float, default=120.0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spawn-index", type=int, default=-1)
    p.add_argument("--goal-index", type=int, default=-1)
    p.add_argument("--route-target-length", type=float, default=500.0)
    p.add_argument("--max-episode-steps", type=int, default=0)
    p.add_argument("--npc-min", type=int, default=20)
    p.add_argument("--npc-max", type=int, default=20)
    p.add_argument("--walker-min", type=int, default=0)
    p.add_argument("--walker-max", type=int, default=0)

    shield = p.add_mutually_exclusive_group()
    shield.add_argument("--use-safety-shield", dest="use_safety_shield", action="store_true")
    shield.add_argument("--no-safety-shield", dest="use_safety_shield", action="store_false")
    p.set_defaults(use_safety_shield=True)

    red = p.add_mutually_exclusive_group()
    red.add_argument("--red-light-assist", dest="red_light_assist", action="store_true")
    red.add_argument("--no-red-light-assist", dest="red_light_assist", action="store_false")
    p.set_defaults(red_light_assist=False)

    p.add_argument(
        "--attention-score-form",
        choices=("auto", "reliability", "paper_ratio"),
        default="auto",
        help="auto preserves the checkpoint setting; paper-aligned runs require reliability",
    )
    p.add_argument("--allow-protocol-mismatch", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-rendering", action="store_true")
    p.add_argument("--no-follow-ego-view", action="store_true")
    p.add_argument("--no-world-reuse", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-step-freq", type=int, default=20)
    p.add_argument("--reset-retries", type=int, default=3)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--flush-every-steps", type=int, default=100)

    logs = p.add_mutually_exclusive_group()
    logs.add_argument("--log-steps", dest="log_steps", action="store_true")
    logs.add_argument("--no-log-steps", dest="log_steps", action="store_false")
    p.set_defaults(log_steps=True)

    writes = p.add_mutually_exclusive_group()
    writes.add_argument("--resume", action="store_true")
    writes.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    if args.episodes < 1:
        p.error("--episodes must be at least 1")
    for lo_name, hi_name in (("npc_min", "npc_max"), ("walker_min", "walker_max")):
        lo, hi = int(getattr(args, lo_name)), int(getattr(args, hi_name))
        if lo < 0 or hi < 0 or lo > hi:
            p.error(f"invalid range: {lo_name}={lo}, {hi_name}={hi}")
    if args.route_target_length <= 50.0:
        p.error("--route-target-length must exceed 50 metres")
    if args.fps < 1 or args.reset_retries < 1 or args.flush_every_steps < 1:
        p.error("fps, reset retries, and flush interval must be positive")
    return args


def canonical_town(name: str) -> str:
    base = str(name).replace("\\", "/").split("/")[-1].split(".")[0]
    return base[:-4] if base.endswith("_Opt") else base


def resolve_protocol(town: str, requested: str) -> str:
    if requested != "auto":
        return requested
    key = canonical_town(town)
    if key == "Town10HD":
        return "source_stress"
    if key == "Town05":
        return "zero_shot"
    if key == "Town02":
        return "cross_town"
    return "extended"


def expected_weather(protocol: str) -> str:
    return "night_rain_fog" if protocol == "source_stress" else "mixed"


def resolve_weather(protocol: str, requested: str) -> str:
    return expected_weather(protocol) if requested == "auto" else requested


def default_checkpoint(protocol: str) -> str:
    name = "target_agent.pt" if protocol == "cross_town" else "source_agent.pt"
    return f"./culrt_carla_0915_aligned/models/{name}"


def resolve_path(path_text: str, module_path: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd() / path, module_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[1].resolve()


def checkpoint_config(car: Any, checkpoint: Path) -> Any:
    cfg = car.Config()
    try:
        try:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(checkpoint), map_location="cpu")
        saved_cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
        if isinstance(saved_cfg, dict):
            for key, value in saved_cfg.items():
                if hasattr(cfg, key):
                    try:
                        setattr(cfg, key, value)
                    except (AttributeError, TypeError, ValueError):
                        pass
    except Exception as exc:
        raise RuntimeError(f"Could not inspect checkpoint {checkpoint}: {exc}") from exc
    finally:
        if "payload" in locals():
            del payload
    return cfg


def apply_eval_overrides(cfg: Any, args: argparse.Namespace, out_dir: Path, weather: str) -> Any:
    cfg.mode_name = "eval"
    cfg.seed = int(args.seed)
    cfg.fps = int(args.fps)
    cfg.tm_port = int(args.tm_port)
    cfg.client_timeout = float(args.server_timeout)
    cfg.debug_mode = bool(args.debug)
    cfg.debug_step_freq = max(1, int(args.debug_step_freq))
    cfg.follow_ego_view = not bool(args.no_follow_ego_view)
    cfg.no_rendering_mode = bool(args.no_rendering)
    cfg.reuse_loaded_world = not bool(args.no_world_reuse)
    cfg.use_safety_shield = bool(args.use_safety_shield)
    cfg.red_light_assist = bool(args.red_light_assist)
    cfg.use_training_curriculum = False
    if args.attention_score_form != "auto":
        cfg.attention_score_form = str(args.attention_score_form)
    cfg.npc_min, cfg.npc_max = sorted((int(args.npc_min), int(args.npc_max)))
    cfg.walker_min, cfg.walker_max = sorted((int(args.walker_min), int(args.walker_max)))
    cfg.fixed_goal_index = int(args.goal_index)
    cfg.target_goal_index = int(args.goal_index)
    cfg.use_fixed_destination = int(args.goal_index) >= 0
    cfg.strict_goal_route = False
    cfg.allow_fallback_route = True
    cfg.enforce_route_length_for_fixed_goal = True
    cfg.fixed_weather = weather
    cfg.target_weather = weather
    cfg.out_dir = str(out_dir)

    target = float(args.route_target_length)
    scale = target / 500.0
    tolerance = max(5.0, 10.0 * scale)
    cfg.route_target_length_m = target
    cfg.route_target_tolerance_m = tolerance
    cfg.route_soft_min_length_m = target - tolerance
    cfg.route_soft_max_length_m = target + tolerance
    cfg.min_route_length_m = max(50.0, target - 2.0 * tolerance)
    cfg.candidate_goal_min_dist_m = min(float(cfg.candidate_goal_min_dist_m), target * 0.20)
    cfg.candidate_goal_max_tries = max(160, int(160 * scale))
    cfg.max_episode_steps = (
        int(args.max_episode_steps)
        if int(args.max_episode_steps) > 0
        else max(2500, int(6000 * scale))
    )
    return cfg


def validate_protocol(
    args: argparse.Namespace,
    protocol: str,
    weather: str,
    checkpoint: Path,
    cfg: Any,
) -> None:
    problems: List[str] = []
    town = canonical_town(args.town)
    expected_town = {"source_stress": "Town10HD", "zero_shot": "Town05", "cross_town": "Town02"}.get(protocol)
    if expected_town and town != expected_town:
        problems.append(f"protocol {protocol} expects {expected_town}, received {town}")
    if protocol != "extended" and weather != expected_weather(protocol):
        problems.append(f"protocol {protocol} expects weather={expected_weather(protocol)}, received {weather}")
    checkpoint_name = checkpoint.name.lower()
    if protocol == "cross_town" and "source_agent" in checkpoint_name:
        problems.append("cross_town requires an adapted target checkpoint, not source_agent.pt")
    if protocol in ("source_stress", "zero_shot") and "target_agent" in checkpoint_name:
        problems.append(f"{protocol} requires the frozen source checkpoint, not target_agent.pt")
    if str(getattr(cfg, "attention_score_form", "reliability")) != "reliability":
        problems.append("paper text/Table I require attention_score_form=reliability")
    if problems:
        message = "Protocol mismatch:\n  - " + "\n  - ".join(problems)
        if args.allow_protocol_mismatch:
            print(f"[WARN] {message}")
        else:
            raise ValueError(message + "\nUse --allow-protocol-mismatch only for a deliberately non-paper run.")


def freeze_agent(agent: Any) -> None:
    modules: List[Any] = [agent.actor, *list(agent.critics), *list(agent.target_critics)]
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


@torch.inference_mode()
def policy_diagnostics(agent: Any, obs: Dict[str, np.ndarray], action: np.ndarray) -> Dict[str, Any]:
    scalars = torch.from_numpy(obs["scalars"][None]).to(agent.device).float()
    edges = torch.from_numpy(obs["edges"][None]).to(agent.device).float()
    mask = torch.from_numpy(obs["mask"][None]).to(agent.device).float()
    act = torch.from_numpy(np.asarray(action, dtype=np.float32)[None]).to(agent.device).float()
    _, logstd, attention, sigma_ale = agent.actor.forward(scalars, edges, mask)
    qs = agent.critic_ensemble_nograd(scalars, edges, mask, act)
    sigma_bar, sigma_epi = agent.compute_sigma_bar(
        sigma_ale.reshape(1), qs.reshape(agent.cfg.n_critics, 1)
    )
    gaussian_entropy = (0.5 * math.log(2.0 * math.pi * math.e) + logstd).sum(dim=-1)
    att = attention.reshape(-1).cpu().numpy().astype(np.float32)
    return {
        "sigma_bar": float(sigma_bar.reshape(-1)[0].cpu().item()),
        "sigma_ale": float(sigma_ale.reshape(-1)[0].cpu().item()),
        "sigma_epi": float(sigma_epi.reshape(-1)[0].cpu().item()),
        "policy_entropy_proxy": float(gaussian_entropy.reshape(-1)[0].cpu().item()),
        "attention": att,
        "top_attention_index": int(np.argmax(att)) if att.size else -1,
        "top_attention_value": float(np.max(att)) if att.size else 0.0,
    }


def available_towns(car: Any, host: str, port: int, timeout: float) -> Dict[str, str]:
    client = car.carla.Client(host, int(port))
    client.set_timeout(float(timeout))
    result: Dict[str, str] = {}
    for item in client.get_available_maps():
        base = str(item).replace("\\", "/").split("/")[-1].split(".")[0]
        result.setdefault(canonical_town(base), base)
    return result


def wait_for_server(car: Any, host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + float(timeout)
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            client = car.carla.Client(host, int(port))
            client.set_timeout(min(5.0, float(timeout)))
            client.get_server_version()
            client.get_world().get_map().name
            return
        except BaseException as exc:
            last_error = exc
            time.sleep(2.0)
    raise TimeoutError(f"CARLA at {host}:{port} did not answer within {timeout:.0f}s: {last_error}")


def make_env_builder(car: Any, args: argparse.Namespace, cfg: Any, weather: str) -> Callable[[], Any]:
    def build() -> Any:
        return car.CarlaReliableTransferEnv(
            host=args.host,
            port=int(args.port),
            town_name=args.town,
            fixed_spawn_index=int(args.spawn_index),
            fixed_goal_index=int(args.goal_index),
            weather_mode=weather,
            cfg=cfg,
        )
    return build


def reset_with_retries(
    env: Optional[Any],
    builder: Callable[[], Any],
    npc_count: int,
    walker_count: int,
    goal_index: int,
    retries: int,
) -> Tuple[Any, Dict[str, np.ndarray], Dict[str, Any]]:
    waits = (2.0, 4.0, 8.0, 12.0)
    current = env
    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            if current is None:
                current = builder()
            obs, info = current.reset(options={
                "npc_count": int(npc_count),
                "walker_count": int(walker_count),
                "goal_index": int(goal_index),
            })
            return current, obs, info
        except BaseException as exc:
            last_error = exc
            print(f"[WARN] reset attempt {attempt}/{retries} failed: {exc}")
            if current is not None:
                try:
                    current.close()
                except BaseException:
                    pass
            current = None
            if attempt < retries:
                time.sleep(waits[min(attempt - 1, len(waits) - 1)])
    raise RuntimeError(f"reset failed after {retries} attempts: {last_error}")


def collision_kind(env: Any) -> str:
    for event in reversed(list(getattr(env, "collision_events", []))):
        try:
            type_id = str(event.other_actor.type_id).lower()
        except BaseException:
            type_id = ""
        if type_id.startswith("walker.pedestrian"):
            return "pedestrian"
        if type_id.startswith("vehicle."):
            return "vehicle"
        if type_id:
            return "static"
    return "unknown"


def local_bounded_score(
    route_completion_pct: float,
    collision: int,
    collision_type: str,
    red_light_events: int,
    timeout: int,
) -> Tuple[float, float]:
    """Paper DS=RC*IS using the observable subset of Leaderboard 2.1 events."""
    collision_coeff = {
        "pedestrian": 1.00,
        "vehicle": 0.70,
        "static": 0.60,
        "unknown": 0.70,
    }.get(collision_type, 0.70)
    weighted = (
        collision_coeff * int(bool(collision))
        + 0.40 * int(red_light_events)
        + 0.40 * int(bool(timeout))
    )
    infraction_score = 1.0 / (1.0 + weighted)
    driving_score = float(np.clip(route_completion_pct, 0.0, 100.0)) * infraction_score
    return float(infraction_score), float(driving_score)


@dataclass
class EpisodeResult:
    protocol: str
    town: str
    weather: str
    episode: int
    seed: int
    checkpoint_role: str
    checkpoint_training_steps: int
    calibrator_fitted: int
    requested_npc: int
    spawned_npc: int
    requested_walkers: int
    spawned_walkers: int
    spawn_index: int
    goal_index: int
    route_length_m: float
    steps: int
    total_reward: float
    success: int
    terminal_reason: str
    route_completion_pct: float
    distance_km: float
    collision: int
    collision_type: str
    off_route: int
    timeout: int
    stuck: int
    red_light_events: int
    intervention_rate: float
    mean_cte_m: float
    mean_abs_route_heading_rad: float
    mean_speed_kmh: float
    min_ttc_s: float
    ttc_observed: int
    mean_sigma_bar: float
    mean_sigma_ale: float
    mean_sigma_epi: float
    mean_policy_entropy_proxy: float
    local_infraction_score: float
    local_driving_score: float
    official_leaderboard_score: int
    server_error: int


STEP_FIELDS = [
    "protocol", "town", "weather", "episode", "t", "sim_time_s",
    "ego_x", "ego_y", "ego_z", "ego_yaw_deg",
    "route_s_m", "route_remaining_m", "route_completion_pct", "cte_m", "route_heading_rad",
    "speed_kmh", "reward",
    "command_throttle", "command_brake", "command_steer",
    "applied_throttle", "applied_brake", "applied_steer",
    "sigma_bar", "sigma_ale", "sigma_epi", "policy_entropy_proxy",
    "attention", "top_attention_index", "top_attention_value", "ttc_s",
    "collision", "off_route", "timeout", "red_light_violation", "intervention", "terminal_reason",
]

ROUTE_FIELDS = [
    "protocol", "town", "weather", "episode", "route_point_index", "route_s_m",
    "x", "y", "z", "yaw_deg",
]

TRAJECTORY_FIELDS = [
    "protocol", "town", "weather", "episode", "t", "sim_time_s",
    "ego_x", "ego_y", "ego_z", "ego_yaw_deg", "route_s_m",
    "route_completion_pct", "cte_m", "route_heading_rad", "terminal_reason",
]


def mean_or(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(fmean(vals)) if vals else float(default)


def std_or(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(stdev(vals)) if len(vals) > 1 else float(default)


def reliability_proxy_ece(rows: Sequence[EpisodeResult], bins: int = 5) -> float:
    """Episode-level completion proxy; explicitly not a final per-step ECE."""
    if not rows:
        return float("nan")
    confidences = np.asarray([1.0 - np.clip(r.mean_sigma_bar, 0.0, 1.0) for r in rows])
    outcomes = np.asarray([float(r.success) for r in rows])
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (confidences >= edges[i]) & (
            confidences <= edges[i + 1] if i == bins - 1 else confidences < edges[i + 1]
        )
        if np.any(mask):
            result += float(mask.mean()) * abs(float(confidences[mask].mean() - outcomes[mask].mean()))
    return float(result)


def summarize(rows: Sequence[EpisodeResult], protocol: str, town: str, weather: str) -> Dict[str, Any]:
    distance = sum(r.distance_km for r in rows)
    denom = max(distance, 1e-9)
    collisions = sum(r.collision for r in rows)
    off_routes = sum(r.off_route for r in rows)
    timeouts = sum(r.timeout for r in rows)
    red_events = sum(r.red_light_events for r in rows)
    ttc_values = [r.min_ttc_s for r in rows if r.ttc_observed and np.isfinite(r.min_ttc_s)]
    return {
        "protocol": protocol,
        "town": town,
        "weather": weather,
        "episodes": len(rows),
        "success_rate_pct": 100.0 * sum(r.success for r in rows) / max(len(rows), 1),
        "mean_local_driving_score": mean_or(r.local_driving_score for r in rows),
        "std_local_driving_score": std_or(r.local_driving_score for r in rows),
        "mean_route_completion_pct": mean_or(r.route_completion_pct for r in rows),
        "std_route_completion_pct": std_or(r.route_completion_pct for r in rows),
        "mean_local_infraction_score": mean_or(r.local_infraction_score for r in rows),
        "std_local_infraction_score": std_or(r.local_infraction_score for r in rows),
        "mean_cte_m": mean_or(r.mean_cte_m for r in rows),
        "std_cte_m": std_or(r.mean_cte_m for r in rows),
        "mean_abs_route_heading_rad": mean_or(r.mean_abs_route_heading_rad for r in rows),
        "std_abs_route_heading_rad": std_or(r.mean_abs_route_heading_rad for r in rows),
        "mean_min_ttc_s": mean_or(ttc_values, float("nan")),
        "std_min_ttc_s": std_or(ttc_values, float("nan")),
        "ttc_observed_episodes": len(ttc_values),
        "mean_intervention_rate": mean_or(r.intervention_rate for r in rows),
        "std_intervention_rate": std_or(r.intervention_rate for r in rows),
        "mean_sigma_bar": mean_or(r.mean_sigma_bar for r in rows),
        "std_sigma_bar": std_or(r.mean_sigma_bar for r in rows),
        "mean_policy_entropy_proxy": mean_or(r.mean_policy_entropy_proxy for r in rows),
        "reliability_proxy_ece": reliability_proxy_ece(rows),
        "total_distance_km": distance,
        "collisions": collisions,
        "collisions_per_km": collisions / denom,
        "off_routes": off_routes,
        "off_routes_per_km": off_routes / denom,
        "timeouts": timeouts,
        "timeouts_per_km": timeouts / denom,
        "red_light_events": red_events,
        "red_light_events_per_km": red_events / denom,
        "server_errors": sum(r.server_error for r in rows),
        "official_leaderboard_score": 0,
    }


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def load_episode_rows(path: Path) -> List[EpisodeResult]:
    if not path.is_file():
        return []
    hints = get_type_hints(EpisodeResult)
    rows: List[EpisodeResult] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            values: Dict[str, Any] = {}
            for name, kind in hints.items():
                text = raw.get(name, "")
                if kind is int:
                    values[name] = int(float(text or 0))
                elif kind is float:
                    values[name] = float(text or "nan")
                else:
                    values[name] = str(text)
            rows.append(EpisodeResult(**values))
    return rows


def route_rows(env: Any, protocol: str, town: str, weather: str, episode: int) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    cumulative = list(getattr(env, "route_cumdist", []))
    for index, wp in enumerate(list(getattr(env, "route_wps", []))):
        try:
            tf = wp.transform
            loc = tf.location
            yaw = tf.rotation.yaw
        except BaseException:
            continue
        result.append({
            "protocol": protocol,
            "town": town,
            "weather": weather,
            "episode": episode,
            "route_point_index": index,
            "route_s_m": cumulative[index] if index < len(cumulative) else "",
            "x": float(loc.x),
            "y": float(loc.y),
            "z": float(loc.z),
            "yaw_deg": float(yaw),
        })
    return result


def combine_csv_files(paths: Sequence[Path], output: Path, fields: Sequence[str]) -> None:
    def rows() -> Iterable[Dict[str, Any]]:
        for path in paths:
            if not path.is_file():
                continue
            with path.open("r", newline="", encoding="utf-8") as handle:
                yield from csv.DictReader(handle)
    write_csv_atomic(output, fields, rows())


def evaluate(
    car: Any,
    agent: Any,
    cfg: Any,
    args: argparse.Namespace,
    protocol: str,
    weather: str,
    out_dir: Path,
    existing: Sequence[EpisodeResult],
    on_episode: Callable[[List[EpisodeResult]], None],
) -> List[EpisodeResult]:
    car.set_global_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    builder = make_env_builder(car, args, cfg, weather)
    completed = {row.episode for row in existing}
    rows = list(existing)
    env: Optional[Any] = None
    calibrator_fitted = int(bool(getattr(agent.sigma_cal, "fitted", False)))
    checkpoint_role = "adapted_target" if protocol == "cross_town" else "source"

    try:
        for episode in range(1, int(args.episodes) + 1):
            requested_npc = rng.randint(int(args.npc_min), int(args.npc_max))
            requested_walkers = rng.randint(int(args.walker_min), int(args.walker_max))
            if episode in completed:
                print(f"[RESUME] episode {episode:02d} already complete")
                continue

            episode_seed = int(args.seed) + episode
            car.set_global_seed(episode_seed)
            cfg.seed = episode_seed
            if env is not None:
                try:
                    env.tm.set_random_device_seed(episode_seed)
                except BaseException:
                    pass
            env, obs, reset_info = reset_with_retries(
                env, builder, requested_npc, requested_walkers,
                int(args.goal_index), int(args.reset_retries),
            )
            actual_town = str(reset_info.get("town", getattr(env, "town_name", "")))
            if canonical_town(actual_town) != canonical_town(args.town):
                raise RuntimeError(f"requested {args.town}, but CARLA loaded {actual_town}")

            route_path = out_dir / "routes" / f"episode_{episode:03d}.csv"
            write_csv_atomic(route_path, ROUTE_FIELDS, route_rows(env, protocol, args.town, weather, episode))

            step_final = out_dir / "steps" / f"episode_{episode:03d}.csv"
            step_part = step_final.with_suffix(".csv.part")
            step_handle = None
            step_writer: Optional[csv.DictWriter] = None
            if args.log_steps:
                step_part.parent.mkdir(parents=True, exist_ok=True)
                step_handle = step_part.open("w", newline="", encoding="utf-8")
                step_writer = csv.DictWriter(step_handle, fieldnames=STEP_FIELDS)
                step_writer.writeheader()

            steps = 0
            total_reward = 0.0
            ctes: List[float] = []
            headings: List[float] = []
            speeds: List[float] = []
            ttc_values: List[float] = []
            sigma_bar_values: List[float] = []
            sigma_ale_values: List[float] = []
            sigma_epi_values: List[float] = []
            entropy_values: List[float] = []
            red_events = 0
            red_active_previous = False
            server_error = 0
            last_info: Dict[str, Any] = {
                "route_completion_pct": 0.0,
                "distance_driven_m": 0.0,
                "success": False,
                "term_reason": "",
            }

            try:
                while True:
                    action = agent.act(obs, deterministic=True)
                    diag = policy_diagnostics(agent, obs, action)
                    try:
                        next_obs, _, terminated, truncated, info = env.step(action)
                    except BaseException as exc:
                        print(f"[WARN] episode {episode} step failed: {exc}")
                        info = dict(last_info)
                        info.update({
                            "success": False,
                            "collision": False,
                            "off_route": False,
                            "timeout": False,
                            "stuck": False,
                            "term_reason": "server_error",
                        })
                        server_error = 1
                        terminated, truncated = False, True
                        next_obs = obs

                    reward = 0.0 if server_error else float(car.build_reward_from_info(
                        info, sigma_bar=float(diag["sigma_bar"]), cfg=cfg
                    ))
                    steps += 1
                    total_reward += reward
                    cte = abs(float(car.safe_float(info.get("dL", 0.0))))
                    heading = abs(float(car.safe_float(info.get("route_heading_err", 0.0))))
                    ctes.append(cte)
                    headings.append(heading)
                    sigma_bar_values.append(float(diag["sigma_bar"]))
                    sigma_ale_values.append(float(diag["sigma_ale"]))
                    sigma_epi_values.append(float(diag["sigma_epi"]))
                    entropy_values.append(float(diag["policy_entropy_proxy"]))
                    ttc_raw = float(car.safe_float(info.get("time_to_conflict", 999.0), 999.0))
                    if 0.0 < ttc_raw < 900.0 and math.isfinite(ttc_raw):
                        ttc_values.append(ttc_raw)

                    try:
                        tf = env.vehicle.get_transform()
                        loc = tf.location
                        ego_x, ego_y, ego_z, ego_yaw = float(loc.x), float(loc.y), float(loc.z), float(tf.rotation.yaw)
                        speed_kmh = 3.6 * float(car.vec3_length(env.vehicle.get_velocity()))
                    except BaseException:
                        ego_x = ego_y = ego_z = ego_yaw = speed_kmh = 0.0
                    speeds.append(speed_kmh)

                    red_active = bool(info.get("red_light_violation", False))
                    if red_active and not red_active_previous:
                        red_events += 1
                    red_active_previous = red_active
                    command = np.asarray(info.get("command_action", action), dtype=np.float32)
                    applied = np.asarray(info.get("applied_action", action), dtype=np.float32)

                    if step_writer is not None:
                        step_writer.writerow({
                            "protocol": protocol,
                            "town": args.town,
                            "weather": weather,
                            "episode": episode,
                            "t": steps,
                            "sim_time_s": steps / float(cfg.fps),
                            "ego_x": ego_x,
                            "ego_y": ego_y,
                            "ego_z": ego_z,
                            "ego_yaw_deg": ego_yaw,
                            "route_s_m": car.safe_float(info.get("route_s", 0.0)),
                            "route_remaining_m": car.safe_float(info.get("goal_dist", 0.0)),
                            "route_completion_pct": car.safe_float(info.get("route_completion_pct", 0.0)),
                            "cte_m": cte,
                            "route_heading_rad": heading,
                            "speed_kmh": speed_kmh,
                            "reward": reward,
                            "command_throttle": float(command[0]),
                            "command_brake": float(command[1]),
                            "command_steer": float(command[2]),
                            "applied_throttle": float(applied[0]),
                            "applied_brake": float(applied[1]),
                            "applied_steer": float(applied[2]),
                            "sigma_bar": diag["sigma_bar"],
                            "sigma_ale": diag["sigma_ale"],
                            "sigma_epi": diag["sigma_epi"],
                            "policy_entropy_proxy": diag["policy_entropy_proxy"],
                            "attention": "|".join(f"{v:.7f}" for v in diag["attention"]),
                            "top_attention_index": diag["top_attention_index"],
                            "top_attention_value": diag["top_attention_value"],
                            "ttc_s": ttc_raw if ttc_raw < 900.0 else "",
                            "collision": int(bool(info.get("collision", False))),
                            "off_route": int(bool(info.get("off_route", False))),
                            "timeout": int(bool(info.get("timeout", False))),
                            "red_light_violation": int(red_active),
                            "intervention": int(bool(info.get("shield_active", False))),
                            "terminal_reason": str(info.get("term_reason", "")),
                        })
                        if steps % int(args.flush_every_steps) == 0:
                            step_handle.flush()

                    last_info = dict(info)
                    obs = next_obs
                    if bool(terminated or truncated):
                        break
            finally:
                if step_handle is not None:
                    step_handle.flush()
                    os.fsync(step_handle.fileno())
                    step_handle.close()
                    os.replace(step_part, step_final)

            collision = int(bool(last_info.get("collision", False)))
            kind = collision_kind(env) if collision else "none"
            timeout = int(bool(last_info.get("timeout", False)))
            completion = float(np.clip(car.safe_float(last_info.get("route_completion_pct", 0.0)), 0.0, 100.0))
            infraction_score, driving_score = local_bounded_score(
                completion, collision, kind, red_events, timeout
            )
            row = EpisodeResult(
                protocol=protocol,
                town=args.town,
                weather=weather,
                episode=episode,
                seed=episode_seed,
                checkpoint_role=checkpoint_role,
                checkpoint_training_steps=int(getattr(agent, "training_steps", 0)),
                calibrator_fitted=calibrator_fitted,
                requested_npc=requested_npc,
                spawned_npc=int(reset_info.get("npc_count", requested_npc)),
                requested_walkers=requested_walkers,
                spawned_walkers=int(reset_info.get("walker_count", requested_walkers)),
                spawn_index=int(reset_info.get("spawn_index", -1) if reset_info.get("spawn_index") is not None else -1),
                goal_index=int(reset_info.get("goal_index", -1) if reset_info.get("goal_index") is not None else -1),
                route_length_m=float(car.safe_float(reset_info.get("route_total_len_m", 0.0))),
                steps=steps,
                total_reward=total_reward,
                success=int(bool(last_info.get("success", False))),
                terminal_reason=str(last_info.get("term_reason", "")),
                route_completion_pct=completion,
                distance_km=float(car.safe_float(last_info.get("distance_driven_m", 0.0))) / 1000.0,
                collision=collision,
                collision_type=kind,
                off_route=int(bool(last_info.get("off_route", last_info.get("off_road", False)))),
                timeout=timeout,
                stuck=int(bool(last_info.get("stuck", False))),
                red_light_events=red_events,
                intervention_rate=float(car.safe_float(last_info.get("intervention_rate", 0.0))),
                mean_cte_m=mean_or(ctes),
                mean_abs_route_heading_rad=mean_or(headings),
                mean_speed_kmh=mean_or(speeds),
                min_ttc_s=min(ttc_values) if ttc_values else float("nan"),
                ttc_observed=int(bool(ttc_values)),
                mean_sigma_bar=mean_or(sigma_bar_values),
                mean_sigma_ale=mean_or(sigma_ale_values),
                mean_sigma_epi=mean_or(sigma_epi_values),
                mean_policy_entropy_proxy=mean_or(entropy_values),
                local_infraction_score=infraction_score,
                local_driving_score=driving_score,
                official_leaderboard_score=0,
                server_error=server_error,
            )
            rows.append(row)
            rows.sort(key=lambda item: item.episode)
            on_episode(rows)
            print(
                f"[EVAL] town={args.town} weather={weather} ep={episode:02d} "
                f"success={row.success} RC={row.route_completion_pct:.1f}% "
                f"IS_local={row.local_infraction_score:.3f} DS_local={row.local_driving_score:.1f} "
                f"reason={row.terminal_reason} NPC={row.spawned_npc} walkers={row.spawned_walkers}"
            )

            if server_error:
                try:
                    env.close()
                except BaseException:
                    pass
                env = None
                wait_for_server(car, args.host, int(args.port), float(args.server_timeout))
                if args.fail_fast:
                    raise RuntimeError(f"CARLA server error in episode {episode}")
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as exc:
                print(f"[WARN] environment close failed: {exc}")
    return rows


def main() -> int:
    args = parse_args()
    car, module_path = load_car_module(args.car_module)
    protocol = resolve_protocol(args.town, args.protocol)
    weather = resolve_weather(protocol, args.weather)
    checkpoint_text = args.checkpoint or default_checkpoint(protocol)
    checkpoint = resolve_path(checkpoint_text, module_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.out_dir:
        out_dir = resolve_path(args.out_dir, module_path)
    else:
        out_dir = resolve_path(
            f"./culrt_carla_0915_aligned/evaluation/{canonical_town(args.town)}_{protocol}_{weather}",
            module_path,
        )
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    episodes_path = out_dir / "episodes.csv"
    if episodes_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Results already exist: {episodes_path}. Use --resume or --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = checkpoint_config(car, checkpoint)
    cfg = apply_eval_overrides(cfg, args, out_dir, weather)
    validate_protocol(args, protocol, weather, checkpoint, cfg)
    car.set_global_seed(int(args.seed))
    wait_for_server(car, args.host, int(args.port), float(args.server_timeout))
    maps = available_towns(car, args.host, int(args.port), min(float(args.server_timeout), 30.0))
    if canonical_town(args.town) not in maps:
        raise ValueError(f"Town {args.town} is unavailable. CARLA reports: {', '.join(sorted(maps.values()))}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    agent = car.SACAgent(device=device, cfg=cfg)
    agent.load(str(checkpoint))
    freeze_agent(agent)
    calibrator_fitted = bool(getattr(agent.sigma_cal, "fitted", False))
    if not calibrator_fitted:
        print(
            "[WARN] checkpoint uncertainty calibrator is not fitted. Evaluation will not fit it "
            "on test data; sigma_bar is recorded but cannot be called held-out calibrated."
        )

    existing = load_episode_rows(episodes_path) if args.resume else []
    if existing:
        first = existing[0]
        if (
            canonical_town(first.town) != canonical_town(args.town)
            or first.weather != weather
            or first.protocol != protocol
        ):
            raise ValueError("Existing episodes.csv does not match this town/weather/protocol")

    episode_fields = list(EpisodeResult.__dataclass_fields__.keys())
    summary_path = out_dir / "summary.csv"
    report_path = out_dir / "summary.json"

    def persist(rows: List[EpisodeResult]) -> None:
        summary = summarize(rows, protocol, args.town, weather)
        write_csv_atomic(episodes_path, episode_fields, (asdict(row) for row in rows))
        write_csv_atomic(summary_path, list(summary.keys()), [summary])
        report = {
            "evaluation_only": True,
            "one_town_per_process": True,
            "protocol": protocol,
            "town": args.town,
            "weather": weather,
            "checkpoint": str(checkpoint),
            "checkpoint_training_steps": int(getattr(agent, "training_steps", 0)),
            "calibrator_fitted_on_training_data": calibrator_fitted,
            "device": str(device),
            "arguments": vars(args),
            "summary": summary,
            "score_definition": {
                "formula": "DS_local_i = RC_i * IS_local_i",
                "infraction_formula": "IS_local_i = 1 / (1 + sum(c_j * event_count_j))",
                "included": ["collision_pedestrian", "collision_vehicle", "collision_static", "red_light", "timeout"],
                "unsupported_official_events": ["stop_sign", "emergency_yield", "minimum_speed", "proportional_offroad"],
                "official_leaderboard_score": False,
            },
            "reliability_note": "reliability_proxy_ece is an episode-level proxy, not final per-step ECE",
        }
        write_json_atomic(report_path, json_safe(report))

    print("Mode: evaluation only")
    print(f"Protocol: {protocol}")
    print(f"Town: {args.town} | weather: {weather} | episodes: {args.episodes}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint role: {'adapted target' if protocol == 'cross_town' else 'source'}")
    print(f"Device: {device} | policy: deterministic, frozen, no gradients")
    print(f"Traffic: NPC=[{args.npc_min},{args.npc_max}] walkers=[{args.walker_min},{args.walker_max}]")
    print(f"Safety shield: {cfg.use_safety_shield} | red-light assist: {cfg.red_light_assist}")

    try:
        rows = evaluate(
            car, agent, cfg, args, protocol, weather, out_dir, existing, persist
        )
    except BaseException:
        if existing and not episodes_path.exists():
            persist(list(existing))
        raise
    if not rows:
        raise RuntimeError("No completed evaluation episodes")
    persist(rows)

    route_files = sorted((out_dir / "routes").glob("episode_*.csv"))
    combine_csv_files(route_files, out_dir / "routes.csv", ROUTE_FIELDS)
    if args.log_steps:
        step_files = sorted((out_dir / "steps").glob("episode_*.csv"))
        combine_csv_files(step_files, out_dir / "steps.csv", STEP_FIELDS)

        def trajectory_rows() -> Iterable[Dict[str, Any]]:
            for path in step_files:
                with path.open("r", newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        yield {field: row.get(field, "") for field in TRAJECTORY_FIELDS}
        write_csv_atomic(out_dir / "trajectories.csv", TRAJECTORY_FIELDS, trajectory_rows())

    summary = summarize(rows, protocol, args.town, weather)
    print(
        f"[DONE] episodes={summary['episodes']} SR={summary['success_rate_pct']:.1f}% "
        f"RC={summary['mean_route_completion_pct']:.2f}±{summary['std_route_completion_pct']:.2f}% "
        f"DS_local={summary['mean_local_driving_score']:.2f}±{summary['std_local_driving_score']:.2f} "
        f"IS_local={summary['mean_local_infraction_score']:.3f} "
        f"coll/km={summary['collisions_per_km']:.4f} timeout/km={summary['timeouts_per_km']:.4f}"
    )
    print(f"Results: {out_dir}")
    return 2 if int(summary["server_errors"]) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
