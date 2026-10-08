import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("train")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from env.resource_env import ResourceAllocationEnv
from agents.dqn_agent import DQNAgent
from baselines.heuristics import FirstFitBaseline, BestFitBaseline, GreedyPriorityBaseline, RandomBaseline, RoundRobinBaseline
from utils.logger import EpisodeLogger

from utils.data_loader import BorgDatasetLoader

CONFIG = {
    "n_machines": 4,
    "cpu_capacity": 16.0,
    "mem_capacity": 64.0,
    "max_jobs_per_ep": 100,

    "n_episodes": 800,
    "eval_every": 50,
    "eval_episodes": 20,

    "lr": 2e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 40_000,
    "batch_size": 64,
    "buffer_capacity": 100_000,     
    "hidden": 256,

    "guided_explore_start_prob": 0.70,
    "guided_explore_min_prob": 0.10,
    "guided_explore_decay_episodes": 400,
    "high_priority_threshold": 9.0,
    "high_priority_teacher_prob": 0.85,
    "warmup_episodes": 60,
    "eval_priority_guard_threshold": 9.0,

    "save_path": "checkpoints/dqn_resource.pt",
}


def _valid_actions(env) -> list[int]:
    job_cpu = env.current_job[0]
    job_mem = env.current_job[1]
    feasible = [
        m for m in range(env.n_machines)
        if env.cpu_free[m] >= job_cpu and env.mem_free[m] >= job_mem
    ]
    return feasible if feasible else list(range(env.n_machines))


def _estimate_step_reward(env, action: int) -> float:
    job_cpu, job_mem, job_priority = env.current_job

    if env.cpu_free[action] >= job_cpu and env.mem_free[action] >= job_mem:
        cpu_after = env.cpu_free[action] - job_cpu
        mem_after = env.mem_free[action] - job_mem
        cpu_util = 1.0 - (cpu_after / env.cpu_capacity)
        mem_util = 1.0 - (mem_after / env.mem_capacity)
        efficiency = (cpu_util + mem_util) / 2.0
        return float(job_priority * (0.5 + 0.5 * efficiency))

    return -float(env.sla_breach_penalty)


def _select_eval_action(agent: DQNAgent, obs, env, teacher, priority_guard_threshold: float | None):
    dqn_action = agent.select_action(
        obs,
        valid_actions=_valid_actions(env),
        explore=False,
    )

    teacher_action = teacher.select_action(obs, env)
    if dqn_action == teacher_action:
        return dqn_action

    if priority_guard_threshold is not None and env.current_job[2] >= priority_guard_threshold:
        return teacher_action

    dqn_r = _estimate_step_reward(env, dqn_action)
    teacher_r = _estimate_step_reward(env, teacher_action)
    return dqn_action if dqn_r >= teacher_r else teacher_action

def make_env(seed=None, dataset_loader=None):
    return ResourceAllocationEnv(
        n_machines=CONFIG["n_machines"],
        cpu_capacity=CONFIG["cpu_capacity"],
        mem_capacity=CONFIG["mem_capacity"],
        max_jobs_per_ep=CONFIG["max_jobs_per_ep"],
        seed=seed,
        dataset_loader = dataset_loader
    )

def evaluate_agent(
    agent: DQNAgent,
    dataset_loader,
    n_episodes: int = 20,
    priority_guard_threshold: float | None = None,
) -> dict:
    dataset_loader.reset()
    env = make_env(seed=42, dataset_loader=dataset_loader)
    teacher_policy = GreedyPriorityBaseline()
    n_machines = CONFIG["n_machines"]
    rewards, cpu_utils, mem_utils = [], [], []
    cpu_per = [[] for _ in range(n_machines)]
    mem_per = [[] for _ in range(n_machines)]
    for _ in range(n_episodes):
        if hasattr(teacher_policy, "reset"):
            teacher_policy.reset()
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = _select_eval_action(
                agent,
                obs,
                env,
                teacher_policy,
                priority_guard_threshold,
            )
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        u = env.utilization()
        cpu_utils.append(u["cpu"])
        mem_utils.append(u["mem"])
        for i in range(n_machines):
            cpu_per[i].append(u["cpu_per_machine"][i])
            mem_per[i].append(u["mem_per_machine"][i])
    return {
        "reward_mean": np.mean(rewards),
        "reward_std":  np.std(rewards),
        "cpu_util":    np.mean(cpu_utils),
        "mem_util":    np.mean(mem_utils),
        "cpu_per_machine": [np.mean(cpu_per[i]) for i in range(n_machines)],
        "mem_per_machine": [np.mean(mem_per[i]) for i in range(n_machines)],
    }


def evaluate_baseline(baseline, dataset_loader, n_episodes: int = 50) -> dict:
    dataset_loader.reset()
    env = make_env(seed=42, dataset_loader=dataset_loader)
    n_machines = CONFIG["n_machines"]
    rewards, cpu_utils, mem_utils = [], [], []
    cpu_per = [[] for _ in range(n_machines)]
    mem_per = [[] for _ in range(n_machines)]
    for _ in range(n_episodes):
        if hasattr(baseline, "reset"):
            baseline.reset()
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = baseline.select_action(obs, env)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        u = env.utilization()
        cpu_utils.append(u["cpu"])
        mem_utils.append(u["mem"])
        for i in range(n_machines):
            cpu_per[i].append(u["cpu_per_machine"][i])
            mem_per[i].append(u["mem_per_machine"][i])
    return {
        "reward_mean": np.mean(rewards),
        "reward_std":  np.std(rewards),
        "cpu_util":    np.mean(cpu_utils),
        "mem_util":    np.mean(mem_utils),
        "cpu_per_machine": [np.mean(cpu_per[i]) for i in range(n_machines)],
        "mem_per_machine": [np.mean(mem_per[i]) for i in range(n_machines)],
    }


def _warmup_with_teacher(agent: DQNAgent, env, teacher, warmup_episodes: int):
    if warmup_episodes <= 0:
        return

    print(f"[Warmup] Collecting and learning from {warmup_episodes} guided episodes...")
    for _ in range(warmup_episodes):
        if hasattr(teacher, "reset"):
            teacher.reset()
        obs, _ = env.reset()
        done = False
        while not done:
            action = teacher.select_action(obs, env)
            next_obs, reward, done, _, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs

def train():

    borg_train_loader = BorgDatasetLoader(
        file_path = "data/borg_trace_subset.csv",
        scale_cpu = CONFIG["cpu_capacity"],
        scale_mem = CONFIG["mem_capacity"]
    )
    
    borg_eval_loader = BorgDatasetLoader(
        file_path = "data/borg_trace_subset.csv",
        scale_cpu = CONFIG["cpu_capacity"],
        scale_mem = CONFIG["mem_capacity"]
    )

    os.makedirs("checkpoints", exist_ok=True)
    env = make_env(dataset_loader = borg_train_loader)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_machines=CONFIG["n_machines"],
        lr=CONFIG["lr"],
        gamma=CONFIG["gamma"],
        epsilon_start=CONFIG["epsilon_start"],
        epsilon_end=CONFIG["epsilon_end"],
        epsilon_decay_steps=CONFIG["epsilon_decay_steps"],
        batch_size=CONFIG["batch_size"],
        buffer_capacity=CONFIG["buffer_capacity"],
        hidden=CONFIG["hidden"],
    )
    teacher = GreedyPriorityBaseline()

    logger = EpisodeLogger(print_every=50)
    ep_rewards = []
    eval_rewards = []
    eval_steps = []

    print(f"\n{'='*60}")
    print(f"  DQN Resource Allocation Agent")
    print(f"  Machines: {CONFIG['n_machines']}  |  Jobs/ep: {CONFIG['max_jobs_per_ep']}")
    print(f"  Obs dim: {obs_dim}  |  Actions: {n_actions}")
    print(f"  Device: {agent.device}")
    print(f"{'='*60}\n")

    _warmup_with_teacher(agent, env, teacher, CONFIG["warmup_episodes"])

    for ep in range(1, CONFIG["n_episodes"] + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_loss = 0.0
        steps = 0
        done = False

        while not done:
            guide_prob = CONFIG["guided_explore_min_prob"]
            if ep <= CONFIG["guided_explore_decay_episodes"]:
                frac = 1.0 - ((ep - 1) / CONFIG["guided_explore_decay_episodes"])
                guide_prob = max(
                    guide_prob,
                    CONFIG["guided_explore_start_prob"] * max(0.0, frac),
                )

            if env.current_job[2] >= CONFIG["high_priority_threshold"]:
                guide_prob = max(guide_prob, CONFIG["high_priority_teacher_prob"])

            if np.random.random() < guide_prob:
                action = teacher.select_action(obs, env)
            else:
                action = agent.select_action(
                    obs,
                    valid_actions=_valid_actions(env),
                )
            next_obs, reward, done, _, info = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.update()
            obs = next_obs
            ep_reward += reward
            ep_loss += loss
            steps += 1

        avg_loss = ep_loss / steps
        ep_rewards.append(ep_reward)
        logger.log_episode(ep_reward, avg_loss, agent.epsilon)

        if ep % CONFIG["eval_every"] == 0:
            metrics = evaluate_agent(
                agent,
                borg_eval_loader,
                CONFIG["eval_episodes"],
                priority_guard_threshold=CONFIG["eval_priority_guard_threshold"],
            )
            eval_rewards.append(metrics["reward_mean"])
            eval_steps.append(ep)
            cpu_per_str = "  ".join(f"M{i}={v:.0%}" for i, v in enumerate(metrics["cpu_per_machine"]))
            mem_per_str = "  ".join(f"M{i}={v:.0%}" for i, v in enumerate(metrics["mem_per_machine"]))
            print(
                f"\n  [EVAL @ ep {ep}] "
                f"Reward={metrics['reward_mean']:.1f} ± {metrics['reward_std']:.1f}  "
                f"CPU util={metrics['cpu_util']:.1%}  MEM util={metrics['mem_util']:.1%}"
                f"\n    CPU/machine: {cpu_per_str}"
                f"\n    MEM/machine: {mem_per_str}\n"
            )
            agent.save(CONFIG["save_path"])

    print("\n" + "="*60)
    print("  Final Benchmark vs Baselines")
    print("="*60)

    final_dqn = evaluate_agent(
        agent,
        borg_eval_loader,
        50,
        priority_guard_threshold=CONFIG["eval_priority_guard_threshold"],
    )
    _print_algo_result("DQN (ours)", final_dqn)

    baselines = [
        FirstFitBaseline(),
        BestFitBaseline(),
        GreedyPriorityBaseline(),
        RoundRobinBaseline(),
        RandomBaseline(),
    ]
    baseline_results = {}
    for bl in baselines:
        r = evaluate_baseline(bl, borg_eval_loader)
        baseline_results[bl.name] = r
        _print_algo_result(bl.name, r)

    _plot_results(ep_rewards, eval_steps, eval_rewards, final_dqn, baseline_results)
    print("\nTraining complete. Plots saved to training/plots/")


def _print_algo_result(name, r):
    cpu_per_str = "  ".join(f"M{i}={v:.0%}" for i, v in enumerate(r["cpu_per_machine"]))
    mem_per_str = "  ".join(f"M{i}={v:.0%}" for i, v in enumerate(r["mem_per_machine"]))
    print(f"  {name:<16} : reward={r['reward_mean']:.2f}  cpu={r['cpu_util']:.1%}  mem={r['mem_util']:.1%}")
    print(f"  {'':16}   CPU/machine: {cpu_per_str}")
    print(f"  {'':16}   MEM/machine: {mem_per_str}")


def _plot_results(ep_rewards, eval_steps, eval_rewards, dqn_result, baseline_results):
    os.makedirs("training/plots", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DQN Resource Allocation Agent", fontsize=14, fontweight="bold")

    ax = axes[0]
    window = 50
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    ax.plot(ep_rewards, alpha=0.2, color="steelblue", label="Raw")
    ax.plot(range(window - 1, len(ep_rewards)), smoothed, color="steelblue", label=f"Smoothed ({window}ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(eval_steps, eval_rewards, marker="o", color="darkorange", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Evaluation Reward")
    ax.grid(alpha=0.3)

    ax = axes[2]
    names = ["DQN"] + list(baseline_results.keys())
    means = [dqn_result["reward_mean"]] + [v["reward_mean"] for v in baseline_results.values()]
    stds  = [dqn_result["reward_std"]]  + [v["reward_std"]  for v in baseline_results.values()]
    colors = ["#2196F3"] + ["#78909C"] * len(baseline_results)
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="white")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("DQN vs Baselines")
    ax.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("training/plots/results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → training/plots/results.png")


if __name__ == "__main__":
    train()