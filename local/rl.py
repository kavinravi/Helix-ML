"""Train standard Gymnasium environments with SB3; evaluate saved policies separately."""
import argparse
import json
import math
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

ALGORITHMS = {cls.__name__: cls for cls in (A2C, DDPG, DQN, PPO, SAC, TD3)}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def environment(config):
    options = {"max_episode_steps": config["maxEpisodeSteps"]} if config.get("maxEpisodeSteps") else {}
    env = gym.make(config["envId"], **options)
    if not env.spec.max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    return env


def device(config):
    value = os.environ.get("HELIX_DEVICE", config.get("device", "cpu"))
    # A successful Torch check does not guarantee every algorithm supports that GPU.
    x = torch.ones(2, device=value, requires_grad=True)
    x.sum().backward()
    return value


class Progress(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episodes.append({"step": self.num_timesteps, "reward": float(info["episode"]["r"]), "length": int(info["episode"]["l"])})
                # shortcut: retain the last 500 episodes; use streaming storage for longer learning curves.
                self.episodes = self.episodes[-500:]
        if self.num_timesteps % 1000 == 0:
            print(f"Training steps: {self.num_timesteps}", flush=True)
        return True


def train(config, output):
    output.mkdir(parents=True, exist_ok=True)
    env = Monitor(environment(config))
    try:
        algorithm = config["algorithm"]
        if algorithm == "DQN" and not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DQN requires discrete actions. Try SAC, TD3, DDPG, PPO or A2C for continuous actions.")
        if algorithm in {"SAC", "TD3", "DDPG"} and not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(f"{algorithm} requires continuous Box actions. Try PPO, A2C or DQN for discrete actions.")
        kwargs = dict(config.get("hyperparameters", {}))
        if "net_arch" in kwargs:
            kwargs["policy_kwargs"] = {"net_arch": kwargs.pop("net_arch")}
        policy = "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "CnnPolicy" if len(env.observation_space.shape or ()) == 3 else "MlpPolicy"
        selected_device = device(config)
        torch.set_num_threads(min(4, int(os.environ.get("OMP_NUM_THREADS", "4"))))
        model = ALGORITHMS[algorithm](policy, env, seed=config["seed"], device=selected_device, verbose=0, **kwargs)
        if model.device.type != selected_device:
            raise ValueError(f"SB3 selected {model.device}, but {selected_device} was requested")
        callback = Progress()
        model.learn(total_timesteps=config["totalTimesteps"], callback=callback, progress_bar=False)
        model.save(output / "policy.zip")
        dump(output / "training.json", {"timesteps": model.num_timesteps, "device": str(model.device), "episodes": callback.episodes,
            "policy": policy, "seed": config["seed"]})
    finally:
        env.close()


def evaluate(config, model_path):
    env = environment(config)
    try:
        model = ALGORITHMS[config["algorithm"]].load(model_path, env=env, device=device(config))
        rewards, lengths = [], []
        for seed in config["seeds"]:
            obs, _ = env.reset(seed=seed)
            state, start = None, True
            total, length = 0.0, 0
            while True:
                action, state = model.predict(obs, state=state, episode_start=np.array([start]), deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                start = False
                total += float(reward)
                length += 1
                if not math.isfinite(total):
                    raise ValueError("Environment returned a non-finite reward")
                if terminated or truncated:
                    break
            rewards.append(total)
            lengths.append(length)
        return {"score": float(np.mean(rewards)), "deviation": float(np.std(rewards)), "episodeReturns": rewards,
                "episodeLengths": lengths, "evaluationEpisodes": len(rewards), "reloadVerified": True}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "train", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.command == "check":
        env = environment(config)
        try:
            env.reset(seed=config["seed"])
            dump(args.output, {"envId": config["envId"], "observationSpace": str(env.observation_space), "actionSpace": str(env.action_space),
                               "maxEpisodeSteps": env.spec.max_episode_steps})
        finally:
            env.close()
    elif args.command == "train":
        train(config, Path(args.output))
    else:
        dump(args.output, evaluate(config, args.model))


if __name__ == "__main__":
    main()
