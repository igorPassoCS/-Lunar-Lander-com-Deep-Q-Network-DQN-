"""Treina uma configuração do DQN e salva métricas/modelo em results/.

Uso: python run_experiments.py <full|no_target|no_replay>
Cada config roda em processo próprio (paralelizável) com seed fixa = 42.
"""
import os
import sys
import json

import torch

import dqn

torch.set_num_threads(4)  # 3 configs em paralelo * 4 threads = 12 cores

SEED = 42
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

CONFIGS = {
    "full":      dqn.DQNConfig(name="full",      use_replay=True,  use_target=True),
    "no_target": dqn.DQNConfig(name="no_target", use_replay=True,  use_target=False),
    "no_replay": dqn.DQNConfig(name="no_replay", use_replay=False, use_target=True),
}


def main(name: str) -> None:
    cfg = CONFIGS[name]
    # Ablações não resolvem -> sem early stop, rodam todos os episódios para
    # evidenciar a (in)estabilidade. A config completa pode parar ao resolver.
    early = name == "full"
    n_ep = 700
    res, agent = dqn.train(cfg, n_episodes=n_ep, seed=SEED, early_stop=early, verbose=True)

    os.makedirs(RESULTS, exist_ok=True)
    dqn.save_result(res, RESULTS)
    eval_score = dqn.evaluate(agent, n_episodes=20, seed=123)
    torch.save(agent.qnet.state_dict(), os.path.join(RESULTS, f"model_{name}.pt"))

    summary = {
        "name": name,
        "episodes_run": len(res.scores),
        "solved_episode": res.solved_episode,
        "best_moving_avg": max(res.moving_avg),
        "final_moving_avg": res.moving_avg[-1],
        "eval_greedy_mean_20ep": eval_score,
    }
    with open(os.path.join(RESULTS, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1])
