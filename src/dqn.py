"""
Deep Q-Network (DQN) para o ambiente LunarLander-v3, implementado do zero.

Nenhuma biblioteca de RL pronta (stable-baselines3, RLlib, keras-rl, ...) é usada:
o Gymnasium entra APENAS como simulador do ambiente. Todo o algoritmo (replay
buffer, target network, erro de Bellman, otimização) é escrito manualmente aqui
usando somente PyTorch para a rede neural.

O módulo é parametrizável para permitir as ablações do experimento comparativo:
    - use_replay=False  -> treina online, só na transição mais recente (sem buffer)
    - use_target=False  -> alvo de Bellman vem da própria rede online (sem target net)
"""
from __future__ import annotations

import os
import random
from collections import deque, namedtuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade (Python, NumPy, PyTorch).

    Necessário porque inicialização dos pesos, amostragem do replay, exploração
    epsilon-greedy e a dinâmica do ambiente são estocásticas. Sem seed fixa as
    curvas de aprendizado não são reproduzíveis e a comparação entre as
    configurações (item 3) ficaria injusta: diferenças poderiam vir do acaso e
    não do componente que estou isolando.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 1) Rede Q: aproxima Q(s, a), substituindo a tabela do Q-learning tabular
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """MLP que mapeia um estado (8 dims) para os Q-valores das 4 ações.

    Arquitetura: 8 -> 128 -> ReLU -> 128 -> ReLU -> 4 (saída linear).
    Saída linear porque Q-valores podem ser negativos e ilimitados.
    """

    def __init__(self, state_size: int = 8, action_size: int = 4, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --------------------------------------------------------------------------- #
# 2) Experience Replay
# --------------------------------------------------------------------------- #
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Memória circular de transições (s, a, r, s', done) amostrada uniformemente.

    Resolve dois problemas: (i) correlação temporal entre amostras consecutivas
    (que viola a hipótese i.i.d. do SGD) e (ii) baixa eficiência amostral (cada
    transição é reutilizada várias vezes). Sem replay, o treino fica instável e
    sujeito a esquecimento catastrófico.
    """

    def __init__(self, capacity: int, batch_size: int):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([t.state for t in batch])).float()
        actions = torch.from_numpy(np.vstack([t.action for t in batch])).long()
        rewards = torch.from_numpy(np.vstack([t.reward for t in batch])).float()
        next_states = torch.from_numpy(np.vstack([t.next_state for t in batch])).float()
        dones = torch.from_numpy(np.vstack([t.done for t in batch]).astype(np.uint8)).float()
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)


# --------------------------------------------------------------------------- #
# Configuração de um experimento
# --------------------------------------------------------------------------- #
@dataclass
class DQNConfig:
    name: str = "full"
    state_size: int = 8
    action_size: int = 4
    hidden: int = 128
    gamma: float = 0.99           # fator de desconto
    lr: float = 5e-4              # taxa de aprendizado (Adam)
    buffer_size: int = 100_000
    batch_size: int = 64
    tau: float = 5e-3            # soft update do target network (Polyak)
    update_every: int = 4         # passos entre atualizações de aprendizado
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    # Ablações:
    use_replay: bool = True
    use_target: bool = True


# --------------------------------------------------------------------------- #
# 3) Agente DQN
# --------------------------------------------------------------------------- #
class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.qnet = QNetwork(cfg.state_size, cfg.action_size, cfg.hidden)
        # Target network: cópia "congelada" usada para calcular o alvo de Bellman.
        # Sem ela (use_target=False) o alvo é calculado da própria rede online.
        self.target_net = QNetwork(cfg.state_size, cfg.action_size, cfg.hidden)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=cfg.lr)

        self.memory = ReplayBuffer(cfg.buffer_size, cfg.batch_size) if cfg.use_replay else None
        self.t_step = 0
        self.last_loss = 0.0

    # --- política epsilon-greedy --- #
    def act(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.cfg.action_size)
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        self.qnet.eval()
        with torch.no_grad():
            q = self.qnet(state_t)
        self.qnet.train()
        return int(q.argmax(dim=1).item())

    # --- registra transição e dispara o aprendizado --- #
    def step(self, state, action, reward, next_state, done) -> None:
        if self.cfg.use_replay:
            self.memory.push(state, action, reward, next_state, done)
            self.t_step = (self.t_step + 1) % self.cfg.update_every
            if self.t_step == 0 and len(self.memory) >= self.cfg.batch_size:
                self._learn(self.memory.sample())
        else:
            # Sem replay: aprende imediatamente apenas da transição mais recente.
            batch = (
                torch.from_numpy(np.vstack([state])).float(),
                torch.from_numpy(np.vstack([action])).long(),
                torch.from_numpy(np.vstack([reward])).float(),
                torch.from_numpy(np.vstack([next_state])).float(),
                torch.from_numpy(np.vstack([done]).astype(np.uint8)).float(),
            )
            self._learn(batch)

    # --- núcleo do DQN: erro de Bellman + atualização dos pesos --- #
    def _learn(self, batch) -> None:
        states, actions, rewards, next_states, dones = batch

        # Q(s, a) atual previsto pela rede online
        q_expected = self.qnet(states).gather(1, actions)

        # Alvo de Bellman: r + gamma * max_a' Q(s', a') * (1 - done)
        bootstrap_net = self.target_net if self.cfg.use_target else self.qnet
        with torch.no_grad():
            q_next = bootstrap_net(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + self.cfg.gamma * q_next * (1.0 - dones)

        loss = F.smooth_l1_loss(q_expected, q_target)  # Huber

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()
        self.last_loss = float(loss.item())

        # Soft update do target network: theta_target <- tau*theta + (1-tau)*theta_target
        if self.cfg.use_target:
            for tp, p in zip(self.target_net.parameters(), self.qnet.parameters()):
                tp.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * tp.data)


# --------------------------------------------------------------------------- #
# Loop de treino
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    config: dict
    scores: list = field(default_factory=list)        # recompensa por episódio
    moving_avg: list = field(default_factory=list)     # média móvel de 100
    losses: list = field(default_factory=list)
    solved_episode: int | None = None                  # 1º episódio com média>=200


def train(
    cfg: DQNConfig,
    n_episodes: int = 700,
    max_t: int = 1000,
    seed: int = 42,
    solve_score: float = 200.0,
    early_stop: bool = True,
    verbose: bool = True,
) -> TrainResult:
    """Treina um agente DQN sob a configuração `cfg`. Retorna métricas por episódio."""
    import gymnasium as gym

    set_seed(seed)
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    agent = DQNAgent(cfg)
    res = TrainResult(config=asdict(cfg))
    eps = cfg.eps_start
    window = deque(maxlen=100)

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0.0
        for _ in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, float(terminated))
            state = next_state
            score += reward
            if done:
                break

        window.append(score)
        avg = float(np.mean(window))
        res.scores.append(score)
        res.moving_avg.append(avg)
        res.losses.append(agent.last_loss)
        eps = max(cfg.eps_end, cfg.eps_decay * eps)

        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"[{cfg.name}] ep {ep:4d} | score {score:8.2f} | média100 {avg:8.2f} | eps {eps:.3f}")

        if res.solved_episode is None and len(window) == 100 and avg >= solve_score:
            res.solved_episode = ep
            if verbose:
                print(f"[{cfg.name}] RESOLVIDO no episódio {ep} (média100={avg:.2f})")
            if early_stop:
                break

    env.close()
    return res, agent


# --------------------------------------------------------------------------- #
# Utilitários de persistência
# --------------------------------------------------------------------------- #
def save_result(res: TrainResult, results_dir: str) -> str:
    import pandas as pd

    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(
        {"episode": np.arange(1, len(res.scores) + 1),
         "score": res.scores,
         "moving_avg_100": res.moving_avg,
         "loss": res.losses}
    )
    path = os.path.join(results_dir, f"metrics_{res.config['name']}.csv")
    df.to_csv(path, index=False)
    return path


def evaluate(agent: DQNAgent, n_episodes: int = 10, seed: int = 123) -> float:
    """Avalia a política greedy (eps=0): mede o desempenho real aprendido."""
    import gymnasium as gym

    env = gym.make("LunarLander-v3")
    scores = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        score, done = 0.0, False
        while not done:
            action = agent.act(state, eps=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)
    env.close()
    return float(np.mean(scores))
