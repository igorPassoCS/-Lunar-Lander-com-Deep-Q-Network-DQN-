# Deep Q-Network (DQN) — Lunar Lander

Implementação **individual e do zero** de um agente Deep Q-Network para o ambiente
`LunarLander-v3` do [Gymnasium](https://gymnasium.farama.org/). O Gymnasium é usado
**apenas como simulador** do ambiente; **todo o algoritmo de RL** (rede Q, experience
replay, target network, erro de Bellman e otimização) é escrito manualmente — nenhuma
biblioteca de RL pronta (stable-baselines3, RLlib, keras-rl, Tianshou, ...) é utilizada.

- **Notebook (entrega principal):** [`dqn_lunar_lander.ipynb`](dqn_lunar_lander.ipynb) — implementação, experimentos e curvas, com saídas já executadas.
- **Código-fonte espelhado:** [`src/dqn.py`](src/dqn.py) (módulo) e [`src/run_experiments.py`](src/run_experiments.py) (runner usado para treinar em paralelo).
- **Resultados:** pasta [`results/`](results/) (CSVs de métricas, gráficos `.png`, pesos `.pt`).

## Como reproduzir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install swig
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install "gymnasium[box2d]"==1.3.0 numpy matplotlib pandas jupyter ipykernel nbconvert
# treinar as 3 configurações (paralelo) e gerar results/
cd src && python run_experiments.py full & python run_experiments.py no_target & python run_experiments.py no_replay &
# ou simplesmente abrir/executar o notebook, que é auto-contido
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 dqn_lunar_lander.ipynb
```

Todos os experimentos usam **seed fixa = 42**.

---

# 1. Compreensão do ambiente

## (a) Variáveis de estado

O estado é um vetor **contínuo de 8 dimensões** (`Box`). Os intervalos abaixo são os
limites do espaço de observação do `LunarLander-v3`:

| # | Variável | Significado físico | Intervalo |
|---|----------|--------------------|-----------|
| 0 | `x` | posição horizontal relativa à plataforma de pouso (origem no centro) | [-2.5, 2.5] |
| 1 | `y` | posição vertical (altura) relativa à plataforma | [-2.5, 2.5] |
| 2 | `vx` | velocidade linear horizontal | [-10, 10] |
| 3 | `vy` | velocidade linear vertical | [-10, 10] |
| 4 | `θ` (ângulo) | orientação/inclinação do módulo, em radianos (0 = nivelado) | [-6.2832, 6.2832] |
| 5 | `ω` | velocidade angular (taxa de rotação) | [-10, 10] |
| 6 | perna esquerda | indicador booleano de contato da perna esquerda com o solo | {0, 1} |
| 7 | perna direita | indicador booleano de contato da perna direita com o solo | {0, 1} |

As duas últimas são efetivamente binárias (0 = sem contato, 1 = tocando o solo); as seis
primeiras são reais. As posições/velocidades estão em unidades normalizadas do simulador,
não em metros/SI.

## (b) Ações disponíveis

Espaço de ações **discreto com 4 opções** (`Discrete(4)`):

| Ação | Significado físico |
|------|--------------------|
| `0` | não fazer nada (sem propulsão) |
| `1` | acionar o motor de **orientação esquerdo** (empurra/gira o módulo para a direita) |
| `2` | acionar o **motor principal** (impulso para cima, contra a gravidade) |
| `3` | acionar o motor de **orientação direito** (empurra/gira o módulo para a esquerda) |

## (c) Estrutura da recompensa

A recompensa combina um termo de *shaping* contínuo (a cada passo) com bônus/penalidades
discretos:

**Aumenta (positiva) quando o agente:**
- se **aproxima** da plataforma de pouso (recompensa cresce conforme a distância diminui);
- **reduz a velocidade** (chegar devagar é melhor que chegar rápido);
- mantém o módulo **nivelado** (pouca inclinação);
- encosta cada perna no solo: **+10 por perna** em contato;
- **+100** ao pousar com sucesso (fim de episódio).

**Diminui (penaliza) quando o agente:**
- está **longe** da plataforma, **rápido** ou muito **inclinado**;
- gasta combustível: **−0,3 por quadro** com o motor principal ligado e **−0,03 por quadro**
  com um motor lateral ligado (incentiva pousar gastando o mínimo);
- **−100** ao **colidir/cair** (fim de episódio).

Ou seja, a recompensa empurra o agente a chegar à plataforma de forma suave, nivelada,
econômica em combustível, e a finalizar tocando as duas pernas sem bater.

## (d) Condições de término de um episódio

- **Colisão (crash):** o corpo do módulo (não as pernas) toca o solo → `terminated=True` (com −100).
- **Saída da área válida:** o módulo sai do campo visível, `|x| > 1` → `terminated=True`.
- **Repouso:** o módulo para de se mover (corpo "adormece" na física do Box2D) → `terminated=True`.
- **Limite de passos (truncamento):** atingidos os 1000 passos do `TimeLimit` sem terminar →
  `truncated=True`.

No código distinguimos os dois sinais: `done = terminated or truncated` encerra o laço, mas
**apenas `terminated` zera o bootstrap de Bellman** (o truncamento por tempo não significa
estado terminal real, então o valor futuro continua válido).

## (e) O que é "pousar com sucesso" e critério de aprendizado

Um pouso bem-sucedido é o módulo **repousar sobre a plataforma com as duas pernas em
contato, em baixa velocidade e nivelado**, sem colidir — episódio que rende o bônus de
+100 e termina com recompensa alta.

O critério padrão de desempenho do ambiente é considerá-lo **"resolvido" quando a média da
recompensa dos últimos 100 episódios consecutivos ≥ 200**. Usamos essa média móvel de 100
como métrica de convergência e, ao final, avaliamos a **política greedy (ε = 0)** em 20
episódios novos para medir o desempenho real aprendido (sem exploração aleatória).

## (f) Por que métodos tabulares não resolvem o Lunar Lander

O Q-learning tabular guarda um valor `Q[s, a]` para cada par estado-ação numa tabela, o que
exige um **conjunto de estados finito e enumerável**. No Lunar Lander, seis das oito
variáveis de estado são **contínuas** (posição, velocidade, ângulo, velocidade angular) →
o espaço de estados é **infinito/incontável**. Não há como indexar uma tabela, e qualquer
discretização sofreria com a **maldição da dimensionalidade** (o número de células cresce
exponencialmente com as 8 dimensões) e nunca veria a maioria dos estados.

A solução é **aproximação de função**: uma rede neural parametrizada `Q(s, a; θ)` que
**generaliza** — aprende uma função suave do estado contínuo para os Q-valores e produz
estimativas razoáveis até para estados nunca visitados. É exatamente o papel do DQN.

---

# 2. Implementação do agente DQN

## (a) Arquitetura da rede neural (justificada)

Multilayer Perceptron (`QNetwork` em [`src/dqn.py`](src/dqn.py)):

```
entrada (8)  →  Linear(8, 128)  → ReLU
             →  Linear(128, 128) → ReLU
             →  Linear(128, 4)   → saída linear (Q de cada ação)
```

- **Dimensão de entrada = 8**: coerente com as 8 variáveis de estado do Lunar Lander.
- **Dimensão de saída = 4**: um Q-valor por ação discreta (arquitetura padrão do DQN, que
  produz todos os `Q(s, ·)` numa só passada — eficiente para o `max` e o `argmax`).
- **Duas camadas ocultas de 128 neurônios**: capacidade suficiente para aproximar uma
  função suave de ℝ⁸ → ℝ⁴ sem inflar parâmetros nem favorecer overfitting; é a configuração
  consagrada para este ambiente.
- **ReLU** nas ocultas: não-linearidade barata, sem saturação de gradiente.
- **Saída linear** (sem ativação): Q-valores podem ser **negativos e ilimitados**, então
  qualquer ativação limitante (sigmoid/tanh) seria inadequada.
- **Otimizador Adam** (lr = 5e-4), **perda de Huber** (`smooth_l1_loss`) sobre o erro de
  Bellman (mais robusta a outliers que o MSE), **γ = 0,99**, e **clipping de gradiente** (10)
  para estabilidade.

### Seed fixada e por que ela é necessária

Fixamos `SEED = 42` em `random`, `numpy`, `torch` e no ambiente
(`env.reset(seed=...)` + `env.action_space.seed(...)`). Várias etapas do DQN são
**estocásticas**: a inicialização dos pesos da rede, a amostragem aleatória do experience
replay, a exploração ε-greedy e a própria dinâmica/condição inicial do ambiente. Sem uma
seed fixa, **duas execuções do mesmo código produzem curvas diferentes**, o que (i) impede
reproduzir os resultados relatados e (ii) **invalida a comparação do item 3**: uma diferença
entre "com" e "sem" um componente poderia ser fruto do acaso, e não do componente isolado.
Com a seed fixa, todas as configurações partem das mesmas condições e a diferença observada
é atribuível ao fator que mudamos.

## (b) Loop de treinamento implementado do zero

O laço (`train()`) é escrito manualmente, sem abstração externa:

1. A cada passo, escolhe a ação por **ε-greedy** e interage com o ambiente.
2. Armazena/usa a transição `(s, a, r, s', terminated)`.
3. **Erro de Bellman:** alvo `y = r + γ · maxₐ' Q(s', a') · (1 − terminated)`; previsão
   `Q(s, a)` obtida com `gather` na rede online.
4. **Perda de Huber** entre previsão e alvo; `loss.backward()`; **clip de gradiente**;
   `optimizer.step()` (Adam) atualiza os pesos.
5. Decai ε (1,0 → 0,01) e registra recompensa e média móvel de 100.

## (c) Experience replay e target network (por que existem)

**Experience Replay** (`ReplayBuffer`, capacidade 100k, lotes de 64):
- *O que resolve:* (i) a **correlação temporal** entre transições consecutivas, que viola a
  hipótese i.i.d. do gradiente estocástico, e (ii) a **baixa eficiência amostral** — cada
  transição é reaproveitada em vários updates.
- *Sem ele:* o agente treina apenas na transição mais recente, com amostras altamente
  correlacionadas, sofrendo **esquecimento catastrófico** e oscilação — como mostra o
  experimento `no_replay`.

**Target Network** (cópia "congelada" atualizada por *soft update* de Polyak, τ = 5e-3):
- *O que resolve:* o problema do **alvo móvel**. Se o alvo de Bellman fosse calculado pela
  mesma rede que está sendo treinada, ele se moveria a cada passo "perseguindo" a própria
  estimativa, gerando um laço de realimentação instável.
- *Sem ela:* os Q-valores **oscilam e podem divergir**; o treino fica menos estável e a
  convergência é mais difícil de sustentar — como mostra o experimento `no_target`.

---

# 3. Análise comparativa de configurações

## (a) Configurações comparadas

Três configurações, **mesma seed (42)** e **mesmos hiperparâmetros**, isolando um componente
por vez (até 700 episódios; a `full` para ao resolver):

| Config | Experience replay | Target network |
|--------|:--:|:--:|
| `full` | ✅ | ✅ |
| `no_target` | ✅ | ❌ (bootstrap da própria rede online) |
| `no_replay` | ❌ (online, só a última transição) | ✅ |

## (b) Curvas de aprendizado

Geradas pelo notebook (recompensa por episódio + média móvel de 100):

- Curvas individuais: [`results/curvas_individuais.png`](results/curvas_individuais.png)
- Comparação direta das médias móveis: [`results/curva_comparativa.png`](results/curva_comparativa.png)

Resultados quantitativos (`results/summary.csv`):

| Config | Resolveu? (média100 ≥ 200) | Melhor média(100) | Média(100) final | Avaliação greedy (20 ep) |
|--------|:--:|:--:|:--:|:--:|
| `full` | ✅ no episódio **473** | 201,0 | 201,0 | 243,5 |
| `no_target` | ❌ | 182,0 | 110,7 | 247,3 |
| `no_replay` | ❌ | 38,0 | 11,1 | 52,7 |

> Valores da execução embutida no notebook (`results/summary.csv`). Dada a mesma seed, o
> treino é reproduzível; pequenas variações entre máquinas/execuções vêm da ordem de redução
> de ponto-flutuante na CPU (nº de threads), mas o **padrão qualitativo é estável**.

## (c) Interpretação

**DQN completo (`full`)** é a única configuração que **resolve** o ambiente, cruzando a
média de 200 no episódio **473** e confirmando o aprendizado na avaliação greedy
(**243,5**, bem acima do limiar). A curva sobe de forma consistente e **estável** — os dois
componentes trabalham juntos: o replay fornece lotes descorrelacionados e o target network
mantém o alvo de Bellman estável enquanto a rede aprende.

**Sem target network (`no_target`)** é o caso mais instrutivo da **instabilidade**: o agente
chega a aprender (melhor média móvel **182**), mas **não sustenta** o desempenho — a média de
100 episódios **oscila e regride fortemente, caindo de 182 para 110,7 no final**, sem nunca
cruzar os 200 de forma consistente. Essa é a assinatura do **alvo móvel**: como o alvo de
Bellman é recalculado pela mesma rede que está sendo otimizada, pequenos erros se realimentam
e a política "balança" em vez de convergir. Curiosamente, a avaliação greedy de um *snapshot*
final saiu alta (**247,3**), o que **reforça** o diagnóstico em vez de contradizê-lo: a
política varia muito de episódio para episódio, então uma fotografia pontual pode parecer boa
enquanto o **desempenho médio sustentado** (o que realmente importa) permanece instável e
abaixo do critério. Sem uma rede-alvo dedicada, o treino não estabiliza.

**Sem experience replay (`no_replay`)** é o caso mais severo: treinar apenas na transição
mais recente deixa as amostras **fortemente correlacionadas** e impede o **reuso** de
experiência. O resultado é um aprendizado **lento, ruidoso e que não converge** (melhor
média **38**, final **11,1**, avaliação greedy **52,7**) — fica muito longe dos 200. Confirma
que o experience replay é, das duas peças, a **mais crítica** para viabilizar o DQN neste
ambiente: sem ele o agente sequer se aproxima de uma política competente.

**Conclusão.** Os experimentos confirmam o papel de cada componente: o **experience replay
é decisivo** para que o DQN aprenda (descorrelaciona e reaproveita amostras), e o **target
network é decisivo para a estabilidade e a convergência sustentada** (elimina o alvo móvel).
O DQN completo — combinando os dois — é o único que aprende uma política de pouso confiável.

---

## Estrutura do repositório

```
.
├── README.md                   # este arquivo (itens 1, 2 e 3 em prosa)
├── requirements.txt            # dependências fixadas
├── dqn_lunar_lander.ipynb       # NOTEBOOK — implementação + experimentos + curvas (executado)
├── src/
│   ├── dqn.py                  # implementação do DQN (módulo espelho do notebook)
│   └── run_experiments.py       # runner para treinar as 3 configs e salvar results/
└── results/
    ├── metrics_{full,no_target,no_replay}.csv   # métricas por episódio
    ├── summary.csv / summary_*.json             # resumo por configuração
    ├── curvas_individuais.png / curva_comparativa.png
    └── model_{full,no_target,no_replay}.pt      # pesos finais
```

## Fora de escopo
- Vídeo explicativo (gravação do aluno; este README e o notebook servem de roteiro).
- Variantes além do DQN baunilha (Double DQN, Dueling DQN, PER) — possíveis trabalhos futuros.
