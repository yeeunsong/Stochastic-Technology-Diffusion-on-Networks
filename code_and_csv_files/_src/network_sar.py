# src/network_sar.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import networkx as nx


@dataclass(frozen=True)
class SARParams:
    beta: float   # adoption per S-A edge per month
    gamma: float  # A->R per A per month
    rho: float    # R->A per R per month


def make_network(
    N: int,
    kind: str,
    seed: int,
    *,
    er_p: float = 0.002,       # ER edge probability
    ba_m: int = 3,             # BA edges per new node
    ws_k: int = 10,            # WS mean degree ~ k
    ws_rewire: float = 0.1     # WS rewiring probability
) -> nx.Graph:
    if kind == "er":
        return nx.erdos_renyi_graph(N, er_p, seed=seed)
    if kind == "sf":  # scale-free-ish via BA
        return nx.barabasi_albert_graph(N, ba_m, seed=seed)
    if kind == "sw":
        if ws_k % 2 != 0:
            raise ValueError("ws_k must be even for watts_strogatz_graph")
        return nx.watts_strogatz_graph(N, ws_k, ws_rewire, seed=seed)
    raise ValueError("kind must be 'er', 'sf', or 'sw'")


def init_states(N: int, A0: int, R0: int, rng: np.random.Generator) -> np.ndarray:
    """
    state codes: 0=S, 1=A, 2=R
    """
    if A0 + R0 > N:
        raise ValueError("Need A0 + R0 <= N")
    state = np.zeros(N, dtype=np.int8)
    perm = rng.permutation(N)
    state[perm[:A0]] = 1
    state[perm[A0:A0+R0]] = 2
    return state


def compute_nA(G: nx.Graph, state: np.ndarray) -> np.ndarray:
    N = len(state)
    nA = np.zeros(N, dtype=np.int32)
    for i in range(N):
        nA[i] = sum(1 for j in G.neighbors(i) if state[j] == 1)
    return nA


def compute_SA_edges(state: np.ndarray, nA: np.ndarray) -> int:
    # computes the total number of S–A edges in the entire network
    return int(nA[state == 0].sum())


def gillespie_network_SAR(
    G: nx.Graph,
    params: SARParams,
    *,
    A0: int = 10,
    R0: int = 0,
    months: int = 60,
    sample_dt: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gillespie algorithm for networked SAR.
    Adoption is along edges: each S-A edge triggers adoption at rate beta.
    """
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()

    # states + caches
    state = init_states(N, A0, R0, rng)
    nA = compute_nA(G, state)

    A_set = set(np.where(state == 1)[0])
    R_set = set(np.where(state == 2)[0])

    A_count = len(A_set)
    R_count = len(R_set)
    S_count = N - A_count - R_count

    SA_edges = compute_SA_edges(state, nA)

    t_end = float(months)
    times = np.arange(0.0, t_end + 1e-9, sample_dt)
    outS = np.zeros(len(times), dtype=int)
    outA = np.zeros(len(times), dtype=int)
    outR = np.zeros(len(times), dtype=int)

    t = 0.0
    idx = 0

    def record_up_to(current_t: float):
        nonlocal idx
        while idx < len(times) and times[idx] <= current_t + 1e-12:
            outS[idx] = S_count
            outA[idx] = A_count
            outR[idx] = R_count
            idx += 1

    record_up_to(t)

    # --- SAR loop ---
    while t < t_end:
        a1 = params.beta * SA_edges
        a2 = params.gamma * A_count
        a3 = params.rho * R_count
        a0 = a1 + a2 + a3

        if a0 <= 0.0:
            t = t_end
            record_up_to(t)
            break

        # next reaction time
        dt = -np.log(rng.random()) / a0
        t_next = t + dt

        if t_next > t_end:
            t = t_end
            record_up_to(t)
            break

        # choose reaction
        u = rng.random() * a0

        if u < a1:
            # Adoption: choose an S node weighted by nA[i]
            S_nodes = np.where(state == 0)[0]
            if SA_edges == 0 or len(S_nodes) == 0:
                t = t_next
                record_up_to(t)
                continue

            weights = nA[S_nodes].astype(float)
            wsum = weights.sum()
            if wsum <= 0:
                t = t_next
                record_up_to(t)
                continue

            r = rng.random() * wsum
            cum = 0.0
            chosen = None
            for node, w in zip(S_nodes, weights):
                cum += w
                if cum >= r:
                    chosen = int(node)
                    break
            i = chosen

            # S -> A
            state[i] = 1
            A_set.add(i)
            S_count -= 1
            A_count += 1

            # i was S contributing nA[i] to SA_edges; remove it
            SA_edges -= int(nA[i])

            # i becomes active: increases nA for neighbors; increases SA_edges for S neighbors
            for j in G.neighbors(i):
                if state[j] == 0:
                    nA[j] += 1
                    SA_edges += 1
                else:
                    nA[j] += 1

            # refresh nA[i] 
            nA[i] = sum(1 for j in G.neighbors(i) if state[j] == 1)

        elif u < a1 + a2:
            # Churn: pick random A node
            if A_count == 0:
                t = t_next
                record_up_to(t)
                continue
            i = int(rng.choice(list(A_set)))

            # A -> R
            state[i] = 2
            A_set.remove(i)
            R_set.add(i)
            A_count -= 1
            R_count += 1

            # i ceases being active: decreases nA for neighbors; decreases SA_edges for S neighbors
            for j in G.neighbors(i):
                if state[j] == 0:
                    nA[j] -= 1
                    SA_edges -= 1
                else:
                    nA[j] -= 1

            nA[i] = sum(1 for j in G.neighbors(i) if state[j] == 1)

        else:
            # Return: pick random R node
            if R_count == 0:
                t = t_next
                record_up_to(t)
                continue
            i = int(rng.choice(list(R_set)))

            # R -> A
            state[i] = 1
            R_set.remove(i)
            A_set.add(i)
            R_count -= 1
            A_count += 1

            # i becomes active: increases nA for neighbors; increases SA_edges for S neighbors
            for j in G.neighbors(i):
                if state[j] == 0:
                    nA[j] += 1
                    SA_edges += 1
                else:
                    nA[j] += 1

            nA[i] = sum(1 for j in G.neighbors(i) if state[j] == 1)

        t = t_next
        record_up_to(t)

    return times, outS, outA, outR