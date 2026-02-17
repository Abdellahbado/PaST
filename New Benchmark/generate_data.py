import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

# -----------------------------
# Utilities
# -----------------------------
def discrete_uniform(a: int, b: int, rng: random.Random) -> int:
    """Inclusive discrete uniform U[a,b]."""
    return rng.randint(a, b)

def sample_intervals_sum_to_T(T: int, choices: List[int], rng: random.Random) -> List[int]:
    """
    Sample interval durations Tk from 'choices' until they sum exactly to T.
    Rejection/backtracking to ensure exact sum.
    """
    # simple backtracking that is fast for these T values
    Tk = []
    remaining = T
    while remaining > 0:
        feasible = [x for x in choices if x <= remaining]
        if not feasible:
            # restart if stuck (rare with choices {2,3,5})
            Tk = []
            remaining = T
            continue
        x = rng.choice(feasible)
        Tk.append(x)
        remaining -= x
    return Tk

def expand_ck_to_ct(Tk: List[int], ck: List[int]) -> List[int]:
    """Expand interval prices ck into per-period prices c'_t of length sum(Tk)."""
    ct = []
    for dur, price in zip(Tk, ck):
        ct.extend([price] * dur)
    return ct

# -----------------------------
# Instance spec
# -----------------------------
@dataclass
class Instance:
    instance_id: int
    paper: str            # "Wang2018" or "Anghinolfi2021"
    scale: str            # "small", "mls", "vls"
    m: int
    n: int
    T: int
    p: List[int]          # length n
    e: List[int]          # length m
    Tk: List[int]         # interval durations, sum(Tk)=T
    ck: List[int]         # interval prices, length len(Tk)
    ct: List[int]         # per-period prices, length T

def generate_instance(instance_id: int,
                      paper: str,
                      scale: str,
                      m: int,
                      n: int,
                      T: int,
                      rng: random.Random,
                      Tk_choices: List[int],
                      p_range: Tuple[int,int],
                      e_range: Tuple[int,int],
                      ck_range: Tuple[int,int]) -> Instance:
    # time intervals
    Tk = sample_intervals_sum_to_T(T, Tk_choices, rng)
    K = len(Tk)
    ck = [discrete_uniform(ck_range[0], ck_range[1], rng) for _ in range(K)]
    ct = expand_ck_to_ct(Tk, ck)

    # job processing times and machine energy rates
    p = [discrete_uniform(p_range[0], p_range[1], rng) for _ in range(n)]
    e = [discrete_uniform(e_range[0], e_range[1], rng) for _ in range(m)]

    return Instance(instance_id, paper, scale, m, n, T, p, e, Tk, ck, ct)

def build_90_instances(seed: int = 12345,
                       Tk_choices_wang=(2,3,5),
                       Tk_choices_vls=(2,3,5),
                       out_json_path: str = "instances_90.json") -> List[Instance]:
    """
    Generates instances 1..90 with the configurations described in:
    - Wang et al. (2018): instances 1..60
    - Anghinolfi et al. (2021): instances 61..90
    """
    rng = random.Random(seed)

    # ---- Wang et al. (2018): 60 instances ----
    wang_small = [(m,n,T) for m in [3,5,7]
                         for n in [6,10,15,20,25]
                         for T in [50,80]]  # 30 [file:5]
    wang_mls   = [(m,n,T) for m in [8,16,25]
                         for n in [30,60,100,150,200]
                         for T in [100,300]]  # 30 [file:5]

    instances: List[Instance] = []
    iid = 1

    for (m,n,T) in wang_small:
        instances.append(
            generate_instance(
                iid, "Wang2018", "small", m, n, T, rng,
                list(Tk_choices_wang),
                p_range=(1,4),          # pj ~ U[1,4] [file:5]
                e_range=(1,3),          # ei in {1,2,3} [file:5]
                ck_range=(1,4)          # ck in {1,2,3,4} [file:5]
            )
        )
        iid += 1

    for (m,n,T) in wang_mls:
        instances.append(
            generate_instance(
                iid, "Wang2018", "mls", m, n, T, rng,
                list(Tk_choices_wang),
                p_range=(1,4),          # [file:5]
                e_range=(1,3),          # [file:5]
                ck_range=(1,4)          # [file:5]
            )
        )
        iid += 1

    # ---- Anghinolfi et al. (2021): 30 VLS instances ----
    vls = [(m,n,T) for m in [25,30,40]
                  for n in [250,300,350,400,500]
                  for T in [350,500]]  # 30 [file:4]

    for (m,n,T) in vls:
        instances.append(
            generate_instance(
                iid, "Anghinolfi2021", "vls", m, n, T, rng,
                list(Tk_choices_vls),
                p_range=(1,12),         # pj ~ U[1,12] [file:4]
                e_range=(1,6),          # eh ~ U[1,6] [file:4]
                ck_range=(1,8)          # ck ~ U[1,8] [file:4]
            )
        )
        iid += 1

    # save
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in instances], f, indent=2)

    return instances

if __name__ == "__main__":
    build_90_instances(seed=20260109, out_json_path="instances_90.json")
    print("Wrote instances_90.json")
