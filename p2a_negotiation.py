"""Phases 2 to 4 of Policy2Agreement for the paper's running example, in plain Python.

The paper's prototype reads ODRL policies, runs Phase 2 in the EYE reasoner and negotiates with its
own MiCRO-Min engine. This module encodes the same norms, the same domain rule and the same compiler
of Eq. (1) directly in Python, and ports the prototype's MiCRO-Min agent and SAOP simulator line by
line, so the negotiation can be re-run without the reasoner. The notebook checks that it reproduces
the agreements and step counts recorded with the prototype (results/table1_prototype.json).

    compile_problem(parties, mu)   Phases 2 and 3: the feasible set Omega_F and each party's
                                   utility u (Eq. 1) and adjusted utility u_hat
    negotiate(problem, order, seed) Phase 4: one MiCRO-Min negotiation under SAOP
    negotiate_all(problem)          6 turn orders x 20 seeds, as for Table 1
"""
from __future__ import annotations

import random
from collections import Counter
from itertools import permutations, product

# The three issues and their finite ordered domains (Section 2.3).
DOMAIN = {"eps": [0.3, 0.5, 0.6, 0.8], "q": [0.8, 0.85, 0.9, 0.92], "t": [10, 15, 20]}
ISSUES = list(DOMAIN)

# Each party's norms are permissions qualified by constraints <issue, operator, value>.
RUNNING_EXAMPLE = {
    "H1": [("eps", "lteq", 0.3), ("q", "gteq", 0.92)],
    "H2": [("eps", "lteq", 0.5), ("t", "lteq", 15)],
    "R": [("q", "gteq", 0.85), ("t", "lteq", 20)],
}
# The variant of Section 4: R requires t >= 15 instead of t <= 20.
VARIANT = {**RUNNING_EXAMPLE, "R": [("q", "gteq", 0.85), ("t", "gteq", 15)]}


def consortium_obligation(outcome: dict) -> bool:
    """The consortium's obligation t >= 15; by lex superior it prunes Omega_F."""
    return outcome["t"] >= 15


def holds(outcome: dict, constraint: tuple) -> bool:
    issue, op, value = constraint
    x = outcome[issue]
    return x <= value if op == "lteq" else x >= value if op == "gteq" else x == value


def domain_rule(constraints: list) -> list[int]:
    """The domain rule of the running example: a party that asks for q >= 0.92 together with
    eps <= 0.3 is flagged; both constraints of the pair are flagged (multiplier mu).
    Returns the indices of the flagged constraints, or [] if the rule does not fire."""
    high_q = [i for i, (j, op, v) in enumerate(constraints) if j == "q" and op == "gteq" and v >= 0.92]
    strict_eps = [i for i, (j, op, v) in enumerate(constraints) if j == "eps" and op == "lteq" and v <= 0.3]
    return sorted(set(high_q + strict_eps)) if high_q and strict_eps else []


def ideal_point(constraints: list) -> dict:
    """Each bound read as a direction of preference: gteq -> the highest value, lteq -> the lowest."""
    ideal = {}
    for issue, op, value in constraints:
        dom = sorted(DOMAIN[issue])
        ideal[issue] = max(dom) if op == "gteq" else min(dom) if op == "lteq" else value
    return ideal


def base_utility(outcome: dict, ideal: dict) -> float:
    """Eq. (1): one minus the mean distance, by position in each domain, from the ideal point."""
    if not ideal:
        return 1.0
    distances = []
    for issue in ideal:
        dom = sorted(DOMAIN[issue])
        distances.append(abs(dom.index(outcome[issue]) - dom.index(ideal[issue])) / (len(dom) - 1))
    return 1.0 - sum(distances) / len(distances)


def compile_problem(parties: dict, mu: float = 0.2, raw: bool = False, omega_filter=None) -> dict:
    """Phases 2 and 3. Only an authority's norms prune (omega_filter); a party's own constraints
    shape its preferences: u_hat = 0 outside its acceptance set (an unflagged constraint is broken)
    and u_hat = mu * u on the region S where all of its flagged constraints hold.
    raw=True leaves u_hat = u everywhere (the 'no domain rule' rows of Table 1)."""
    outcomes = [dict(zip(ISSUES, values)) for values in product(*(DOMAIN[j] for j in ISSUES))]
    omega_f = [o for o in outcomes if omega_filter is None or omega_filter(o)]
    utilities = {}
    for party, constraints in parties.items():
        flagged = set(domain_rule(constraints))
        ideal = ideal_point(constraints)
        table = []
        for o in omega_f:
            u = base_utility(o, ideal)
            multipliers = []
            if not raw:
                if any(not holds(o, c) for i, c in enumerate(constraints) if i not in flagged):
                    multipliers.append(0.0)
                if flagged and all(holds(o, constraints[i]) for i in flagged):
                    multipliers.append(mu)
            u_hat = u
            for m in multipliers:
                u_hat *= m
            table.append({"u": u, "u_hat": u_hat, "multipliers": multipliers})
        utilities[party] = table
    return {"parties": list(parties), "constraints": parties, "omega_f": omega_f, "utilities": utilities}


PRIVACY_TIE_BREAK = 1e-6


def privacy_first(problem: dict, party: str = "H1") -> dict:
    """H1 states, without any number, that privacy comes before accuracy: its ordering becomes
    lexicographic (lower eps first, then higher q), with the rule's multipliers kept."""
    eps_dom, q_dom = sorted(DOMAIN["eps"]), sorted(DOMAIN["q"])
    new = {**problem, "utilities": dict(problem["utilities"])}
    table = []
    for o, row in zip(problem["omega_f"], problem["utilities"][party]):
        d_eps = eps_dom.index(o["eps"]) / (len(eps_dom) - 1)
        d_q = (len(q_dom) - 1 - q_dom.index(o["q"])) / (len(q_dom) - 1)
        factor = 1.0
        for m in row["multipliers"]:
            factor *= m
        table.append({**row, "u_hat": (1 - d_eps - PRIVACY_TIE_BREAK * d_q) * factor})
    new["utilities"][party] = table
    return new


# ------------------------------------------------------------------ Phase 4: MiCRO-Min under SAOP
# A line-by-line port of the prototype's engine. Offers are tuples of strings in issue order, as in
# the prototype, and a repeated offer is drawn with a seeded choice over the sorted candidates.

TOLERANCE = 1e-9
MAX_TURNS = 2000


def offer_of(outcome: dict) -> tuple:
    return tuple(str(outcome[j]) for j in ISSUES)


class MiCROMin:
    """MiCRO-Min, the multilateral form of MiCRO cited in the paper. Offers are sorted once by the
    agent's own utility. It proposes its next best offer only if it has not proposed or accepted
    more distinct offers than every other party (m <= n_min); otherwise it repeats one of its own.
    It accepts the standing offer if that is at least as good as the worst offer it would propose.
    Among offers tied with its next offer, it prefers one another party has already put forward.
    As in the prototype, the reservation value is 0, so an agent that has conceded that far could
    accept an outcome outside its acceptance set; the notebook checks that no agreement does."""

    def __init__(self, index: int, n_agents: int, offers: list, utility: dict, rng: random.Random):
        self.i, self.others = index, [k for k in range(n_agents) if k != index]
        self.u, self.rng, self.reservation = utility, rng, 0.0
        self.sorted_offers = sorted(offers, key=lambda o: utility[o], reverse=True)   # stable sort

    def _counts(self, transcript):
        counts = {k: set() for k in [self.i] + self.others}
        for idx, action, offer in transcript:
            if action in ("propose", "accept") and offer is not None:
                counts[idx].add(offer)
        return counts

    def _next_fresh_rank(self, mine):
        for rank, offer in enumerate(self.sorted_offers):
            if offer not in mine:
                return rank
        return None

    def act(self, transcript, standing):
        counts = self._counts(transcript)
        mine = counts[self.i]
        ready_to_concede = len(mine) <= min(len(counts[k]) for k in self.others)
        rank = self._next_fresh_rank(mine)
        nxt = self.sorted_offers[rank] if rank is not None else None
        below_reservation = nxt is None or self.u[nxt] < self.reservation

        if (below_reservation or not ready_to_concede) and mine:
            proposal = self.rng.choice(sorted(mine))
        elif nxt is None:
            proposal = self.rng.choice(sorted(mine)) if mine else self.sorted_offers[-1]
        else:
            proposal = self._reproposing(nxt, rank, counts)

        lowest = self._lowest_acceptable(mine, nxt, ready_to_concede)
        threshold = self.reservation if lowest is None else max(self.reservation, self.u[lowest])
        if standing is not None and self.u[standing] >= threshold - TOLERANCE:
            return "accept", standing
        return "propose", proposal

    def _reproposing(self, nxt, rank, counts):
        mine = counts[self.i]
        for candidate in self.sorted_offers[rank:]:
            if self.u[candidate] < self.u[nxt] - TOLERANCE:
                break
            if candidate in mine:
                continue
            if any(candidate in counts[k] for k in self.others):
                return candidate
        return nxt

    def _lowest_acceptable(self, mine, nxt, ready_to_concede):
        if ready_to_concede:
            return nxt
        owned = [r for r, o in enumerate(self.sorted_offers) if o in mine]
        return self.sorted_offers[max(owned)] if owned else None


def negotiate(problem: dict, order=None, seed: int = 0):
    """One negotiation under the Stacked Alternating Offers Protocol. Agents take turns in `order`;
    each proposes an offer or accepts the standing one, and an offer accepted by every agent (its
    proposer included) is the agreement. Returns (agreement as an outcome dict or None, steps), where
    steps counts the turns after the first one, as the prototype reports them."""
    order = list(order or problem["parties"])
    offers = [offer_of(o) for o in problem["omega_f"]]
    rng = random.Random(seed)
    agents = [MiCROMin(k, len(order), offers,
                       {offer_of(o): row["u_hat"] for o, row in zip(problem["omega_f"], problem["utilities"][p])}, rng)
              for k, p in enumerate(order)]
    transcript, standing, accepted_by = [], None, set()
    for turn in range(MAX_TURNS):
        k = turn % len(agents)
        action, offer = agents[k].act(transcript, standing)
        transcript.append((k, action, offer))
        if action == "propose":
            standing, accepted_by = offer, {k}
        else:
            accepted_by.add(k)
        if len(accepted_by) == len(agents):
            return next(o for o in problem["omega_f"] if offer_of(o) == standing), turn
    return None, MAX_TURNS


def negotiate_all(problem: dict, seeds=range(20)) -> dict:
    """Every turn order x every seed, as for Table 1 (6 x 20 = 120 negotiations)."""
    agreements, steps, deadlocks = Counter(), [], 0
    for order in permutations(problem["parties"]):
        for seed in seeds:
            agreement, n = negotiate(problem, order, seed)
            if agreement is None:
                deadlocks += 1
                continue
            agreements[(agreement["eps"], agreement["q"], agreement["t"])] += 1
            steps.append(n)
    return {"agreements": agreements, "mean_steps": sum(steps) / len(steps) if steps else None,
            "deadlocks": deadlocks}


def utility_at(problem: dict, party: str, outcome: tuple) -> tuple[float, float]:
    """(u, u_hat) of a party at an outcome (eps, q, t)."""
    for o, row in zip(problem["omega_f"], problem["utilities"][party]):
        if (o["eps"], o["q"], o["t"]) == tuple(outcome):
            return row["u"], row["u_hat"]
    raise KeyError(outcome)
