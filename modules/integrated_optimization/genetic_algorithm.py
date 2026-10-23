"""A bounded real-valued genetic algorithm with feasibility-first selection.

The caller supplies objective values and summed normalized constraint violations.
Independent uniform initialization, binary tournaments, simulated binary
crossover, polynomial mutation, and elitist survival drive the search.
"""

import numpy as np


def _order(objective, violation):
    """Feasible candidates first; otherwise prefer smaller total violation."""
    return np.lexsort((objective, violation, violation > 0.0))


def _sbx(first, second, lower, upper, rng, eta=15.0):
    """Bounded simulated binary crossover (0.9 per pair, 0.5 per variable)."""
    low_parent = np.minimum(first, second)
    high_parent = np.maximum(first, second)
    distance = high_parent - low_parent
    active = (
        (rng.random((len(first), 1)) < 0.9)
        & (rng.random(first.shape) < 0.5)
        & (distance > 1e-14)
    )
    denominator = np.maximum(distance, 1e-14)
    random = rng.random(first.shape)
    exponent = 1.0 / (eta + 1.0)

    beta = 1.0 + 2.0 * (low_parent - lower) / denominator
    alpha = 2.0 - beta ** (-(eta + 1.0))
    beta_q = np.where(
        random <= 1.0 / alpha,
        (random * alpha) ** exponent,
        (1.0 / (2.0 - random * alpha)) ** exponent,
    )
    child_low = 0.5 * (low_parent + high_parent - beta_q * distance)

    beta = 1.0 + 2.0 * (upper - high_parent) / denominator
    alpha = 2.0 - beta ** (-(eta + 1.0))
    beta_q = np.where(
        random <= 1.0 / alpha,
        (random * alpha) ** exponent,
        (1.0 / (2.0 - random * alpha)) ** exponent,
    )
    child_high = 0.5 * (low_parent + high_parent + beta_q * distance)

    swap = rng.random(first.shape) < 0.5
    child_first = np.where(active, np.where(swap, child_high, child_low), first)
    child_second = np.where(active, np.where(swap, child_low, child_high), second)
    return np.clip(child_first, lower, upper), np.clip(child_second, lower, upper)


def _mutate(candidates, lower, upper, rng, eta=20.0):
    """Bounded polynomial mutation with probability 1 / dimension."""
    span = upper - lower
    distance_lower = (candidates - lower) / span
    distance_upper = (upper - candidates) / span
    random = rng.random(candidates.shape)
    exponent = 1.0 / (eta + 1.0)
    left = (
        2.0 * random
        + (1.0 - 2.0 * random) * (1.0 - distance_lower) ** (eta + 1.0)
    )
    right = (
        2.0 * (1.0 - random)
        + 2.0 * (random - 0.5) * (1.0 - distance_upper) ** (eta + 1.0)
    )
    # Only the corresponding branch is used; clip roundoff in the unused one.
    delta = np.where(
        random <= 0.5,
        np.maximum(left, 0.0) ** exponent - 1.0,
        1.0 - np.maximum(right, 0.0) ** exponent,
    )
    selected = rng.random(candidates.shape) < 1.0 / candidates.shape[1]
    return np.clip(candidates + selected * delta * span, lower, upper)


def minimize(evaluate, lower, upper, *, seed, population_size=96, generations=200):
    """Minimize a black-box objective subject to caller-supplied constraints.

    ``evaluate(x)`` receives an ``(N, D)`` array and returns two ``(N,)``
    arrays: objective and nonnegative summed normalized violation. Zero
    violation means feasible. Lower and upper are finite vectors with strict
    lower < upper in every coordinate. Initial candidates are independently
    uniform inside those bounds.

    Selection uses binary tournaments. Each generation evaluates one new
    population, then retains the best ``population_size`` candidates from
    parents and offspring together. The search runs for the configured number
    of generations and returns the selected candidate and convergence history.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)
    population = rng.uniform(lower, upper, size=(population_size, len(lower)))
    objective, violation = (np.asarray(values, dtype=float) for values in evaluate(population))
    order = _order(objective, violation)
    population, objective, violation = population[order], objective[order], violation[order]
    evaluations = population_size
    history = []

    def record(generation):
        history.append({
            "generation": generation,
            "evaluations": evaluations,
            "best_objective": float(objective[0]),
            "best_violation": float(violation[0]),
            "feasible_count": int(np.count_nonzero(violation <= 0.0)),
        })

    record(0)
    pair_count = (population_size + 1) // 2
    for generation in range(1, generations + 1):
        # Population order is already the feasibility-first rank, so the
        # lower index wins a binary tournament without comparing floats.
        tournaments = rng.integers(0, population_size, size=(2 * pair_count, 2))
        parents = population[np.min(tournaments, axis=1)]
        first, second = _sbx(parents[0::2], parents[1::2], lower, upper, rng)
        children = np.stack((first, second), axis=1).reshape(-1, len(lower))[:population_size]
        children = _mutate(children, lower, upper, rng)
        child_objective, child_violation = (
            np.asarray(values, dtype=float) for values in evaluate(children)
        )
        evaluations += population_size
        combined = np.concatenate((population, children))
        combined_objective = np.concatenate((objective, child_objective))
        combined_violation = np.concatenate((violation, child_violation))
        survivors = _order(combined_objective, combined_violation)[:population_size]
        population = combined[survivors]
        objective = combined_objective[survivors]
        violation = combined_violation[survivors]
        record(generation)

    return {
        "best_x": population[0].copy(),
        "best_objective": float(objective[0]),
        "best_violation": float(violation[0]),
        "evaluations": evaluations,
        "history": history,
        "settings": {
            "seed": int(seed),
            "population_size": int(population_size),
            "generations": int(generations),
            "initialization": "independent uniform sampling within bounds",
            "selection": "binary tournament; feasibility-first ranking",
            "survival": "elitist parent-plus-offspring ranking",
            "crossover": "bounded simulated binary crossover",
            "crossover_probability_per_pair": 0.9,
            "crossover_probability_per_variable": 0.5,
            "crossover_eta": 15.0,
            "mutation": "bounded polynomial mutation",
            "mutation_probability_per_variable": 1.0 / len(lower),
            "mutation_eta": 20.0,
            "random_resets": False,
        },
    }
