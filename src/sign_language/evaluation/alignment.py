"""Deterministic minimum-edit alignment for per-gloss recall."""


def align(reference, hypothesis):
    costs = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        costs[i][0] = i
    for j in range(len(hypothesis) + 1):
        costs[0][j] = j
    for i, word in enumerate(reference, 1):
        for j, predicted in enumerate(hypothesis, 1):
            costs[i][j] = min(costs[i-1][j-1] + (word != predicted),
                              costs[i-1][j] + 1, costs[i][j-1] + 1)
    i, j = len(reference), len(hypothesis)
    pairs = []
    while i or j:
        if i and j and costs[i][j] == costs[i-1][j-1] + (reference[i-1] != hypothesis[j-1]):
            pairs.append((reference[i-1], hypothesis[j-1]))
            i, j = i-1, j-1
        elif i and costs[i][j] == costs[i-1][j] + 1:
            pairs.append((reference[i-1], None))
            i -= 1
        else:
            pairs.append((None, hypothesis[j-1]))
            j -= 1
    return list(reversed(pairs))
