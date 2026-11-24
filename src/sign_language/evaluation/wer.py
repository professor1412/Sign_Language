"""Corpus gloss WER. Deterministic ties: diagonal, deletion, insertion."""


def edit_counts(reference, hypothesis):
    # Each cell stores distance, substitutions, deletions, insertions.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, word in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, prediction in enumerate(hypothesis, 1):
            cost, s, d, ins = previous[j-1]
            different = int(word != prediction)
            diagonal = (cost + different, s + different, d, ins)
            cost, s, d, ins = previous[j]
            deletion = (cost + 1, s, d + 1, ins)
            cost, s, d, ins = current[j-1]
            insertion = (cost + 1, s, d, ins + 1)
            current.append(min((diagonal, deletion, insertion), key=lambda item: item[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {"reference_words": len(reference), "errors": errors, "substitutions": substitutions,
            "deletions": deletions, "insertions": insertions}


def corpus_wer(references, hypotheses):
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Need equally sized, nonempty reference and prediction collections")
    total = dict.fromkeys(("reference_words", "errors", "substitutions", "deletions", "insertions"), 0)
    for ref, hyp in zip(references, hypotheses):
        for key, value in edit_counts(ref, hyp).items():
            total[key] += value
    total["wer_percent"] = 100 * total["errors"] / total["reference_words"] if total["reference_words"] else None
    total["videos"] = len(references)
    return total
