"""Dependency-free categorical calibration; targets are puzzle solution moves."""
import math


def log_probs(scores, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError("scores must be nonempty and finite")
    maximum = max(scores)
    shifted = [(s - maximum) / temperature for s in scores]
    z = math.log(sum(math.exp(s) for s in shifted))
    return [s - z for s in shifted]


def validate(rows):
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    for row in rows:
        log_probs(row['scores'])
        if type(row['label']) is not int or not 0 <= row['label'] < len(row['scores']):
            raise ValueError("label outside candidate set")


def nll(rows, temperature):
    return sum(-log_probs(r['scores'], temperature)[r['label']] for r in rows) / len(rows)


def fit_temperature(validation):
    """Minimize validation NLL over T=[0.05,20]; never takes test data."""
    validate(validation)
    lo, hi = math.log(.05), math.log(20.)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(validation, math.exp(a)), nll(validation, math.exp(b))
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(validation, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(validation, math.exp(b))
    candidates = [1., .05, 20., math.exp((lo + hi) / 2)]
    return min(candidates, key=lambda t: nll(validation, t))


def metrics(rows, temperature=1., bins=10):
    validate(rows)
    if bins < 1:
        raise ValueError("bins must be positive")
    buckets = [[] for _ in range(bins)]
    observations, brier, top3 = [], 0., 0
    for row in rows:
        probs = [math.exp(p) for p in log_probs(row['scores'], temperature)]
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        correct, confidence = int(order[0] == row['label']), probs[order[0]]
        top3 += row['label'] in order[:3]
        brier += sum((p - int(i == row['label'])) ** 2 for i, p in enumerate(probs))
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, correct))
        observations.append((confidence, correct))
    reliability, ece = [], 0.
    for i, bucket in enumerate(buckets):
        conf = sum(x[0] for x in bucket) / len(bucket) if bucket else None
        acc = sum(x[1] for x in bucket) / len(bucket) if bucket else None
        if bucket:
            ece += len(bucket) / len(rows) * abs(conf - acc)
        reliability.append(dict(lower=i / bins, upper=(i+1) / bins,
                                count=len(bucket), confidence=conf, accuracy=acc))
    # Include all ties at the cutoff; report actual coverage rather than breaking ties arbitrarily.
    ranked = sorted(observations, reverse=True)
    selective = []
    for coverage in (.25, .5, .75, 1.):
        threshold = ranked[math.ceil(len(rows) * coverage) - 1][0]
        kept = [correct for confidence, correct in observations if confidence >= threshold]
        selective.append(dict(requested_coverage=coverage, coverage=len(kept)/len(rows),
                              threshold=threshold, error=1-sum(kept)/len(kept)))
    return dict(n=len(rows), accuracy=sum(x[1] for x in observations)/len(rows),
                top3=top3/len(rows), nll=nll(rows, temperature), brier=brier/len(rows),
                ece=ece, temperature=temperature, reliability=reliability,
                selective=selective, forced_positions=sum(len(r['scores']) == 1 for r in rows))
