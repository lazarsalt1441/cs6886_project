"""Separate calibration from holdout timing. No architecture-specific winner table."""
import math
import random
import statistics
import time


def select(medians, baseline, min_gain=0.05):
    if baseline not in medians or not medians:
        raise ValueError('Missing baseline')
    if not 0 <= min_gain < 1:
        raise ValueError('Invalid minimum gain')
    if any(not math.isfinite(x) or x <= 0 for x in medians.values()):
        raise ValueError('Invalid latency')
    best = min(medians, key=medians.get)
    return best if medians[best] <= medians[baseline] * (1-min_gain) else baseline


def calibrate(invocations, baseline, trials=21, min_gain=0.05):
    if trials <= 0:
        raise ValueError('Calibration requires observations')
    rng = random.Random(731)
    samples = {v: [] for v in invocations}
    for _ in range(trials):
        order = list(invocations)
        rng.shuffle(order)
        for v in order:
            start = time.perf_counter_ns()
            invocations[v]()  # Caller must synchronize, e.g. output to_host().
            samples[v].append((time.perf_counter_ns()-start)/1000)
    medians = {v: statistics.median(s) for v,s in samples.items()}
    return dict(selected=select(medians,baseline,min_gain), samples_us=samples,
                medians_us=medians, min_gain=min_gain,
                note='Selection uses calibration only, not subsequent holdout samples.')
