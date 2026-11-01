#!/usr/bin/env python3
"""Recalculate IREE-only per-study comparisons from committed raw results.

Run from any directory; outputs go beside this script under study_results/.
No benchmark is rerun. Speedup = baseline latency / transformed latency.
"""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'study_results'
ROWS = []
SOURCES = {}
EXCLUDED = []


def load(path):
    return json.loads((ROOT / path).read_text())


def add(study, optimization, case, baseline, optimized, before, after,
        unit, attribution, source, note=''):
    assert before > 0 and after > 0
    ROWS.append(dict(study=study, optimization=optimization, case=str(case),
                     baseline=baseline, optimized=optimized,
                     baseline_time=before, optimized_time=after, unit=unit,
                     speedup=before / after, attribution=attribution,
                     source=source, note=note))


def medians(study, optimization, case, times, baseline, variants, source,
            attribution='isolated', note=''):
    for variant in variants:
        add(study, optimization, case, baseline, variant, times[baseline],
            times[variant], 'us', attribution, source, note)


def main():
    source = 'compiler/research2/results/final-timings.json'
    data = load(source)
    SOURCES['CPU cache operators'] = dict(source=source, timing=data['method'])
    for g in data['groups']:
        times = g['holdout']['median_us']
        if g['family'] == 'winner':
            for v, label, attr in [
                ('factored', 'Winner factoring', 'isolated'),
                ('factor-conditional', 'Winner factoring + conditional forwarding', 'combined'),
                ('factor-masked', 'Winner factoring + masked forwarding', 'combined')]:
                medians('CPU cache operators', label, g['case'], times,
                        g['best_baseline'], [v], source, attr)
        else:
            for v, label in [('forwarded', 'Conditional forwarding'),
                             ('patch-forwarded', 'Masked gather forwarding')]:
                medians('CPU cache operators', label, g['family'] + ':' + g['case'],
                        times, g['best_baseline'], [v], source)

    source = 'compiler/research2/results/extension-timings.json'
    data = load(source)
    SOURCES['CPU extensions'] = dict(source=source, timing=data['method'])
    for g in data['groups']:
        times, base = g['holdout']['median_us'], g['best_baseline']
        if g['family'] == 'chain':
            variants = [('coalesced', 'Update coalescing', 'isolated'),
                        ('coalesced-forwarded', 'Coalescing + conditional forwarding', 'combined'),
                        ('coalesced-masked', 'Coalescing + masked forwarding', 'combined')]
            note = 'One-token coalescing is a no-op.' if g['case'] == 1 else ''
        else:
            variants = [('protected', 'Snapshot protection', 'correctness repair'),
                        ('root-cow', 'Borrowed-buffer COW', 'correctness repair'),
                        ('forwarded', 'Snapshot protection + conditional forwarding', 'combined'),
                        ('masked', 'Snapshot protection + masked forwarding', 'combined')]
            note = 'Compared with fastest correct snapshot baseline; incorrect stock excluded.'
        for variant, label, attr in variants:
            medians('CPU extensions', label, g['family'] + ':' + str(g['case']),
                    times, base, [variant], source, attr, note)

    source = 'compiler/research/results/final-timings.json'
    data = load(source)
    SOURCES['CPU projections'] = dict(source=source, timing=data['method'])
    for g in data['groups']:
        baseline = {'decomposed': 'strong', 'rotary': 'native', 'portfolio': 'original'}[g['family']]
        optimized = g['selected'] if g['family'] == 'portfolio' else 'optimized'
        medians('CPU projections', {'decomposed': 'Key/value projection placement',
                'rotary': 'Partial RoPE + projection placement',
                'portfolio': 'Projection portfolio selection'}[g['family']],
                g['case'], g['median_us'], baseline, [optimized], source,
                'combined' if g['family'] == 'rotary' else 'pipeline',
                'Unabsorbed strong baseline; not an optimized MLA serving baseline.')

    source = 'compiler/research/results/final-ablation-timings.json'
    SOURCES['CPU projection ablations'] = dict(source=source,
        timing='Recorded per-variant microsecond medians; separate ablation run.')
    for g in load(source):
        times = g['median_us']
        if g['family'] == 'fallback':
            medians('CPU projection ablations', 'Correct contraction fallback',
                    g['case'], times, 'guarded', ['fallback'], source,
                    'alternative correct lowering',
                    'Compared with correct guarded fused lowering, not incorrect stock.')
        else:
            for variant in times:
                if variant != 'baseline':
                    medians('CPU projection ablations', 'Projection ablation',
                            g['case'], times, 'baseline', [variant], source,
                            'ablation', 'Variant names identify key-only/value-only and policy controls.')

    prefix = 'results/initial_gpu/job_colab_t4_001/artifacts/'
    source = prefix + 'research/results/gpu-sm_75-runtime.json'
    study = 'T4 earlier projection operators'
    SOURCES[study] = dict(source=source,
        timing='Synchronized invocation, execution and all output downloads; resident input upload excluded. Separate from final resident-output model timings.')
    for g in load(source):
        if not all(g['variants'][v].get('correct', False) for v in ['baseline', 'optimized']):
            EXCLUDED.append(dict(study=study, case=g['case'], source=source,
                                 reason='At least one variant failed numerical validation.'))
            continue
        medians(study, 'Partial RoPE + projection placement' if g['kind'] == 'rotary'
                else 'Key/value projection placement' if g['kind'] == 'decomposed'
                else 'Native projection/lowering portfolio',
                g['case'], g['synchronized_host_median_us'], 'baseline',
                ['optimized'], source, 'combined pipeline')
    for family in ['attention', 'block', 'winner']:
        source = prefix + f'research2/results/gpu-sm_75-runtime-{family}.json'
        study = f'T4 earlier cache {family} operators'
        SOURCES[study] = dict(source=source,
            timing='Synchronized invocation, execution and all output downloads; best correct available IREE control chosen as baseline. No pooling with final resident-output timings.')
        for g in load(source):
            times = g['synchronized_host_median_us']
            valid = {v for v in times if g['variants'][v].get('correct', False)}
            baseline = min((v for v in ['strong', 'aggressive', 'lifted-aggressive',
                            'lifted-normalized'] if v in valid), key=times.get)
            for variant, label in [('forwarded', 'Conditional forwarding'),
                                   ('patch-forwarded', 'Masked gather forwarding')]:
                if variant not in valid:
                    EXCLUDED.append(dict(study=study, case=g['case'], variant=variant,
                                         source=source, reason='Not compiled/correct.'))
                    continue
                medians(study, ('Winner factoring + ' + label.lower()) if family == 'winner'
                        else label, g['case'], times, baseline, [variant], source,
                        'combined' if family == 'winner' else 'isolated')
    source = prefix + 'research2/results/gpu-extensions-sm_75-runtime.json'
    study = 'T4 earlier update-chain operators'
    SOURCES[study] = dict(source=source,
        timing='Synchronized invocation plus all output downloads; resident input upload excluded. Combined coalescing/forwarding forms, not coalescing-only ablations.')
    for g in load(source):
        times = g['host_median_us']
        valid = {v for v in times if g['variants'][v].get('correct', False)}
        if not {'baseline', 'fused'} & valid:
            EXCLUDED.append(dict(study=study, case=g.get('tokens', g['family']),
                                 source=source, reason='No correct baseline.'))
            continue
        base = min({'baseline', 'fused'} & valid, key=times.get)
        for v in ['conditional', 'masked']:
            if v in valid:
                medians(study, ('Coalescing' if g['family'] == 'chain' else 'Snapshot protection') + ' + ' + v + ' forwarding',
                        g['family'] + ':' + str(g.get('tokens', '')), times, base, [v], source,
                        'combined', 'One-token coalescing is a no-op.' if g.get('tokens') == 1 else '')

    for track, stamp in [('A', '20261101_074854_B_main'),
                         ('B', '20261101_081149_B_main')]:
        source = f'results/dram/{track}/{stamp}/main/summary.json'
        data = load(source)
        study = f'T4 final DRAM study {track}'
        SOURCES[study] = dict(source=source,
            timing='Unprofiled resident-output context-synchronized latency; median of repetition medians. Model: three prompts; synthetic: one case. Not pooled with earlier operator timings.')
        rows = data['valid_latency_rows']
        # torch_compile remains in raw data only; all comparisons here are IREE.
        cases = sorted({r['case'] for r in rows})
        for case in cases:
            if case.startswith(('A_n', 'B_n')):
                continue
            ts = {b: statistics.median(r['median_ms'] for r in rows
                    if r['case'] == case and r['backend'] == b and r.get('valid', True))
                  for b in ['iree_stable', 'iree_forwarded']}
            add(study, 'Conditional forwarding', case, 'iree_stable',
                'iree_forwarded', ts['iree_stable'], ts['iree_forwarded'],
                'ms', 'no-match control' if case == 'cache_dense' else 'isolated',
                source, 'Dense case is a no-rewrite timing control; not an optimization gain.' if case == 'cache_dense' else '')
        for n in [2048, 3072, 4096]:
            selected = [r for r in rows if r['case'].startswith(f'{track}_n{n}_')]
            if not selected:
                continue
            opt = 'iree_commit' if track == 'B' else 'iree_forwarded'
            if not any(r['backend'] == opt for r in selected):
                continue  # Ordinary decode has identical IR; no invented optimized measurement.
            ts = {b: statistics.median(r['median_ms'] for r in selected
                                      if r['backend'] == b) for b in ['iree_stable', opt]}
            add(study, 'Winner commit factoring' if track == 'B' else 'Ordinary decode control',
                f'context={n}', 'iree_stable', opt, ts['iree_stable'], ts[opt],
                'ms', 'isolated' if track == 'B' else 'no-match control', source,
                'Both variants use identical trig repair and strong compiler flags.')

    source = 'results/repair_results_B/repair_results_B/latency.json'
    data = load(source)
    SOURCES['T4 earlier fusion ablation'] = dict(source=source,
        timing='Earlier repaired-model run; median of valid repetition medians, kept separate from final DRAM study.')
    for n in [2048, 3072, 4096]:
        ts = {b: statistics.median(r['median_ms'] for r in data
                if r['context'] == n and r['backend'] == b and r['valid'])
              for b in ['iree_stable_commit', 'iree_stable_commit_aggressive']}
        add('T4 earlier fusion ablation', 'Additional aggressive fusion', n,
            'iree_stable_commit', 'iree_stable_commit_aggressive',
            ts['iree_stable_commit'], ts['iree_stable_commit_aggressive'],
            'ms', 'standard IREE flag ablation', source)

    repairs = []
    for track in ['A', 'B']:
        source = f'results/repair_results_{track}/repair_results_{track}/latency.json'
        data = load(source)
        valid = {(r['context'], r['prompt'], r['repetition'], r['backend']): r
                 for r in data if r['valid']}
        ratios = [r['median_ms'] / valid[(*key[:3], 'iree_baseline')]['median_ms']
                  for key, r in valid.items() if key[3] == 'iree_stable'
                  and (*key[:3], 'iree_baseline') in valid]
        repairs.append(dict(optimization='Stable trig', track=track, source=source,
            matched_valid_pairs=len(ratios),
            median_latency_overhead_percent=100 * (statistics.median(ratios) - 1),
            note='Matched valid original cases only; repair improves natural-case correctness from 5/9 to 9/9. Not a speed optimization.'))

    OUT.mkdir(exist_ok=True)
    suite_means = []
    for study, optimization in [('CPU projections', 'Key/value projection placement'),
            ('CPU projections', 'Partial RoPE + projection placement'),
            ('T4 earlier projection operators', 'Key/value projection placement'),
            ('T4 earlier projection operators', 'Partial RoPE + projection placement')]:
        values = [r['speedup'] for r in ROWS if r['study'] == study and r['optimization'] == optimization]
        suite_means.append(dict(study=study, optimization=optimization, cases=len(values),
                                geometric_mean_speedup=math.exp(statistics.mean(map(math.log, values)))))
    result = dict(definition='speedup = baseline latency / optimized latency; <1 is slower',
                  sources=SOURCES, comparisons=ROWS, excluded_comparisons=EXCLUDED,
                  suite_geometric_means=suite_means, correctness_repairs=repairs,
                  unattributed=[dict(optimization=o, result=note) for o, note in [
                      ('Gather-index normalization', 'Correctness enabler; standalone speedup not isolated.'),
                      ('Native phase guard', '20/20 CPU audit cases pass vs 6/20 stock; no valid speedup against wrong output.'),
                      ('Original GQA head reuse', 'Historical default CPU 18.47 to 9.94 us (1.86x); no added gain established over both upstream flags plus fusion.')]])
    (OUT / 'per_study_results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (OUT / 'per_study_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(ROWS)
    lines = ['# Per-study IREE results', '',
        'Generated by `python report/summarize_studies.py` from committed raw measurements. '
        'No new timing runs. Speedup is baseline / optimized latency; below 1 means slower. '
        'Rows marked combined cannot identify the standalone contribution of each pass. '
        'Representative PDF rows are not suite-wide guarantees. CPU and GPU timings are not pooled.', '']
    lines += ['## Suite summaries', '', '| Study | Transformation | Cases | Geometric-mean speedup |',
              '|---|---|---:|---:|']
    lines += [f"| {r['study']} | {r['optimization']} | {r['cases']} | {r['geometric_mean_speedup']:.3f}× |" for r in suite_means]
    lines += ['']
    for study, metadata in SOURCES.items():
        lines += [f'## {study}', '', metadata['timing'], '',
                  f"Source path (after artifact unpacking): `{metadata['source']}`", '',
                  '| Transformation / case | Baseline → optimized | Time before → after | Speedup | Attribution |',
                  '|---|---|---:|---:|---|']
        for r in ROWS:
            if r['study'] == study:
                lines.append(f"| {r['optimization']}: `{r['case']}` | {r['baseline']} → {r['optimized']} | {r['baseline_time']:.3f} → {r['optimized_time']:.3f} {r['unit']} | {r['speedup']:.3f}× | {r['attribution']} |")
        lines += ['']
    lines += ['## Correctness repairs and non-isolated effects', '']
    lines += [f"* Stable trig, track {r['track']}: {r['matched_valid_pairs']} valid matched pairs; median latency overhead {r['median_latency_overhead_percent']:.3f}%. {r['note']}" for r in repairs]
    lines += [f"* {r['optimization']}: {r['result']}" for r in result['unattributed']]
    lines += ['', '## Excluded comparisons', '']
    lines += [f"* {r['study']}, `{r['case']}`: {r['reason']}" for r in EXCLUDED]
    lines += ['', '## Interpretation', '',
        '* Coalescing alone is slightly slower on the 4–16-token cases. Its benefit is enabling forwarding; the combined result is 2.17–2.34×.',
        '* Snapshot gains include forwarding. The protection/COW passes restore semantics and cannot be credited with the entire combined gain.',
        '* Partial-RoPE rows include projection placement; no standalone RoPE speedup was isolated.',
        '* Dense no-rewrite GPU regressions and CPU negative cases are retained. Uncontrolled clocks and kernel differences limit causal timing attribution.',
        '* Final model gains concern a custom two-candidate step in FP32 SmolLM2-135M on T4, not general autoregressive serving throughput.', '']
    (OUT / 'per_study_results.md').write_text('\n'.join(lines))
    assert len(ROWS) > 100
    assert all(math.isfinite(r['speedup']) and r['speedup'] > 0 for r in ROWS)
    assert all(not r['baseline'].startswith('torch') and not r['optimized'].startswith('torch') for r in ROWS)
    print(json.dumps(dict(comparisons=len(ROWS), repairs=repairs), indent=2))


if __name__ == '__main__':
    main()
