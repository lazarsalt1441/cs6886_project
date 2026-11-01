"""Runnable NumPy counterparts of the report's explanatory transformations.

These check the displayed array laws on concrete inputs. They are not compiler
pass tests, a substitute for the C++ regression suites, or GPU benchmarks.
Run: python report/examples.py
"""
import json
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def update(c, p, u):
    out = c.copy()
    out[p:p + len(u)] = u
    return out

def main():
    rng = np.random.default_rng(6886)
    checks = []
    def check(name, before, after, exact=False):
        if exact:
            np.testing.assert_array_equal(before, after)
        else:
            np.testing.assert_allclose(before, after, atol=1e-12, rtol=1e-12)
        checks.append(dict(example=name, passed=True))

    # 1. Flat query heads vs explicit KV-head/sharing-group loops.
    q = rng.normal(size=(8, 3, 5))
    k = rng.normal(size=(2, 7, 5))
    v = rng.normal(size=(2, 7, 4))
    kr, vr = np.repeat(k, 4, axis=0), np.repeat(v, 4, axis=0)
    expected = softmax(q @ kr.swapaxes(-2, -1) / np.sqrt(5)) @ vr
    grouped_q = q.reshape(2, 4, 3, 5)
    scores = np.einsum('hgmd,hnd->hgmn', grouped_q, k) / np.sqrt(5)
    actual = np.einsum('hgmn,hnv->hgmv', softmax(scores), v)
    check('grouped_GQA_without_repeat', expected, actual.reshape(8, 3, 4))

    c = rng.normal(size=(12, 2, 3))
    u = rng.normal(size=(2, 2, 3))
    p, ids = 4, np.array([0, 4, 5, 11])
    expected = update(c, p, u)[ids]
    conditional = np.stack([u[i-p] if p <= i < p+len(u) else c[i] for i in ids])
    check('conditional_read_forwarding', expected, conditional, exact=True)
    inside = (p <= ids) & (ids < p + len(u))
    safe = np.where(inside, ids-p, 0)
    masked = np.where(inside[:, None, None], u[safe], c[ids])
    check('masked_gather_forwarding_with_safe_indices', expected, masked, exact=True)

    a, b = rng.normal(size=(1, 2, 3)), rng.normal(size=(1, 2, 3))
    for winner in [False, True]:
        before = np.where(winner, np.concatenate([c, a]), np.concatenate([c, b]))
        after = np.concatenate([c, np.where(winner, a, b)])
        check('winner_commit_' + str(winner), before, after, exact=True)

    c1 = update(c, 4, a)
    c2 = update(c1, 5, b)
    joined = update(c, 4, np.concatenate([a, b]))
    check('adjacent_update_coalescing', c2, joined, exact=True)
    # An observed intermediate remains a different value: it cannot be replaced
    # by the final version. The C++ pass rejects that additional use.
    assert not np.array_equal(c1, joined)

    vector_ids = np.array([[[3], [1], [7]], [[2], [0], [9]]])
    scalar_ids = vector_ids.reshape(2, 3)
    expected = np.stack([np.stack([c[vector_ids[f,n,0]] for n in range(3)]) for f in range(2)])
    check('unit_component_axis_gather_normalization', expected, c[scalar_ids], exact=True)

    original = c.copy()
    bad = c.copy()
    wrong_new = bad                 # unsafe alias of the returned old buffer
    wrong_new[4:5] = a
    assert not np.array_equal(bad, original)
    safe_new = update(original, 4, a)
    check('snapshot_old_buffer_preserved', c, original, exact=True)
    assert not np.array_equal(safe_new, original)

    latent = rng.normal(size=(7, 3))
    wk = rng.normal(size=(3, 5))
    query = rng.normal(size=(2, 5))
    check('key_projection_placement', query @ (latent @ wk).T, (query @ wk.T) @ latent.T)
    probs = softmax(query @ (latent @ wk).T)
    wv = rng.normal(size=(3, 4))
    check('value_projection_placement', probs @ (latent @ wv), (probs @ latent) @ wv)

    qr, kr = rng.normal(size=(2, 2)), rng.normal(size=(7, 2))
    old_scores = np.concatenate([query, qr], axis=1) @ np.concatenate([latent @ wk, kr], axis=1).T
    new_scores = (query @ wk.T) @ latent.T + qr @ kr.T
    check('partial_RoPE_preserves_original_scale', softmax(old_scores / np.sqrt(7)), softmax(new_scores / np.sqrt(7)))
    assert not np.allclose(softmax(new_scores / np.sqrt(5)), softmax(old_scores / np.sqrt(7)))

    partial_scores = np.zeros((16, 2))
    partial_scores[0, 0] = np.log(9)
    correct = softmax(partial_scores.sum(axis=0))[0]
    wrong = np.exp(partial_scores[:, 0]).sum() / np.exp(partial_scores).sum()
    check('native_phase_counterexample_correct_result', correct, 0.9)
    check('native_phase_counterexample_wrong_result', wrong, 0.6)

    x = np.array([-9000, -8192, -4096, -33, 0, 32, 33, 4096, 8192, 9000], dtype=np.float32)
    xd = x.astype(np.float64)
    reduced = (xd - np.floor(xd / (2*np.pi) + 0.5) * (2*np.pi)).astype(np.float32)
    active = (np.abs(x) > 32) & (np.abs(x) <= 8192)
    out = np.where(active, np.sin(reduced), np.sin(x))
    np.testing.assert_array_equal(out[~active], np.sin(x)[~active])
    assert np.max(np.abs(out[active] - np.sin(xd[active]))) < 3e-7
    checks.append(dict(example='bounded_trig_example_and_exact_fallback', passed=True))
    print(json.dumps(dict(scope='Illustrative array-law checks only; not compiler/GPU execution', checks=checks), indent=2))

if __name__ == '__main__':
    main()
