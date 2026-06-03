"""
Lesson 00 — The Functional Model
================================================================================

This is the foundation. Every other JAX transform (jit, grad, vmap, pmap) assumes
the function it wraps is *pure* and operates on *immutable* data. Get this right and
the rest of JAX is composition.

Four ideas:

1. ARRAYS ARE IMMUTABLE
   You cannot do `x[0] = 5` on a JAX array. Instead you produce a NEW array:
       y = x.at[0].set(5)
   `x` is unchanged; `y` is a new array. This is the same discipline as functional
   programming: no in-place mutation, only transformation.

2. PURE FUNCTIONS
   A pure function (a) returns the same output for the same input, and (b) has no
   side effects (no printing, no mutating globals, no I/O that affects the result).
   JAX *traces* your function with abstract placeholder values to build a computation
   graph. Side effects happen during tracing (once), not during execution — so an
   impure function will silently misbehave under jit. Purity is not stylistic here;
   it's a correctness requirement.

3. EXPLICIT RANDOMNESS (PRNG KEYS)
   NumPy hides RNG state in a global. That's a side effect, so JAX bans it. Instead
   you pass an explicit `key`. To get independent randomness you *split* a key:
       key, subkey = jax.random.split(key)
   Reusing the same key gives identical "random" numbers. Splitting is how you get
   fresh, independent streams — deterministically and reproducibly.

4. PYTREES
   Model parameters are nested containers of arrays — e.g.
   `{"layer1": {"w": ..., "b": ...}, "layer2": {...}}`. JAX calls any such nested
   structure of lists/tuples/dicts-of-arrays a "pytree". `jax.tree.map(f, tree)`
   applies `f` to every leaf (array) while preserving the structure. This is how an
   optimizer updates *all* params in one line, regardless of model shape.

Fill in every TODO, then run:  python 01-jax/lesson_00_functional_model.py
"""

import jax
import jax.numpy as jnp


# ------------------------------------------------------------------------------
# Exercise 1 — Immutable update
# ------------------------------------------------------------------------------
def zero_out_negatives(x):
    """Return a NEW array equal to `x` but with every negative element replaced by 0.

    Constraint: do NOT use a Python loop and do NOT mutate `x`. Use JAX's functional
    update syntax (`x.at[...]` ... ) and/or a boolean mask. There are a couple of
    clean one-liners — find one.
    """
    # TODO: implement
    raise NotImplementedError


# ------------------------------------------------------------------------------
# Exercise 2 — Make this pure
# ------------------------------------------------------------------------------
# The function below is IMPURE: it mutates a global counter and depends on it.
# Under jit this would capture the counter's value at trace time and never change.
#
# Rewrite it as `running_sum_pure(state, x)` that takes the current running total
# (`state`, a scalar array) and the new value `x`, and returns the NEW running total.
# No globals, no mutation. This "thread the state through" pattern is how JAX handles
# everything stateful: optimizers, RNG, RNN hidden states, etc.

_counter = 0.0  # do not use this in your solution


def running_sum_impure(x):  # <- the bad example, for contrast. Don't edit.
    global _counter
    _counter += x
    return _counter


def running_sum_pure(state, x):
    """Return the new running total given the previous `state` and new value `x`."""
    # TODO: implement
    raise NotImplementedError


# ------------------------------------------------------------------------------
# Exercise 3 — Independent randomness via key splitting
# ------------------------------------------------------------------------------
def two_independent_batches(key, shape):
    """Given one PRNG `key`, return a tuple `(a, b)` of two arrays, each of `shape`,
    drawn from a standard normal, that are STATISTICALLY INDEPENDENT of each other.

    Hint: if you call jax.random.normal with the same key twice you get the SAME
    numbers. You must split the key first.
    """
    # TODO: implement
    raise NotImplementedError


# ------------------------------------------------------------------------------
# Exercise 4 — Pytrees: one SGD step over arbitrary params
# ------------------------------------------------------------------------------
def sgd_step(params, grads, lr):
    """Apply one SGD update to a pytree of parameters.

    `params` and `grads` are pytrees with identical structure (nested dicts of arrays).
    Return a new pytree where every leaf is  `p - lr * g`.

    Constraint: this must work for ANY nesting structure without you knowing the keys.
    Use jax.tree.map over the two trees together. (One line.)
    """
    # TODO: implement
    raise NotImplementedError


# ==============================================================================
# Checker — do not edit below this line
# ==============================================================================
def _check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    return cond


def main():
    print("Lesson 00 — functional model\n")
    results = []

    # Ex1
    try:
        x = jnp.array([-2.0, 3.0, -1.0, 0.0, 5.0])
        y = zero_out_negatives(x)
        ok = (
            jnp.allclose(y, jnp.array([0.0, 3.0, 0.0, 0.0, 5.0]))
            and jnp.allclose(x, jnp.array([-2.0, 3.0, -1.0, 0.0, 5.0]))  # x unchanged
        )
        results.append(_check("ex1 zero_out_negatives", bool(ok), f"got {y}"))
    except NotImplementedError:
        results.append(_check("ex1 zero_out_negatives", False, "not implemented"))

    # Ex2
    try:
        s = jnp.array(0.0)
        for v in [1.0, 2.0, 3.0]:
            s = running_sum_pure(s, v)
        # purity: calling with the same args twice gives the same answer
        a = running_sum_pure(jnp.array(10.0), 5.0)
        b = running_sum_pure(jnp.array(10.0), 5.0)
        ok = jnp.allclose(s, 6.0) and jnp.allclose(a, b) and jnp.allclose(a, 15.0)
        results.append(_check("ex2 running_sum_pure", bool(ok), f"sum={s}, a={a}"))
    except NotImplementedError:
        results.append(_check("ex2 running_sum_pure", False, "not implemented"))

    # Ex3
    try:
        key = jax.random.key(0)
        a, b = two_independent_batches(key, (10000,))
        same_shape = a.shape == (10000,) and b.shape == (10000,)
        not_identical = not bool(jnp.allclose(a, b))
        # independent draws have near-zero correlation
        corr = jnp.corrcoef(a, b)[0, 1]
        low_corr = bool(jnp.abs(corr) < 0.1)
        ok = same_shape and not_identical and low_corr
        results.append(_check("ex3 two_independent_batches", ok, f"corr={corr:.3f}"))
    except NotImplementedError:
        results.append(_check("ex3 two_independent_batches", False, "not implemented"))

    # Ex4
    try:
        params = {"l1": {"w": jnp.ones((2, 2)), "b": jnp.zeros((2,))},
                  "l2": {"w": jnp.full((2,), 3.0)}}
        grads = {"l1": {"w": jnp.ones((2, 2)), "b": jnp.ones((2,))},
                 "l2": {"w": jnp.ones((2,))}}
        new = sgd_step(params, grads, lr=0.1)
        ok = (
            jnp.allclose(new["l1"]["w"], 0.9)
            and jnp.allclose(new["l1"]["b"], -0.1)
            and jnp.allclose(new["l2"]["w"], 2.9)
        )
        results.append(_check("ex4 sgd_step", bool(ok)))
    except NotImplementedError:
        results.append(_check("ex4 sgd_step", False, "not implemented"))

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed.")
    if passed == len(results):
        print("All green. Write a sentence in NOTES.md on why purity is required for "
              "tracing, then ping me to review and move to lesson 01 (grad).")


if __name__ == "__main__":
    main()
