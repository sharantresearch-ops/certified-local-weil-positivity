# Certified local Weil positivity

Interval-arithmetic certificate for a computer-assisted theorem:

> For every `0 < a <= 3/5`, the exact compact-support zeta-Weil form, restricted to the
> complex-even `cosh(x/2)`-moment-null closed form domain, satisfies
> `T_a[f] >= (3/100000) ||f||_2^2`.

At `a = 3/5` the arithmetic part contains exactly the prime-power atoms `n = 2, 3`. The paper
also gives the corresponding normalized shorted-graph Poisson-frame Loewner inequality on the
same even sector.

Sharan Thota. Paper in `paper/`, certificate and validator in `certificate/`.

## What this is not

This is a local support theorem. It says nothing about positivity at larger support radius,
nothing about the unrestricted or non-even localized operator, and it does not prove RH.

The finite-dimensional interval calculation controls the complement of an infinite-dimensional
trace-class operator. No Ritz eigenvalue of the target Weil form is used as a lower bound
anywhere. That is the step that usually goes wrong in arguments of this shape, so I am saying
it twice.

## Replaying it

Python 3.11 and `python-flint` 0.9.0 (FLINT 3.6.0). The frozen certificate was produced at
256-bit Arb working precision.

```
python -m pip install -r requirements.txt
```

The fast path does no quadrature. It checks the canonical proof payload and replays the final
serialized interval-matrix `LDL^*` decision:

```
mkdir build
python certificate/validate.py \
  --ledger certificate/ledger.json \
  --output build/recomputed.json \
  --precision 256

cmp build/recomputed.json \
  certificate/schur-lmi.json
sha256sum -c SHA256SUMS
```

The recomputed JSON has to come out byte-identical. What it should contain:

- canonical proof-payload SHA-256 `950A5E6D261E548A40AB776D15F67741660A9786614E6B000B0CD493378A2285`
- minimum interval `LDL^*` pivot, lower endpoint `0.009793348856651610171448814290609394...`
- coercive reserve `3/100000`

CI runs exactly this on every push.

## Regenerating from scratch

Slow. The generator rigorously recomputes all 1,980 Gram/compression entries plus the sign,
root-neighborhood, trace-tail and interval-positivity checks, across several processes. Budget
minutes at minimum, hardware depending. Output lands in the ignored `build/` so the frozen
artifacts stay put.

```
python certificate/certify.py \
  --workers 8 --precision-bits 256 --tolerance 1e-14 \
  --eval-limit 500000 --json build/fresh-ledger.json

python certificate/validate.py \
  --ledger build/fresh-ledger.json \
  --output build/fresh-schur-lmi.json --precision 256
```

Parallel scheduling moves the wall clock, not the payload.

Two different reserve numbers show up and it is worth knowing which is which. The generator
prints its own margin, near `2.404e-5`, which comes from the coarse 2x2 block bound it uses
internally. The theorem's `3/100000` is the sharper matrix Schur bound and comes out of the
validator. Both are valid lower bounds; the paper states the second.
