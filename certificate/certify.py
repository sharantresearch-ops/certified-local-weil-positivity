"""Rigorous Arb certificate for the full-rescue block at a=3/5.

The proof is an exact continuum operator bound on the complex-even,
cosh(x/2)-moment-null space. It does not use a finite Ritz value as a
lower bound for the target Weil form.

Worker processes enclose the two finite compression matrices. Every
support, sign, tail, trace, generalized matrix, and final-margin decision
is an active Arb endpoint check.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
from fractions import Fraction
from pathlib import Path

from flint import acb, arb, arb_mat, ctx


def q(text: str) -> Fraction:
    return Fraction(text)


A_RADIUS = q("3/5")
RANK = 44
MU = q("1")
TAIL_POINT = q("256")
LIPSCHITZ = q("8")
MAX_PANEL = q("1/2")

ALPHA_A_BOUND = q("1013/1000")
ALPHA_C_BOUND = q("1307/1000")
ALPHA_S_TIGHT = q("999961/1000000")
ALPHA_S_FALLBACK = q("499981/500000")
TAU_A_BOUND = q("1/200000")
TAU_C_BOUND = q("1/500000")

TRACE_A_UPPER_TARGET = q("41701319/10000000")
CAPTURE_A_LOWER_TARGET = q("20850637/5000000")
TRACE_C_UPPER_TARGET = q("10081295/2000000")
CAPTURE_C_LOWER_TARGET = q("50406457/10000000")

ROOT_BRACKETS = [
    (q("3.0191096348"), q("3.0191096349")),
    (q("3.8008263447"), q("3.8008263448")),
    (q("12.5552694277"), q("12.5552694278")),
    (q("15.7324326807"), q("15.7324326808")),
    (q("19.0865793993"), q("19.0865793994")),
    (q("22.8037531082"), q("22.8037531083")),
    (q("23.0575169855"), q("23.0575169856")),
    (q("26.8022968756"), q("26.8022968757")),
    (q("29.5626235078"), q("29.5626235079")),
    (q("33.6272004434"), q("33.6272004435")),
    (q("35.8937544390"), q("35.8937544391")),
    (q("44.4358191060"), q("44.4358191061")),
    (q("46.8111409577"), q("46.8111409578")),
    (q("62.0476580051"), q("62.0476580052")),
    (q("64.0340849762"), q("64.0340849763")),
    (q("73.4213566062"), q("73.4213566063")),
    (q("74.4660486479"), q("74.4660486480")),
    (q("79.8053009760"), q("79.8053009761")),
    (q("81.0059160359"), q("81.0059160360")),
    (q("90.5999156651"), q("90.5999156652")),
    (q("91.9867618696"), q("91.9867618697")),
    (q("108.0467921239"), q("108.0467921240")),
    (q("109.3276524164"), q("109.3276524165")),
    (q("125.8502082546"), q("125.8502082547")),
    (q("126.2810340061"), q("126.2810340062")),
    (q("154.1683954859"), q("154.1683954860")),
    (q("154.5122487723"), q("154.5122487724")),
]

# Rational closed cores strictly inside the first six favorable bands.
RESCUE_CORES = [
    (q("3.01910965"), q("3.80082633")),
    (q("12.55526944"), q("15.73243267")),
    (q("19.08657941"), q("22.80375310")),
    (q("23.05751700"), q("26.80229686")),
    (q("29.56262352"), q("33.62720043")),
    (q("35.89375445"), q("44.43581909")),
]

ADVERSE_CORES = [(q("0"), ROOT_BRACKETS[0][0])] + [
    (ROOT_BRACKETS[j][1], ROOT_BRACKETS[j + 1][0])
    for j in range(1, len(ROOT_BRACKETS) - 1, 2)
]

SIGNED_SEGMENTS = []
for j in range(len(ROOT_BRACKETS) - 1):
    SIGNED_SEGMENTS.append(
        (
            ROOT_BRACKETS[j][1],
            ROOT_BRACKETS[j + 1][0],
            -1 if j % 2 == 0 else 1,
        )
    )
SIGNED_SEGMENTS.insert(0, (q("0"), ROOT_BRACKETS[0][0], 1))
SIGNED_SEGMENTS.append((ROOT_BRACKETS[-1][1], TAIL_POINT, -1))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def aball(value: Fraction | int) -> arb:
    if isinstance(value, int):
        return arb(value)
    return arb(value.numerator) / value.denominator


def lower(value: arb) -> str:
    return value.lower().str(60)


def upper(value: arb) -> str:
    return value.upper().str(60)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


# The following globals are initialized independently in every process.
a: arb
mu: arb
pi: arb
log2: arb
log3: arb
log4: arb
amplitude2: arb
amplitude3: arb
moment_norm_sq: arb
feature_points: list[acb]
integration_tol: arb
integration_eval_limit: int


def setup(precision_bits: int, tolerance: str, eval_limit: int) -> None:
    global a, mu, pi, log2, log3, log4
    global amplitude2, amplitude3, moment_norm_sq, feature_points
    global integration_tol, integration_eval_limit

    ctx.prec = precision_bits
    a = aball(A_RADIUS)
    mu = aball(MU)
    pi = arb.pi()
    log2 = arb(2).log()
    log3 = arb(3).log()
    log4 = arb(4).log()
    amplitude2 = arb(2).sqrt() * log2
    amplitude3 = 2 * log3 / arb(3).sqrt()
    moment_norm_sq = a + a.sinh()
    feature_points = [
        acb(arb(5 * (2 * j + 1)) * pi / 6, 0) for j in range(RANK)
    ]
    integration_tol = arb(tolerance)
    integration_eval_limit = eval_limit


def g_infty_real(t: arb) -> arb:
    return acb(arb(1) / 4, t / 2).digamma().real - pi.log()


def deficit_at_fraction(tq: Fraction) -> arb:
    t = aball(tq)
    return (
        mu
        - g_infty_real(t)
        + amplitude2 * (log2 * t).cos()
        + amplitude3 * (log3 * t).cos()
    )


def analytic_g_infty(t: acb) -> acb:
    z0 = acb(arb(1) / 4, 0)
    ii = acb(0, 1)
    return (
        (z0 + ii * t / 2).digamma()
        + (z0 - ii * t / 2).digamma()
    ) / 2 - pi.log()


def analytic_deficit(t: acb) -> acb:
    return (
        mu
        - analytic_g_infty(t)
        + amplitude2 * (log2 * t).cos()
        + amplitude3 * (log3 * t).cos()
    )


def moment_pairing(t: acb) -> acb:
    return 2 * (
        (arb(1) / 2) * (a / 2).sinh() * (a * t).cos()
        + t * (a / 2).cosh() * (a * t).sin()
    ) / (t * t + arb(1) / 4)


def projected_kernel(s: acb, t: acb) -> acb:
    raw = a * ((a * (s - t)).sinc() + (a * (s + t)).sinc())
    return raw - moment_pairing(s) * moment_pairing(t) / moment_norm_sq


def split_panels(
    left: Fraction, right: Fraction, max_width: Fraction = MAX_PANEL
) -> list[tuple[Fraction, Fraction]]:
    width = right - left
    count = (
        width.numerator * max_width.denominator
        + width.denominator * max_width.numerator
        - 1
    ) // (width.denominator * max_width.numerator)
    count = max(1, count)
    return [
        (left + width * j / count, left + width * (j + 1) / count)
        for j in range(count)
    ]


ADVERSE_PANELS = [
    panel for left, right in ADVERSE_CORES for panel in split_panels(left, right)
]
RESCUE_PANELS = [
    panel for left, right in RESCUE_CORES for panel in split_panels(left, right)
]


def integrate_real(callback, left: Fraction, right: Fraction) -> arb:
    value = acb.integral(
        callback,
        aball(left),
        aball(right),
        rel_tol=integration_tol,
        abs_tol=integration_tol,
        eval_limit=integration_eval_limit,
    )
    require(value.imag.contains(0), "an Arb integral lost reality")
    return value.real


def trace_integral(kind: str) -> arb:
    panels = ADVERSE_PANELS if kind == "A" else RESCUE_PANELS
    sign = 1 if kind == "A" else -1
    total = arb(0)
    for left, right in panels:
        total += integrate_real(
            lambda t, analytic: sign
            * analytic_deficit(t)
            * projected_kernel(t, t),
            left,
            right,
        )
    return total / pi


def matrix_entry(kind: str, i: int, j: int) -> arb:
    panels = ADVERSE_PANELS if kind == "A" else RESCUE_PANELS
    sign = 1 if kind == "A" else -1
    si = feature_points[i]
    sj = feature_points[j]
    total = arb(0)

    def integrand(t: acb, analytic: bool) -> acb:
        del analytic
        return (
            sign
            * analytic_deficit(t)
            * projected_kernel(si, t)
            * projected_kernel(sj, t)
        )

    for left, right in panels:
        total += integrate_real(integrand, left, right)
    return total / pi


def matrix_worker(payload: tuple) -> list[tuple[str, int, int, str]]:
    worker_id, tasks, precision_bits, tolerance, eval_limit = payload
    setup(precision_bits, tolerance, eval_limit)
    results: list[tuple[str, int, int, str]] = []
    started = time.time()
    for index, (kind, i, j) in enumerate(tasks, 1):
        value = matrix_entry(kind, i, j)
        results.append((kind, i, j, value.str(90)))
        if index % 10 == 0 or index == len(tasks):
            print(
                f"rank{RANK} rescue worker {worker_id}: {index}/{len(tasks)} "
                f"entries after {time.time()-started:.1f}s",
                flush=True,
            )
    return results


def ldl_pivots(matrix: list[list[arb]]) -> list[arb] | None:
    n = len(matrix)
    lower_matrix = [[arb(0) for _ in range(n)] for _ in range(n)]
    pivots: list[arb] = []
    for i in range(n):
        diagonal = matrix[i][i]
        for k in range(i):
            diagonal -= (
                lower_matrix[i][k] * lower_matrix[i][k] * pivots[k]
            )
        if not diagonal.lower() > 0:
            return None
        pivots.append(diagonal)
        lower_matrix[i][i] = arb(1)
        for j in range(i + 1, n):
            value = matrix[j][i]
            for k in range(i):
                value -= (
                    lower_matrix[j][k]
                    * lower_matrix[i][k]
                    * pivots[k]
                )
            lower_matrix[j][i] = value / pivots[i]
    return pivots


def matrix_difference(
    level: arb,
    gram: list[list[arb]],
    positive: list[list[arb]],
    negative: list[list[arb]] | None = None,
) -> list[list[arb]]:
    n = len(gram)
    output: list[list[arb]] = []
    for i in range(n):
        row = []
        for j in range(n):
            value = level * gram[i][j] - positive[i][j]
            if negative is not None:
                value += negative[i][j]
            row.append(value)
        output.append(row)
    return output


def matrix_json(matrix: list[list[arb]]) -> list[list[dict[str, str]]]:
    return [
        [
            {
                "ball": matrix[i][j].str(60),
                "lower": lower(matrix[i][j]),
                "upper": upper(matrix[i][j]),
            }
            for j in range(len(matrix))
        ]
        for i in range(len(matrix))
    ]


def certify(
    precision_bits: int,
    tolerance: str,
    eval_limit: int,
    workers: int,
) -> dict:
    setup(precision_bits, tolerance, eval_limit)
    started = time.time()

    derivative_bound = (
        arb(9)
        * (pi * pi + 8 * arb.const_catalan())
        / (16 * arb(3).sqrt())
        + amplitude2 * log2
        + amplitude3 * log3
    )
    require(
        derivative_bound.upper() < aball(LIPSCHITZ).lower(),
        "the global two-atom derivative bound is not below 8",
    )
    require((2 * a).lower() > log3.upper(), "n=3 is not active")
    require((2 * a).upper() < log4.lower(), "n=4 is not excluded")
    tail_margin = (
        g_infty_real(aball(TAIL_POINT))
        - amplitude2
        - amplitude3
        - mu
    )
    require(tail_margin.lower() > 0, "the t>=256 adverse tail was not excluded")

    root_values = []
    for index, (left_q, right_q) in enumerate(ROOT_BRACKETS):
        left_value = deficit_at_fraction(left_q)
        right_value = deficit_at_fraction(right_q)
        if index % 2 == 0:
            require(
                left_value.lower() > 0 and right_value.upper() < 0,
                f"root bracket {index} failed +,- signs",
            )
        else:
            require(
                left_value.upper() < 0 and right_value.lower() > 0,
                f"root bracket {index} failed -,+ signs",
            )
        root_values.append((left_value, right_value))

    def certify_sign(
        left_q: Fraction, right_q: Fraction, sign: int, depth: int = 0
    ) -> tuple[int, int]:
        midpoint = (left_q + right_q) / 2
        radius = (right_q - left_q) / 2
        value = deficit_at_fraction(midpoint)
        allowance = aball(LIPSCHITZ * radius)
        if sign > 0 and value.lower() > allowance.upper():
            return 1, depth
        if sign < 0 and value.upper() < -allowance.upper():
            return 1, depth
        require(depth < 100, f"sign subdivision failed on [{left_q},{right_q}]")
        left_count, left_depth = certify_sign(left_q, midpoint, sign, depth + 1)
        right_count, right_depth = certify_sign(midpoint, right_q, sign, depth + 1)
        return left_count + right_count, max(left_depth, right_depth)

    sign_data = []
    for left_q, right_q, sign in SIGNED_SEGMENTS:
        leaves, depth = certify_sign(left_q, right_q, sign)
        sign_data.append(
            {
                "left": str(left_q),
                "right": str(right_q),
                "deficit_sign": sign,
                "leaves": leaves,
                "max_depth": depth,
            }
        )

    rescue_sign_data = []
    for left_q, right_q in RESCUE_CORES:
        left_value = deficit_at_fraction(left_q)
        right_value = deficit_at_fraction(right_q)
        require(
            left_value.upper() < 0 and right_value.upper() < 0,
            "a rescue-core endpoint is not strictly favorable",
        )
        leaves, depth = certify_sign(left_q, right_q, -1)
        rescue_sign_data.append(
            {
                "left": str(left_q),
                "right": str(right_q),
                "left_deficit_upper": upper(left_value),
                "right_deficit_upper": upper(right_value),
                "leaves": leaves,
                "max_depth": depth,
            }
        )

    print(
        f"rank{RANK} rescue sign/tail preflight passed after {time.time()-started:.1f}s",
        flush=True,
    )

    gram = [[arb(0) for _ in range(RANK)] for _ in range(RANK)]
    for i in range(RANK):
        for j in range(i, RANK):
            value = projected_kernel(feature_points[i], feature_points[j])
            require(value.imag.contains(0), "the feature Gram matrix lost reality")
            gram[i][j] = value.real
            gram[j][i] = value.real
    gram_pivots = ldl_pivots(gram)
    require(gram_pivots is not None, "the exact feature Gram matrix is not PD")

    trace_a_core = trace_integral("A")
    trace_c = trace_integral("C")
    root_error_q = sum(
        (
            LIPSCHITZ
            * 2
            * A_RADIUS
            * (right_q - left_q)
            * (right_q - left_q)
            for left_q, right_q in ROOT_BRACKETS
        ),
        Fraction(0),
    )
    root_trace_error = aball(root_error_q) / pi
    trace_a_upper = trace_a_core.upper() + root_trace_error.upper()

    require(
        trace_a_upper < aball(TRACE_A_UPPER_TARGET).lower(),
        "the adverse full-trace rational target failed",
    )
    require(
        trace_c.upper() < aball(TRACE_C_UPPER_TARGET).lower(),
        "the rescue full-trace rational target failed",
    )
    print(
        f"rank{RANK} rescue scalar traces passed after {time.time()-started:.1f}s; "
        f"launching {workers} matrix workers",
        flush=True,
    )

    tasks = [
        (kind, i, j)
        for kind in ("A", "C")
        for i in range(RANK)
        for j in range(i, RANK)
    ]
    panel_weights = {"A": len(ADVERSE_PANELS), "C": len(RESCUE_PANELS)}
    chunks: list[list[tuple[str, int, int]]] = [[] for _ in range(workers)]
    loads = [0 for _ in range(workers)]
    for task in sorted(tasks, key=lambda row: panel_weights[row[0]], reverse=True):
        index = min(range(workers), key=lambda k: loads[k])
        chunks[index].append(task)
        loads[index] += panel_weights[task[0]]

    payloads = [
        (index, chunks[index], precision_bits, tolerance, eval_limit)
        for index in range(workers)
    ]
    raw_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(matrix_worker, payload) for payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            raw_results.extend(future.result())

    expected_keys = set(tasks)
    actual_keys = [(kind, i, j) for kind, i, j, _ in raw_results]
    require(
        len(actual_keys) == len(set(actual_keys)),
        "the worker matrix results contain duplicate entries",
    )
    require(
        set(actual_keys) == expected_keys,
        "the worker matrix result keys do not match the requested entries",
    )

    matrices = {
        "A": [[arb(0) for _ in range(RANK)] for _ in range(RANK)],
        "C": [[arb(0) for _ in range(RANK)] for _ in range(RANK)],
    }
    for kind, i, j, value_text in raw_results:
        value = arb(value_text)
        matrices[kind][i][j] = value
        matrices[kind][j][i] = value
    adverse = matrices["A"]
    rescue = matrices["C"]
    expected_results = RANK * (RANK + 1)
    require(
        len(raw_results) == expected_results,
        f"the worker matrix result count is not {expected_results}",
    )

    gram_mat = arb_mat(gram)
    adverse_mat = arb_mat(adverse)
    rescue_mat = arb_mat(rescue)
    gram_inverse = gram_mat.inv()
    captured_a = (gram_inverse * adverse_mat).trace()
    captured_c = (gram_inverse * rescue_mat).trace()
    require(
        captured_a.lower() > aball(CAPTURE_A_LOWER_TARGET).upper(),
        "the adverse captured-trace rational target failed",
    )
    require(
        captured_c.lower() > aball(CAPTURE_C_LOWER_TARGET).upper(),
        "the rescue captured-trace rational target failed",
    )

    tau_a_upper = trace_a_upper - captured_a.lower()
    tau_c_upper = trace_c.upper() - captured_c.lower()
    require(tau_a_upper.lower() >= 0, "the adverse complement trace became negative")
    require(tau_c_upper.lower() >= 0, "the rescue complement trace became negative")
    require(
        tau_a_upper < aball(TAU_A_BOUND).lower(),
        "the adverse complement-trace cap failed",
    )
    require(
        tau_c_upper < aball(TAU_C_BOUND).lower(),
        "the rescue complement-trace cap failed",
    )

    effective_alpha_a = aball(ALPHA_A_BOUND) - root_trace_error.upper()
    alpha_a_pivots = ldl_pivots(
        matrix_difference(effective_alpha_a, gram, adverse)
    )
    require(alpha_a_pivots is not None, "the alpha_A generalized LDL failed")

    alpha_c_pivots = ldl_pivots(
        matrix_difference(aball(ALPHA_C_BOUND), gram, rescue)
    )
    require(alpha_c_pivots is not None, "the alpha_C generalized LDL failed")

    alpha_s_choice = None
    alpha_s_pivots = None
    for candidate in (ALPHA_S_TIGHT, ALPHA_S_FALLBACK):
        effective = aball(candidate) - root_trace_error.upper()
        pivots = ldl_pivots(
            matrix_difference(effective, gram, adverse, rescue)
        )
        if pivots is not None:
            alpha_s_choice = candidate
            alpha_s_pivots = pivots
            break
    require(alpha_s_choice is not None, "both alpha_S generalized LDL levels failed")
    require(alpha_s_pivots is not None, "alpha_S pivots were not retained")

    rho_bound = (
        (aball(TAU_A_BOUND) * aball(ALPHA_A_BOUND)).sqrt()
        + (aball(TAU_C_BOUND) * aball(ALPHA_C_BOUND)).sqrt()
    )
    alpha_s_ball = aball(alpha_s_choice)
    tau_a_ball = aball(TAU_A_BOUND)
    block_upper = (
        alpha_s_ball
        + tau_a_ball
        + (
            (alpha_s_ball - tau_a_ball) ** 2
            + 4 * rho_bound * rho_bound
        ).sqrt()
    ) / 2
    coercive_reserve = mu - block_upper
    require(
        coercive_reserve.lower() > 0,
        "the active outward-rounded full-rescue block margin failed",
    )

    print(
        f"rank{RANK} rescue certificate passed after {time.time()-started:.1f}s; "
        f"reserve lower {lower(coercive_reserve)}",
        flush=True,
    )

    return {
        "status": "rigorous python-flint/Arb full-rescue continuum certificate",
        "theorem": (
            "The exact compact-support Weil form with active n=2,3 atoms is "
            "coercive on the complex-even cosh(x/2)-moment-null closed form "
            "domain in L2(-3/5,3/5)."
        ),
        "parameters": {
            "a": fraction_text(A_RADIUS),
            "mu": fraction_text(MU),
            "rank": RANK,
            "precision_bits": precision_bits,
            "integration_tolerance": tolerance,
            "integration_eval_limit": eval_limit,
            "workers": workers,
            "maximum_exact_panel_width": fraction_text(MAX_PANEL),
            "adverse_panels": len(ADVERSE_PANELS),
            "rescue_panels": len(RESCUE_PANELS),
            "features": [f"{5*(2*j+1)}*pi/6" for j in range(RANK)],
        },
        "support_and_sign": {
            "active_prime_powers": [2, 3],
            "2a_lower": lower(2 * a),
            "log3_upper": upper(log3),
            "2a_upper": upper(2 * a),
            "log4_lower": lower(log4),
            "derivative_bound_upper": upper(derivative_bound),
            "rational_lipschitz": fraction_text(LIPSCHITZ),
            "tail_margin_lower_at_256": lower(tail_margin),
            "root_brackets": [
                {
                    "left": str(bracket[0]),
                    "right": str(bracket[1]),
                    "left_deficit": values[0].str(60),
                    "right_deficit": values[1].str(60),
                }
                for bracket, values in zip(ROOT_BRACKETS, root_values)
            ],
            "signed_segments": sign_data,
            "rescue_cores": rescue_sign_data,
        },
        "trace_enclosures": {
            "adverse_core_trace": trace_a_core.str(60),
            "adverse_root_trace_error_upper": upper(root_trace_error),
            "adverse_full_trace_upper": upper(trace_a_upper),
            "adverse_captured_trace_lower": lower(captured_a),
            "tau_A_upper": upper(tau_a_upper),
            "rescue_trace": trace_c.str(60),
            "rescue_captured_trace_lower": lower(captured_c),
            "tau_C_upper": upper(tau_c_upper),
        },
        "rational_operator_bounds": {
            "alpha_A": fraction_text(ALPHA_A_BOUND),
            "alpha_C": fraction_text(ALPHA_C_BOUND),
            "alpha_S": fraction_text(alpha_s_choice),
            "tau_A": fraction_text(TAU_A_BOUND),
            "tau_C": fraction_text(TAU_C_BOUND),
            "rho_upper": upper(rho_bound),
            "block_L_upper": upper(block_upper),
            "coercive_reserve_lower": lower(coercive_reserve),
        },
        "matrix_checks": {
            "gram_LDL_pivot_lowers": [lower(value) for value in gram_pivots],
            "alpha_A_LDL_pivot_lowers": [
                lower(value) for value in alpha_a_pivots
            ],
            "alpha_C_LDL_pivot_lowers": [
                lower(value) for value in alpha_c_pivots
            ],
            "alpha_S_LDL_pivot_lowers": [
                lower(value) for value in alpha_s_pivots
            ],
            "gram": matrix_json(gram),
            "adverse_compression_core": matrix_json(adverse),
            "rescue_compression": matrix_json(rescue),
        },
        "proof_logic": [
            "The exact multiplier is split as T=I-A+C_full with A=(1-b)_+.",
            "C is integrated only on six sign-certified rational closed cores, so 0<=C<=C_full and T>=I-(A-C).",
            "The twenty-seven root brackets and every complementary segment are sign-certified using the global derivative bound below 8.",
            "The monotone archimedean lower envelope excludes every adverse frequency beyond 256.",
            f"The rank-{RANK} feature span lies exactly in the cosh-moment-null continuum space; G is its exact Gram matrix.",
            "Continuum trace identities give tau_A and tau_C; interval generalized LDL gives alpha_A, alpha_C, and alpha_S.",
            "Positive-block factorization gives rho=sqrt(tau_A alpha_A)+sqrt(tau_C alpha_C) and the two-by-two block majorant L.",
            "The final Arb endpoint check proves L<1. No finite Ritz value of the target Weil form is used as a lower bound.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision-bits", type=int, default=256)
    parser.add_argument("--tolerance", default="1e-14")
    parser.add_argument("--eval-limit", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="print the complete ledger after writing it",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().with_name("ledger.json"),
    )
    args = parser.parse_args()
    if args.precision_bits <= 0:
        parser.error("--precision-bits must be positive")
    if args.eval_limit <= 0:
        parser.error("--eval-limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    try:
        tolerance = Fraction(args.tolerance)
    except (ValueError, ZeroDivisionError) as exc:
        parser.error(f"--tolerance must be a positive number: {exc}")
    if tolerance <= 0:
        parser.error("--tolerance must be positive")
    result = certify(
        args.precision_bits,
        args.tolerance,
        args.eval_limit,
        args.workers,
    )
    # Canonical UTF-8/LF output makes the frozen ledger byte-reproducible
    # across Windows and POSIX runners.
    rendered = json.dumps(result, indent=2) + "\n"
    args.json.write_bytes(rendered.encode("utf-8"))
    if args.print_json:
        print(rendered, end="")
    else:
        print(f"wrote canonical certificate ledger: {args.json}", flush=True)


if __name__ == "__main__":
    main()
