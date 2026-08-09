"""Post-process the certified a=3/5 rank-44 ledger with a matrix Schur LMI.

No quadrature here. It reads the outward-rounded Gram and compression
matrices from the supplied 256-bit Arb ledger and checks the matrix-valued
residual theorem of Section 4 of paper/certified_local_weil_positivity.tex.

Every decision is made from Arb endpoints. The adverse root-neighborhood
operator is charged by its certified trace, hence by the same number times
the identity in Loewner order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from flint import arb, ctx


DEFAULT_LEDGER = Path(__file__).with_name("ledger.json")
DEFAULT_OUTPUT = Path(__file__).with_name("schur-lmi.json")
EXPECTED_PROOF_PAYLOAD_SHA256 = (
    "950A5E6D261E548A40AB776D15F67741660A9786614E6B000B0CD493378A2285"
)

# Exact rational constants used in the interval LMI below.
ETA = Fraction(1, 16)
LAMBDA = Fraction(99997, 100000)  # 0.99997; reserve = 3/100000.


def exact(value: Fraction | int) -> arb:
    if isinstance(value, int):
        return arb(value)
    return arb(value.numerator) / value.denominator


def load_matrix(payload: dict, name: str) -> list[list[arb]]:
    return [
        [arb(entry["ball"]) for entry in row]
        for row in payload["matrix_checks"][name]
    ]


def proof_payload_sha256(payload: dict) -> str:
    """Hash the complete theorem-bearing JSON payload."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


def ldl_pivots(matrix: list[list[arb]]) -> list[arb] | None:
    """Unpivoted interval LDL^T; success proves strict positivity."""

    n = len(matrix)
    lower = [[arb(0) for _ in range(n)] for _ in range(n)]
    pivots: list[arb] = []
    for i in range(n):
        diagonal = matrix[i][i]
        for k in range(i):
            diagonal -= lower[i][k] * lower[i][k] * pivots[k]
        if not diagonal.lower() > 0:
            return None
        pivots.append(diagonal)
        lower[i][i] = arb(1)
        for j in range(i + 1, n):
            value = matrix[j][i]
            for k in range(i):
                value -= lower[j][k] * lower[i][k] * pivots[k]
            lower[j][i] = value / pivots[i]
    return pivots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precision", type=int, default=256)
    args = parser.parse_args()

    ctx.prec = args.precision
    raw = args.ledger.read_bytes()
    ledger_sha256 = hashlib.sha256(raw).hexdigest().upper()
    payload = json.loads(raw)
    payload_sha256 = proof_payload_sha256(payload)
    if payload_sha256 != EXPECTED_PROOF_PAYLOAD_SHA256:
        raise RuntimeError(
            "the input does not have the certified rank-44 a=3/5 proof payload"
        )
    parameters = payload.get("parameters", {})
    if not (
        parameters.get("a") == "3/5"
        and parameters.get("mu") == "1/1"
        and parameters.get("rank") == 44
        and parameters.get("precision_bits") == 256
    ):
        raise RuntimeError("the certified ledger parameters are not the expected ones")

    gram = load_matrix(payload, "gram")
    adverse_core = load_matrix(payload, "adverse_compression_core")
    rescue = load_matrix(payload, "rescue_compression")
    n = len(gram)
    if n != 44 or any(len(row) != n for row in gram):
        raise RuntimeError("the input ledger is not the expected rank-44 ledger")

    trace_data = payload["trace_enclosures"]
    epsilon = arb(trace_data["adverse_root_trace_error_upper"]).upper()
    tau_a = arb(trace_data["tau_A_upper"]).upper()
    tau_c = arb(trace_data["tau_C_upper"]).upper()
    eta = exact(ETA)
    level = exact(LAMBDA)
    if not level > tau_a:
        raise RuntimeError("the Schur denominator is not positive")

    denominator = level - tau_a
    coefficient_a = (1 + eta) * tau_a / denominator
    coefficient_c = (1 + 1 / eta) * tau_c / denominator

    # In coordinates of the nonorthogonal feature basis:
    #   P A P <= B_A,core + epsilon G,
    #   P(A-C)P <= B_A,core + epsilon G - B_C,
    # and the matrix Young residual is
    #   (1+eta) tau_A PAP + (1+eta^{-1}) tau_C PCP.
    lmi: list[list[arb]] = []
    for i in range(n):
        row: list[arb] = []
        for j in range(n):
            adverse_upper = adverse_core[i][j] + epsilon * gram[i][j]
            row.append(
                level * gram[i][j]
                - (adverse_upper - rescue[i][j])
                - coefficient_a * adverse_upper
                - coefficient_c * rescue[i][j]
            )
        lmi.append(row)

    pivots = ldl_pivots(lmi)
    if pivots is None:
        raise RuntimeError("the matrix Schur LMI interval LDL test failed")

    reserve = exact(1) - level
    result = {
        "status": f"rigorous {args.precision}-bit Arb post-processing certificate",
        # Store only the basename so that serialized output is path-independent.
        "input_ledger": args.ledger.name,
        "input_ledger_sha256": ledger_sha256,
        "input_proof_payload_sha256": payload_sha256,
        "postprocess_precision_bits": args.precision,
        "rank": n,
        "eta": f"{ETA.numerator}/{ETA.denominator}",
        "lambda": f"{LAMBDA.numerator}/{LAMBDA.denominator}",
        "tau_A_upper_endpoint": tau_a.str(60),
        "tau_C_upper_endpoint": tau_c.str(60),
        "adverse_root_operator_error_upper_endpoint": epsilon.str(60),
        "minimum_LDL_pivot_lower": min(p.lower() for p in pivots).str(60),
        "coercive_reserve": f"{(1 - LAMBDA).numerator}/{(1 - LAMBDA).denominator}",
        "coercive_reserve_decimal": reserve.str(60),
        "theorem": (
            "At a=3/5 on the complex-even cosh(x/2)-moment-null closed "
            "form domain, the exact n=2,3 compact-support Weil form is "
            "bounded below by (3/100000) times the L2 identity."
        ),
        "proof_logic": [
            "The certified ledger encloses G, the adverse core compression, the retained rescue compression, both complement traces, and the adverse root trace error.",
            "For B>=0, (QBP)^*(QBP) <= ||QBQ|| PBP <= Tr(QBQ) PBP.",
            "Matrix Young bounds the signed cross residual Q(A-C)P without replacing PAP and PCP by scalar norms.",
            "The displayed interval LDL positivity is the Schur sufficient condition A-C <= lambda I.",
            "The exact full Weil form dominates I-(A-C), giving reserve 1-lambda=3/100000.",
        ],
    }
    # Canonical UTF-8/LF output makes the result byte-reproducible across
    # Windows and POSIX runners.
    rendered = json.dumps(result, indent=2) + "\n"
    args.output.write_bytes(rendered.encode("utf-8"))
    print(rendered, end="")


if __name__ == "__main__":
    main()
