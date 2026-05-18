"""Constraint specification for likelihood ratio testing on ClaSSE birth kernels."""

from dataclasses import dataclass, field


@dataclass
class KernelConstraint:
    """Encodes a linear inequality on birth kernel entries:

    lhs_coeff * B[lhs_i, lhs_j] >= rhs_coeff * B[rhs_i, rhs_j] + offset

    Special cases:
      Ratio test:   lhs_coeff=1, rhs_i>=0, rhs_j>=0, offset=0
                    => B[i,j] >= k * B[p,q]
      Lower bound:  lhs_coeff=1, rhs_i=-1, rhs_j=-1, rhs_coeff=0, offset=c
                    => B[i,j] >= c

    Both models in the LRT (unconstrained and null) always respect the potency
    support mask. This constraint adds an *extra* restriction on top.
    """

    lhs_i: int
    lhs_j: int
    rhs_i: int       # -1 for scalar RHS (no second kernel entry)
    rhs_j: int       # -1 for scalar RHS
    lhs_coeff: float = 1.0
    rhs_coeff: float = 1.0
    offset: float = 0.0
    label: str = ""

    @classmethod
    def ratio(cls, i: int, j: int, p: int, q: int, k: float, label: str = "") -> "KernelConstraint":
        """B[i,j] >= k * B[p,q]."""
        if label == "":
            label = f"B[{i},{j}] >= {k} * B[{p},{q}]"
        return cls(lhs_i=i, lhs_j=j, rhs_i=p, rhs_j=q, rhs_coeff=k, label=label)

    @classmethod
    def lower_bound(cls, i: int, j: int, c: float, label: str = "") -> "KernelConstraint":
        """B[i,j] >= c."""
        if label == "":
            label = f"B[{i},{j}] >= {c}"
        return cls(lhs_i=i, lhs_j=j, rhs_i=-1, rhs_j=-1, rhs_coeff=0.0, offset=c, label=label)

    def validate(self, support_mask) -> None:
        """Check that the constraint references structurally non-zero entries.

        Args:
            support_mask: Boolean tensor (K, K) from DaughterKernelBuilder.
        """
        if not support_mask[self.lhs_i, self.lhs_j].item():
            raise ValueError(
                f"LHS B[{self.lhs_i},{self.lhs_j}] is potency-forbidden (structurally zero). "
                "Cannot impose a positivity constraint on it."
            )
        if self.rhs_i >= 0 and not support_mask[self.rhs_i, self.rhs_j].item():
            raise ValueError(
                f"RHS B[{self.rhs_i},{self.rhs_j}] is potency-forbidden (structurally zero). "
                "The constraint is trivially satisfied — this is likely a configuration error."
            )

    def is_satisfied(self, B, tol: float = 1e-5) -> bool:
        """Check whether the constraint holds on a given birth kernel.

        Args:
            B: Tensor of shape (K, K).
            tol: Numerical tolerance (allow slight violation up to tol).

        Returns:
            True if satisfied.
        """
        lhs = float(self.lhs_coeff) * float(B[self.lhs_i, self.lhs_j].item())
        rhs = (float(self.rhs_coeff) * float(B[self.rhs_i, self.rhs_j].item())
               if self.rhs_i >= 0 else 0.0)
        return lhs >= rhs + self.offset - tol

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "lhs_i": self.lhs_i,
            "lhs_j": self.lhs_j,
            "rhs_i": self.rhs_i,
            "rhs_j": self.rhs_j,
            "lhs_coeff": self.lhs_coeff,
            "rhs_coeff": self.rhs_coeff,
            "offset": self.offset,
        }


def constraint_from_names(
    state2idx: dict,
    from_state: str,
    to_state: str,
    from_state2: str = None,
    to_state2: str = None,
    k: float = 1.0,
    min_val: float = None,
    label: str = "",
) -> KernelConstraint:
    """Construct a KernelConstraint from human-readable state names.

    Args:
        state2idx: Mapping from state label string to integer index.
        from_state: Row state for the LHS entry.
        to_state: Column state for the LHS entry.
        from_state2: Row state for the RHS entry (ratio test only).
        to_state2: Column state for the RHS entry (ratio test only).
        k: Ratio multiplier (ratio test only).
        min_val: Minimum value for lower-bound constraint. If set, overrides
            ratio test parameters and returns a lower-bound constraint.
        label: Human-readable label; auto-generated if empty.
    """
    i = state2idx[from_state]
    j = state2idx[to_state]
    if min_val is not None:
        if label == "":
            label = f"B[{from_state},{to_state}] >= {min_val}"
        return KernelConstraint.lower_bound(i, j, min_val, label=label)
    if from_state2 is None or to_state2 is None:
        raise ValueError("Must provide from_state2 and to_state2 for a ratio constraint, "
                         "or min_val for a lower-bound constraint.")
    p = state2idx[from_state2]
    q = state2idx[to_state2]
    if label == "":
        label = f"B[{from_state},{to_state}] >= {k} * B[{from_state2},{to_state2}]"
    return KernelConstraint.ratio(i, j, p, q, k, label=label)
