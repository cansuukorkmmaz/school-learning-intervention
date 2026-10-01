"""Human approval rules. The system never approves its own recommendations."""
import pandas as pd


def apply_teacher_decision(recommendations: pd.DataFrame, recommendation_id: str, decision: str) -> pd.DataFrame:
    allowed = {"APPROVED", "DISMISSED", "PENDING"}
    if decision not in allowed:
        raise ValueError(f"Decision must be one of {sorted(allowed)}")
    result = recommendations.copy()
    mask = result.recommendation_id.eq(recommendation_id)
    if not mask.any():
        raise KeyError(f"Unknown recommendation_id: {recommendation_id}")
    result.loc[mask, "status"] = decision
    return result
