def calculate_risk_score(
    severity,
    confidence,
    growth_rate,
    crisis_post_count
):
    """
    Calculate overall crisis risk score from 0 to 100.
    """

    # -----------------------------
    # 1. Severity score
    # -----------------------------
    severity_scores = {
        "LOW": 20,
        "MEDIUM": 50,
        "HIGH": 80,
        "CRITICAL": 100
    }

    severity = str(severity).upper()

    severity_score = severity_scores.get(
        severity,
        20
    )

    # -----------------------------
    # 2. Confidence score
    # -----------------------------
    confidence = float(confidence)

    # Handle confidence supplied as 0-1
    if confidence <= 1:
        confidence *= 100

    confidence = min(
        max(confidence, 0),
        100
    )

    # -----------------------------
    # 3. Growth score
    # -----------------------------
    if growth_rate >= 100:
        growth_score = 100

    elif growth_rate >= 50:
        growth_score = 80

    elif growth_rate >= 30:
        growth_score = 60

    elif growth_rate >= 0:
        growth_score = 40

    else:
        growth_score = 20

    # -----------------------------
    # 4. Post-volume score
    # -----------------------------
    if crisis_post_count >= 50:
        volume_score = 100

    elif crisis_post_count >= 25:
        volume_score = 80

    elif crisis_post_count >= 10:
        volume_score = 60

    elif crisis_post_count >= 5:
        volume_score = 40

    else:
        volume_score = 20

    # -----------------------------
    # 5. Final weighted score
    # -----------------------------
    risk_score = (
        severity_score * 0.35
        + confidence * 0.20
        + growth_score * 0.25
        + volume_score * 0.20
    )

    risk_score = round(
        min(max(risk_score, 0), 100),
        2
    )

    # -----------------------------
    # 6. Risk level
    # -----------------------------
    if risk_score >= 80:
        risk_level = "CRITICAL"

    elif risk_score >= 60:
        risk_level = "HIGH"

    elif risk_score >= 40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # -----------------------------
    # 7. Early warning
    # -----------------------------
    if risk_score >= 80:
        warning = "IMMEDIATE EARLY WARNING"

    elif risk_score >= 60:
        warning = "EARLY WARNING"

    elif risk_score >= 40:
        warning = "MONITOR CLOSELY"

    else:
        warning = "NORMAL MONITORING"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "early_warning": warning
    }


def print_risk_result(result):

    print("\n" + "=" * 70)
    print("CRISIS RISK ASSESSMENT")
    print("=" * 70)

    print(
        f"Risk Score    : "
        f"{result['risk_score']}/100"
    )

    print(
        f"Risk Level    : "
        f"{result['risk_level']}"
    )

    print(
        f"Early Warning : "
        f"{result['early_warning']}"
    )

    print("=" * 70)


if __name__ == "__main__":

    # Example crisis situation
    result = calculate_risk_score(
        severity="HIGH",
        confidence=94,
        growth_rate=108.33,
        crisis_post_count=25
    )

    print_risk_result(result)