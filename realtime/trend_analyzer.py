from collections import Counter


def calculate_growth_rate(previous_count, current_count):
    """
    Calculate percentage growth between two time periods.
    """

    if previous_count == 0:
        if current_count > 0:
            return 100.0
        return 0.0

    growth_rate = (
        (current_count - previous_count)
        / previous_count
    ) * 100

    return round(growth_rate, 2)


def determine_trend(growth_rate):
    """
    Determine crisis trend based on growth rate.
    """

    if growth_rate >= 100:
        return "RAPIDLY GROWING"

    elif growth_rate >= 30:
        return "GROWING"

    elif growth_rate > -30:
        return "STABLE"

    else:
        return "DECREASING"


def analyze_crisis_trend(time_counts):
    """
    Analyze crisis-post counts over time.

    Example:
        [3, 6, 12, 25]
    """

    if len(time_counts) < 2:
        return {
            "growth_rate": 0.0,
            "trend": "INSUFFICIENT DATA"
        }

    previous_count = time_counts[-2]
    current_count = time_counts[-1]

    growth_rate = calculate_growth_rate(
        previous_count,
        current_count
    )

    trend = determine_trend(growth_rate)

    return {
        "previous_count": previous_count,
        "current_count": current_count,
        "growth_rate": growth_rate,
        "trend": trend
    }


def analyze_events(events):
    """
    Analyze the growth of crisis events.

    Each event should contain:
        post_count
    """

    if not events:
        return {
            "total_crisis_posts": 0,
            "event_count": 0,
            "trend": "NO CRISIS"
        }

    total_posts = sum(
        event.get("post_count", 0)
        for event in events
    )

    event_count = len(events)

    return {
        "total_crisis_posts": total_posts,
        "event_count": event_count
    }


def print_trend_result(result):

    print("\n" + "=" * 70)
    print("CRISIS TREND ANALYSIS")
    print("=" * 70)

    print(
        f"Previous Crisis Posts : "
        f"{result.get('previous_count', 0)}"
    )

    print(
        f"Current Crisis Posts  : "
        f"{result.get('current_count', 0)}"
    )

    print(
        f"Growth Rate           : "
        f"{result.get('growth_rate', 0)}%"
    )

    print(
        f"Crisis Trend          : "
        f"{result.get('trend', 'UNKNOWN')}"
    )


if __name__ == "__main__":

    # Simulated real-time crisis-post counts
    crisis_counts = [
        3,
        6,
        12,
        25
    ]

    print("\nCrisis posts over time:")

    for i, count in enumerate(crisis_counts):
        print(
            f"Time {i + 1}: "
            f"{count} crisis posts"
        )

    result = analyze_crisis_trend(
        crisis_counts
    )

    print_trend_result(result)