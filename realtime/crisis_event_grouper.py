import re
from collections import Counter


def extract_keywords(text, top_n=5):
    """
    Extract simple important keywords from a crisis post.
    """

    if not isinstance(text, str):
        return []

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep words only
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)

    # Basic stopwords
    stopwords = {
        "this", "that", "with", "from", "have",
        "there", "they", "their", "about", "were",
        "been", "will", "what", "when", "where",
        "which", "people", "after", "into",
        "some", "more", "very", "just"
    }

    words = [
        word for word in words
        if word not in stopwords
    ]

    counts = Counter(words)

    return [word for word, count in counts.most_common(top_n)]


def detect_event_name(posts):
    """
    Generate a simple event name from related posts.
    """

    all_keywords = []

    for post in posts:
        all_keywords.extend(extract_keywords(post))

    if not all_keywords:
        return "Unknown Crisis Event"

    keyword_counts = Counter(all_keywords)

    top_keywords = [
        word for word, count in keyword_counts.most_common(3)
    ]

    return " ".join(word.title() for word in top_keywords)


def group_crisis_posts(posts, similarity_threshold=0.30):
    """
    Group similar crisis posts using keyword overlap.

    Returns:
        List of crisis events.
    """

    events = []

    for post in posts:

        if not isinstance(post, str) or not post.strip():
            continue

        post_keywords = set(extract_keywords(post))

        if not post_keywords:
            continue

        assigned = False

        for event in events:

            event_keywords = set()

            for existing_post in event["posts"]:
                event_keywords.update(
                    extract_keywords(existing_post)
                )

            if not event_keywords:
                continue

            intersection = post_keywords.intersection(event_keywords)
            union = post_keywords.union(event_keywords)

            similarity = len(intersection) / len(union)

            if similarity >= similarity_threshold:

                event["posts"].append(post)
                event["keywords"] = list(
                    set(event["keywords"]).union(post_keywords)
                )

                assigned = True
                break

        if not assigned:

            events.append({
                "posts": [post],
                "keywords": list(post_keywords)
            })

    # Generate event names
    for i, event in enumerate(events, start=1):

        event["event_id"] = i

        event["event_name"] = detect_event_name(
            event["posts"]
        )

        event["post_count"] = len(
            event["posts"]
        )

    return events


def print_events(events):
    """
    Display detected crisis events.
    """

    print("\n" + "=" * 70)
    print("CRISIS EVENT GROUPING")
    print("=" * 70)

    if not events:
        print("No crisis events detected.")
        return

    for event in events:

        print(f"\nEvent ID      : {event['event_id']}")
        print(f"Event Name    : {event['event_name']}")
        print(f"Related Posts : {event['post_count']}")
        print(f"Keywords      : {', '.join(event['keywords'])}")

        print("Posts:")

        for post in event["posts"]:
            print(f"  - {post}")


if __name__ == "__main__":

    test_posts = [
        "Heavy flooding reported in Hyderabad",
        "Hyderabad streets are flooded after heavy rain",
        "People are stranded because of Hyderabad floods",
        "Rescue teams are helping flood victims",
        "Stock market prices increased today",
        "New technology conference announced"
    ]

    events = group_crisis_posts(test_posts)

    print_events(events)