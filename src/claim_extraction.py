import re

VERB_LIST = [
    "is", "was", "are", "were", "has", "have", "had", "won", "built",
    "discovered", "created", "developed", "designed", "composed",
    "written", "located", "died", "lived", "founded", "released", "stands",
    "orbits", "invented", "authored", "directed", "synthesized", "calculated",
    "established", "contains", "consists", "produces", "generates", "published",
    "defeated", "weighs", "measured", "cost", "costs", "serves", "began",
    "ended", "started", "formed", "joined", "rules", "ruled", "elected",
    "became", "produces", "contains", "cured", "patented", "identified"
]
VERB_PATTERN = r"(?:" + "|".join(VERB_LIST) + r")"



def clean_sentence(text):
    """Clean markdown bullet points, list numbers, header tags, and excess whitespace."""
    if not text:
        return ""
    
    # Strip markdown headers (e.g. "### Heading")
    cleaned = re.sub(r"^#+\s*", "", text.strip())
    
    # Strip leading bullets or list enumerations like "1.", "1)", "-", "*", "•"
    cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s+", "", cleaned.strip())
    
    # Remove excess whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    
    # Ignore standalone header labels ending in a colon (e.g. "Solar System Facts:")
    if cleaned.endswith(":") and len(cleaned.split()) <= 6:
        return ""
        
    return cleaned.strip()


def decompose_compound_sentence(sentence):
    """
    Decompose compound sentences into discrete atomic propositions.
    Handles:
    1. Shared subject clauses: "Subject1 Verb1 ... and Verb2 ..."
       e.g. "The Eiffel Tower was completed in 1889 and is located in London"
    2. Independent clauses: "Subject1 Verb1 ... and Subject2 Verb2 ..."
       e.g. "Python was designed by Guido van Rossum and C++ was developed by Bjarne Stroustrup"
    """
    sentence = sentence.strip().rstrip(".!?")
    if not sentence:
        return []

    # Check for semicolon-separated clauses
    if ";" in sentence:
        semi_parts = [p.strip() for p in sentence.split(";") if p.strip()]
        if len(semi_parts) > 1:
            claims = []
            for sp in semi_parts:
                claims.extend(decompose_compound_sentence(sp))
            return claims

    # Check for shared-subject compound clause: "Subject ... and <verb> ..."
    # Example: "The Eiffel Tower was completed in 1889 and is located in London"
    # Example: "Albert Einstein won the Nobel Prize and was born in Germany"
    shared_match = re.match(
        rf"^(.+?)\s+and\s+({VERB_PATTERN})\s+(.+)$",
        sentence,
        flags=re.IGNORECASE
    )

    if shared_match:
        first_part = shared_match.group(1).strip()
        verb = shared_match.group(2).strip()
        second_part = shared_match.group(3).strip()

        # Extract subject from before the first verb in first_part
        subject_match = re.match(
            rf"^(.*?)(?:\s+{VERB_PATTERN}\b)",
            first_part,
            flags=re.IGNORECASE
        )

        if subject_match:
            subject = subject_match.group(1).strip()
            if subject and len(subject.split()) <= 6:
                return [
                    first_part.rstrip(".!?") + ".",
                    f"{subject} {verb} {second_part}".strip().rstrip(".!?") + "."
                ]

    # Check for two independent clauses: "Subject1 Verb1 ... and Subject2 Verb2 ..."
    # Example: "Gold has the chemical symbol Au and water is composed of hydrogen and oxygen"
    indep_match = re.match(
        rf"^(.+?\b{VERB_PATTERN}\b.+?)(?:,\s*|\s+)and\s+([a-zA-Z0-9_\+\#\s]+?\b{VERB_PATTERN}\b.+)$",
        sentence,
        flags=re.IGNORECASE
    )

    if indep_match:
        part1 = indep_match.group(1).strip()
        part2 = indep_match.group(2).strip()
        first_word = part2.split()[0].lower() if part2.split() else ""
        # If the first word of part2 is a verb, it was already handled or is invalid
        if first_word not in VERB_LIST and len(part1.split()) >= 3 and len(part2.split()) >= 3:
            part2_cap = part2[0].upper() + part2[1:] if part2 else part2
            return [
                part1.rstrip(".!?") + ".",
                part2_cap.rstrip(".!?") + "."
            ]

    return [sentence.rstrip(".!?") + "."]


def extract_claims(text):
    """
    Extract individual factual claims from an AI-generated answer.

    Handles:
    - Multi-line / bulleted / numbered answers
    - Markdown headers and titles
    - Compound sentences ("The Eiffel Tower was completed in 1889 and is located in London")
    - Independent multi-subject clauses ("Python was created by X and C++ was created by Y")
    - Standard sentence boundaries
    """
    if not text or not text.strip():
        return []

    # Split lines first
    raw_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    raw_sentences = []

    for line in raw_lines:
        line_clean = clean_sentence(line)
        if not line_clean:
            continue
        # Split sentences within line
        parts = re.split(r"(?<=[.!?])\s+", line_clean)
        for part in parts:
            part_clean = clean_sentence(part)
            if part_clean:
                raw_sentences.append(part_clean)

    claims = []

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 3:
            continue

        decomposed = decompose_compound_sentence(sentence)
        for claim in decomposed:
            if len(claim.split()) >= 3:
                claims.append(claim)

    return claims