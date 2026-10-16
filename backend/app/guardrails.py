"""Conservative citation checks; these do not prove semantic correctness."""

import re

ABSTENTION = "I could not find a supported answer in the selected documents."


def check_answer(answer: str, source_count: int) -> tuple[str, set[int]]:
    """Reject missing/out-of-range citations instead of manufacturing evidence."""
    answer = answer.strip()
    if not source_count or answer == ABSTENTION:
        return ABSTENTION, set()
    numbers = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    if not numbers or any(number < 1 or number > source_count for number in numbers):
        return ABSTENTION, set()
    return answer, numbers
