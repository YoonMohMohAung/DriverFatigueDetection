def fatigue_level(yawn_count, perclos):
    score = 0

    if yawn_count >= 3:
        score += 2
    if perclos > 0.25:
        score += 2

    if score <= 1:
        return "LOW"
    elif score == 2:
        return "MEDIUM"
    else:
        return "HIGH"
