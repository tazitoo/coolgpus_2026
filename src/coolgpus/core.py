def determine_segment(t, temps, speeds):
    """Determine which segment of the piecewise linear fan curve t belongs to."""
    segments = zip(temps[:-1], temps[1:], speeds[:-1], speeds[1:])
    for temp_a, temp_b, speed_a, speed_b in segments:
        if t < temp_a:
            break
        if temp_a <= t < temp_b:
            break
    return temp_a, temp_b, speed_a, speed_b


def min_speed(t, temps, speeds):
    temp_a, temp_b, speed_a, speed_b = determine_segment(t, temps, speeds)
    load = (t - temp_a) / float(temp_b - temp_a)
    return int(min(max(speed_a + (speed_b - speed_a) * load, speed_a), speed_b))


def max_speed(t, temps, speeds, hyst):
    return min_speed(t + hyst, temps, speeds)


def target_speed(current_speed, temp, temps, speeds, hyst):
    lo = min_speed(temp, temps, speeds)
    hi = max_speed(temp, temps, speeds, hyst)
    return min(max(current_speed, lo), hi), lo, hi


def clamp(v, lo, hi):
    return min(hi, max(lo, v))
