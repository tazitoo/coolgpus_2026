import pytest
from coolgpus.core import clamp, determine_segment, min_speed, max_speed, target_speed


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert clamp(5, 0, 10) == 5

    def test_below_min(self):
        assert clamp(-1, 0, 10) == 0

    def test_above_max(self):
        assert clamp(15, 0, 10) == 10

    def test_at_lower_boundary(self):
        assert clamp(0, 0, 10) == 0

    def test_at_upper_boundary(self):
        assert clamp(10, 0, 10) == 10

    def test_equal_bounds(self):
        assert clamp(5, 5, 5) == 5

    def test_negative_range(self):
        assert clamp(-5, -10, -1) == -5
        assert clamp(-20, -10, -1) == -10
        assert clamp(0, -10, -1) == -1


# ---------------------------------------------------------------------------
# determine_segment
# ---------------------------------------------------------------------------

class TestDetermineSegment:
    """Two-point curve: temps=[20, 80], speeds=[30, 100]."""

    def test_below_min_temp(self):
        # t < first temp_a => breaks immediately on first segment
        ta, tb, sa, sb = determine_segment(10, [20, 80], [30, 100])
        assert (ta, tb, sa, sb) == (20, 80, 30, 100)

    def test_at_min_temp(self):
        ta, tb, sa, sb = determine_segment(20, [20, 80], [30, 100])
        assert (ta, tb, sa, sb) == (20, 80, 30, 100)

    def test_in_range(self):
        ta, tb, sa, sb = determine_segment(50, [20, 80], [30, 100])
        assert (ta, tb, sa, sb) == (20, 80, 30, 100)

    def test_at_max_temp_boundary(self):
        # t == temp_b for last segment; loop ends without break, returns last segment
        ta, tb, sa, sb = determine_segment(80, [20, 80], [30, 100])
        assert (ta, tb, sa, sb) == (20, 80, 30, 100)

    def test_above_max_temp(self):
        ta, tb, sa, sb = determine_segment(100, [20, 80], [30, 100])
        assert (ta, tb, sa, sb) == (20, 80, 30, 100)


class TestDetermineSegmentMulti:
    """Three-point curve: temps=[20, 50, 80], speeds=[30, 60, 100]."""

    def test_first_segment(self):
        ta, tb, sa, sb = determine_segment(30, [20, 50, 80], [30, 60, 100])
        assert (ta, tb) == (20, 50)
        assert (sa, sb) == (30, 60)

    def test_second_segment(self):
        ta, tb, sa, sb = determine_segment(60, [20, 50, 80], [30, 60, 100])
        assert (ta, tb) == (50, 80)
        assert (sa, sb) == (60, 100)

    def test_at_segment_boundary(self):
        # t == 50 falls into second segment (temp_a <= t < temp_b)
        ta, tb, sa, sb = determine_segment(50, [20, 50, 80], [30, 60, 100])
        assert (ta, tb) == (50, 80)
        assert (sa, sb) == (60, 100)

    def test_below_all(self):
        ta, tb, sa, sb = determine_segment(5, [20, 50, 80], [30, 60, 100])
        assert (ta, tb) == (20, 50)
        assert (sa, sb) == (30, 60)

    def test_above_all(self):
        ta, tb, sa, sb = determine_segment(90, [20, 50, 80], [30, 60, 100])
        assert (ta, tb) == (50, 80)
        assert (sa, sb) == (60, 100)


# ---------------------------------------------------------------------------
# min_speed
# ---------------------------------------------------------------------------

class TestMinSpeed:
    """Two-point curve: temps=[20, 80], speeds=[30, 100]."""

    temps = [20, 80]
    speeds = [30, 100]

    def test_at_low_temp(self):
        assert min_speed(20, self.temps, self.speeds) == 30

    def test_at_high_temp(self):
        # t=80 is at/above last boundary; clamped to speed_b
        assert min_speed(80, self.temps, self.speeds) == 100

    def test_midpoint(self):
        # load = (50-20)/(80-20) = 0.5 => 30 + 70*0.5 = 65
        assert min_speed(50, self.temps, self.speeds) == 65

    def test_below_min_temp_clamps(self):
        # t=0 => load negative => clamped to speed_a
        assert min_speed(0, self.temps, self.speeds) == 30

    def test_above_max_temp_clamps(self):
        # t=100 => load > 1 => clamped to speed_b
        assert min_speed(100, self.temps, self.speeds) == 100

    def test_quarter(self):
        # load = (35-20)/60 = 0.25 => 30 + 70*0.25 = 47.5 => int(47) = 47
        assert min_speed(35, self.temps, self.speeds) == 47

    def test_three_quarter(self):
        # load = (65-20)/60 = 0.75 => 30 + 70*0.75 = 82.5 => int(82) = 82
        assert min_speed(65, self.temps, self.speeds) == 82


class TestMinSpeedMultiSegment:
    """Three-point curve: temps=[20, 50, 80], speeds=[30, 60, 100]."""

    temps = [20, 50, 80]
    speeds = [30, 60, 100]

    def test_first_segment_mid(self):
        # segment [20,50], load = (35-20)/30 = 0.5 => 30 + 30*0.5 = 45
        assert min_speed(35, self.temps, self.speeds) == 45

    def test_second_segment_mid(self):
        # segment [50,80], load = (65-50)/30 = 0.5 => 60 + 40*0.5 = 80
        assert min_speed(65, self.temps, self.speeds) == 80

    def test_at_segment_boundary(self):
        # t=50 => segment [50,80], load=0 => speed=60
        assert min_speed(50, self.temps, self.speeds) == 60


# ---------------------------------------------------------------------------
# max_speed
# ---------------------------------------------------------------------------

class TestMaxSpeed:
    """max_speed(t) == min_speed(t + hyst)."""

    temps = [20, 80]
    speeds = [30, 100]
    hyst = 5

    def test_equals_min_speed_plus_hyst(self):
        for t in [0, 20, 35, 50, 65, 80, 100]:
            assert max_speed(t, self.temps, self.speeds, self.hyst) == \
                   min_speed(t + self.hyst, self.temps, self.speeds)

    def test_max_ge_min(self):
        # max_speed >= min_speed for any temp (since hyst >= 0)
        for t in range(0, 110, 5):
            lo = min_speed(t, self.temps, self.speeds)
            hi = max_speed(t, self.temps, self.speeds, self.hyst)
            assert hi >= lo

    def test_zero_hysteresis(self):
        for t in [20, 50, 80]:
            assert max_speed(t, self.temps, self.speeds, 0) == \
                   min_speed(t, self.temps, self.speeds)


# ---------------------------------------------------------------------------
# target_speed
# ---------------------------------------------------------------------------

class TestTargetSpeed:
    temps = [20, 80]
    speeds = [30, 100]
    hyst = 5

    def test_ramp_up_from_zero(self):
        # current_speed=0 is below lo, so result should be lo
        speed, lo, hi = target_speed(0, 50, self.temps, self.speeds, self.hyst)
        assert speed == lo
        assert speed == min_speed(50, self.temps, self.speeds)

    def test_stay_in_hysteresis_band(self):
        # If current_speed is between lo and hi, it should stay unchanged
        lo_val = min_speed(50, self.temps, self.speeds)
        hi_val = max_speed(50, self.temps, self.speeds, self.hyst)
        mid = (lo_val + hi_val) // 2
        speed, lo, hi = target_speed(mid, 50, self.temps, self.speeds, self.hyst)
        assert speed == mid
        assert lo <= speed <= hi

    def test_clamp_above_hi(self):
        # current_speed above hi should be clamped down to hi
        speed, lo, hi = target_speed(200, 50, self.temps, self.speeds, self.hyst)
        assert speed == hi

    def test_clamp_below_lo(self):
        # current_speed below lo should be clamped up to lo
        speed, lo, hi = target_speed(0, 50, self.temps, self.speeds, self.hyst)
        assert speed == lo

    def test_returns_lo_hi(self):
        speed, lo, hi = target_speed(50, 50, self.temps, self.speeds, self.hyst)
        assert lo == min_speed(50, self.temps, self.speeds)
        assert hi == max_speed(50, self.temps, self.speeds, self.hyst)

    def test_at_low_temp(self):
        speed, lo, hi = target_speed(0, 10, self.temps, self.speeds, self.hyst)
        assert speed == 30  # clamped to min speed

    def test_at_high_temp(self):
        speed, lo, hi = target_speed(0, 90, self.temps, self.speeds, self.hyst)
        assert speed == 100  # lo == hi == 100 at extreme temp
