import math
import pytest

from torchmil.utils.annealing_scheduler import (
    AnnealingScheduler,
    ConstantAnnealingScheduler,
    LinearAnnealingScheduler,
    CyclicalAnnealingScheduler,
)


def test_annealing_scheduler_base():
    """
    Tests the base AnnealingScheduler class.
    """
    scheduler = AnnealingScheduler()
    assert scheduler.coef == 1.0  # Default coefficient
    assert scheduler() == 1.0  # __call__ returns current coefficient

    # step() should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        scheduler.step()


def test_constant_annealing_scheduler():
    """
    Tests the ConstantAnnealingScheduler.
    """
    # Test with default coefficient
    scheduler_default = ConstantAnnealingScheduler()
    assert scheduler_default.coef == 1.0
    scheduler_default.step()
    assert scheduler_default.coef == 1.0  # Should remain constant

    # Test with custom coefficient
    scheduler_custom = ConstantAnnealingScheduler(coef=0.5)
    assert scheduler_custom.coef == 0.5
    assert scheduler_custom() == 0.5
    scheduler_custom.step()
    assert scheduler_custom.coef == 0.5  # Should remain constant


def test_linear_annealing_scheduler():
    """
    Tests the LinearAnnealingScheduler.
    """
    # Test basic linear progression
    scheduler = LinearAnnealingScheduler(coef_init=0.0, coef_end=1.0, n_steps=10)
    assert scheduler.coef == 0.0
    assert scheduler.step_count == 0

    # Step 1
    scheduler.step()
    assert scheduler.step_count == 1
    assert math.isclose(scheduler.coef, 0.1)

    # Step 5
    for _ in range(4):  # Already did 1 step, so 4 more for a total of 5
        scheduler.step()
    assert scheduler.step_count == 5
    assert math.isclose(scheduler.coef, 0.5)

    # Step 10 (end)
    for _ in range(5):  # Already did 5 steps, so 5 more for a total of 10
        scheduler.step()
    assert scheduler.step_count == 10
    assert math.isclose(scheduler.coef, 1.0)

    # Step beyond n_steps
    scheduler.step()
    assert scheduler.step_count == 11
    # The coefficient will continue to increase linearly beyond coef_end if steps exceed n_steps
    assert math.isclose(scheduler.coef, 1.1)

    # Test with different initial and end coefficients
    scheduler2 = LinearAnnealingScheduler(coef_init=10.0, coef_end=20.0, n_steps=5)
    assert scheduler2.coef == 10.0
    scheduler2.step()  # step 1
    assert math.isclose(scheduler2.coef, 12.0)  # 10 + (1 * 10/5) = 12
    scheduler2.step()  # step 2
    assert math.isclose(scheduler2.coef, 14.0)  # 10 + (2 * 10/5) = 14
    scheduler2.step()  # step 3
    assert math.isclose(scheduler2.coef, 16.0)
    scheduler2.step()  # step 4
    assert math.isclose(scheduler2.coef, 18.0)
    scheduler2.step()  # step 5
    assert math.isclose(scheduler2.coef, 20.0)


def test_cyclical_annealing_scheduler():
    """
    Tests the CyclicalAnnealingScheduler.
    """
    # Test a simple cycle without warmup
    # cycle_len = 4, min_coef = 0, max_coef = 1, cycle_prop = 0.5
    # cut_step = 4 * 0.5 = 2
    # Steps:
    # 0: coef = 0 (initial)
    # 1: mod_step = 0 < cut_step (0.5 * 0 / 2 = 0) -> coef = 0
    # 2: mod_step = 1 < cut_step (0.5 * 1 / 2 = 0.5) -> coef = 0.5
    # 3: mod_step = 2 >= cut_step -> coef = 1.0
    # 4: mod_step = 3 >= cut_step -> coef = 1.0
    # 5: mod_step = 0 < cut_step (0.5 * 0 / 2 = 0) -> coef = 0 (start of new cycle)

    scheduler = CyclicalAnnealingScheduler(
        cycle_len=4, min_coef=0.0, max_coef=1.0, cycle_prop=0.5, warmup_steps=0
    )
    assert scheduler.coef == 0.0
    assert scheduler.step_count == 0

    # Step 1 (mod_step = 0)
    scheduler.step()
    assert scheduler.step_count == 1
    assert math.isclose(scheduler.coef, 0.0)

    # Step 2 (mod_step = 1)
    scheduler.step()
    assert scheduler.step_count == 2
    assert math.isclose(scheduler.coef, 0.5)

    # Step 3 (mod_step = 2)
    scheduler.step()
    assert scheduler.step_count == 3
    assert math.isclose(scheduler.coef, 1.0)

    # Step 4 (mod_step = 3)
    scheduler.step()
    assert scheduler.step_count == 4
    assert math.isclose(scheduler.coef, 1.0)

    # Step 5 (mod_step = 0 - start of new cycle)
    scheduler.step()
    assert scheduler.step_count == 5
    assert math.isclose(scheduler.coef, 0.0)

    # Test with warmup steps
    # warmup_steps = 2
    # cycle_len = 4, min_coef = 0, max_coef = 1, cycle_prop = 0.5
    # cut_step = 2
    scheduler_warmup = CyclicalAnnealingScheduler(
        cycle_len=4, min_coef=0.0, max_coef=1.0, cycle_prop=0.5, warmup_steps=2
    )
    assert scheduler_warmup.coef == 0.0
    assert scheduler_warmup.step_count == 0

    # Step 1 (within warmup)
    scheduler_warmup.step()
    assert scheduler_warmup.warmup_step_count == 1
    assert scheduler_warmup.step_count == 0
    assert math.isclose(
        scheduler_warmup.coef, 0.0
    )  # Coef remains min_coef during warmup

    # Step 2 (within warmup)
    scheduler_warmup.step()
    assert scheduler_warmup.warmup_step_count == 2
    assert scheduler_warmup.step_count == 0
    assert math.isclose(
        scheduler_warmup.coef, 0.0
    )  # Coef remains min_coef during warmup

    # Step 3 (first step after warmup, mod_step = 0)
    scheduler_warmup.step()
    assert scheduler_warmup.warmup_step_count == 2
    assert scheduler_warmup.step_count == 1
    assert math.isclose(scheduler_warmup.coef, 0.0)

    # Step 4 (mod_step = 1)
    scheduler_warmup.step()
    assert scheduler_warmup.warmup_step_count == 2
    assert scheduler_warmup.step_count == 2
    assert math.isclose(scheduler_warmup.coef, 0.5)

    # Step 5 (mod_step = 2)
    scheduler_warmup.step()
    assert scheduler_warmup.warmup_step_count == 2
    assert scheduler_warmup.step_count == 3
    assert math.isclose(scheduler_warmup.coef, 1.0)

    # Test with different cycle_prop
    # cycle_len = 10, min_coef = 0, max_coef = 10, cycle_prop = 0.8
    # cut_step = 10 * 0.8 = 8
    # Coef increases from 0 to 10 over 8 steps, then stays at 10 for 2 steps.
    scheduler_prop = CyclicalAnnealingScheduler(
        cycle_len=10, min_coef=0.0, max_coef=10.0, cycle_prop=0.8
    )
    assert scheduler_prop.coef == 0.0

    # Steps 1-8 (increasing phase)
    for i in range(1, 9):
        scheduler_prop.step()
        expected_coef = (i - 1) * 10.0 / 8.0  # (mod_step) * coef_dif / cut_step
        assert math.isclose(scheduler_prop.coef, expected_coef)

    # Steps 9-10 (constant phase)
    scheduler_prop.step()  # Step 9 (mod_step = 8)
    assert math.isclose(scheduler_prop.coef, 10.0)
    scheduler_prop.step()  # Step 10 (mod_step = 9)
    assert math.isclose(scheduler_prop.coef, 10.0)

    # Step 11 (start of new cycle, mod_step = 0)
    scheduler_prop.step()
    assert math.isclose(scheduler_prop.coef, 0.0)
