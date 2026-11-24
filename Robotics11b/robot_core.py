import time

import numpy as np

from robot_core_original import *
from robot_core_original import RobotArm as OriginalRobotArm


class ArrivalTimeout(RuntimeError):
    pass


def raw_ticks(raw_deg):
    return np.array([deg_to_ticks(value) for value in raw_deg], dtype=int)


def tip_from_raw(raw_deg):
    return fk_frames(raw_to_model_rad(raw_deg))[1][:3, 3].copy()


class RobotArm(OriginalRobotArm):
    def wait_until_stopped(self, target, timeout_s=8.0):
        target = np.asarray(target, dtype=float)
        start = time.monotonic()
        previous = self.read_raw_deg()
        history = [(0.0, previous.copy())]
        stable_count = 0

        while time.monotonic() - start < timeout_s:
            time.sleep(POSITION_POLL_INTERVAL_S)
            current = self.read_raw_deg()
            elapsed = time.monotonic() - start
            history.append((elapsed, current.copy()))

            close = np.all(
                np.abs(current - target) <= ARRIVAL_TOLERANCE_DEG
            )
            steady = np.all(
                np.abs(current - previous) <= STABILITY_TOLERANCE_DEG
            )

            if close and steady:
                stable_count += 1
                if stable_count >= REQUIRED_STABLE_READINGS:
                    return current
            else:
                stable_count = 0

            previous = current

        last_time = history[-1][0]
        last_second = np.array([
            raw for elapsed, raw in history if elapsed >= last_time - 1.0
        ])

        trace = []
        next_sample = 0.0
        for index, (elapsed, raw) in enumerate(history):
            if elapsed >= next_sample or index == len(history) - 1:
                trace.append(f"{elapsed:.1f}s={raw[1]:.2f}")
                next_sample = elapsed + 0.5

        raise ArrivalTimeout(
            f"Target not reached within {timeout_s:.1f} s.\n"
            f"  Requested [deg]: {target.round(2)}\n"
            f"  Measured  [deg]: {previous.round(2)}\n"
            f"  Error     [deg]: {(previous - target).round(2)}\n"
            f"  Tolerance [deg]: {ARRIVAL_TOLERANCE_DEG}\n"
            f"  Q2 samples [deg]: {' | '.join(trace)}\n"
            f"  Q2 final second [deg]: "
            f"{last_second[:, 1].min():.2f} to "
            f"{last_second[:, 1].max():.2f}"
        )