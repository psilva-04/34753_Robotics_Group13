import glob
import os
import sys
import time

import numpy as np


try:
    import dynamixel_sdk as dxl
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    candidates = (
        glob.glob(
            os.path.join(project_dir, "MyRobot", "DynamixelSDK*", "python", "src")
        )
        + glob.glob(
            os.path.join(project_dir, "DynamixelSDK*", "python", "src")
        )
    )

    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    import dynamixel_sdk as dxl


# AX-12A hardware settings
DEVICENAME = "COM6"
BAUDRATE = 1_000_000
PROTOCOL_VERSION = 1.0
DXL_IDS = [1, 2, 3, 4]

ADDR_TORQUE_ENABLE = 24
ADDR_CW_MARGIN = 26
ADDR_CCW_MARGIN = 27
ADDR_CW_SLOPE = 28
ADDR_CCW_SLOPE = 29
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_TORQUE_LIMIT = 34
ADDR_PRESENT_POSITION = 36
ADDR_PUNCH = 48

DEG_PER_TICK_REAL = 300.0 / 1023.0
SCALE_029_TO_REAL = DEG_PER_TICK_REAL / 0.29

RAW_MIN_DEG = 5.0
RAW_MAX_DEG = 295.0

# Joint order: base, shoulder, elbow, wrist.
ARRIVAL_TOLERANCE_DEG = np.array([2.0, 4.0, 2.0, 2.0])
STABILITY_TOLERANCE_DEG = 0.45
REQUIRED_STABLE_READINGS = 3
POSITION_POLL_INTERVAL_S = 0.12

# Robot dimensions in millimetres
D1 = 50.0
A2 = 93.0
A3 = 93.0

# Stylus translation in the rotated wrist frame: [DX, -DY, 0].
DX = 20.0
DY = 84.0

# Measured upright pose: both links vertical, stylus pointing toward +r.
Q_UPRIGHT_MODEL = np.array([0.0, 90.0, 0.0, 0.0])
Q_UPRIGHT_RAW = np.array([506, 512, 516, 211]) * DEG_PER_TICK_REAL


def load_signs():
    values = np.load("p11_joint_signs.npy")
    if values.shape != (4,) or not np.isin(values, [-1, 1]).all():
        raise ValueError("Invalid joint signs. Repeat step1_verify_robot.py.")
    return values


SIGNS = load_signs() if os.path.isfile("p11_joint_signs.npy") else None


def checked_signs():
    if SIGNS is None:
        raise ValueError("Joint signs are missing. Run step1_verify_robot.py first.")
    return SIGNS


def deg_to_ticks(deg):
    return int(np.clip(round(deg / DEG_PER_TICK_REAL), 0, 1023))


def ticks_to_deg(ticks):
    return ticks * DEG_PER_TICK_REAL


def raw_to_model_rad(q_raw_deg):
    delta_raw = np.asarray(q_raw_deg, dtype=float) - Q_UPRIGHT_RAW
    q_model_deg = Q_UPRIGHT_MODEL + checked_signs() * delta_raw
    return np.deg2rad(q_model_deg)


def model_to_raw_deg(q_model_rad):
    delta_model = np.rad2deg(q_model_rad) - Q_UPRIGHT_MODEL
    q_raw = Q_UPRIGHT_RAW + checked_signs() * delta_model

    if q_raw.shape != (4,) or not np.isfinite(q_raw).all():
        raise ValueError(f"Invalid joint target: {q_raw}")

    for joint, angle in enumerate(q_raw, start=1):
        if not RAW_MIN_DEG <= angle <= RAW_MAX_DEG:
            raise ValueError(
                f"Joint {joint} exceeds the raw command limits "
                f"[{RAW_MIN_DEG}, {RAW_MAX_DEG}] deg: {angle:.2f} deg. "
                f"Full target: {q_raw.round(2)}"
            )

    return q_raw


def load_pose_real(real_file, legacy_029_file):
    if os.path.exists(real_file):
        return np.load(real_file).astype(float)

    # Older pose files used 0.29 degrees per encoder tick.
    return np.load(legacy_029_file).astype(float) * SCALE_029_TO_REAL


def dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def fk_frames(q_rad):
    q1, q2, q3, q4 = q_rad

    T3_0 = (
        dh(q1, D1, 0.0, np.pi / 2)
        @ dh(q2, 0.0, A2, 0.0)
        @ dh(q3, 0.0, A3, 0.0)
    )

    tool_offset = np.eye(4)
    tool_offset[:3, 3] = [DX, -DY, 0.0]
    T_tip_3 = dh(q4, 0.0, 0.0, 0.0) @ tool_offset

    return T3_0, T3_0 @ T_tip_3


def ik_vertical(xyz_mm, theta_deg=0.0):
    x, y, z = np.asarray(xyz_mm, dtype=float)
    theta = np.deg2rad(theta_deg)

    q1 = np.arctan2(y, x)
    radius = np.hypot(x, y)

    wrist_radius = radius - (DX * np.cos(theta) + DY * np.sin(theta))
    wrist_height = z - D1 - (DX * np.sin(theta) - DY * np.cos(theta))

    cos_q3 = (
        wrist_radius**2 + wrist_height**2 - A2**2 - A3**2
    ) / (2.0 * A2 * A3)

    if not np.isfinite(cos_q3) or abs(cos_q3) > 1.0:
        raise ValueError(
            f"Target outside the geometric workspace: {np.round(xyz_mm, 1)} mm"
        )

    # Elbow-up solution. At theta=0 the stylus points downward.
    q3 = -np.arccos(np.clip(cos_q3, -1.0, 1.0))
    q2 = (
        np.arctan2(wrist_height, wrist_radius)
        - np.arctan2(A3 * np.sin(q3), A2 + A3 * np.cos(q3))
    )
    q4 = theta - q2 - q3

    return np.array([q1, q2, q3, q4])


def pixel_to_base_3d(u, v, K, T_cam_0, T_tab_0, height_above_table_mm=0.0):
    # The input pixel comes from an image undistorted with the same K.
    ray_cam = np.linalg.solve(K, np.array([u, v, 1.0]))
    camera_origin = T_cam_0[:3, 3]
    ray_base = T_cam_0[:3, :3] @ ray_cam

    normal_up = T_tab_0[:3, 2].copy()
    if normal_up[2] < 0:
        normal_up = -normal_up

    plane_point = T_tab_0[:3, 3] + height_above_table_mm * normal_up
    denominator = np.dot(normal_up, ray_base)

    if abs(denominator) < 1e-8:
        raise ValueError("The camera ray is parallel to the target plane.")

    distance = np.dot(normal_up, plane_point - camera_origin) / denominator
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Invalid intersection or target behind the camera.")

    return camera_origin + distance * ray_base


class RobotArm:
    def __init__(self, port=DEVICENAME):
        self.port = dxl.PortHandler(port)
        self.pkt = dxl.PacketHandler(PROTOCOL_VERSION)

        if not self.port.openPort():
            raise RuntimeError(f"Could not open {port}. Check the USB connection.")

        try:
            if not self.port.setBaudRate(BAUDRATE):
                raise RuntimeError(f"Could not set the baud rate to {BAUDRATE}.")
            self.setup_servos()
        except BaseException:
            self.port.closePort()
            raise

    def _w1(self, dxl_id, addr, val):
        result, error = self.pkt.write1ByteTxRx(
            self.port, dxl_id, addr, int(val)
        )
        if result != 0 or error != 0:
            raise IOError(
                f"DXL 1-byte write failed: ID={dxl_id}, address={addr}, "
                f"result={result}, error={error}"
            )

    def _w2(self, dxl_id, addr, val):
        result, error = self.pkt.write2ByteTxRx(
            self.port, dxl_id, addr, int(val)
        )
        if result != 0 or error != 0:
            raise IOError(
                f"DXL 2-byte write failed: ID={dxl_id}, address={addr}, "
                f"result={result}, error={error}"
            )

    def _r2(self, dxl_id, addr):
        value, result, error = self.pkt.read2ByteTxRx(
            self.port, dxl_id, addr
        )
        if result != 0 or error != 0:
            raise IOError(
                f"DXL 2-byte read failed: ID={dxl_id}, address={addr}, "
                f"result={result}, error={error}"
            )
        return value

    def setup_servos(self, speed=60):
        # Use each servo's current position as its initial target.
        for servo_id in DXL_IDS:
            present = self._r2(servo_id, ADDR_PRESENT_POSITION)
            self._w2(servo_id, ADDR_GOAL_POSITION, present)
            self._w2(servo_id, ADDR_MOVING_SPEED, speed)
            self._w1(servo_id, ADDR_TORQUE_ENABLE, 1)

    def set_torque(self, enable):
        value = 1 if enable else 0
        for servo_id in DXL_IDS:
            self._w1(servo_id, ADDR_TORQUE_ENABLE, value)

    def set_speed(self, speed):
        value = int(np.clip(speed, 10, 300))
        for servo_id in DXL_IDS:
            self._w2(servo_id, ADDR_MOVING_SPEED, value)

    def read_raw_deg(self):
        return np.array([
            ticks_to_deg(self._r2(servo_id, ADDR_PRESENT_POSITION))
            for servo_id in DXL_IDS
        ], dtype=float)

    def wait_until_stopped(self, target, timeout_s=8.0):
        target = np.asarray(target, dtype=float)
        deadline = time.monotonic() + timeout_s
        previous = self.read_raw_deg()
        stable_count = 0

        while time.monotonic() < deadline:
            time.sleep(POSITION_POLL_INTERVAL_S)
            current = self.read_raw_deg()

            within_tolerance = np.all(
                np.abs(current - target) <= ARRIVAL_TOLERANCE_DEG
            )
            stationary = np.all(
                np.abs(current - previous) <= STABILITY_TOLERANCE_DEG
            )

            if within_tolerance and stationary:
                stable_count += 1
                if stable_count >= REQUIRED_STABLE_READINGS:
                    return current
            else:
                stable_count = 0

            previous = current

        raise RuntimeError(
            f"Target not reached within {timeout_s:.1f} s.\n"
            f"  Requested [deg]: {target.round(2)}\n"
            f"  Measured  [deg]: {previous.round(2)}\n"
            f"  Error     [deg]: {(previous - target).round(2)}\n"
            f"  Tolerance [deg]: {ARRIVAL_TOLERANCE_DEG}"
        )

    def move_raw_deg(self, q_raw_deg, speed=80, wait=True):
        q_raw_deg = np.asarray(q_raw_deg, dtype=float)

        if (
            q_raw_deg.shape != (4,)
            or not np.isfinite(q_raw_deg).all()
            or np.any(q_raw_deg < RAW_MIN_DEG)
            or np.any(q_raw_deg > RAW_MAX_DEG)
        ):
            raise ValueError(f"Invalid raw joint command: {q_raw_deg}")

        self.set_speed(speed)
        for servo_id, angle in zip(DXL_IDS, q_raw_deg):
            self._w2(servo_id, ADDR_GOAL_POSITION, deg_to_ticks(angle))

        if wait:
            return self.wait_until_stopped(q_raw_deg)
        return None

    def move_xyz(self, xyz_mm, theta_deg=0.0, speed=80, wait=True):
        q_model = ik_vertical(xyz_mm, theta_deg=theta_deg)
        q_raw = model_to_raw_deg(q_model)
        measured = self.move_raw_deg(q_raw, speed=speed, wait=wait)
        return q_raw, measured

    def close(self, disable_torque=False):
        # Closing the serial port leaves torque enabled by default.
        try:
            if disable_torque:
                self.set_torque(False)
        finally:
            self.port.closePort()


def vertical_segment(start, end, theta):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    if not np.allclose(start[:2], end[:2]):
        raise ValueError("A vertical segment requires constant X and Y.")

    # Check every waypoint before starting the vertical move.
    count = max(1, int(np.ceil(abs(end[2] - start[2]) / 4.0)))
    points = np.linspace(start, end, count + 1)

    for point in points:
        model_to_raw_deg(ik_vertical(point, theta))

    return points


def model_signature():
    return np.r_[
        D1, A2, A3, DX, DY,
        Q_UPRIGHT_RAW, Q_UPRIGHT_MODEL, checked_signs(),
    ]