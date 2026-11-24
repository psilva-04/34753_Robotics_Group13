import csv
import time
from pathlib import Path

import cv2
import numpy as np

from robot_core import (
    RobotArm,
    model_signature,
    pixel_to_base_3d,
    tip_from_raw,
    vertical_segment,
)


SMARTIE_HEIGHT_MM = 16.0 / 3.0
HOVER_MM = 25.0
FINAL_STEP_MM = 2.0
STEP_PAUSE_S = 0.1

DESCENT_SPEED = 45
LIFT_SPEED = 55

H1_MAX = 13
H2_MIN = 180
S_MIN = 119
V_MIN = 65
MIN_RADIUS_PX = 19
CROP_BOTTOM_PX = 0

LOG_PATH = Path("p11_step3_trials.csv")


def yes(question):
    return input(f"{question} (y/n): ").strip().lower() == "y"


def load_settings():
    with np.load("camera_calibration.npz") as calib:
        K = calib["matrix"]
        dist = calib["distortion"]
        camera = int(calib["camera_index"].item())
        size = tuple(int(value) for value in calib["image_size"].ravel())

    with np.load("p11_extrinsics.npz") as ext:
        if (
            not np.array_equal(ext["model"], model_signature())
            or not np.array_equal(ext["K"], K)
            or not np.array_equal(ext["dist"], dist)
            or int(ext["camera_index"].item()) != camera
        ):
            raise ValueError("Robot model or camera calibration differs from Step 2.")

        return (
            K, dist, camera, size,
            ext["T_cam_0"], ext["T_tab_0"],
            ext["Q_OBS"], ext["Q_REST"], float(ext["THETA_DEG"].item()),
        )


def capture_undistorted(camera, size, K, dist):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera)

    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open the camera.")

        width, height = size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        for _ in range(10):
            cap.read()

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture the observation image.")

        if frame.shape[:2] != (height, width):
            raise RuntimeError("Captured resolution differs from calibration.")

        return cv2.undistort(frame, K, dist)
    finally:
        cap.release()


def detect_red_smarties(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    low = cv2.inRange(hsv, (0, S_MIN, V_MIN), (H1_MAX, 255, 255))
    high = cv2.inRange(hsv, (H2_MIN, S_MIN, V_MIN), (180, 255, 255))
    mask = cv2.bitwise_or(low, high)

    if CROP_BOTTOM_PX > 0:
        mask[-CROP_BOTTOM_PX:, :] = 0

    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing)

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (5, 5), 0)

    diameter = 2 * MIN_RADIUS_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    peaks = (
        (distance == cv2.dilate(distance, kernel))
        & (distance >= MIN_RADIUS_PX)
    ).astype(np.uint8)

    _, _, _, centers = cv2.connectedComponentsWithStats(peaks)
    pixels = [(float(u), float(v)) for u, v in centers[1:]]
    return pixels, mask


def find_targets(pixels, K, camera_base, board_base):
    targets = []
    for u, v in pixels:
        top = pixel_to_base_3d(
            u, v, K, camera_base, board_base,
            height_above_table_mm=SMARTIE_HEIGHT_MM,
        )
        targets.append((u, v, top))

    # Visit targets in order of base rotation.
    targets.sort(key=lambda target: np.arctan2(target[2][1], target[2][0]))
    return targets


def confirm_targets(image, targets):
    preview = image.copy()

    for number, (u, v, _) in enumerate(targets, 1):
        center = (round(u), round(v))
        cv2.circle(preview, center, MIN_RADIUS_PX, (0, 255, 0), 2)
        cv2.circle(preview, center, 3, (255, 0, 0), -1)
        cv2.putText(
            preview, str(number), (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
        )

    cv2.imwrite("smarties_detected.png", preview)
    window = "Visit order | ENTER accept | ESC cancel"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window, preview)

    try:
        while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(30) & 255
            if key in (10, 13):
                return True
            if key == 27:
                break
        return False
    finally:
        cv2.destroyAllWindows()


def plan_moves(targets, theta):
    plans = []

    for number, (u, v, top) in enumerate(targets, 1):
        hover = top + np.array([0.0, 0.0, HOVER_MM])
        precontact = top + np.array([0.0, 0.0, FINAL_STEP_MM])

        # Check every vertical segment before visiting the first target.
        vertical_segment(hover, precontact, theta)
        vertical_segment(precontact, top, theta)

        plans.append((number, u, v, hover, precontact, top))
        print(
            f"Target {number}: pixel=({u:.1f}, {v:.1f}), "
            f"commanded top XYZ={top.round(2)} mm"
        )

    return plans


def log_trial(number, u, v, top, joints, measured_tip):
    header = [
        "timestamp", "mode", "target_id", "u_px", "v_px",
        "smartie_height_mm", "target_x_mm", "target_y_mm", "target_z_mm",
        "cmd_end_z_mm", "q1_deg", "q2_deg", "q3_deg", "q4_deg",
        "fk_x_mm", "fk_y_mm", "fk_z_mm", "observed_result",
    ]
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        "uniform_height",
        number,
        round(u, 2),
        round(v, 2),
        SMARTIE_HEIGHT_MM,
        *np.round(top, 2),
        round(float(top[2]), 2),
        *np.round(joints, 2),
        *np.round(measured_tip, 2),
        "Calculated endpoint reached. Physical contact not independently sensed.",
    ]

    with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(header)
        writer.writerow(row)


def main():
    K, dist, camera, size, camera_base, board_base, obs, rest, theta = (
        load_settings()
    )

    print(f"Common Smartie height: {SMARTIE_HEIGHT_MM:.3f} mm")
    if not yes("Clear path for REST and OBS?"):
        return

    bot = RobotArm()
    try:
        bot.move_raw_deg(rest, speed=75)
        bot.move_raw_deg(obs, speed=75)

        image = capture_undistorted(camera, size, K, dist)
        pixels, mask = detect_red_smarties(image)
        cv2.imwrite("smarties_mask.png", mask)
        print(f"Detected {len(pixels)} red centers.")

        if not pixels:
            print("No targets to visit.")
            return

        targets = find_targets(pixels, K, camera_base, board_base)
        if not confirm_targets(image, targets):
            print("Run cancelled.")
            return

        plans = plan_moves(targets, theta)
        if not yes(f"Start automatic run for all {len(plans)} targets?"):
            return

        start = time.monotonic()

        for number, u, v, hover, precontact, top in plans:
            print(f"\nTarget {number}/{len(plans)}")

            bot.move_xyz(hover, theta_deg=theta, speed=LIFT_SPEED)
            bot.move_xyz(precontact, theta_deg=theta, speed=DESCENT_SPEED)
            time.sleep(STEP_PAUSE_S)

            # The endpoint is the camera-derived top at the common height.
            bot.move_xyz(top, theta_deg=theta, speed=DESCENT_SPEED)
            time.sleep(STEP_PAUSE_S)

            joints = bot.read_raw_deg()
            measured_tip = tip_from_raw(joints)
            print(f"Commanded XYZ [mm]: {top.round(2)}")
            print(f"Encoder FK XYZ [mm]: {measured_tip.round(2)}")

            bot.move_xyz(precontact, theta_deg=theta, speed=LIFT_SPEED)
            bot.move_xyz(hover, theta_deg=theta, speed=LIFT_SPEED)

            log_trial(number, u, v, top, joints, measured_tip)

        elapsed = time.monotonic() - start
        print(f"\nCompleted {len(plans)} commanded endpoints in {elapsed:.1f} s.")

        if yes("Clear path to return to REST?"):
            bot.move_raw_deg(rest, speed=60)
    finally:
        bot.close(disable_torque=False)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"\nStep 3 stopped: {exc}")