import csv
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

import step2_original as original
from robot_core import (
    ArrivalTimeout,
    RobotArm,
    ik_vertical,
    load_pose_real,
    model_signature,
    model_to_raw_deg,
    pixel_to_base_3d,
    raw_ticks,
    tip_from_raw,
)


SESSION = Path("p11_step2_session.npz")
RESULT = Path("p11_extrinsics.npz")
START = np.array([165.0, 0.0, 50.0])
THETA = 0.0

LABELS = (
    "O / point 1 / red",
    "X / point 4 / green",
    "Y / point 11 / blue",
    "Validation / point 8 / row 2 column 3",
)

DIRECTIONS = {
    "w": np.array([1, 0, 0]),
    "s": np.array([-1, 0, 0]),
    "a": np.array([0, 1, 0]),
    "d": np.array([0, -1, 0]),
    "q": np.array([0, 0, 1]),
    "e": np.array([0, 0, -1]),
}
STEPS = {"1": 10.0, "2": 2.0, "3": 0.5}


class OperatorStop(Exception):
    pass


def require_yes(question):
    answer = input(f"{question} [y/n] > ").strip().lower()
    if answer not in ("y", "s", "yes", "sim"):
        raise OperatorStop("Stopped by operator.")


def atomic_save(path, **data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **data)
    os.replace(temporary, path)


def load_archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def photo_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_session(state, settings):
    for key, value in settings.items():
        if not np.array_equal(state[key], value):
            raise ValueError(f"Session setting changed: {key}. Start a new session.")

    photo = str(state["photo"].item())
    if photo_hash(photo) != str(state["photo_hash"].item()):
        raise ValueError("The session photograph has changed. Start a new session.")


def show_points(frame, corners):
    view = frame.copy()
    names = {0: "1 O", 3: "4 X", 10: "11 Y", 7: "8 CHECK"}
    colors = {
        0: (0, 0, 255),
        3: (0, 230, 0),
        10: (255, 0, 0),
        7: (0, 230, 255),
    }
    for i, point in enumerate(corners.reshape(-1, 2)):
        u, v = np.rint(point).astype(int)
        color = colors.get(i, (255, 160, 0))
        cv2.circle(view, (u, v), 6, color, 2)
        cv2.putText(
            view, names.get(i, str(i + 1)), (u + 7, v - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )
    if not cv2.imwrite("calib_corners_OXY.png", view):
        raise IOError("Could not save the labelled image.")
    name = "Board points: press SPACE or ENTER to accept, ESC to stop"
    cv2.imshow(name, view)
    try:
        while original.window_is_open(name):
            key = cv2.waitKey(30) & 255
            if key in (10, 13, 32):
                return
            if key == 27:
                break
        raise OperatorStop("Board image was not accepted.")
    finally:
        original.close_window(name)


def board_pose(corners, K, dist):
    obj = np.zeros((15, 3), np.float32)
    obj[:, :2] = np.mgrid[0:5, 0:3].T.reshape(-1, 2) * 30.0
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist)
    if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise ValueError("Invalid camera pose.")

    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    errors = np.linalg.norm((projected - corners).reshape(-1, 2), axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    print(f"Reprojection RMS: {rms:.2f} px. Maximum: {errors.max():.2f} px.")

    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    return transform, rms


def move_checked(bot, target, speed):
    for attempt in range(2):
        try:
            return bot.move_xyz(target, theta_deg=THETA, speed=speed)
        except ArrivalTimeout as exc:
            print(exc)
            if attempt == 1:
                raise OperatorStop(
                    "The same target failed twice. Completed touches are saved."
                ) from exc
            answer = input(
                "If the tip is clear, the arm is stationary and the direction "
                "was correct, type RETRY. Press ENTER to stop > "
            ).strip().upper()
            if answer != "RETRY":
                raise OperatorStop("Completed touches are saved.") from exc


def teleop(bot, label, start):
    pos = start.copy()
    step = 2.0
    print(f"\nTarget: {label}")
    print("w/s: +X/-X | a/d: +Y/-Y | q/e: +Z/-Z")
    print("1: 10 mm | 2: 2 mm | 3: 0.5 mm | stop: end session")
    print("Empty ENTER records contact at the selected point.")

    while True:
        cmd = input(
            f"{label} | target {pos.round(1)} | {step:g} mm > "
        ).strip().lower()

        if cmd == "stop":
            raise OperatorStop("Completed touches are saved.")
        if cmd in STEPS:
            step = STEPS[cmd]
            continue
        if cmd == "":
            raw = bot.read_raw_deg()
            measured = tip_from_raw(raw)
            print("Touch FK [mm]:", measured.round(2))
            return pos, raw, measured
        if cmd not in DIRECTIONS:
            print("Enter one command at a time.")
            continue

        target = pos + step * DIRECTIONS[cmd]
        try:
            old_goal = model_to_raw_deg(ik_vertical(pos, THETA))
            new_goal = model_to_raw_deg(ik_vertical(target, THETA))
        except ValueError as exc:
            print(exc)
            continue

        tick_change = raw_ticks(new_goal) - raw_ticks(old_goal)
        before = bot.read_raw_deg()
        _, after = move_checked(bot, target, 45)
        delta_fk = tip_from_raw(after) - tip_from_raw(before)
        print("Requested XYZ change [mm]:", (target - pos).round(2))
        print("Goal change [ticks]:", tick_change)
        print("Encoder change [ticks]:", raw_ticks(after) - raw_ticks(before))
        print("Tip change from encoder FK [mm]:", delta_fk.round(2))
        pos = target


def build_board(points):
    origin, px, py = points[:3]
    vx = px - origin
    vy = py - origin
    lx, ly = np.linalg.norm(vx), np.linalg.norm(vy)
    if not np.isfinite([lx, ly]).all() or min(lx, ly) < 1:
        raise ValueError("Invalid or coincident touches.")

    angle = np.degrees(np.arccos(np.clip(vx @ vy / (lx * ly), -1, 1)))
    print(f"OX: {lx:.2f} mm | OY: {ly:.2f} mm | angle: {angle:.2f} deg")
    if abs(lx - 90) > 4 or abs(ly - 60) > 4 or abs(angle - 90) > 2:
        print("Geometry outside tolerance. Continuing to independent point 8 validation.")

    x = vx / lx
    z = np.cross(x, vy)
    z /= np.linalg.norm(z)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    transform[:3, 3] = origin
    return transform


def run():
    with np.load("camera_calibration.npz") as data:
        K = data["matrix"]
        dist = data["distortion"]
        camera = int(data["camera_index"].item())
        size = tuple(int(n) for n in data["image_size"].ravel())

    rest = load_pose_real("q_rest_real.npy", "rest_raw_degrees.npy")
    obs = load_pose_real("q_obs_real.npy", "observation_raw_degrees.npy")
    settings = dict(
        model=model_signature(), K=K, dist=dist, Q_REST=rest,
        Q_OBS=obs, camera_index=camera, image_size=np.array(size),
    )

    print("1: original board_obs.png and original 15 corners")
    print("2: new photograph and new corners")
    print("3: resume saved touches from this version")
    mode = input("Mode > ").strip()
    if mode not in ("1", "2", "3"):
        return

    state = None
    if mode == "3":
        state = load_archive(SESSION)
        check_session(state, settings)
        photo = str(state["photo"].item())
        frame = cv2.imread(photo)
        original.check_image_size(frame, size)
        show_points(frame, state["corners"])
        require_yes("Same board, base, camera mounting and physical points?")
        print("Completed touches:", len(state["points"]))
    elif SESSION.exists():
        require_yes("Start a new set and replace the previous checkpoint?")

    if mode == "1":
        photo = "board_obs.png"
        frame = cv2.imread(photo)
        original.check_image_size(frame, size)
        corners = original.refine_grid(frame, original.RECOVERED_CLICKS)
        show_points(frame, corners)
        require_yes("Original photograph, correct corners and unchanged setup?")

    require_yes("Tip clear and paths to REST and OBS clear?")
    bot = RobotArm()
    try:
        bot.move_raw_deg(rest, speed=75)
        obs_measured = bot.move_raw_deg(obs, speed=75)

        if mode == "2":
            import uuid

            photo = f"board_obs_{uuid.uuid4().hex[:12]}.png"
            frame = original.capture_frame(camera, size)
            print("Select 3 consecutive rows, 5 corners per row, left to right.")
            print("ENTER: confirm | U: undo | R: restart | ESC: stop")
            corners = original.select_grid(frame)
            if corners is None:
                return
            show_points(frame, corners)
            if not cv2.imwrite(photo, frame):
                raise IOError("Could not save the new photograph.")

        if state is None:
            camera_board, rms = board_pose(corners, K, dist)
            state = dict(
                **settings, photo=photo, photo_hash=photo_hash(photo),
                corners=corners, T_tab_cam=camera_board, rms=rms,
                Q_OBS_MEASURED=obs_measured,
                points=np.empty((0, 3)), commands=np.empty((0, 3)),
                joints=np.empty((0, 4)),
            )
            atomic_save(SESSION, **state)

        count = len(state["points"])
        if count >= 3:
            build_board(state["points"])

        pos = START.copy()
        if count < 4:
            require_yes("Clear path from OBS to [165, 0, 50] mm?")
            move_checked(bot, pos, 60)

        for index in range(count, 4):
            print("Next point:", LABELS[index])
            if index == 3:
                build_board(state["points"])

            command, raw, measured = teleop(bot, LABELS[index], pos)
            state["points"] = np.vstack((state["points"], measured))
            state["commands"] = np.vstack((state["commands"], command))
            state["joints"] = np.vstack((state["joints"], raw))
            atomic_save(SESSION, **state)
            print("Saved:", LABELS[index])

            with open("p11_touches.csv", "a", newline="") as stream:
                csv.writer(stream).writerow(
                    [LABELS[index], *command, *raw, *measured]
                )

            require_yes("Clear to lift 30 mm from the commanded contact position?")
            pos = command + np.array([0.0, 0.0, 30.0])
            move_checked(bot, pos, 45)

        board = build_board(state["points"])
        camera_base = board @ np.linalg.inv(state["T_tab_cam"])

        pixel = state["corners"][7:8].reshape(1, 1, 2)
        u, v = cv2.undistortPoints(pixel, K, dist, P=K).reshape(2)
        predicted = pixel_to_base_3d(u, v, K, camera_base, board)
        observed = state["points"][3]
        delta = observed - predicted
        lateral = float(np.linalg.norm(board[:3, :2].T @ delta))
        normal = float(abs(board[:3, 2] @ delta))

        print("Point 8 predicted [mm]:", predicted.round(2))
        print("Point 8 touched, encoder FK [mm]:", observed.round(2))
        print(f"Point 8 in-plane error: {lateral:.2f} mm")
        print(f"Point 8 normal error: {normal:.2f} mm")

        require_yes("Accept this validation error and save the Step 3 calibration?")
        atomic_save(
            RESULT,
            T_cam_0=camera_base, T_tab_0=board,
            Q_OBS=obs, Q_REST=rest, Q_OBS_MEASURED=state["Q_OBS_MEASURED"],
            THETA_DEG=THETA, model=settings["model"], K=K, dist=dist,
            camera_index=camera, reprojection_rms_px=state["rms"],
            validation_point8_fk_mm=observed,
            validation_point8_predicted_mm=predicted,
            validation_point8_lateral_mm=lateral,
            validation_point8_normal_mm=normal,
        )
        print("Step 2 complete. Saved p11_extrinsics.npz.")
        if input("Clear path to REST? [y/n] > ").strip().lower() in ("y", "s"):
            bot.move_raw_deg(rest, speed=60)
    finally:
        bot.close(disable_torque=False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except (OperatorStop, ArrivalTimeout, ValueError, OSError) as exc:
        print(f"\nStep 2 stopped: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted. Previously completed touches remain in the checkpoint.")