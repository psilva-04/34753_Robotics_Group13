import csv
import os

import cv2
import numpy as np

from robot_core import (
    RobotArm,
    fk_frames,
    load_pose_real,
    model_signature,
    pixel_to_base_3d,
    raw_to_model_rad,
    vertical_segment,
)


# Região de 15 cantos visíveis em Q_OBS.
GRID_COLS = 5
GRID_ROWS = 3
SQUARE_MM = 30.0

# Numeração dos cliques:
#   1   2   3   4   5
#   6   7   8   9  10
#  11  12  13  14  15
IDX_O = 0     # Clique 1
IDX_X = 3     # Clique 4: 90 mm de O
IDX_Y = 10    # Clique 11: 60 mm de O

THETA_DEG = 0.0
START_XYZ = np.array([165.0, 0.0, 50.0])
LIFT_AFTER_TOUCH_MM = 30.0

# Cliques da fotografia board_obs.png que produziu RMS = 0.79 px.
# Para o clique 13 usa-se a última posição registada no terminal.
RECOVERED_CLICKS = [
    (116, 213), (311, 207), (507, 202), (699, 196), (891, 189),
    (108, 405), (307, 398), (509, 393), (703, 386), (902, 381),
    (98, 609),  (305, 599), (510, 594), (711, 587), (917, 582),
]


def yes(question):
    return input(f"{question} (s/n): ").strip().lower() == "s"


def close_window(name):
    # Em Windows, carregar no X pode destruir a janela antes desta chamada.
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def window_is_open(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def check_image_size(frame, image_size):
    if frame is None:
        raise RuntimeError("A imagem não foi carregada.")
    actual = (frame.shape[1], frame.shape[0])
    if actual != image_size:
        raise RuntimeError(
            f"Imagem com resolução {actual}. A calibração usa {image_size}."
        )


def capture_frame(camera_index, image_size):
    width, height = image_size
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmara {camera_index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(10):
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Falha ao capturar a fotografia.")
    finally:
        cap.release()

    check_image_size(frame, image_size)
    return frame


def refine_grid(frame, clicks):
    if len(clicks) != GRID_COLS * GRID_ROWS:
        raise ValueError("São necessários exatamente 15 cantos.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = np.array(clicks, dtype=np.float32).reshape(-1, 1, 2)
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )

    grid = corners.reshape(GRID_ROWS, GRID_COLS, 2)
    if np.any(np.diff(grid[:, :, 0], axis=1) <= 0):
        raise ValueError("Cada fila deve avançar da esquerda para a direita.")
    if np.any(np.diff(grid[:, :, 1], axis=0) <= 0):
        raise ValueError("As filas devem avançar de cima para baixo.")
    return corners


def select_grid(frame):
    name = "15 cantos - ENTER confirma, U desfaz, R recomeca, ESC sai"
    clicks = []
    height, width = frame.shape[:2]

    def redraw():
        view = frame.copy()
        for i, (x, y) in enumerate(clicks, 1):
            cv2.circle(view, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                view, str(i), (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
            )
        cv2.imshow(name, view)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= 15:
            return
        if not (12 <= x < width - 12 and 12 <= y < height - 12):
            print("Ponto demasiado perto da margem.")
            return
        clicks.append((x, y))
        print(f"Clique {len(clicks):2d}: ({x}, {y})")
        redraw()

    print("Seleciona 3 filas de 5 cantos. Em cada fila, da esquerda para a direita.")
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, on_click)
    redraw()
    try:
        while window_is_open(name):
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if key in (ord("u"), ord("U")) and clicks:
                clicks.pop()
                redraw()
            elif key in (ord("r"), ord("R")):
                clicks.clear()
                redraw()
            elif key in (10, 13):
                if len(clicks) == 15:
                    return refine_grid(frame, clicks)
                print(f"Tens {len(clicks)} de 15 cantos.")
        return None
    finally:
        close_window(name)


def estimate_board_pose(frame, corners, K, dist):
    objp = np.zeros((GRID_ROWS * GRID_COLS, 3), dtype=np.float32)
    objp[:, :2] = (
        np.mgrid[0:GRID_COLS, 0:GRID_ROWS].T.reshape(-1, 2) * SQUARE_MM
    )
    objp -= objp[IDX_O].copy()

    solved, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not solved or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise RuntimeError("solvePnP não encontrou uma pose válida.")

    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    errors = np.linalg.norm((projected - corners).reshape(-1, 2), axis=1)
    rms_px = float(np.sqrt(np.mean(errors**2)))
    worst = int(np.argmax(errors))
    print(f"\nRMS de reprojeção dos 15 cantos: {rms_px:.2f} px")
    print(f"Maior erro: clique {worst + 1}, {errors[worst]:.2f} px")

    rotation, _ = cv2.Rodrigues(rvec)
    T_tab_cam = np.eye(4)
    T_tab_cam[:3, :3] = rotation
    T_tab_cam[:3, 3] = tvec.ravel()

    view = frame.copy()
    for i, point in enumerate(corners[:, 0], 1):
        u, v = np.rint(point).astype(int)
        cv2.circle(view, (u, v), 4, (0, 180, 255), -1)
        cv2.putText(
            view, str(i), (u + 6, v + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1,
        )

    cv2.drawFrameAxes(view, K, dist, rvec, tvec, SQUARE_MM, 2)
    for label, idx, color in (
        ("O", IDX_O, (0, 0, 255)),
        ("X", IDX_X, (0, 255, 0)),
        ("Y", IDX_Y, (255, 0, 0)),
    ):
        u, v = np.rint(corners[idx, 0]).astype(int)
        cv2.circle(view, (u, v), 9, color, 2)
        cv2.putText(
            view, label, (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
        )

    cv2.imwrite("calib_corners_OXY.png", view)
    name = "O=1 vermelho, X=4 verde, Y=11 azul - ESPACO para continuar"
    cv2.imshow(name, view)
    print("Localiza os pontos 1, 4 e 11 na mesa. Prime ESPAÇO na janela.")
    try:
        while window_is_open(name):
            key = cv2.waitKey(30) & 0xFF
            if key in (10, 13, 32, 27):
                break
    finally:
        close_window(name)

    return T_tab_cam, rms_px


def teleop_touch(bot, label, start_xyz):
    pos = np.array(start_xyz, dtype=float)
    step = 5.0
    print(f"\nGuiar a ponta até {label}.")
    print("w/s: +X/-X | a/d: +Y/-Y | q/e: subir/descer")
    print("1/2/3: passo de 10 / 2 / 0.5 mm")
    print("Escreve o comando e prime ENTER. ENTER vazio grava o toque.")

    bot.move_xyz(pos, theta_deg=THETA_DEG, speed=60)

    directions = {
        "w": np.array([1, 0, 0]),
        "s": np.array([-1, 0, 0]),
        "a": np.array([0, 1, 0]),
        "d": np.array([0, -1, 0]),
        "q": np.array([0, 0, 1]),
        "e": np.array([0, 0, -1]),
    }
    steps = {"1": 10.0, "2": 2.0, "3": 0.5}

    while True:
        cmd = input(
            f"{label} pos={pos.round(1)} [passo={step} mm] > "
        ).strip().lower()

        if cmd == "":
            joints = bot.read_raw_deg()
            T_tip_0 = fk_frames(raw_to_model_rad(joints))[1]
            measured = T_tip_0[:3, 3].copy()
            print("Ponta estimada pela FK dos encoders:", measured.round(2))
            with open("p11_touches.csv", "a", newline="") as out:
                csv.writer(out).writerow([label, *pos, *joints, *measured])
            return measured

        if cmd in steps:
            step = steps[cmd]
            continue
        if cmd not in directions:
            print("Comando inválido.")
            continue

        target = pos + step * directions[cmd]
        try:
            bot.move_xyz(target, theta_deg=THETA_DEG, speed=45)
        except ValueError as exc:
            print(exc)
            continue
        pos = target


def build_board_frame(pO, pX, pY):
    vx = pX - pO
    vy = pY - pO
    lx = np.linalg.norm(vx)
    ly = np.linalg.norm(vy)
    if not np.isfinite([lx, ly]).all() or min(lx, ly) < 1.0:
        raise ValueError("Toques inválidos ou coincidentes.")

    cosine = np.clip(np.dot(vx, vy) / (lx * ly), -1.0, 1.0)
    angle = float(np.rad2deg(np.arccos(cosine)))
    expected_x = 3 * SQUARE_MM
    expected_y = 2 * SQUARE_MM

    print("\nVerificação dos três toques:")
    print(f"O→X: {lx:.2f} mm | esperado: {expected_x:.2f} mm")
    print(f"O→Y: {ly:.2f} mm | esperado: {expected_y:.2f} mm")
    print(f"Ângulo X–O–Y: {angle:.2f}° | esperado: 90.00°")

    if (
        abs(lx - expected_x) > 4.0
        or abs(ly - expected_y) > 4.0
        or abs(angle - 90.0) > 2.0
    ):
        raise ValueError(
            "Geometria fora dos critérios de 4 mm / 2°. "
            "Verifica os toques físicos nos pontos 1, 4 e 11."
        )

    x_hat = vx / lx
    z_hat = np.cross(x_hat, vy)
    z_hat /= np.linalg.norm(z_hat)
    y_hat = np.cross(z_hat, x_hat)

    T_tab_0 = np.eye(4)
    T_tab_0[:3, :3] = np.column_stack([x_hat, y_hat, z_hat])
    T_tab_0[:3, 3] = pO
    return T_tab_0


def select_validation_marks(image):
    name = "Validacao - clica nas marcas e prime ENTER"
    marks = []
    view = image.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            marks.append((x, y))
            cv2.circle(view, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(name, view)

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, on_click)
    cv2.imshow(name, view)
    try:
        while window_is_open(name):
            key = cv2.waitKey(30) & 0xFF
            if key in (10, 13):
                return marks
            if key == 27:
                return []
        return []
    finally:
        close_window(name)


def validate_marks(bot, q_rest, q_obs, camera_index, image_size, K, dist,
                   T_cam_0, T_tab_0):
    while yes("Fazer teste em marcas independentes?"):
        h_text = input("Altura da marca relativa ao plano do padrão [mm]: ")
        h_test = float((h_text or "0").replace(",", "."))

        bot.move_raw_deg(q_rest, speed=75)
        bot.move_raw_deg(q_obs, speed=75)
        frame = capture_frame(camera_index, image_size)
        undist = cv2.undistort(frame, K, dist)
        marks = select_validation_marks(undist)

        for i, (u, v) in enumerate(marks, 1):
            p = pixel_to_base_3d(
                u, v, K, T_cam_0, T_tab_0,
                height_above_table_mm=h_test,
            )
            print(f"Marca {i}: alvo estimado {p.round(2)} mm")
            path = vertical_segment(
                p + [0, 0, 25], p + [0, 0, 1], THETA_DEG
            )
            for waypoint in path:
                bot.move_xyz(waypoint, theta_deg=THETA_DEG, speed=45)

            while True:
                try:
                    err = float(input("Erro lateral observado [mm]: ").replace(",", "."))
                    if np.isfinite(err) and err >= 0:
                        break
                except ValueError:
                    pass
                print("Introduz uma distância maior ou igual a zero.")

            new_file = not os.path.exists("p11_validation.csv")
            with open("p11_validation.csv", "a", newline="") as out:
                writer = csv.writer(out)
                if new_file:
                    writer.writerow([
                        "mark", "height_mm", "u", "v", "x", "y", "z",
                        "lateral_error_mm",
                    ])
                writer.writerow([i, h_test, u, v, *p, err])

            for waypoint in path[-2::-1]:
                bot.move_xyz(waypoint, theta_deg=THETA_DEG, speed=45)


def main():
    signature = model_signature()
    with np.load("camera_calibration.npz") as calib:
        K = calib["matrix"]
        dist = calib["distortion"]
        camera_index = int(calib["camera_index"].item())
        image_size = tuple(int(n) for n in calib["image_size"].ravel())

    q_rest = load_pose_real("q_rest_real.npy", "rest_raw_degrees.npy")
    q_obs = load_pose_real("q_obs_real.npy", "observation_raw_degrees.npy")

    print("\n1 — Recuperar board_obs.png e os 15 cliques do ensaio de 0.79 px")
    print("2 — Tirar fotografia nova e selecionar 15 cantos")
    mode = input("Escolhe 1 ou 2: ").strip()
    if mode not in ("1", "2"):
        print("Cancelado.")
        return

    # Na recuperação, ler a foto guardada antes de qualquer movimento.
    if mode == "1":
        frame = cv2.imread("board_obs.png")
        if frame is None:
            raise FileNotFoundError(
                "board_obs.png não está nesta pasta. "
                "Coloca aqui a fotografia original ou usa o modo 2."
            )
        check_image_size(frame, image_size)
        corners = refine_grid(frame, RECOVERED_CLICKS)
        T_tab_cam, rms_px = estimate_board_pose(frame, corners, K, dist)

        if not yes(
            "É a fotografia original, os números coincidem com os cantos "
            "e o tabuleiro, a base e a montagem da câmara estão no mesmo sítio?"
        ):
            print("Terminado. Para uma montagem diferente, usa a opção 2.")
            return

    if not yes("Percurso livre para REST e OBS?"):
        return

    bot = RobotArm()
    try:
        bot.move_raw_deg(q_rest, speed=75)
        q_obs_meas = bot.move_raw_deg(q_obs, speed=75)

        if mode == "2":
            frame = capture_frame(camera_index, image_size)
            # Nome distinto para preservar a fotografia associada ao modo 1.
            if not cv2.imwrite("board_obs_new.png", frame):
                raise RuntimeError("Não foi possível guardar board_obs_new.png.")
            print("Fotografia nova guardada em board_obs_new.png.")

            while True:
                corners = select_grid(frame)
                if corners is None:
                    return
                T_tab_cam, rms_px = estimate_board_pose(frame, corners, K, dist)
                if yes("Os 15 pontos estão bem marcados e identificaste O, X e Y?"):
                    break
                if not yes("Voltar a selecionar na mesma fotografia?"):
                    return

        if not yes("Tabuleiro imóvel e percurso livre de OBS até [165,0,50]?"):
            return

        pO = teleop_touch(bot, "O = ponto 1 (vermelho)", START_XYZ)
        above_O = pO + [0, 0, LIFT_AFTER_TOUCH_MM]
        bot.move_xyz(above_O, theta_deg=THETA_DEG, speed=70)

        pX = teleop_touch(bot, "X = ponto 4 (verde)", above_O)
        above_X = pX + [0, 0, LIFT_AFTER_TOUCH_MM]
        bot.move_xyz(above_X, theta_deg=THETA_DEG, speed=70)

        pY = teleop_touch(bot, "Y = ponto 11 (azul)", above_X)
        bot.move_xyz(
            pY + [0, 0, LIFT_AFTER_TOUCH_MM],
            theta_deg=THETA_DEG,
            speed=70,
        )

        T_tab_0 = build_board_frame(pO, pX, pY)
        T_cam_0 = T_tab_0 @ np.linalg.inv(T_tab_cam)

        np.savez(
            "p11_extrinsics.npz",
            T_cam_0=T_cam_0,
            T_tab_0=T_tab_0,
            Q_OBS=q_obs,
            Q_REST=q_rest,
            Q_OBS_MEASURED=q_obs_meas,
            THETA_DEG=THETA_DEG,
            model=signature,
            K=K,
            dist=dist,
            camera_index=camera_index,
            reprojection_rms_px=rms_px,
        )
        print("\nExtrínsecas guardadas em p11_extrinsics.npz.")

        validate_marks(
            bot, q_rest, q_obs, camera_index, image_size,
            K, dist, T_cam_0, T_tab_0,
        )

        if yes("Trajeto livre para REST?"):
            bot.move_raw_deg(q_rest, speed=60)
    finally:
        try:
            cv2.destroyAllWindows()
        finally:
            bot.close(disable_torque=False)


if __name__ == "__main__":
    main()