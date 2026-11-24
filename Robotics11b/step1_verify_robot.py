import numpy as np

import robot_core
from robot_core import RobotArm, Q_UPRIGHT_RAW, load_pose_real


def main():
    if input("Braço apoiado, zona livre para ir à pose vertical? (s/n): ").lower() != "s":
        return

    bot = RobotArm()
    try:
        print("A mover para a postura vertical medida.")
        bot.move_raw_deg(Q_UPRIGHT_RAW, speed=60)
        input(
            "Verifica os elos verticais e a ponta horizontal para +r. "
            "ENTER para testar os sinais..."
        )

        # +r é o sentido horizontal da base para a zona frontal de trabalho.
        prompts = [
            "q1: anti-horário visto de cima",
            "q2: partindo da vertical, o braço superior inclina para -r",
            "q3: partindo da vertical, o antebraço inclina para -r",
            "q4: com a ponta inicialmente para +r, a ponta sobe",
        ]

        signs = []
        for idx, positive in enumerate(prompts):
            target = Q_UPRIGHT_RAW.copy()
            target[idx] += 15.0
            input(f"ENTER para +15 graus raw em q{idx + 1}.")
            bot.move_raw_deg(target, speed=45)
            print(f"Sentido positivo no modelo: {positive}.")
            answer = input(
                "O movimento coincide (+) ou é oposto (-)? Outra tecla termina: "
            ).strip()
            bot.move_raw_deg(Q_UPRIGHT_RAW, speed=45)
            if answer not in ("+", "-"):
                return
            signs.append(1 if answer == "+" else -1)

        robot_core.SIGNS = np.asarray(signs)
        print("SIGNS =", signs)
        if input("Confirmar este conjunto e guardar? (s/n): ").strip().lower() != "s":
            return
        np.save("p11_joint_signs.npy", robot_core.SIGNS)

        if input("Trajeto livre para testar os dois alvos no ar? (s/n): ").strip().lower() == "s":
            for pt in [[165, 0, 50], [165, 40, 50]]:
                print(f"\nA mover a ponta para {pt} mm, com theta=0°.")
                q_cmd, q_meas = bot.move_xyz(pt, theta_deg=0.0, speed=60)
                print("Comandado raw:", np.round(q_cmd, 2))
                print("Medido raw:", np.round(q_meas, 2))
                print("Erro estático [°]:", np.round(q_meas - q_cmd, 2))
                input("Confirma a orientação e a direção do movimento. ENTER...")

        if input("Verificar as poses REST e OBS? (s/n): ").strip().lower() == "s":
            q_rest = load_pose_real("q_rest_real.npy", "rest_raw_degrees.npy")
            q_obs = load_pose_real("q_obs_real.npy", "observation_raw_degrees.npy")

            print("Q_REST:", np.round(q_rest, 2))
            bot.move_raw_deg(q_rest, speed=60)
            input("Pose REST correta e percurso livre? ENTER para OBS...")

            print("Q_OBS:", np.round(q_obs, 2))
            bot.move_raw_deg(q_obs, speed=60)
            input("Confirma a pose OBS. ENTER...")

        if input("Percurso livre para REST? (s/n): ").lower() == "s":
            q_rest = load_pose_real("q_rest_real.npy", "rest_raw_degrees.npy")
            bot.move_raw_deg(q_rest, speed=60)
    finally:
        bot.close(disable_torque=False)


if __name__ == "__main__":
    main()