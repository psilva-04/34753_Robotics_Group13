import numpy as np
from Problem3 import problem3_circle, dh_transform

np.set_printoptions(precision=9, suppress=True)


def linear_jacobian(q):
    q1, q2, q3, q4 = q

    T00 = np.eye(4)
    T01 = dh_transform(q1, 50, 0, np.pi / 2)
    T02 = T01 @ dh_transform(q2, 0, 93, 0)
    T03 = T02 @ dh_transform(q3, 0, 93, 0)
    T04 = T03 @ dh_transform(q4, 0, 50, 0)

    o4_0 = T04[:3, 3]
    Jv = np.zeros((3, 4))

    for i, T in enumerate([T00, T01, T02, T03]):
        z_axis = T[:3, 2]
        origin = T[:3, 3]
        Jv[:, i] = np.cross(z_axis, o4_0 - origin)

    return Jv


def inverse_velocity(q, v):
    # Same horizontal-orientation condition used in Problem 5.
    # theta_dot = q2_dot + q3_dot + q4_dot = 0.
    system = np.vstack([
        linear_jacobian(q),
        [0.0, 1.0, 1.0, 1.0]
    ])

    return np.linalg.solve(system, np.append(v, 0.0))


def quintic_coefficients(q_start, q_end, dq_start, dq_end, T):
    # Internal coefficient order: c0, c1, c2, c3, c4, c5.
    matrix = np.array([
        [1, 0,    0,       0,        0,        0],
        [0, 1,    0,       0,        0,        0],
        [0, 0,    2,       0,        0,        0],
        [1, T, T**2,    T**3,     T**4,     T**5],
        [0, 1,  2*T,  3*T**2,   4*T**3,   5*T**4],
        [0, 0,    2,     6*T,  12*T**2,  20*T**3]
    ], dtype=float)

    boundary_values = np.vstack([
        q_start,
        dq_start,
        np.zeros(4),
        q_end,
        dq_end,
        np.zeros(4)
    ])

    # One row per joint, one column per polynomial coefficient.
    return np.linalg.solve(matrix, boundary_values).T


def evaluate_polynomial(coefficients, t):
    q = coefficients @ np.array([
        1, t, t**2, t**3, t**4, t**5
    ])

    dq = coefficients @ np.array([
        0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4
    ])

    ddq = coefficients @ np.array([
        0, 0, 2, 6*t, 12*t**2, 20*t**3
    ])

    return q, dq, ddq


def problem6():
    _, _, configurations = problem3_circle()

    knot_indices = np.array([0, 9, 18, 27, 36])
    q_knots = configurations[knot_indices]

    v_knots = np.array([
        [0.0,   0.0,   0.0],
        [0.0, -27.0,   0.0],
        [0.0,   0.0, -27.0],
        [0.0,  27.0,   0.0],
        [0.0,   0.0,   0.0]
    ])

    dq_knots = np.array([
        inverse_velocity(q, v)
        for q, v in zip(q_knots, v_knots)
    ])

    coefficients = {}

    for i, segment in enumerate(["A", "B", "C", "D"]):
        coefficients[segment] = quintic_coefficients(
            q_start=q_knots[i],
            q_end=q_knots[i + 1],
            dq_start=dq_knots[i],
            dq_end=dq_knots[i + 1],
            T=2.0
        )

    return knot_indices, q_knots, dq_knots, v_knots, coefficients


if __name__ == "__main__":
    (
        knot_indices,
        q_knots,
        dq_knots,
        v_knots,
        coefficients
    ) = problem6()

    print("\nPROBLEM 6")
    print("Each segment uses local time: 0 <= t <= 2 s.")
    print("Joint accelerations are zero at every knot.")
    print("The stylus remains horizontal.")

    print("\nJOINT POSITIONS AT THE KNOTS [rad]")
    for j, q in zip(knot_indices, q_knots):
        print(f"q({j:2d}) = {q}")

    print("\nJOINT VELOCITIES AT THE KNOTS [rad/s]")
    for j, dq in zip(knot_indices, dq_knots):
        print(f"dq({j:2d}) = {dq}")

    print("\nPOLYNOMIAL COEFFICIENTS")
    print("q_i(t) = c5*t^5 + c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0")
    print("Angles in rad; local time in s.")
    print("Coefficient ck has units rad/s^k.")

    for segment, C in coefficients.items():
        print(f"\nSEGMENT {segment}")
        print(
            f"{'Joint':>6}"
            f"{'c5':>16}{'c4':>16}{'c3':>16}"
            f"{'c2':>16}{'c1':>16}{'c0':>16}"
        )

        for joint in range(4):
            print(
                f"{joint + 1:6d}"
                + "".join(
                    f"{value:16.9f}"
                    for value in C[joint, ::-1]
                )
            )

        print("\nAssignment notation:")
        for joint in range(4):
            print(
                ", ".join(
                    f"{segment}{joint + 1}{power}"
                    f" = {C[joint, power]:.9f}"
                    for power in range(5, -1, -1)
                )
            )

    max_q_error = 0.0
    max_dq_error = 0.0
    max_ddq_error = 0.0
    max_v_error = 0.0

    for i, segment in enumerate(["A", "B", "C", "D"]):
        C = coefficients[segment]

        for t, knot in [(0.0, i), (2.0, i + 1)]:
            q, dq, ddq = evaluate_polynomial(C, t)
            v = linear_jacobian(q) @ dq

            max_q_error = max(
                max_q_error,
                np.max(np.abs(q - q_knots[knot]))
            )
            max_dq_error = max(
                max_dq_error,
                np.max(np.abs(dq - dq_knots[knot]))
            )
            max_ddq_error = max(
                max_ddq_error,
                np.max(np.abs(ddq))
            )
            max_v_error = max(
                max_v_error,
                np.max(np.abs(v - v_knots[knot]))
            )

            np.testing.assert_allclose(
                q, q_knots[knot], rtol=0, atol=1e-10
            )
            np.testing.assert_allclose(
                dq, dq_knots[knot], rtol=0, atol=1e-10
            )
            np.testing.assert_allclose(
                ddq, np.zeros(4), rtol=0, atol=1e-10
            )
            np.testing.assert_allclose(
                v, v_knots[knot], rtol=0, atol=1e-8
            )

        # Verify theta(t) = q2(t) + q3(t) + q4(t) = 0.
        np.testing.assert_allclose(
            C[1] + C[2] + C[3],
            np.zeros(6),
            rtol=0,
            atol=1e-10
        )

    print("\nBOUNDARY-CONDITION VERIFICATION")
    print(f"Maximum joint position error:     {max_q_error:.3e} rad")
    print(f"Maximum joint velocity error:     {max_dq_error:.3e} rad/s")
    print(f"Maximum joint acceleration error: {max_ddq_error:.3e} rad/s^2")
    print(f"Maximum tip velocity error:       {max_v_error:.3e} mm/s")
    print("All boundary conditions satisfied.")
    print("theta(t) = 0 throughout all four segments.")