
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

L, TOOL, H, R = 93.0, 50.0, 50.0, 32.0


def inverse_kinematics(phi):
    """Joint positions along the circle."""
    y = R*np.cos(phi)
    z = 120 + R*np.sin(phi)
    rho = np.hypot(150, y)
    a, b = rho - TOOL, z - H

    q3 = -np.arccos(np.clip((a*a+b*b-2*L*L)/(2*L*L), -1, 1))
    q2 = np.arctan2(b, a) - np.arctan2(np.sin(q3), 1+np.cos(q3))
    q1 = np.arctan2(y, 150)

    return np.column_stack((q1, q2, q3, -q2-q3))


def forward_kinematics(q):
    """Stylus position in frame 0."""
    q1, q2, q3, q4 = q.T
    theta = q2 + q3 + q4

    rho = L*(np.cos(q2)+np.cos(q2+q3)) + TOOL*np.cos(theta)
    z = H + L*(np.sin(q2)+np.sin(q2+q3)) + TOOL*np.sin(theta)

    return np.column_stack((rho*np.cos(q1), rho*np.sin(q1), z))


def knot_data(n):
    """Calculate knot positions and velocities."""
    phi = np.linspace(0, 2*np.pi, n)
    times = np.linspace(0, 8, n)
    q = inverse_kinematics(phi)
    dq = np.zeros_like(q)

    for k in range(1, n-1):
        y = R*np.cos(phi[k])
        rho = np.hypot(150, y)
        dy = -27*np.sin(phi[k])
        dz = 27*np.cos(phi[k])
        drho = y*dy/rho

        dq[k, 0] = 150*dy/rho**2

        q2, q3 = q[k, 1:3]

        J = L*np.array([
            [-np.sin(q2)-np.sin(q2+q3), -np.sin(q2+q3)],
            [np.cos(q2)+np.cos(q2+q3), np.cos(q2+q3)]
        ])

        dq[k, 1:3] = np.linalg.solve(J, [drho, dz])
        dq[k, 3] = -dq[k, 1]-dq[k, 2]

    return times, q, dq


def quintic(s, j, times, q, dq):
    """Degree-5 interpolation."""
    T = times[j+1]-times[j]
    q0, qf = q[j], q[j+1]
    v0, vf = T*dq[j], T*dq[j+1]

    d = qf-q0-v0
    w = vf-v0
    s = np.asarray(s)[:, None]

    return (q0 + v0*s
            + (10*d-4*w)*s**3
            + (-15*d+7*w)*s**4
            + (6*d-3*w)*s**5)


def septic(s, j, times, q, dq, c):
    """Degree-7 interpolation."""
    s = np.asarray(s)
    b = 64*s**3*(1-s)**3
    B = np.column_stack((b, b*(2*s-1)))

    d = B @ c.reshape(2, 3)

    return quintic(s, j, times, q, dq) + np.column_stack((
        d, -d[:, 1]-d[:, 2]
    ))


def contour_error(p):
    """Distance to the circle."""
    return np.hypot(
        p[:, 0]-150,
        np.hypot(p[:, 1], p[:, 2]-120)-R
    )


def optimize(times, q, dq):
    """Optimize degree-7 coefficients."""
    s_fit = np.linspace(0, 1, 41)[1:-1]
    coefficients = []

    for j in range(len(times)-1):

        def residual(c):
            p = forward_kinematics(
                septic(s_fit, j, times, q, dq, c)
            )
            return np.column_stack((
                p[:, 0]-150,
                np.hypot(p[:, 1], p[:, 2]-120)-R
            )).ravel()

        result = least_squares(residual, np.zeros(6))
        coefficients.append(result.x)

    return coefficients


def sample(times, q, dq, coefficients=None):
    """Sample a complete trajectory."""
    t = np.linspace(0, 8, 4001)
    joints = np.zeros((len(t), 4))

    segments = np.clip(
        np.searchsorted(times, t, side='right')-1,
        0, len(times)-2
    )

    for j in range(len(times)-1):
        mask = segments == j
        s = (t[mask]-times[j])/(times[j+1]-times[j])

        if coefficients is None:
            joints[mask] = quintic(s, j, times, q, dq)
        else:
            joints[mask] = septic(s, j, times, q, dq, coefficients[j])

    return t, joints, forward_kinematics(joints)


# Benchmark

results = {}

for n in (5, 13, 37):
    times, q, dq = knot_data(n)
    c7 = optimize(times, q, dq)

    for method, coefficients in (("Quintic", None), ("Degree 7 + LS", c7)):

        t, joints, p = sample(times, q, dq, coefficients)
        e = contour_error(p)

        accel = np.gradient(
            np.gradient(joints, t, axis=0),
            t, axis=0
        )

        results[(method, n)] = (
            p,
            np.mean(e**2),
            np.sqrt(np.mean(e**2)),
            np.max(e),
            np.max(np.abs(accel))
        )


# Print results

print("\nPROBLEM 7 — CONTOURING ERROR")
print(f"{'Method':18} {'Knots':>5} {'MSE [mm²]':>12} "
      f"{'RMSE [mm]':>12} {'Max [mm]':>11} "
      f"{'Max |qdd| [rad/s²]':>20}")
print("-"*88)

for method in ("Quintic", "Degree 7 + LS"):
    for n in (5, 13, 37):
        p, mse, rmse, maximum, accel = results[(method, n)]

        print(f"{method:18} {n:5d} {mse:12.3e} "
              f"{rmse:12.6f} {maximum:11.6f} {accel:20.4f}")


# Plot 1: Quintic with 5, 13 and 37 knots
# Plot 2: Degree 7 with 5, 13 and 37 knots

phi = np.linspace(0, 2*np.pi, 1000)

for method in ("Quintic", "Degree 7 + LS"):

    plt.figure(figsize=(7, 7))

    plt.plot(
        R*np.cos(phi),
        120+R*np.sin(phi),
        'k--',
        linewidth=2,
        label='Ideal circle'
    )

    for n in (5, 13, 37):
        p = results[(method, n)][0]

        plt.plot(
            p[:, 1],
            p[:, 2],
            label=f'{method} — {n} knots'
        )

    plt.xlabel('y [mm]')
    plt.ylabel('z [mm]')
    plt.title(f'Problem 7 — {method}')
    plt.axis('equal')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    name = "quintic" if method == "Quintic" else "degree7"
    plt.savefig(f"problem7_{name}_2D.png", dpi=180)

plt.show()
