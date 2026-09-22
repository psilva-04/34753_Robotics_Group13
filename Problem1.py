import sympy as sp


def dh_transform(theta, d, a, alpha):
    """Standard DH transformation."""
    ct = sp.cos(theta)
    st = sp.sin(theta)
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)

    return sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])


def direct_kinematics():
    "calculating the homogeneous matrices"

    theta1, theta2, theta3, theta4 = sp.symbols(
        'theta1 theta2 theta3 theta4',
        real=True
    )

    base_height = sp.Integer(50)
    link_2 = sp.Integer(93)
    link_3 = sp.Integer(93)
    stylus_link = sp.Integer(50)

    T01 = dh_transform(theta1, base_height, 0, sp.pi/2)
    T12 = dh_transform(theta2, 0, link_2, 0)
    T23 = dh_transform(theta3, 0, link_3, 0)
    T34 = dh_transform(theta4, 0, stylus_link, 0)

    T04 = sp.simplify(T01 * T12 * T23 * T34)

    T45 = sp.Matrix([
        [1, 0, 0, -15],
        [0, 1, 0,  45],
        [0, 0, 1,   0],
        [0, 0, 0,   1]
    ])

    T05 = sp.simplify(T04 * T45)

    return T01, T12, T23, T34, T04, T45, T05


T01, T12, T23, T34, T04, T45, T05 = direct_kinematics()

print("T_4^0 =")
sp.pprint(T04)

print("\nT_5^0 =")
sp.pprint(T05)