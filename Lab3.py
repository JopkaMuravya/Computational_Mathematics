import numpy as np
import sympy as sp
from math import factorial


def lagrange_derivative(
        x_values: np.ndarray,
        y_values: np.ndarray,
        k: int,
        m: int,
        h: float
) -> float:
    """
    Calculates the k-th derivative of the Lagrange polynomial at a point x_m on a uniform grid.
    :param x_values: (np.ndarray) - array of interpolation nodes.
    :param y_values: (np.ndarray) - array of function values in nodes.
    :param k: (int) - derivative order.
    :param m: (int) - index of the point at which the derivative is calculated (x_m = x_values[m]).
    :param h: (float) - grid spacing.
    :return: (float) - The value of the k-th derivative of the Lagrange polynomial at the point x_m
    """

    n = len(x_values) - 1

    total = 0.0
    for i in range(n + 1):
        term = y_values[i] * derivative_basis_poly(n, m, i, k, h)
        total += term

    return total


def derivative_basis_poly(
        n: int,
        m: int,
        i: int,
        k: int,
        h: float
) -> float:
    """
    Compute the k-th derivative of the i-th Lagrange basis polynomial at x_m.
    :param n: (int) - degree of the polynomial (number of nodes - 1).
    :param m: (int) - index of the point at which the derivative is calculated (x_m = x_values[m]).
    :param i: (int) -  index of the basis polynomial l_i(x).
    :param k: (int) - derivative order.
    :param h: (float) - grid spacing.
    :return: - The value of k-th derivative of the i-th Lagrange basis polynomial at x_m.
    """

    numerator_poly = [1.0]

    for j in range(n + 1):
        if i == j:
            continue

        new_poly = [0.0] * (len(numerator_poly) + 1)
        for p in range(len(numerator_poly)):
            new_poly[p] += numerator_poly[p] * (-j * h)
            new_poly[p + 1] += numerator_poly[p] * 1.0
        numerator_poly = new_poly

    denominator = 1.0
    for j in range(n + 1):
        if j == i:
            continue
        denominator *= (i - j) * h

    poly_derivatives = [numerator_poly]
    for d in range(1, k + 1):
        deriv = [0.0] * (len(poly_derivatives[-1]) - 1)
        for p in range(1, len(poly_derivatives[-1])):
            deriv[p - 1] = poly_derivatives[-1][p] * p
        poly_derivatives.append(deriv)

    value = 0.0
    for p in range(len(poly_derivatives[k])):
        value += poly_derivatives[k][p] * (m * h) ** p

    return value / denominator


def remainder_term(
        x_values: np.ndarray,
        func_symp: sp.Expr,
        k: int,
        m: int
) -> tuple[float, float]:
    """
    Calculates the theoretical estimate of the remainder of the k-th order of differentiation
    :param x_values: (np.ndarray) - array of interpolation nodes.
    :param func_symp: (sympy.Expr) - Symbolic expression of the function to be interpolated
    :param k: (int) - derivative order.
    :param m: (int) - index of the point at which the derivative is calculated (x_m = x_values[m]).
    :return: (tuple[float, float]) - value the theoretical estimate of the remainder
                                    of the k-th order of differentiation
    """
    x = sp.symbols('x')

    n_plus = len(x_values)

    f_deriv = func_symp.diff(x, n_plus)
    f_deriv_func = sp.lambdify(x, f_deriv, 'numpy')

    x_min = min(x_values)
    x_max = max(x_values)
    x_grid = np.linspace(x_min, x_max, 1000)
    deriv_values = np.abs(f_deriv_func(x_grid))

    max_deriv = np.max(deriv_values)
    min_deriv = np.min(deriv_values)

    omega = sp.prod([x - x_j for x_j in x_values])
    omega_k = sp.diff(omega, x, k)
    omega_k_func = sp.lambdify(x, omega_k)
    omega_k_xm = omega_k_func(x_values[m])

    r_estimate_min = (min_deriv / factorial(n_plus)) * omega_k_xm
    r_estimate_max = (max_deriv / factorial(n_plus)) * omega_k_xm

    return abs(r_estimate_min), abs(r_estimate_max)


x = sp.symbols('x')
f_sympy = x**2 + sp.ln(x)
f = sp.lambdify(x, f_sympy, 'numpy')

a = 16
b = 19
point_cnt = 4
proizv = 2

x_point = np.linspace(16, 19, point_cnt)
y_point = f(x_point)

f_symp_der = f_sympy.diff(x, proizv)
f_symp_der_func = sp.lambdify(x, f_symp_der, 'numpy')

for s in range(len(x_point)):
    l_value = lagrange_derivative(x_point, y_point, proizv, s, 1)
    true_value = f_symp_der_func(x_point[s])
    abs_error = abs(l_value - true_value)
    r_min, r_max = remainder_term(x_point, f_sympy, proizv, s)
    print(f"f' = {true_value}")
    print(f"L' = {l_value}")
    print(f"Error: {abs_error}")
    print(f"R_min = {r_min}")
    print(f"R_max = {r_max}")
    if r_min < abs_error < r_max:
        print("min(R) < |L' - f'| < max(R) true")
    else:
        print("min(R) < |L' - f'| < max(R) false")
    print("--------------------------------------------", '\n')