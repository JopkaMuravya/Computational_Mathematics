import sympy as sp
import numpy as np
import pandas as pd
import math

x = sp.symbols('x')
f_sympy = x**2 + sp.ln(x)
f = sp.lambdify(x, f_sympy, 'numpy')

a = 0.4
b = 0.9
n = 10
h = (b - a) / n
x_point = np.linspace(a, b, n + 1)
y_point = f(x_point)

x_star1 = 0.52
x_star2 = 0.42
x_star3 = 0.87
x_star4 = 0.67


def finite_differences_table(
        x_values: np.ndarray,
        y_values: np.ndarray
) -> tuple[pd.DataFrame, list]:
    """
    Builds a finite difference table as a DataFrame and list.
    :param x_values: (np.ndarray) - Array of x values.
    :param y_values: (np.ndarray) - Array of values of the function f(x).
    :return: DataFrame and list with finite difference table.
    """
    differences = [y_values.copy()]

    for order in range(1, len(y_values)):
        last_order_diff = differences[-1]
        current_order_diff = []

        for i in range(len(last_order_diff) - 1):
            current_order_diff.append(last_order_diff[i+1] - last_order_diff[i])

        differences.append(current_order_diff)

    table_data = {
        'x': x_values,
        'f(x)': differences[0]
    }

    for order in range(1, len(differences)):
        column_data = differences[order] + [None] * (len(x_values) - len(differences[order]))
        table_data[f'Δ^{order}f'] = column_data

    return pd.DataFrame(table_data), differences


def newton_forward_interpolation(
        x_values: np.ndarray,
        differences: list,
        x_star: float,
        h: float,
        order: int
) -> float:
    """
    Forward interpolation using Newton's formula.
    :param x_values: (np.ndarray) - Array of x values.
    :param differences: (list) - list of lists of finite differences of different orders.
    :param x_star: (float) - Point for interpolation
    :param h: (float) - grid spacing
    :param order: (int) - interpolation polynomial order
    :return: (float) - Interpolated value and order
    """

    t = (x_star - x_values[0]) / h

    result = differences[0][0]
    t_product = 1

    for i in range(1, order + 1):
        t_product *= (t - (i - 1))
        result += (t_product / math.factorial(i)) * differences[i][0]

    return result


def newton_backward_interpolation(
        x_values: np.ndarray,
        differences: list,
        x_star: float,
        h: float,
        order: int
) -> float:
    """
    Backward interpolation using Newton's formula.
    :param x_values: (np.ndarray) - Array of x values.
    :param differences: (list) - list of lists of finite differences of different orders
    :param x_star: (float) - Point for interpolation
    :param h: (float) - grid spacing
    :param order: (int) - interpolation polynomial order
    :return:  (float) - Interpolated value
    """

    t = (x_star - x_values[-1]) / h

    result = differences[0][-1]
    t_product = 1

    for i in range(1, order + 1):
        t_product *= (t + (i - 1))
        if i >= len(differences):
            break
        if (-i - 1) >= len(differences[i]):
            break
        result += (t_product / math.factorial(i)) * differences[i][-i - 1]

    return result


def stirling_interpolation(
        x_values: np.ndarray,
        differences: list,
        x_star: float,
        h: float
) -> tuple[float, int]:
    """
    Stirling's interpolation formula for equally spaced nodes (grid center).
    :param x_values: (np.ndarray) - Array of x values
    :param differences: (list) - list of lists of finite differences of different orders
    :param x_star: (float) - Point for interpolation
    :param h: (float) - grid spacing
    :return: (tuple[float, int]) - Interpolated value and order
    """

    len_x = len(x_values)
    center_idx = (len_x - 1) // 2
    x0 = x_values[center_idx]
    t = (x_star - x0) / h

    max_order = min(center_idx, len_x - center_idx - 1)
    result = differences[0][center_idx]

    for k in range(1, max_order + 1):
        if k % 2 == 0:
            m = k // 2
            numerator = 1.0
            for j in range(m):
                numerator *= (t ** 2 - j ** 2)
            coeff = numerator / math.factorial(k)
            diff = differences[k][center_idx - m]
        else:
            m = (k - 1) // 2
            numerator = t
            for j in range(1, m + 1):
                numerator *= (t ** 2 - j ** 2)
            coeff = numerator / math.factorial(k)
            diff = (differences[k][center_idx - m - 1] + differences[k][center_idx - m]) / 2

        result += coeff * diff

    return result, max_order


def remainder_term(
        f_sympy_func: sp.Expr,
        x_selected: list,
        x_st: float,
        order: int
) -> tuple[float, float]:
    """
    Calculates the minimum and maximum values of the remainder of a Lagrange interpolation.
    :param f_sympy_func: (sympy.Expr) - Symbolic expression of the function to be interpolated.
    :param x_selected: (list) - List of selected interpolation nodes used to construct the polynomial.
    :param x_st: (float) - The point at which the remainder term is evaluated.
    :param order: (int) - The order of the Lagrange interpolation polynomial.
    :return: A tuple of two elements: R_min: minimum value of the remainder. R_max: maximum value of the remainder.
    """

    n_plus = order + 1
    f_deriv = f_sympy_func.diff(x, n_plus)
    f_deriv_func = sp.lambdify(x, f_deriv, 'numpy')

    x_min = min(min(x_selected), x_st)
    x_max = max(max(x_selected), x_st)
    x_grid = np.linspace(x_min, x_max, 1000)
    deriv_values = np.abs(f_deriv_func(x_grid))

    max_deriv = np.max(deriv_values)
    min_deriv = np.min(deriv_values)

    omega = np.prod([x_st - xi for xi in x_selected[:order + 1]])
    factorial = math.factorial(n_plus)

    r_min = (min_deriv / factorial) * abs(omega)
    r_max = (max_deriv / factorial) * abs(omega)

    return r_min, r_max


df, dif = finite_differences_table(x_point, y_point)
print("Table of finite differences:")
print(df.to_string(float_format="%.8f", na_rep='-'))

print('\n')
print("#-----------------Ньютон вперед в точке x** = 0.42--------#")
value_x2 = newton_forward_interpolation(x_point, dif, x_star2, h, 10)
f_true2 = f(x_star2)
r_er2 = abs(value_x2 - f_true2)
r_min2, r_max2 = remainder_term(f_sympy, x_point, x_star2, 10)

print(f"f(x**) = {f_true2}")
print(f"L(x**) = {value_x2}")
print(f"Error: {r_er2}")
print(f"R_min2 = {r_min2}")
print(f"R_max2 = {r_max2}")
if r_min2 < r_er2 < r_max2:
    print("min(R) < |L - f| < max(R) true")
else:
    print("min(R) < |L - f| < max(R) false")

print('\n')
print("#-----------------Ньютон назад в точке x*** = 0.87--------#")
value_x3 = newton_backward_interpolation(x_point, dif, x_star3, h, 5)
f_true3 = f(x_star3)
r_er3 = abs(value_x3 - f_true3)
r_min3, r_max3 = remainder_term(f_sympy, x_point, x_star3, 5)
print(x_point, x_point[-6:])

print(f"f(x**) = {f_true3}")
print(f"L(x**) = {value_x3}")
print(f"Error: {r_er3}")
print(f"R_min2 = {r_min3}")
print(f"R_max2 = {r_max3}")
if r_min3 < r_er3 < r_max3:
    print("min(R) < |L - f| < max(R) true")
else:
    print("min(R) < |L - f| < max(R) false")

print('\n')
print("#-----------------Стирлинг в точке x**** = 0.67--------#")
value_x4, order4 = stirling_interpolation(x_point, dif, x_star4, h)
f_true4 = f(x_star4)
r_er4 = abs(value_x4 - f_true4)
center_idx = (len(x_point) - 1) // 2
selected_nodes = x_point[center_idx-order4//2-1 : center_idx+order4//2+2]
r_min4, r_max4 = remainder_term(f_sympy, selected_nodes, x_star4, order4)

print(f"f(x**) = {f_true4}")
print(f"L(x**) = {value_x4}")
print(f"Error: {r_er4}")
print(f"R_min2 = {r_min4}")
print(f"R_max2 = {r_max4}")
if r_min4 < r_er4 < r_max4:
    print("min(R) < |L - f| < max(R) true")
else:
    print("min(R) < |L - f| < max(R) false")
