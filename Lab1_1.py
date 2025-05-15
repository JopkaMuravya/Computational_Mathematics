import sympy as sp
import numpy as np
import math
from scipy.optimize import minimize_scalar

x = sp.symbols('x')
f_sympy = x**2 + sp.ln(x)
f = sp.lambdify(x, f_sympy, 'numpy')

a = 0.4
b = 0.9
n = 10
h = (b - a) / n
x_point = np.linspace(a, b, n + 1)
y_point = f(x_point)
x_star = 0.52


def lagrange_interpolation(
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_st: float,
        order: int
) -> tuple[float, list]:
    """
    Performs interpolation using the Lagrange formula for a given order
    :param x_values: (np.ndarray) - List of x values (interpolation nodes)
    :param y_values: (np.ndarray) - List of y values (function values at nodes)
    :param x_st: (float) - The point at which the interpolated value is calculated
    :param order: (int) - Interpolation order
    :return: tuple[float, list]: Interpolated value in x_st and list of nodes used
    """

    n_nodes = order + 1
    if n_nodes > len(x_values):
        raise ValueError(f"Not enough points for order {order}. At least {n_nodes} points required.")
    if order < 0:
        raise ValueError("The order must be a non-negative integer.")

    points = list(zip(x_values, y_values))
    points.sort(key=lambda point: abs(point[0] - x_st))

    selected_points = points[:n_nodes]
    x_selected = [p[0] for p in selected_points]
    y_selected = [p[1] for p in selected_points]

    result = 0.0
    for i in range(n_nodes):
        part = y_selected[i]
        for j in range(n_nodes):
            if i != j:
                part *= (x_st - x_selected[j])/(x_selected[i] - x_selected[j])
        result += part
    return result, x_selected


def lagrange_remainder_term(
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

    interval = (min([x_st] + x_selected), max([x_st] + x_selected))

    max_res = minimize_scalar(lambda x_val: -f_deriv_func(x_val), bounds=interval, method='bounded')
    max_deriv = -max_res.fun

    min_res = minimize_scalar(f_deriv_func, bounds=interval, method='bounded')
    min_deriv = min_res.fun

    omega = 1.0
    for xi in x_selected:
        omega *= (x_st - xi)

    factorial = math.factorial(n_plus)
    r_min = abs((min_deriv / factorial) * omega)
    r_max = abs((max_deriv / factorial) * omega)

    return min(r_min, r_max), max(r_min, r_max)


def divided_differences(
        x_selected: list,
        y_selected: list
) -> np.ndarray:
    """
    Computes a table of divided differences for only the selected nodes.
    :param x_selected: (list) - Selected interpolation nodes.
    :param y_selected: (list) - Function values at selected nodes.
    :return: (np.ndarray) - Divided difference table (k x k), where k = len(x_selected).
    """

    le = len(x_selected)
    table = np.zeros((le, le))
    table[:, 0] = y_selected

    for j in range(1, le):
        for i in range(le - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_selected[i + j] - x_selected[i])

    return table


def newton_interpolation(
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_st: float,
        order: int
) -> float:
    """
    Calculates the value of the Newton interpolation polynomial of a given order at the point x_st.
    :param x_values: (np.ndarray) - List of x values (interpolation nodes)
    :param y_values: (np.ndarray) - List of y values (function values at nodes)
    :param x_st: (float) - The point at which the interpolated value is calculated
    :param order: (int) - Interpolation order
    :return: (float) - The value of the interpolation polynomial at the point x_st
    """

    if order >= len(x_values):
        raise ValueError("The order of interpolation is higher than the number of available points")

    n_nodes = order + 1
    points = list(zip(x_values, y_values))
    points.sort(key=lambda point: abs(point[0] - x_st))

    selected_points = points[:n_nodes]
    x_selected = [p[0] for p in selected_points]
    y_selected = [p[1] for p in selected_points]

    dd_table = divided_differences(x_selected, y_selected)

    result = dd_table[0, 0]
    product = 1.0

    for k in range(1, order + 1):
        product *= (x_st - x_selected[k - 1])
        result += dd_table[0, k] * product

    return result


f_true = f(x_star)
print("#-------------------------Линейное интерполирование Лагранжа---------------------------#")
l_value_1, x_use1 = lagrange_interpolation(x_point, y_point, x_star, 1)
r_er1 = abs(l_value_1 - f_true)
r_min1, r_max1 = lagrange_remainder_term(f_sympy, x_use1, x_star, 1)

print("The value obtained using the Lagrange interpolation formula: ", l_value_1)
print("Real value of func:", f_true)
print("Minimum value of R1:", r_min1)
print("Maximum value of R1:", r_max1)
print("Lagrange linear interpolation error:", r_er1)

if r_min1 < r_er1 < r_max1:
    print("Actual error is within the theoretical remainder bounds")
else:
    print("Actual error is outside the theoretical remainder bounds")

if r_er1 <= 10 ** (-4):
    print("Interpolation is allowed (linear interpolation error <= 0.0001)")
else:
    print("Interpolation is not allowed (linear interpolation error > 0.0001)")

print('\n')
print("#-------------------------Квадратичное интерполирование Лагранжа---------------------------#")
l_value_2, x_use2 = lagrange_interpolation(x_point, y_point, x_star, 2)
r_er2 = abs(l_value_2 - f_true)
r_min2, r_max2 = lagrange_remainder_term(f_sympy, x_use2, x_star, 2)

print("The value obtained using the Lagrange interpolation formula: ", l_value_2)
print("Real value of func:", f_true)
print("Minimum value of R1:", r_min2)
print("Maximum value of R1:", r_max2)
print("Lagrange linear interpolation error:", r_er2)

if r_min2 < r_er2 < r_max2:
    print("Actual error is within the theoretical remainder bounds")
else:
    print("Actual error is outside the theoretical remainder bounds")

if r_er2 <= 10 ** (-5):
    print("Interpolation is allowed (quadratic interpolation error <= 0.00001)")
else:
    print("Interpolation is not allowed (quadratic interpolation error > 0.00001)")

print('\n')
print("#-------------------------Многочлен Ньютона первого порядка---------------------------#")
n_value_1 = newton_interpolation(x_point, y_point, x_star, 1)
print("the value of first order Newton interpolation polynomial: ", n_value_1)

print('\n')
print("#-------------------------Многочлен Ньютона второго порядка---------------------------#")
n_value_2 = newton_interpolation(x_point, y_point, x_star, 2)
print("the value of second order Newton interpolation polynomial: ", n_value_2)

print('\n')
print("#-------------------------Сравнение---------------------------#")

print(f'value obtained using the Lagrange interpolation formula: {l_value_1},\n'
        f'value of first order Newton interpolation polynomial: {n_value_1},\n'
        f'their difference {l_value_1 - n_value_1}')
print()
print(f'value obtained using the quadratic Lagrange interpolation formula: {l_value_2},\n'
        f'value of second order Newton interpolation polynomial: {n_value_2},\n'
        f'their difference {l_value_2 - n_value_2}')
