from typing import List, Tuple
import pandas as pd
import sympy as sp
from scipy.optimize import minimize_scalar

x = sp.symbols('x')
f_sympy = x**2 + sp.ln(x)


def ura_pobeda(
        a: float,
        b: float,
        x_st: float,
        func
) -> None:
    """
    Victory function, you transmit data, magic happens, you get the result
    :param a: float - Starting point of the interval
    :param b: float - End point of the interval
    :param x_st: float - interpolation point
    :param func: Function for calculating values
    :return: The function does not return anything but makes the interaction more enjoyable for the user
    """

    f = sp.lambdify(x, func, 'numpy')

    table = []
    interval = []

    grid(a, b, x_st, table, interval, f)

    print('\n')
    print("#-------------------------Линейное интерполирование Лагранжа---------------------------#")

    L1 = lagrange_interpolation(x_st, interval, f)
    print("The value obtained using the Lagrange interpolation formula: ", L1)

    R1_min_abs, R1_max_abs = R1_score(interval, func)
    print("Minimum value of R1 by modulus: ", R1_min_abs)
    print("Maximum value of R1 by modulus: ", R1_max_abs)

    R1_x_st = L1 - f(x_st)
    print("Lagrange linear interpolation error: ", abs(R1_x_st))

    if R1_min_abs < abs(R1_x_st) < R1_max_abs:
        print("Actual error is within the theoretical remainder bounds")
    else:
        print("Actual error is outside the theoretical remainder bounds")

    if abs(R1_x_st) <= 10**(-4):
        print("Interpolation is allowed (linear interpolation error <= 0.0001)")
    else:
        print("Interpolation is not allowed (linear interpolation error > 0.0001)")

    print('\n')
    print("#-------------------------Квадратичное интерполирование Лагранжа---------------------------#")
    L2 = lagrange_interpolation_second(x_st, interval, f)
    print("The value obtained using the quadratic Lagrange interpolation formula: ", L2)

    R2_min_abs, R2_max_abs = R2_score(interval, func)
    print("Minimum value of R2 by modulus: ", R2_min_abs)
    print("Maximum value of R2 by modulus: ", R2_max_abs)

    R2_x_st = L2 - f(x_st)
    print("Lagrange quadratic interpolation error: ", abs(R2_x_st))

    if R2_min_abs < abs(R2_x_st) < R2_max_abs:
        print("Actual error is within the theoretical remainder bounds")
    else:
        print("Actual error is outside the theoretical remainder bounds")

    if abs(R2_x_st) <= 10**(-5):
        print("Interpolation is allowed (quadratic interpolation error <= 0.00001)")
    else:
        print("Interpolation is not allowed (quadratic interpolation error > 0.00001)")

    print('\n')
    print("#-------------------------Многочлен Ньютона первого порядка---------------------------#")

    N1 = newton_first_order(x_st, interval, f)
    print("the value of first order Newton interpolation polynomial: ", N1)

    print('\n')
    print("#-------------------------Многочлен Ньютона второго порядка---------------------------#")

    N2 = newton_second_order(x_st, interval, f)
    print("the value of second order Newton interpolation polynomial: ", N2)

    print('\n')
    print("#-------------------------Сравнение---------------------------#")

    print(f'value obtained using the Lagrange interpolation formula: {L1},\n'
          f'value of first order Newton interpolation polynomial: {N1},\n'
          f'their difference {L1 - N1}')
    print()
    print(f'value obtained using the quadratic Lagrange interpolation formula: {L2},\n'
          f'value of second order Newton interpolation polynomial: {N2},\n'
          f'their difference {L2 - N2}')

def grid(
        a: float,
        b: float,
        x_st: float,
        table: List[dict],
        interval: List[float],
        func
) -> None:
    """
    This function creates a grid with a step (h = (b - a) / 10) and calculates the values at the nodes and also
    finds the required interval for interpolation
    :param a: float - Starting point of the interval
    :param b: float - End point of the interval
    :param x_st: float - interpolation point
    :param table: List[dict] - an array containing dictionaries for constructing a table
    :param interval: List[float] - array for interval values that includes x_st
    :param func: Function for calculating values
    :return: The function does not return anything
    """

    h = (b - a) / 10
    for i in range(11):
        x_i = a + i * h
        y = func(x_i)
        table.append({
            'i': i,
            'x_i': x_i,
            'f(x_i)': y
        })
    for i in range(11):
        if x_st < table[i]['x_i']:
            interval.append(table[i - 1]['x_i'])
            interval.append(table[i]['x_i'])
            interval.append(table[i - 2]['x_i'])
            break
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(pd.DataFrame(table), '\n')


def lagrange_interpolation(
        x_st: float,
        interval: List[float],
        func
) -> float:
    """
    This function is a linear interpolation function at point x with a step of (b - a)/10 between points a and b.
    :param x_st: float - interpolation point
    :param interval: List[float] - array for interval values that includes x_st
    :param func: Function for calculating values
    :return: L1 - the value of the interpolation function at point x
    """

    L1 = (func(interval[0]) * (x_st - interval[1]) / (interval[0] - interval[1])
          + func(interval[1]) * (x_st - interval[0]) / (interval[1] - interval[0]))

    return L1

def R1_score(
        interval: List[float],
        func
) -> Tuple[float, float]:
    """
    This function finds the second derivative of a given function, its maximum and minimum value,
    and then calculates the maximum and minimum value of R1 modulo
    :param interval: List[float] - array for interval values that includes x_st
    :param func: Function for calculating values
    :return: Tuple[float, float] - minimum and maximum of R1 modulo
    """

    first_derivative_sympy = sp.diff(func, x)
    second_derivative_sympy = sp.diff(first_derivative_sympy, x)
    second_derivative = sp.lambdify(x, second_derivative_sympy, 'numpy')

    res_min_derivative = minimize_scalar(second_derivative, bounds=(interval[0], interval[1]), method='bounded')
    min_val_second_derivative = res_min_derivative.fun

    res_max_derivative = minimize_scalar(lambda y: -second_derivative(y), bounds=(interval[0], interval[1]),
                                         method='bounded')
    max_val_second_derivative = -res_max_derivative.fun

    res_min_R11 = minimize_scalar(lambda y: min_val_second_derivative * (y - interval[0]) * (y - interval[1]) / 2,
                                  bounds=(interval[0], interval[1]), method='bounded')
    R11_min = res_min_R11.fun

    res_min_R12 = minimize_scalar(lambda y: max_val_second_derivative * (y - interval[0]) * (y - interval[1]) / 2,
                                  bounds=(interval[0], interval[1]), method='bounded')
    R12_min = res_min_R12.fun

    res_max_R11 = minimize_scalar(lambda y: -(min_val_second_derivative * (y - interval[0]) * (y - interval[1]) / 2),
                                  bounds=(interval[0], interval[1]), method='bounded')
    R11_max = -res_max_R11.fun

    res_max_R12 = minimize_scalar(lambda y: -(max_val_second_derivative * (y - interval[0]) * (y - interval[1]) / 2),
                                  bounds=(interval[0], interval[1]), method='bounded')
    R12_max = -res_max_R12.fun

    R1_min = min(abs(R11_min), abs(R12_min), abs(R11_max), abs(R12_max))
    R1_max = max(abs(R11_min), abs(R12_min), abs(R11_max), abs(R12_max))

    return R1_min, R1_max


def lagrange_interpolation_second(
        x_st: float,
        interval: List[float],
        func
) -> float:
    """
    This function is a quadratic interpolation at point x with a step of (b - a)/10 between points a and b.
    :param x_st: float - interpolation point
    :param interval: List[float] - array for interval values that includes x_st
    :param func: Function for calculating values
    :return: L2 - the value of the interpolation function at point x
    """

    L2 = ((func(interval[2]) * (x_st - interval[0]) * (x_st - interval[1])) / ((interval[2] - interval[0]) * (interval[2] - interval[1]))
          + (func(interval[0]) * (x_st - interval[2]) * (x_st - interval[1])) / ((interval[0] - interval[2]) * (interval[0] - interval[1]))
          + (func(interval[1]) * (x_st - interval[2]) * (x_st - interval[0])) / ((interval[1] - interval[2]) * (interval[1] - interval[0])))

    return L2


def R2_score(
        interval: List[float],
        func
) -> Tuple[float, float]:
    """
    This function finds the third derivative of a given function, its maximum and minimum values,
    and then calculates the maximum and minimum values of the modulus R2.
    :param interval: List[float] - array for interval values that includes x_st
    :param func: Function for calculating values
    :return: Tuple[float, float] - minimum and maximum of R2 modulo
    """

    first_derivative_sympy = sp.diff(func, x)
    second_derivative_sympy = sp.diff(first_derivative_sympy, x)
    third_derivative_sympy = sp.diff(second_derivative_sympy, x)
    third_derivative = sp.lambdify(x, third_derivative_sympy, 'numpy')

    res_min_derivative = minimize_scalar(third_derivative, bounds=(interval[2], interval[1]), method='bounded')
    min_val_third_derivative = res_min_derivative.fun

    res_max_derivative = minimize_scalar(lambda y: -third_derivative(y), bounds=(interval[2], interval[1]),
                                         method='bounded')
    max_val_third_derivative = -res_max_derivative.fun

    res_min_R21 = minimize_scalar(lambda y: min_val_third_derivative * (y - interval[2]) * (y - interval[0]) * (y - interval[1]) / 6,
                                  bounds=(interval[2], interval[1]), method='bounded')
    R21_min = res_min_R21.fun

    res_min_R22 = minimize_scalar(lambda y: max_val_third_derivative * (y - interval[2]) * (y - interval[0]) * (y - interval[1]) / 6,
                                  bounds=(interval[2], interval[1]), method='bounded')
    R22_min = res_min_R22.fun

    res_max_R21 = minimize_scalar(lambda y: -(min_val_third_derivative * (y - interval[2]) * (y - interval[0]) * (y - interval[1]) / 6),
                                  bounds=(interval[2], interval[1]), method='bounded')
    R21_max = -res_max_R21.fun

    res_max_R22 = minimize_scalar(lambda y: -(max_val_third_derivative * (y - interval[2]) * (y - interval[0]) * (y - interval[1]) / 6),
                                  bounds=(interval[2], interval[1]), method='bounded')
    R22_max = -res_max_R22.fun

    R2_min = min(abs(R21_min), abs(R22_min), abs(R21_max), abs(R22_max))
    R2_max = max(abs(R21_min), abs(R22_min), abs(R21_max), abs(R22_max))

    return R2_min, R2_max


def first_order_divided_difference(
        x0: float,
        x1: float,
        func
) -> float:
    """
    Function that calculates the divided difference of the first order
    :param x0: float - first parameter of divided difference
    :param x1: float - second parameter of divided difference
    :param func: Function for calculating values
    :return:  f_xi_xi1 - divided difference value
    """

    f_xi_xi1 = (func(x1) - func(x0)) / (x1 - x0)

    return f_xi_xi1


def newton_first_order(
        x_st: float,
        interval: List[float],
        func
) -> float:
    """
    The function calculates the first order Newton interpolation polynomial at the point x_st
    :param x_st: float - interpolation point
    :param interval: List[float] - array for interval values that includes x_st
    :param func: Function for calculating values
    :return: N1 - The value of Newton's interpolation polynomial
    """

    N1 = func(interval[0]) + first_order_divided_difference(interval[0], interval[1], func) * (x_st - interval[0])

    return N1


def second_order_divided_difference(
        interval: List[float],
        func
) -> float:
    """
    Function that calculates the divided difference of the second order
    :param interval: List[float] - array for interval values that includes x_st
    :param func: Function for calculating values
    :return: f_xi1_xi_xi1 - divided difference value
    """

    f_xi1_xi_xi1 = (first_order_divided_difference(interval[0], interval[1], func) - first_order_divided_difference(interval[2], interval[0], func)) / (interval[1] - interval[2])

    return f_xi1_xi_xi1


def newton_second_order(
        x_st: float,
        interval: List[float],
        func
) -> float:
    """
    The function calculates the second order Newton interpolation polynomial at the point x_st
    :param x_st: float - interpolation point
    :param interval: List[float] - array for interval values that includes x_st
    :param func: Function for calculating values
    :return: N2 - The value of Newton's interpolation polynomial
    """

    N2 = func(interval[2]) + first_order_divided_difference(interval[2], interval[0], func) * (x_st - interval[2]) + second_order_divided_difference(interval, func) * (x_st - interval[2]) * (x_st - interval[0])

    return N2


ura_pobeda(0.4, 0.9, 0.52, f_sympy)
