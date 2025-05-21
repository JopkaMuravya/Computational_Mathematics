import numpy as np
from scipy.integrate import quad


def func(
        x: float
) -> float:
    """
    Function that calculates the value at point x.
    :param x: (float) - the point at which the value is calculated.
    :return: (float) - the value of the function at point x.
    """

    return x**2 + np.log(x)


def left_rectangles(
        f: callable,
        a: float,
        b: float,
        n: int
) -> float:
    """
    Calculates the integral of a function using the composite formula of the left rectangles.
    :param f: (function) - Function to integrate.
    :param a: (float) - Lower limit of integration.
    :param b: (float) - Upper limit of integration.
    :param n: (int) - Number of partition intervals.
    :return: (float) - Approximate value of the integral.
    """

    h = (b - a)/n
    return h * sum(f(a + i * h) for i in range(n))


def right_rectangles(
        f: callable,
        a: float,
        b: float,
        n: int
) -> float:
    """
    Calculates the integral of a function using the composite formula of the right rectangles.
    :param f: (function) - Function to integrate.
    :param a: (float) - Lower limit of integration.
    :param b: (float) - Upper limit of integration.
    :param n: (int) - Number of partition intervals.
    :return: (float) - Approximate value of the integral.
    """

    h = (b - a)/n
    return h * sum(f(a + (i + 1) * h) for i in range(n))


def central_rectangles(
        f: callable,
        a: float,
        b: float,
        n: int
) -> float:
    """
    Calculates the integral of a function using the composite formula of the central rectangles.
    :param f: (function) - Function to integrate.
    :param a: (float) - Lower limit of integration.
    :param b: (float) - Upper limit of integration.
    :param n: (int) - Number of partition intervals.
    :return: (float) - Approximate value of the integral.
    """

    h = (b - a)/n
    return h * sum(f(a + (i + 0.5) * h) for i in range(n))


def trapezoid(
        f: callable,
        a: float,
        b: float,
        n: int
) -> float:
    """
    Calculates the integral of a function using the composite formula of the trapezoid.
    :param f: (function) - Function to integrate.
    :param a: (float) - Lower limit of integration.
    :param b: (float) - Upper limit of integration.
    :param n: (int) - Number of partition intervals.
    :return: (float) - Approximate value of the integral.
    """

    h = (b - a) / n
    return h * (0.5 * (f(a) + f(b)) + sum(f(a + i * h) for i in range(1, n)))


def simpson(
        f: callable,
        a: float,
        b: float,
        n: int
) -> float:
    """
    Calculates the integral of a function using Simpson's compound formula.
    Requires an even number of intervals.
    :param f: (function) - Function to integrate.
    :param a: (float) - Lower limit of integration.
    :param b: (float) - Upper limit of integration.
    :param n: (int) - Number of partition intervals (must be even).
    :return: (float) - Approximate value of the integral.
    """
    if n % 2 != 0:
        raise ValueError("Number of partition intervals must be even")

    h = (b - a) / n
    return h / 6 * (f(a) + f(b) + 2 * sum(f(a + i * h) for i in range(1, n))
                                + 4 * sum(f(a + (j + 0.5) * h) for j in range(n)))


def weddle(
        f: callable,
        a: float,
        b: float,
        n: int
) -> float:
    """
    Calculates the integral using Weddle's formula (requires n multiple of 6).
    :param f: (function) - Function to integrate.
    :param a: (float) - Lower limit of integration.
    :param b: (float) - Upper limit of integration.
    :param n: (int) - Number of partition intervals (must be a multiple of 6).
    :return: (float) - Approximate value of the integral.
    """

    if n % 6 != 0:
        raise ValueError("The number of intervals n must be a multiple of 6")

    h = (b - a) / n
    total = 0.0

    pattern = [1, 5, 1, 6, 1, 5, 2]

    for i in range(n + 1):
        if i % 6 == 0 and i != 0 and i != n:
            coeff = 2
        else:
            coeff = pattern[i % 6]

        total += coeff * f(a + i * h)

    return 0.3 * h * total


def newton_cotes(
        f: callable,
        a: float,
        b: float,
        n: int
) -> float:
    """
    Calculates the integral of a function using the Newton-Cotes formula.
    :param f: (function) - Function to integrate.
    :param a: (float) - Lower limit of integration.
    :param b: (float) - Upper limit of integration.
    :param n: (int) - Degree of formula (number of points - 1).
    :return: (float) - Approximate value of the integral.
    """

    coeffs = []
    if n == 1:
        coeffs = [1/2, 1/2]
    elif n == 2:
        coeffs = [1/6, 4/6, 1/6]
    elif n == 3:
        coeffs = [1/8, 3/8, 3/8, 1/8]
    elif n == 4:
        coeffs = [7/90, 32/90, 12/90, 32/90, 7/90]
    elif n == 5:
        coeffs = [19/288, 75/288, 50/288, 50/288, 75/288, 19/288]
    elif n == 6:
        coeffs = [41/840, 216/840, 27/840, 272/840, 27/840, 216/840, 41/840]

    h = (b - a) / n
    nodes = [a + i * h for i in range(n + 1)]
    return (b - a) * sum(c * f(x) for c, x in zip(coeffs, nodes))


def gauss(
        f: callable,
        a: float,
        b: float,
        n: int
) -> float:
    """
    Calculates the integral of a function using the Gauss quadrature formula.
    :param f: (function) - Function to integrate.
    :param a: (float) - Lower limit of integration.
    :param b: (float) - Upper limit of integration.
    :param n: (int) - Number of partition intervals.
    :return: (float) - Approximate value of the integral.
    """

    nodes = []
    weights = []
    if n == 1:
        nodes = [0]
        weights = [2]
    elif n == 2:
        nodes = [-0.577350, 0.577350]
        weights = [1, 1]
    elif n == 3:
        nodes = [-0.774597, 0, 0.774597]
        weights = [5/9, 8/9, 5/9]
    elif n == 4:
        nodes = [-0.861136, -0.339981, 0.339981, 0.861136]
        weights = [0.347855, 0.652145, 0.652145, 0.347855]

    return (b - a) / 2 * sum(w * f((b + a) / 2 + (b - a) / 2 * t) for t, w in zip(nodes, weights))


low = 0.4
up = 0.9
cnt_int = 12

true_value, _ = quad(func, low, up)

methods = [
    ("Левые прямоугольники", left_rectangles),
    ("Правые прямоугольники", right_rectangles),
    ("Центральные прямоугольники", central_rectangles),
    ("Трапеции", trapezoid),
    ("Симпсон", simpson),
    ("Веддля", weddle),
    ("Ньютон-Кортес", newton_cotes),
    ("Гаусс", gauss)
]
print("Истинное значение =", true_value)
for name, method in methods:
    if name == "Ньютон-Кортес":
        value = method(func, low, up, 4)
    elif name == "Гаусс":
        value = method(func, low, up, 3)
    else:
        value = method(func, low, up, cnt_int)
    print(f"{name} = {value}, error = {abs(value - true_value)}")
