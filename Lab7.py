import numpy as np
import matplotlib.pyplot as plt

def separate_roots(
        func: callable,
        start: float,
        end: float,
        bins: int
) -> list:
    """
    Separates the interval [a, b] into subintervals where the function f(x) changes sign.
    :param func: (callable) - Function for which roots are being located. Must be continuous on [a, b].
    :param start: (float) - Left boundary of the interval.
    :param end: (float) - Right boundary of the interval.
    :param bins: (int) - Number of subintervals to divide [a, b] into.
    :return: list: A list of tuples, where each tuple (x_i, x_{i+1}) defines an interval
              in which the function changes sign and a root is expected to exist.
    """

    x = np.linspace(start, end, bins)
    intervals = []
    for i in range(len(x) - 1):
        if np.sign(func(x[i])) != np.sign(func(x[i + 1])):
            intervals.append((x[i], x[i + 1]))
    return intervals


def choose_initial_points(
        func: callable,
        d2_func: callable,
        start: float,
        end: float
) -> tuple[float, float]:
    """
     Chooses proper starting points for the Newton and secant methods based on second derivative.
    :param func: (callable) -  Function for which the root is being sought.
    :param d2_func: (callable) - Second derivative of the function f(x).
    :param start: (float) -  Left endpoint of the interval.
    :param end: (flat) -  Right endpoint of the interval.
    :return: tuple[float, float], where first float -  start point for Newton's method,
                                        second float - other endpoint for the secant method.
    """

    func_start = func(start)
    func_end = func(end)

    d2_func_start = d2_func(start)
    d2_func_end = d2_func(end)

    if func_start * d2_func_start > 0:
        ans = [start, end]
    elif func_end * d2_func_end:
        ans = [end, start]
    else:
        raise ValueError("For Newton's method, the following condition must be met: f(x) * f''(x) > 0")

    return ans[0], ans[1]


def newton_method(
        func: callable,
        d_func: callable,
        start_x: float,
        eps: float
) -> float:
    """
    Finds a root of the equation f(x) = 0 using the Newton method (method of tangents).
    :param func: (callable) - Function for which the root is being sought.
    :param d_func: (callable) - First derivative of the function f(x).
    :param start_x: (float) - Initial approximation.
    :param eps: (float) - Tolerance for stopping criterion (based on difference between iterations).
    :return: (float) - Approximated root of the function with given accuracy.
    """

    x_prev = start_x
    while True:
        x_next = x_prev - func(x_prev) / d_func(x_prev)
        if abs(x_next - x_prev) < eps:
            return x_next
        x_prev = x_next


def secant_method(
        func: callable,
        x1: float,
        x2: float,
        eps: float
) -> float:
    """
    Finds a root of the equation f(x) = 0 using the secant method (method of chords).
    :param func: (callable) - Function for which the root is being sought.
    :param x1: (float) - First initial approximation.
    :param x2: (float) - Second initial approximation.
    :param eps: (float) - Tolerance for stopping criterion (based on difference between iterations).
    :return: (float) - Approximated root of the function with given accuracy.
    """

    while True:
        if func(x1) == func(x2):
            raise ValueError("The secant method cannot work: f(x0) = f(x1). Choose other starting points.")
        x3 = x2 - func(x2) * (x2 - x1) / (func(x2) - func(x1))
        if abs(x3 - x2) < eps:
            return x3
        x1 = x2
        x2 = x3


def combined_method(
        func: callable,
        d_func: callable,
        x_newton_st: float,
        x_secant_st: float,
        eps: float
) -> tuple[float, float, float]:
    """
    Finds a root of the equation f(x) = 0 using combined Newton and secant methods.
    The iteration stops when the difference between Newton and secant approximations becomes less than epsilon.
    :param func: (callable) - Function for which the root is being sought.
    :param d_func: (callable) - First derivative of the function f(x).
    :param x_newton_st: (float) - Left boundary of the interval.
    :param x_secant_st: (float) - Right boundary of the interval.
    :param eps: (float) - Tolerance for stopping criterion (based on difference between methods' approximations).
    :return: (tuple[float, float, float]) - Approximated roots of the function with given accuracy,
                                            where first float - root newton,
                                            second float - root secant,
                                            third float - root arithmetic mean of newton's root and secant
    """
    x_newton_pv = x_newton_st
    x_secant_pv = x_secant_st
    while True:
        x_newton_nt = x_newton_pv - func(x_newton_pv) / d_func(x_newton_pv)

        if func(x_secant_pv) == func(x_newton_pv):
            raise ValueError("The secant method cannot work: f(x0) = f(x1). Choose other starting points.")
        x_secant_nt = (x_newton_pv - func(x_newton_pv) * (x_newton_pv - x_secant_pv) /
                                                        (func(x_newton_pv) - func(x_secant_pv)))

        if abs(x_newton_nt - x_secant_nt) < eps:
            return x_newton_nt, x_secant_nt, (x_newton_nt + x_secant_nt) / 2

        x_secant_pv = x_newton_pv
        x_newton_pv = x_newton_nt


def f(x):
    return 0.5 * x**2 - np.cos(2 * x)


def df(x):
    return x + 2 * np.sin(2 * x)


def d2f(x):
    return 1 + 4 * np.cos(2 * x)


a = -5
b = 5
eps = 1e-6
root_intervals = separate_roots(f, a, b, bins=1000)

x_gr = np.linspace(a, b, 1000)
y_gr = f(x_gr)

plt.figure(figsize=(10, 6))
plt.plot(x_gr, y_gr, label='f(x) = 0.5x² - cos(2x)')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')

count = 0
for start_i, end_i in root_intervals:
    x_newton, x_secant = choose_initial_points(f, d2f, start_i, end_i)
    root_newton = newton_method(f, df, x_newton, eps)
    root_secant = secant_method(f, x_secant, x_newton, eps)
    root_newton_comb, root_secant_comb, root_mean = combined_method(f, df, x_newton, x_secant, eps)

    print(f'Интервал: [{start_i}, {end_i}] Корень (метод касательных) = {root_newton} Корень (метод хорд) = {root_secant}')
    print(f'Интервал: [{start_i}, {end_i}] Корень (метод касательных) = {root_newton_comb} Корень (метод хорд) = {root_secant_comb} Корень (среднее арифметическое) = {root_mean}')
    print()
    if count == 0:
        plt.scatter(root_newton, 0, color='red', marker="x", label='Метод Ньютона')
        plt.scatter(root_secant, 0, color='green', marker="x", label='Метод хорд')
        plt.scatter(root_newton_comb, 0, color='red', marker="o", label='Метод Ньютона комб')
        plt.scatter(root_secant_comb, 0, color='green', marker="o", label='Метод хорд комб')
        plt.scatter(root_mean, 0, color='blue', marker="o", label='Среднее арифметическое комб')

        count += 1
    else:
        plt.scatter(root_newton, 0, color='red', marker="x")
        plt.scatter(root_secant, 0, color='green', marker="x")
        plt.scatter(root_newton_comb, 0, color='red', marker="o")
        plt.scatter(root_secant_comb, 0, color='green', marker="o")
        plt.scatter(root_mean, 0, color='blue', marker="o")

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции и найденные корни')
plt.legend()
plt.grid(True)
plt.show()
