import time
import random
import matplotlib.pyplot as plt


def monotone_sweep(
        A: list,
        B: list,
        C: list,
        F: list
) -> list:
    """
    Solves a system of linear equations with a tridiagonal matrix using the monotone sweep method.
    :param A: (list) - list of lower diagonal coefficients, with A[0] = 0.
    :param B: (list) - list of main diagonal coefficients.
    :param C: (list) - list of upper diagonal coefficients, with C[-1] = 0.
    :param F: (list) - list of right hand sides of equations.
    :return: (list) - list of system solutions.
    """

    N = len(F)

    if len(A) != N or len(B) != N or len(C) != N:
        raise ValueError("The lengths of the coefficient arrays must match the length of F")

    if A[0] != 0 or C[-1] != 0:
        raise ValueError("A[0] and C[-1] must be equal to 0 according to the condition")

    alpha = [0] * N
    beta = [0] * N

    alpha[1] = -C[0] / B[0]
    beta[1] = F[0] / B[0]

    for k in range(1, N - 1):
        denominator = B[k] + A[k] * alpha[k]
        beta[k + 1] = (F[k] - A[k] * beta[k]) / denominator
        alpha[k + 1] = -C[k] / denominator

    U = [0] * N
    U[-1] = (F[-1] - A[-1] * beta[-1]) / (A[-1] * alpha[-1] + B[-1])

    for k in range(N - 2, -1, -1):
        U[k] = alpha[k + 1] * U[k + 1] + beta[k + 1]

    return U


a = [0, 1, 1]
b = [2, 3, 4]
c = [1, 1, 0]
f = [5, 15, 23]

u = monotone_sweep(a, b, c, f)
print('Решение системы:')
for i, j in enumerate(u):
    print(f'U[{i}] = {j}')


def generate_tridiagonal_system(N: int) -> tuple[list, list, list, list]:
    """
    Generates a random tridiagonal system of linear equations.
    :param N: (int) - size of the system.
    :return: (tuple) - tuple of four lists (A, B, C, F), where:
    A - lower diagonal coefficients (A[0] = 0),
    B - main diagonal coefficients,
    C - upper diagonal coefficients (C[-1] = 0),
    F - right-hand side vector.
    """

    AN = [0.0] + [random.uniform(0.1, 1.0) for _ in range(N - 1)]
    BN = [random.uniform(1.0, 2.0) for _ in range(N)]
    CN = [random.uniform(0.1, 1.0) for _ in range(N - 1)] + [0.0]
    FN = [random.uniform(1.0, 10.0) for _ in range(N)]

    return AN, BN, CN, FN


def measure_time(
        N: int,
        num_trials=10
) -> float:
    """
    Measures the average running time of monotone_sweep for a matrix of size N.
    :param N: (int) - matrix size.
    :param num_trials: (int) - number of trials to average, default is 10.
    :return: (float) - average run time in seconds.
    """
    total_time = 0.0

    for _ in range(num_trials):
        A, B, C, F = generate_tridiagonal_system(N)

        start_time = time.perf_counter()
        monotone_sweep(A, B, C, F)
        end_time = time.perf_counter()

        total_time += (end_time - start_time)

    return total_time / num_trials


matrix_sizes = [100, 1000, 5000, 10000, 20000, 50000]
times = []

for n in matrix_sizes:
    avg_time = measure_time(n)
    times.append(avg_time)
    print(f"N = {n:6d}, время = {avg_time:.6f} сек")

plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, times, 'o-', label='Время выполнения')
plt.xlabel('Размер матрицы (N)')
plt.ylabel('Среднее время (сек)')
plt.title('Зависимость времени работы monotone_sweep от размера матрицы')
plt.grid(True)
plt.legend()
plt.show()