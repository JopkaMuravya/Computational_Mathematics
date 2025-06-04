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
