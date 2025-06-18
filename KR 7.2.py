import numpy as np
import matplotlib.pyplot as plt


def value_derivative_func(x):
    return (x * np.cos(x) - (np.sin(x) + 1)) / x**2


x = np.linspace(0, 3, 500)
f = np.sin(x) - x**2 + 1

plt.figure(figsize=(10, 6))
plt.plot(x, f, label='f(x) = sin(x) - x**2 + 1', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('f(x) = sin(x) - x**2 + 1', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

x_points = np.linspace(1,2, 11)
ans = []

for x in x_points:
    value = value_derivative_func(x)
    if abs(value) < 1:
        ans.append(x)

ans = np.array(ans).tolist()
print(ans)