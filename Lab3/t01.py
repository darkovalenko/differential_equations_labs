import sys
import numpy as np
import matplotlib.pyplot as plt

# Лабораторна робота 3, Задача 1. Варіант 3.
# x' = 3x - y,  y' = 2x + 7y
A = np.array([[3, -1],
              [2, 7]])

# =====================================================================
# 1) Знайти, під яким кутом фазові траєкторії перетинають пряму y = x
# =====================================================================
# На прямій y = x маємо x = y.
# Вектор швидкості: dx/dt = 3x - x = 2x, dy/dt = 2x + 7x = 9x
# Кутовий коефіцієнт дотичної до фазової траєкторії: k1 = (dy/dt)/(dx/dt) = 9/2 = 4.5
k1 = 9 / 2
# Кутовий коефіцієнт прямої y = x: k2 = 1.0
k2 = 1.0

# Формула кута між двома прямими: tan(theta) = |(k1 - k2) / (1 + k1*k2)|
tan_theta = abs((k1 - k2) / (1 + k1 * k2))
theta_rad = np.arctan(tan_theta)
theta_deg = np.degrees(theta_rad)

print("--- Пункт 1 ---")
print(f"tg(theta) = {tan_theta:.4f} (= 7/11)")
print(f"Кут перетину з прямою y = x: {theta_deg:.2f}° (або {theta_rad:.4f} рад)\n")

# =====================================================================
# 2) Для фазових траєкторій, що примикають до початку координат,
# знайти кут, під яким вони примикають до точки (0, 0)
# =====================================================================
# Такі траєкторії спрямовані вздовж власних векторів матриці системи.
eigenvalues, eigenvectors = np.linalg.eig(A.astype(float))
eigenvalues = np.real_if_close(eigenvalues)
eigenvectors = np.real_if_close(eigenvectors)

print("--- Пункт 2 ---")
print(f"Власні числа: {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")
tr, det = np.trace(A), np.linalg.det(A)
print(f"tr A = {tr:.0f}, det A = {det:.0f}, D = tr^2 - 4det = {tr**2 - 4*det:.0f} > 0")
if np.all(eigenvalues > 0):
    kind = "Нестійкий вузол"
elif np.all(eigenvalues < 0):
    kind = "Стійкий вузол"
else:
    kind = "Сідло"
print(f"Тип точки (0, 0): {kind}")
print("Кути примикання до (0,0) (визначаються власними векторами):")
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]  # Власний вектор
    angle_rad = np.arctan2(v[1], v[0])
    angle_deg = np.degrees(angle_rad)
    slope = v[1] / v[0]
    print(f"Власне число λ = {eigenvalues[i]:.4f}:")
    print(f"  Власний вектор: [{v[0]:.4f}, {v[1]:.4f}]")
    print(f"  Кутовий коефіцієнт: k = {slope:.4f}")
    print(f"  Кут нахилу: {angle_deg:.2f}° (або {angle_rad:.4f} рад)")

# =====================================================================
# 3) Зобразити фазові траєкторії
# =====================================================================
print("\n--- Пункт 3 ---")
print("Генерую фазовий портрет...")

Y, X = np.mgrid[-5:5:100j, -5:5:100j]

U = 3 * X - Y
V = 2 * X + 7 * Y

plt.figure(figsize=(8, 8))

plt.streamplot(X, Y, U, V, color='royalblue', linewidth=1, density=1.5,
               arrowstyle='->', arrowsize=1.5)

x_vals = np.linspace(-5, 5, 100)
plt.plot(x_vals, x_vals, color='green', linestyle='--', label='Пряма $y = x$')

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    slope = v[1] / v[0]
    plt.plot(x_vals, slope * x_vals, color='red', linestyle='-.',
             label=f'Власний напрямок (λ={eigenvalues[i]:.2f})')

plt.title(r'Фазовий портрет системи: $\dot{x} = 3x - y$, $\dot{y} = 2x + 7y$ (Нестійкий вузол)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.legend(loc='upper left')

if '--save' in sys.argv:
    plt.savefig('task1.png', dpi=130, bbox_inches='tight')
else:
    plt.show()
