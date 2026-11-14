import numpy as np
import matplotlib.pyplot as plt
import sys
import sympy as sp

x, y = sp.symbols('x y', real=True)

# Лабораторна робота 3, Задача 2. Варіант 3.
# x' = x - y,  y' = y^2 + x^2 - 2
P = x - y
Q = y ** 2 + x ** 2 - 2

print("АНАЛІТИЧНА ЧАСТИНА\n" + "=" * 50)

# =====================================================================
# 1) Знайти всі положення рівноваги системи та для кожного записати
#    відповідну систему першого наближення (лінеаризовану)
# =====================================================================
print("--- Пункт 1 ---")
# Знаходимо положення рівноваги (прирівнюємо похідні до нуля)
eq_points = sp.solve([P, Q], (x, y))
print(f"Знайдені положення рівноваги (x, y): {eq_points}")

# Знаходимо матрицю Якобі (для системи першого наближення)
# J = [[dP/dx, dP/dy],
#      [dQ/dx, dQ/dy]]
J = sp.Matrix([[sp.diff(P, x), sp.diff(P, y)],
               [sp.diff(Q, x), sp.diff(Q, y)]])

points_data = []

for idx, pt in enumerate(eq_points):
    pt_x, pt_y = pt
    # Підставляємо координати точки в матрицю Якобі
    J_evaluated = J.subs({x: pt_x, y: pt_y})

    print(f"\nПоложення рівноваги {idx + 1}: ({pt_x}, {pt_y})")
    print("Матриця Якобі (система першого наближення):")
    sp.pprint(J_evaluated)

    points_data.append({'pt': pt, 'J': J_evaluated})

# =====================================================================
# 3) Визначити тип усіх положень рівноваги
# =====================================================================
print("\n--- Пункт 3 ---")
for idx, data in enumerate(points_data):
    pt = data['pt']
    J_matrix = np.array(data['J']).astype(np.float64)

    # Знаходимо власні числа
    eigenvalues = np.linalg.eigvals(J_matrix)
    print(f"\nТочка ({pt[0]}, {pt[1]}):")
    if abs(np.imag(eigenvalues[0])) > 1e-12:
        print(f"Власні числа: λ1,2 = {eigenvalues[0].real:.4f} ± {abs(eigenvalues[0].imag):.4f}i")
    else:
        print(f"Власні числа: λ1 = {eigenvalues[0].real:.4f}, λ2 = {eigenvalues[1].real:.4f}")

    # Визначаємо тип за власними числами
    l1, l2 = eigenvalues[0], eigenvalues[1]

    if abs(np.imag(l1)) > 1e-12:
        real_part = np.real(l1)
        if real_part < 0:
            print("Тип: Стійкий фокус (рух по спіралях до точки)")
        elif real_part > 0:
            print("Тип: Нестійкий фокус (рух по спіралях від точки)")
        else:
            print("Тип: Центр")
    else:
        l1, l2 = l1.real, l2.real
        if l1 < 0 and l2 < 0:
            print("Тип: Стійкий вузол")
        elif l1 > 0 and l2 > 0:
            print("Тип: Нестійкий вузол")
        elif (l1 > 0 > l2) or (l1 < 0 < l2):
            print("Тип: Сідло")
        else:
            print("Тип: Вироджений випадок")

# =====================================================================
# ГРАФІЧНА ЧАСТИНА
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# =====================================================================
# 2) Зобразити на фазовій площині напрямки векторного поля у точках
#    (m, n), де m = -6, ..., 6, n = -6, ..., 6
# =====================================================================

m_vals = np.arange(-6, 7)
n_vals = np.arange(-6, 7)
M, N = np.meshgrid(m_vals, n_vals)

# Обчислюємо вектори швидкостей у вузлах сітки
U_grid = M - N
V_grid = N ** 2 + M ** 2 - 2

# Нормуємо вектори (щоб відобразити лише напрямки)
norm = np.hypot(U_grid, V_grid)
norm[norm == 0] = 1  # Уникаємо ділення на нуль в особливих точках
U_dir = U_grid / norm
V_dir = V_grid / norm

ax1.quiver(M, N, U_dir, V_dir, color='teal', pivot='mid')
ax1.set_title(r'Пункт 2: Напрямки векторного поля на сітці $[-6, 6] \times [-6, 6]$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)

for data in points_data:
    pt_x, pt_y = float(data['pt'][0]), float(data['pt'][1])
    ax1.plot(pt_x, pt_y, 'ro', markersize=8)

# =====================================================================
# 4) Зобразити фазові траєкторії в околах положень рівноваги
# =====================================================================
# Створюємо щільнішу сітку для плавних кривих
x_dense = np.linspace(-4, 4, 400)
y_dense = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x_dense, y_dense)

U_dense = X - Y
V_dense = Y ** 2 + X ** 2 - 2

# Будуємо лінії струму (фазові траєкторії)
ax2.streamplot(X, Y, U_dense, V_dense, color=np.sqrt(U_dense ** 2 + V_dense ** 2),
               cmap='viridis', density=1.5, linewidth=1, arrowsize=1.2)

ax2.set_title(r'Пункт 4: Фазові траєкторії (загальний вигляд)')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)

# Позначаємо положення рівноваги
for idx, data in enumerate(points_data):
    pt_x, pt_y = float(data['pt'][0]), float(data['pt'][1])
    ax2.plot(pt_x, pt_y, 'ro', markersize=8, markeredgecolor='black',
             label=f'Рівновага {idx + 1}: ({pt_x:.0f}, {pt_y:.0f})')

ax2.legend(loc='best')

plt.tight_layout()
if '--save' in sys.argv:
    plt.savefig('task2.png', dpi=130, bbox_inches='tight')
else:
    plt.show()
