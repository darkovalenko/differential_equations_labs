"""
Лабораторна робота 1, Задача 1. Варіант 3.
y' = f(x, y) = y^2 - y + x
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp

# ---------------- АНАЛІТИЧНА ЧАСТИНА ----------------
print("=== Задача 1. y' = y^2 - y + x ===\n")
print("f(x, y) = y^2 - y + x,  f_x = 1,  f_y = 2y - 1")
print("y'' = f_x + f_y * f = 1 + (2y - 1)(y^2 - y + x)\n")

print("1) Поле напрямків: у точці (m, n/2) нахил k = (n/2)^2 - n/2 + m,  m, n = -6..6\n")

print("2) Точки максимуму: y' = 0, y'' < 0.")
print("   y' = 0  =>  x = y - y^2.  На цій кривій y'' = 1 + (2y-1)*0 = 1 > 0.")
print("   Отже, усі точки екстремуму - мінімуми; множина точок максимуму порожня (∅).\n")

print("3) Ізокліни f = k:  x = k + y - y^2,  k = ±1/2, ±1, ±3/2")
print("   (параболи з віссю симетрії y = 1/2, гілками вліво)\n")

print("4) Зростання: y' > 0  <=>  x > y - y^2;   спадання: x < y - y^2\n")

print("5) Точки перегину: y'' = 0  <=>  (2y - 1)(y^2 - y + x) = -1")
print("   =>  x = y - y^2 - 1/(2y - 1),  y ≠ 1/2  (дві гілки, асимптота y = 1/2)\n")

print("6) Опуклість вгору: y'' < 0  <=>  (2y - 1)(y^2 - y + x) < -1\n")

print("7) Наближені інтегральні криві через (0,-2), (0,0), (0,1)")


# ---------------- ГРАФІЧНА ЧАСТИНА ----------------
def f(x, y):
    return y ** 2 - y + x


def ypp(x, y):
    return 1 + (2 * y - 1) * f(x, y)


XMIN, XMAX, YMIN, YMAX = -6.5, 6.5, -3.5, 3.5
fig, ax = plt.subplots(figsize=(14, 10))

# Області опуклості вгору (штрихування) - малюємо першими, щоб були на фоні
xg = np.linspace(XMIN, XMAX, 500)
yg = np.linspace(YMIN, YMAX, 500)
Xg, Yg = np.meshgrid(xg, yg)
ax.contourf(Xg, Yg, ypp(Xg, Yg), levels=[-1e9, 0], colors='none', hatches=['\\\\'])

# 4) Області зростання / спадання
y_line = np.linspace(YMIN, YMAX, 600)
x_null = y_line - y_line ** 2
ax.fill_betweenx(y_line, x_null, XMAX, color='lightgreen', alpha=0.25)
ax.fill_betweenx(y_line, XMIN, x_null, color='lightcoral', alpha=0.25)

# 1) Поле напрямків у точках (m, n/2)
m = np.arange(-6, 7)
n = np.arange(-6, 7)
M, N = np.meshgrid(m, n / 2)
K = f(M, N)
L = 0.35  # половина довжини штриха (всі штрихи однакової довжини)
dx = L / np.sqrt(1 + K ** 2)
dy = K * dx
for xi, yi, ddx, ddy in zip(M.ravel(), N.ravel(), dx.ravel(), dy.ravel()):
    ax.plot([xi - ddx, xi + ddx], [yi - ddy, yi + ddy], color='dimgray', lw=1.3)
ax.plot(M, N, 'k.', ms=2)

# 3) Ізокліни
iso_colors = {-1.5: '#1f3b99', -1: '#3a6fd8', -0.5: '#86a8ef',
              0.5: '#f2b36b', 1: '#e57f1f', 1.5: '#a84f00'}
for k, c in iso_colors.items():
    ax.plot(k + y_line - y_line ** 2, y_line, '--', color=c, lw=1.4, label=f'Ізокліна k = {k:+g}')

# 2) Нуль-ізокліна (лінія екстремумів - тільки мінімуми)
ax.plot(x_null, y_line, 'k-', lw=2.5, label="Нуль-ізокліна y'=0 (точки мінімуму)")

# 5) Точки перегину
for y_br in (np.linspace(YMIN, 0.49, 400), np.linspace(0.51, YMAX, 400)):
    ax.plot(y_br - y_br ** 2 - 1 / (2 * y_br - 1), y_br, 'r-', lw=2.5)
ax.plot([], [], 'r-', lw=2.5, label="Точки перегину (y''=0)")


# 7) Інтегральні криві
def rhs(x, y):
    return [f(x, y[0])]


def escape(x, y):
    return abs(y[0]) - 8


escape.terminal = True

curve_colors = ['blue', 'darkviolet', 'darkorange']
for (x0, y0), c in zip([(0, -2), (0, 0), (0, 1)], curve_colors):
    xs, ys = [], []
    for x_end in (XMIN, XMAX):
        sol = solve_ivp(rhs, [x0, x_end], [y0], events=escape, max_step=0.02, rtol=1e-8)
        xs.append(sol.t)
        ys.append(sol.y[0])
    x_all = np.concatenate((xs[0][::-1], xs[1][1:]))
    y_all = np.concatenate((ys[0][::-1], ys[1][1:]))
    ax.plot(x_all, y_all, color=c, lw=3, label=f'Інтегральна крива через ({x0}, {y0})')
    ax.plot(x0, y0, 'ko', ms=7, zorder=5)

# Легенда для областей
handles, labels = ax.get_legend_handles_labels()
handles += [mpatches.Patch(color='lightgreen', alpha=0.5, label="Зростання (y' > 0)"),
            mpatches.Patch(color='lightcoral', alpha=0.5, label="Спадання (y' < 0)"),
            mpatches.Patch(facecolor='none', edgecolor='k', hatch='\\\\', label="Опуклість вгору (y'' < 0)"),
            plt.Line2D([], [], ls='none', label='Точки максимуму: ∅')]
ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10)

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title("Задача 1 (варіант 3):  y' = y² − y + x", fontsize=14)
ax.grid(True, ls=':')
plt.tight_layout()

if '--save' in sys.argv:
    plt.savefig('task1.png', dpi=150, bbox_inches='tight')
else:
    plt.show()
