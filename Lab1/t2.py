"""
Лабораторна робота 1, Задача 2. Варіант 3.
y' = g(y) = y^2 - 3y + 2 = (y - 1)(y - 2)
"""
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------- АНАЛІТИЧНА ЧАСТИНА ----------------
print("=== Задача 2. y' = g(y) = y^2 - 3y + 2 = (y-1)(y-2) ===\n")

def g(y):
    return y ** 2 - 3 * y + 2

pts = [0, 1, -1, 2, -2, 3, -3, 4, -4]
print("1) Векторне поле: вектор з початком y0 і кінцем y0 + g(y0)")
for y0 in sorted(pts):
    print(f"   y0 = {y0:+d}:  g(y0) = {g(y0):3d}  ->  вектор [{y0}, {y0 + g(y0)}]"
          + ("  (стаціонарна точка, вектора немає)" if g(y0) == 0 else ""))

print("\n2) Нулі g: y1 = 1, y2 = 2 (положення рівноваги).")
print("   Зростання: y ∈ (-∞, 1) ∪ (2, +∞);  спадання: y ∈ (1, 2).")
print("   y'' = g'(y) g(y) = (2y - 3)(y - 1)(y - 2)  =>  перегин нестаціонарних кривих на y = 3/2.")

print("\n3) Відокремлюємо змінні:  dy/((y-1)(y-2)) = dx  =>  ln|(y-2)/(y-1)| = x + C")
print("   З умовою y(0) = y0:  (y-2)/(y-1) = C e^x,  C = (y0-2)/(y0-1)")
print("   y(x, y0) = (2 - C e^x) / (1 - C e^x)")
print("   • y0 = 1 або y0 = 2: y ≡ y0, I = (-∞, +∞)")
print("   • 1 < y0 < 2: C < 0, знаменник ≠ 0, I = (-∞, +∞)")
print("   • y0 < 1: C > 1, асимптота x* = ln((y0-1)/(y0-2)) < 0, I = (x*, +∞)")
print("   • y0 > 2: 0 < C < 1, асимптота x* = ln((y0-1)/(y0-2)) > 0, I = (-∞, x*)")

print("\n4) Графіки для y0 = 0 (< 1), y0 = 1.5 (між нулями), y0 = 3 (> 2)")

# 5) I = (-∞, 1) - зростання, J = (1, 2) - спадання, b = 1
# y(x, 1.5) - 1 = 1/(1 + e^x),  y(x, 0.5) - 1 = -1/(3e^x - 1)
# різниця = 4u / (3u^2 + 2u - 1), u = e^x;  = 1e-3  =>  3u^2 - 3998u - 1 = 0
u = (3998 + math.sqrt(3998 ** 2 + 12)) / 6
x_target = math.log(u)
print("\n5) I = (-∞, 1), J = (1, 2), b = 1:  порівнюємо y(x, 1.5) і y(x, 0.5)")
print("   |y(x,1.5) - y(x,0.5)| = 4e^x / (3e^{2x} + 2e^x - 1)")
print(f"   = 10^-3  =>  3u^2 - 3998u - 1 = 0,  u = e^x  =>  x = {x_target:.4f}")
print(f"   Для всіх x > {x_target:.4f} різниця менша за 10^-3.")


def y_exact(x, y0):
    x = np.asarray(x, dtype=float)
    if y0 in (1, 2):
        return np.full_like(x, float(y0))
    C = (y0 - 2) / (y0 - 1)
    den = 1 - C * np.exp(x)
    y = (2 - C * np.exp(x)) / den
    y[np.abs(den) < 0.02] = np.nan   # не з'єднуємо гілки через асимптоту
    return y


# ---------------- ГРАФІЧНА ЧАСТИНА ----------------
fig, axs = plt.subplots(1, 3, figsize=(19, 7))

# --- 1. Векторне поле на прямій y ---
ax = axs[0]
ax.axvline(0, color='k', lw=1)
for i, y0 in enumerate(sorted(pts)):
    ax.plot(0, y0, 'ko', ms=5)
    ax.annotate(f'{y0}', (0, y0), xytext=(-18, -4), textcoords='offset points', fontsize=9)
    gy = g(y0)
    if gy == 0:
        ax.plot(0, y0, 'o', ms=11, mfc='none', mec='red', mew=2)
        continue
    xo = 0.3 + 0.25 * i          # зсуваємо вектори вбік, щоб не накладались
    ax.annotate('', xy=(xo, y0 + gy), xytext=(xo, y0),
                arrowprops=dict(arrowstyle='->', color='teal', lw=2))
    ax.plot([0, xo], [y0, y0], ':', color='gray', lw=0.8)
    ax.text(xo + 0.05, y0 + gy, f'g={gy}', fontsize=8, va='center')
ax.plot([], [], 'o', mfc='none', mec='red', mew=2, label='Стаціонарні точки y=1, y=2')
ax.plot([], [], color='teal', lw=2, label='Вектор від y0 до y0+g(y0)')
ax.set_xlim(-0.6, 3)
ax.set_ylim(-5, 36)
ax.set_xticks([])
ax.set_ylabel('y')
ax.set_title('1. Векторне поле на осі y')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, axis='y', ls=':')

# --- 2-4. Монотонність, перегин, графіки y(x, y0) ---
ax = axs[1]
x = np.linspace(-4, 4, 2000)
ax.axhspan(-10, 1, color='lightgreen', alpha=0.2)
ax.axhspan(1, 2, color='lightcoral', alpha=0.2)
ax.axhspan(2, 10, color='lightgreen', alpha=0.2)
ax.plot(x, y_exact(x, 1), 'r-', lw=2, label='y ≡ 1 (рівновага)')
ax.plot(x, y_exact(x, 2), 'g-', lw=2, label='y ≡ 2 (рівновага)')
ax.axhline(1.5, color='purple', ls=':', lw=1.5, label='Перегин y = 3/2')
for y0, c in [(0, 'blue'), (1.5, 'darkorange'), (3, 'brown')]:
    ax.plot(x, y_exact(x, y0), color=c, lw=2.5, label=f'y(x, {y0})')
    xs = math.log((y0 - 1) / (y0 - 2)) if y0 not in (1.5,) else None
    if xs is not None:
        ax.axvline(xs, color=c, ls='--', lw=1, alpha=0.7)
        ax.text(xs, -2.8, f'x*={xs:.2f}', color=c, fontsize=8, ha='center')
for y0 in (-1, 0.5, 1.2, 1.8, 2.5):
    ax.plot(x, y_exact(x, y0), color='gray', lw=0.8, alpha=0.6)
ax.set_xlim(-4, 4)
ax.set_ylim(-3, 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2–4. Монотонність, перегин, графіки y(x, y0)\n(зелене: зростання, червоне: спадання)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, ls=':')

# --- 5. Збіжність y(x, b±1/2), b = 1 ---
ax = axs[2]
xc = np.linspace(0, 9, 800)
ax.plot(xc, y_exact(xc, 1.5), 'b-', lw=2, label='y(x, 1.5)')
ax.plot(xc, y_exact(xc, 0.5), 'm-', lw=2, label='y(x, 0.5)')
ax.axhline(1, color='r', ls='--', label='y = 1 (b)')
ax.axvline(x_target, color='orange', ls='--', label=f'x ≈ {x_target:.4f}')
ax.annotate(r'$|y(x,1.5)-y(x,0.5)| < 10^{-3}$', xy=(x_target, 1.0),
            xytext=(x_target - 5, 1.35), arrowprops=dict(arrowstyle='->'))
ax.set_ylim(0.4, 1.6)
ax.set_xlabel('x')
ax.set_title('5. Збіжність розв’язків до y = 1')
ax.legend(loc='lower right')
ax.grid(True, ls=':')

fig.suptitle("Задача 2 (варіант 3):  y' = y² − 3y + 2", fontsize=14)
plt.tight_layout()

if '--save' in sys.argv:
    plt.savefig('task2.png', dpi=150, bbox_inches='tight')
else:
    plt.show()
