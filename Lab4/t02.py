import sys
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def solve_task_2():
    print("=" * 40)
    print("ЗАДАЧА 2: Теорема Пеано та інтегральна воронка")
    print("=" * 40)

    # Варіант 3: y' = 4(y-1)^(3/4), y(0) = 1,
    # K = {(x, y): y >= 1, x^2 + y^2 - 2y <= 3}  (півкруг: x^2 + (y-1)^2 <= 4, y >= 1)

    # 1) Умови теореми Пеано та проміжок існування
    print("1) Перевірка умов теореми Пеано:")
    print("   g(y) = 4*(y-1)^(3/4) неперервна при y >= 1, тобто в усій області K.")
    print("   Отже, умови теореми Пеано виконані, розв'язок існує.")
    # Прямокутник П = {|x| <= a, |y-1| <= b}, що лежить у колі x^2+(y-1)^2 = 4: a^2 + b^2 <= 4.
    a, b = 1, sp.sqrt(3)
    M = 4 * b ** sp.Rational(3, 4)  # max g на |y-1| <= b
    h = min(float(a), float(b / M))
    print(f"   Беремо a = {a}, b = sqrt(3) (кут (1; 1+sqrt(3)) лежить на колі).")
    print(f"   M = 4*b^(3/4) = {float(M):.4f},  h = min(a, b/M) = {h:.4f}")
    print(f"   Теорема гарантує існування розв'язку на [-{h:.4f}, {h:.4f}].")
    print("-" * 40)

    # 2) Знаходження розв'язків
    print("2) Знаходимо розв'язки (u = y - 1):  u' = 4*u^(3/4),  u(0) = 0")
    print("   - Очевидний розв'язок: u = 0, тобто y = 1")
    print("   - Розділяємо змінні: du / u^(3/4) = 4 dx  =>  4*u^(1/4) = 4x + C")
    print("     Із u(0) = 0: C = 0, отже u = x^4,  y = 1 + x^4  (при x >= 0, бо u^(1/4) >= 0)")
    # Перевірка підстановкою
    xp = sp.symbols('x', positive=True)
    check = sp.simplify(sp.diff(xp ** 4, xp) - 4 * (xp ** 4) ** sp.Rational(3, 4))
    print(f"   Перевірка: (x^4)' - 4*(x^4)^(3/4) = {check} (при x >= 0)")
    print("   При x < 0 розв'язок u >= 0 неспадний (u' >= 0) і u(0) = 0, тому u = 0.")
    print("   Усі розв'язки: y_c(x) = 1 при x <= c;  y_c(x) = 1 + (x - c)^4 при x >= c,  c >= 0.")
    print("   Максимальний розв'язок:  y_max(x) = 1 при x <= 0;  1 + x^4 при x > 0")
    print("   Мінімальний розв'язок:   y_min(x) = 1 (усюди)")

    # Точка виходу y_max з області K: x^2 + x^8 = 4
    x_star = brentq(lambda s: s ** 2 + s ** 8 - 4, 0, 2)
    print(f"   y_max виходить на межу K при x^2 + x^8 = 4:  x* ≈ {x_star:.4f}")
    print(f"   y_min = 1 доходить до межі K при x = ±2.")
    print("-" * 40)

    # 3) Графічне зображення
    print("3) Побудова геометричної фігури G...")

    xs = np.linspace(-2, 2, 2000)
    circ_top = 1 + np.sqrt(np.clip(4 - xs ** 2, 0, None))  # верхня межа K
    y_min = np.ones_like(xs)
    y_max = np.where(xs > 0, 1 + xs ** 4, 1.0)
    y_max_in_K = np.minimum(y_max, circ_top)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Межа області K: півколо та відрізок y = 1
    th = np.linspace(0, np.pi, 300)
    ax.plot(2 * np.cos(th), 1 + 2 * np.sin(th), 'k--', label='Межа області $K$')
    ax.plot([-2, 2], [1, 1], 'k--')

    # Фігура G
    ax.fill_between(xs, y_min, y_max_in_K, color='skyblue', alpha=0.5,
                    label='Фігура $G$ (інтегральна воронка)')

    # Проміжні розв'язки y_c
    for c, col in [(0.4, 'orange'), (0.8, 'purple')]:
        xc = np.linspace(c, 2, 800)
        yc = 1 + (xc - c) ** 4
        mask = xc ** 2 + (yc - 1) ** 2 <= 4
        ax.plot(xc[mask], yc[mask], color=col, linewidth=1.5, label=f'$y_c(x)$, $c={c}$')

    # Максимальний та мінімальний розв'язки
    xm = np.linspace(0, x_star, 400)
    ax.plot(np.linspace(-2, 0, 50), np.ones(50), 'b-', linewidth=2.5)
    ax.plot(xm, 1 + xm ** 4, 'b-', linewidth=2.5, label='$y_{max}(x)$')
    ax.plot(xs, y_min, 'r-', linewidth=2, label='$y_{min}(x)=1$')
    ax.plot([0], [1], 'ko', markersize=6)
    ax.plot([x_star], [1 + x_star ** 4], 'bo', markersize=6)

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(0.6, 3.3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, linestyle=':', linewidth=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(r"Геометрична фігура $G$: $y' = 4(y-1)^{3/4}$, $y(0)=1$")
    ax.legend(loc='center left')

    if '--save' in sys.argv:
        plt.savefig('task2.png', dpi=130, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    solve_task_2()
