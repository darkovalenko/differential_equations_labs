import sympy as sp
import math


def solve_task_1():
    print("=" * 40)
    print("ЗАДАЧА 1: Теорема Пікара та послідовні наближення")
    print("=" * 40)

    x, y, t, h1 = sp.symbols('x y t h1')

    # Варіант 3
    f = x ** 2 - y ** 2
    x0, y0 = 2, 0
    a, b = 1, 1

    print(f"Функція: f(x, y) = {f}")
    print(f"Область П: |x - {x0}| <= {a}, |y - {y0}| <= {b}")
    print("-" * 40)

    # 1) Знайти сталу Ліпшиця в П
    df_dy = sp.diff(f, y)
    print(f"1) Частинна похідна df/dy = {df_dy}")
    # Максимум |df/dy| = |2y| на проміжку |y| <= 1 дорівнює 2
    L = 2
    print(f"   Стала Ліпшиця L = {L}")

    # Умови теореми Пікара: f та df/dy неперервні в П.
    # M = max |x^2 - y^2| у П: x in [1, 3], y in [-1, 1].
    # x^2 - y^2 змінюється від 1 - 1 = 0 до 9 - 0 = 9, тому M = 9
    M = 3 ** 2 - 0 ** 2
    print(f"   Максимум модуля функції в П: M = {M}")

    # 2) Проміжок збіжності [x0 - h, x0 + h]
    h = min(a, sp.Rational(b, M))
    print(f"2) h = min(a, b/M) = min({a}, {b}/{M}) = {h} ≈ {float(h):.4f}")
    print(f"   Теорема гарантує збіжність на проміжку: [{x0} - {h}, {x0} + {h}]")
    print("-" * 40)

    # 3) Третє пікарівське наближення
    print("3) Пікарівські наближення:")
    y_approx = [sp.sympify(y0)]  # y0(x) = 0

    for i in range(1, 4):
        # y_n(x) = y0 + \int_{x0}^{x} f(t, y_{n-1}(t)) dt
        integrand = f.subs({x: t, y: y_approx[-1].subs(x, t)})
        y_next = y0 + sp.integrate(integrand, (t, x0, x))
        y_approx.append(sp.expand(y_next))
        print(f"   y_{i}(x) = {y_approx[-1]}")

    print("-" * 40)

    # 4) Оцінка похибки
    n = 3
    # Формула оцінки: Delta_n <= M * L**(n-1) * h**n / n!
    delta_3 = (M * L ** (n - 1) * h ** n) / math.factorial(n)
    print(f"4) Оцінка похибки третього наближення Delta_3 <= {delta_3} ≈ {float(delta_3):.6f}")

    # При якому n виконується нерівність Delta_n <= 10^-3
    target_error = 1e-3
    for n_test in range(1, 10):
        delta_n = (M * L ** (n_test - 1) * h ** n_test) / math.factorial(n_test)
        print(f"   Delta_{n_test} <= {float(delta_n):.6f}")
        if delta_n <= target_error:
            print(f"   Нерівність Delta_n <= 10^-3 виконується починаючи з n = {n_test}")
            break

    print("-" * 40)

    # 5) Знайти h1 для похибки 10^-6
    # (M * L**2 * h1**3) / 3! <= 10^-6
    print("5) Шукаємо h1 in (0, h) таке, що Delta_{h1, 3} <= 10^-6")
    expr = (M * L ** 2 * h1 ** 3) / math.factorial(3)

    solutions = sp.solve(expr - sp.Rational(1, 10 ** 6), h1)

    # Вибираємо дійсний позитивний корінь
    h1_val = None
    for sol in solutions:
        if sol.is_real and sol > 0:
            h1_val = sol.evalf()
            break

    print(f"   h1 <= {h1_val:.6f}")


if __name__ == "__main__":
    solve_task_1()
