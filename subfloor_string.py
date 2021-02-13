import math

# Constants of the problem:
D = 15.0    # [m] length of the beam
δ = 0.01    # [m] slop in the middle
g = 9.81    # [m.s-2] gravity
λ = 1.5e-3  # [kg.m-1] density of rope

# Numerically-stable cosh(x) - a
def coshm1(x):
    if abs(x) > 0.5:
        return math.cosh(x) - 1.0

    res = 0.0
    for i in range(1, 10):
        res += (x ** (2 * i)) / math.factorial(2 * i)

    return res

# Catenary equation:
def y(a, x):
    return a * coshm1(x / a)

# Bisection solver
a0 = 0.0
a1 = 1e5

for _ in range(10):
    a = (a0 + a1) / 2.0

    slop = y(a, D / 2.0)
    print('a=%.3e [m] slop=%.3e [m]' % (a, slop))

    if slop > δ:
        a0 = a
    else:
        a1 = a

# Weight:
w = a * λ

print('w=%.3e [kg]' % (w))
