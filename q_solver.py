import sympy as sp
import json
from sympy import symbols, sin, cos, pi, exp, integrate, diff, sqrt, solve, simplify, latex, binomial, tan, log

t = symbols('t', real=True, positive=True)
x = symbols('x', real=True)
p = symbols('p', real=True, positive=True)
tau = symbols('tau', real=True, positive=True)

print("Q1: Kinematics")
a_func = 2*sin(2*t)
v_func = -cos(2*t) + 1
s_func = -sin(2*t)/2 + t
disp_at_pi = s_func.subs(t, pi)
v_max_val = 2
print(f"Displacement: {latex(disp_at_pi)}, v_max: {v_max_val}")

print("\nQ2: Area and Volume")
upper = 2
lower = exp(x)
area = integrate(upper - lower, (x, 0, log(2)))
vol = 2*pi*integrate(x * (upper - lower), (x, 0, log(2)))
vol_simp = simplify(vol)
print(f"Area: {latex(area)}, Volume: {latex(vol_simp)}")

print("\nQ3: Probability")
E_X = 3/p
Var_Y = 10*p*(1-p)
print(f"E(X): {latex(E_X)}, Var(Y): {latex(Var_Y)}")

print("\nQ4: Projectile Motion")
g_val = 10
x_pos = 6*tau
y_pos = 8*tau - 5*tau**2
landing_eq = y_pos - x_pos/sqrt(3)
times = solve(landing_eq, tau)
t_flight = [t for t in times if t != 0][0]
t_flight_simp = simplify(t_flight)
x_land = x_pos.subs(tau, t_flight_simp)
dist = simplify(x_land / cos(pi/6))
print(f"Time: {latex(t_flight_simp)}, Distance: {latex(dist)}")
