import sympy as sp

def verify():
    k, t, n = sp.symbols('k t n', real=True, positive=True)
    
    # Parametric equations
    x = sp.exp(-k*t) * sp.cos(t)
    y = sp.exp(-k*t) * sp.sin(t)
    
    # Derivatives
    dx_dt = sp.diff(x, t)
    dy_dt = sp.diff(y, t)
    
    # Speed
    speed = sp.simplify(sp.sqrt(dx_dt**2 + dy_dt**2))
    print("Speed:", speed)
    
    # Arc length L
    L = sp.integrate(speed, (t, 0, sp.oo))
    print("Arc length L:", L)
    
    # Area A_n
    # A = 1/2 \int r^2 d\theta
    # r = exp(-kt), theta = t
    r2 = sp.exp(-2*k*t)
    A_n = sp.integrate(1/2 * r2, (t, 2*sp.pi*(n-1), 2*sp.pi*n))
    print("Area A_n:", A_n)
    
    # Common ratio
    A_n_plus_1 = A_n.subs(n, n+1)
    ratio = sp.simplify(A_n_plus_1 / A_n)
    print("Common ratio:", ratio)
    
    # Total area S
    S = sp.integrate(1/2 * r2, (t, 0, sp.oo))
    print("Total area S:", S)
    
    # If L = 5*S, find k
    eq = sp.Eq(L, 5*S)
    k_sol = sp.solve(eq, k)
    print("k_sol for L = 5*S:", k_sol)
    
if __name__ == "__main__":
    verify()
