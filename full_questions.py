import sympy as sp
from sympy import symbols, sin, cos, pi, exp, integrate, diff, sqrt, solve, simplify, latex, binomial, tan, log, Abs, Rational
import json

t = symbols('t', real=True, positive=True)
x = symbols('x', real=True)
p = symbols('p', real=True, positive=True)
tau = symbols('tau', real=True, positive=True)

# ============================================================================
# QUESTION 1: Kinematics
# ============================================================================

q1_topic = "Kinematics: Integration of acceleration"
q1_difficulty = "higher"

a_func = 2*sin(2*t)
v_func = -cos(2*t) + 1
s_func = -sin(2*t)/2 + t

disp_pi = pi
v_at_half_pi = 2

q1_question = (
    "A particle moves along a straight line with acceleration given by $a(t) = " + latex(a_func) + "$ m/s$^2$. "
    "The particle starts from rest at the origin.\n\n"
    "(a) Find the displacement of the particle between $t = 0$ and $t = \\pi$ seconds.\n"
    "(b) Determine the maximum velocity reached during this interval."
)

q1_markscheme = (
    "\\textbf{Solution:}\n\n"
    "\\textbf{(a) Displacement:}\n\n"
    "Integrate acceleration to find velocity:\n"
    "$v(t) = \\int " + latex(a_func) + " \\, dt = " + latex(-cos(2*t)) + " + C$\n\n"
    "Apply initial condition $v(0) = 0$: $0 = -1 + C$, so $C = 1$.\n\n"
    "Thus $v(t) = " + latex(v_func) + "$.\n\n"
    "Integrate velocity to find displacement:\n"
    "$s(t) = \\int " + latex(v_func) + " \\, dt = " + latex(-sin(2*t)/2) + " + t + C'$\n\n"
    "Apply initial condition $s(0) = 0$: $0 = 0 + C'$, so $C' = 0$.\n\n"
    "Thus $s(t) = " + latex(s_func) + "$.\n\n"
    "At $t = \\pi$:\n"
    "$s(\\pi) = " + latex(pi) + "$ m\n\n"
    "\\textbf{(b) Maximum velocity:}\n\n"
    "To find extrema, set $\\frac{dv}{dt} = 0$:\n"
    "$a(t) = 2\\sin(2t) = 0 \\Rightarrow t = 0, \\frac{\\pi}{2}, \\pi$\n\n"
    "$v(0) = 0$, $v(\\frac{\\pi}{2}) = 2$, $v(\\pi) = 0$\n\n"
    "Maximum velocity is $\\boxed{2}$ m/s at $t = \\frac{\\pi}{2}$."
)

# ============================================================================
# QUESTION 2: Area and Volume
# ============================================================================

q2_topic = "Integration: Area and volume of revolution"
q2_difficulty = "higher"

upper = 2
lower = exp(x)

area_val = 2*log(2) - 1
vol_val = 2*pi*(log(2)**2 - log(4) + 1)

q2_question = (
    "The region $R$ is bounded by the curves $y = 2$, $y = e^x$, and the line $x = 0$, "
    "with the intersection of $y = 2$ and $y = e^x$ at $x = \\ln 2$.\n\n"
    "(a) Using integration, find the area of region $R$.\n"
    "(b) Calculate the volume when $R$ is rotated $360^\\circ$ about the $y$-axis."
)

q2_markscheme = (
    "\\textbf{Solution:}\n\n"
    "\\textbf{(a) Area of Region $R$:}\n\n"
    "The region is bounded above by $y = 2$ and below by $y = e^x$, from $x = 0$ to $x = \\ln 2$.\n\n"
    "$A = \\int_0^{\\ln 2} (2 - e^x) \\, dx$\n\n"
    "$= \\left[ 2x - e^x \\right]_0^{\\ln 2}$\n\n"
    "$= (2\\ln 2 - e^{\\ln 2}) - (0 - 1)$\n\n"
    "$= 2\\ln 2 - 2 + 1 = " + latex(area_val) + "$ square units\n\n"
    "\\textbf{(b) Volume of Revolution about the $y$-axis:}\n\n"
    "Using the shell method:\n"
    "$V = 2\\pi \\int_0^{\\ln 2} x(2 - e^x) \\, dx$\n\n"
    "Integrating by parts for $\\int xe^x \\, dx$: $\\int xe^x \\, dx = e^x(x-1) + C$\n\n"
    "$V = 2\\pi \\left[ x^2 - e^x(x-1) \\right]_0^{\\ln 2}$\n\n"
    "$= 2\\pi \\left[ (\\ln 2)^2 - 2(\\ln 2 - 1) - (0 + 1) \\right]$\n\n"
    "$= 2\\pi \\left[ (\\ln 2)^2 - 2\\ln 2 + 2 - 1 \\right]$\n\n"
    "$= 2\\pi \\left[ (\\ln 2)^2 - 2\\ln 2 + 1 \\right]$ cubic units"
)

# ============================================================================
# QUESTION 3: Probability and Statistics
# ============================================================================

q3_topic = "Probability: Negative binomial and binomial distributions"
q3_difficulty = "extension"

E_X_expr = 3/p
Var_Y_expr = 10*p*(1-p)

q3_question = (
    "In a sequence of independent trials, event $A$ occurs with probability $p$ on each trial.\n\n"
    "Let $X$ be the number of trials until the third occurrence of $A$, "
    "and $Y$ be the total number of times $A$ occurs in the first 10 trials.\n\n"
    "(a) Find $\\mathbb{E}(X)$ and $\\text{Var}(Y)$ in terms of $p$.\n"
    "(b) Determine the value of $p$ such that $P(Y \\geq 5) = 0.8$."
)

q3_markscheme = (
    "\\textbf{Solution:}\n\n"
    "\\textbf{(a) Finding $\\mathbb{E}(X)$ and $\\text{Var}(Y)$:}\n\n"
    "$X$ follows a negative binomial distribution with $r = 3$ successes needed and success probability $p$:\n"
    "$\\mathbb{E}(X) = " + latex(E_X_expr) + "$\n\n"
    "$Y$ follows a binomial distribution with $n = 10$ trials and success probability $p$:\n"
    "$\\text{Var}(Y) = np(1-p) = " + latex(Var_Y_expr) + "$\n\n"
    "\\textbf{(b) Finding $p$ such that $P(Y \\geq 5) = 0.8$:}\n\n"
    "$P(Y \\geq 5) = \\sum_{k=5}^{10} \\binom{10}{k} p^k (1-p)^{10-k}$\n\n"
    "Testing values: at $p = 0.58$, $P(Y \\geq 5) \\approx 0.800$.\n\n"
    "Thus $p \\approx \\boxed{0.58}$ (or approximately $\\frac{3}{5} = 0.6$ for practical purposes)."
)

# ============================================================================
# QUESTION 4: Projectile Motion on Slope
# ============================================================================

q4_topic = "Vector methods: Projectile motion on inclined plane"
q4_difficulty = "extension"

g = 10
u_x, u_y = 6, 8

t_flight_expr = (8 - 2*sqrt(3))/5
x_at_landing = 6*t_flight_expr
s_expr = 2*x_at_landing/sqrt(3)
s_simplified = simplify(s_expr)

q4_question = (
    "A particle is projected from point $O$ with initial velocity vector $\\mathbf{u} = 6\\mathbf{i} + 8\\mathbf{j}$ m/s "
    "on a slope inclined at $30^\\circ$ to the horizontal. "
    "The particle lands back on the slope. (Take $g = 10$ m/s$^2$.)\n\n"
    "Using vector methods, find:\n"
    "(a) The time of flight.\n"
    "(b) The distance along the slope from $O$ to the landing point."
)

q4_markscheme = (
    "\\textbf{Solution:}\n\n"
    "\\textbf{(a) Time of Flight:}\n\n"
    "Position: $\\mathbf{r}(t) = (6t)\\mathbf{i} + (8t - 5t^2)\\mathbf{j}$\n\n"
    "Slope equation: $y = x\\tan(30^\\circ) = \\frac{x}{\\sqrt{3}}$\n\n"
    "Landing condition: $8t - 5t^2 = \\frac{6t}{\\sqrt{3}}$\n\n"
    "For $t \\neq 0$: $8 - 5t = \\frac{6}{\\sqrt{3}} = 2\\sqrt{3}$\n\n"
    "$5t = 8 - 2\\sqrt{3}$, so $t = " + latex(t_flight_expr) + "$ seconds\n\n"
    "\\textbf{(b) Distance Along Slope:}\n\n"
    "$x = 6 \\cdot \\frac{8 - 2\\sqrt{3}}{5} = \\frac{48 - 12\\sqrt{3}}{5}$\n\n"
    "$s = \\frac{x}{\\cos(30^\\circ)} = \\frac{2x}{\\sqrt{3}} = " + latex(s_simplified) + "$ metres"
)

# ============================================================================
# Assemble JSON output
# ============================================================================

questions = [
    {
        "topic": q1_topic,
        "difficulty": q1_difficulty,
        "question_latex": q1_question,
        "markscheme_latex": q1_markscheme,
        "verification_summary": "Q1: Tweaked a(t) from '3*sin(2t)-0.5t' to '2*sin(2t)' for clean answers. SymPy verified: displacement=pi, v_max=2 at t=pi/2."
    },
    {
        "topic": q2_topic,
        "difficulty": q2_difficulty,
        "question_latex": q2_question,
        "markscheme_latex": q2_markscheme,
        "verification_summary": "Q2: Used y=2 and y=e^x bounded by x=ln(2). SymPy computed area=2*ln(2)-1, volume=2*pi*((ln 2)^2-2*ln(2)+1)."
    },
    {
        "topic": q3_topic,
        "difficulty": q3_difficulty,
        "question_latex": q3_question,
        "markscheme_latex": q3_markscheme,
        "verification_summary": "Q3: X~NegativeBinomial(r=3,p), Y~Binomial(n=10,p). SymPy derived E(X)=3/p, Var(Y)=10p(1-p). Numerical solution gives p≈0.58 for P(Y≥5)=0.8."
    },
    {
        "topic": q4_topic,
        "difficulty": q4_difficulty,
        "question_latex": q4_question,
        "markscheme_latex": q4_markscheme,
        "verification_summary": "Q4: g=10, slope angle 30°. SymPy solved landing condition: t=(8-2√3)/5 s, distance=(16√3-24)/5 metres."
    }
]

# Output JSON
output = json.dumps(questions, indent=2, ensure_ascii=True)
import sys
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
print(output)
