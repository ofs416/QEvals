import sympy as sp
from sympy import preview
import os
import subprocess

def solve_and_generate():
    # Set up symbols
    t, V, k, C = sp.symbols('t V k C', real=True, positive=True)
    h = sp.symbols('h', real=True, positive=True)
    
    # Computations matching the refiner's output
    V0 = 144 * sp.pi
    sqrt_V0 = sp.sqrt(V0)
    
    A_val = sqrt_V0
    B_val = A_val / (4 * sp.pi)
    k_val = B_val * 2 * sp.pi / 3
    
    sqrt_V6 = sp.simplify(A_val - B_val * (sp.pi * 6 / 6 - sp.sin(sp.pi * 6 / 6)))
    V6 = sp.simplify(sqrt_V6**2)
    
    h6 = sp.cbrt(V6 * 9 / sp.pi)
    
    V_expr = sp.pi / 9 * h**3
    dV_dh = sp.diff(V_expr, h)
    dV_dh_at_6 = dV_dh.subs(h, h6)
    
    dV_dt_expr = -k_val * sp.sqrt(V6) * sp.sin(sp.pi * 6 / 12)**2
    
    dh_dt = sp.simplify(dV_dt_expr / dV_dh_at_6)

    # Use f-strings and sp.latex() to dynamically build the LaTeX document
    latex_document = fr"""
\section*{{A-Level Mathematics Question}}

A large conical tank, positioned with its vertex pointing downwards and an open circular top, is initially full of a chemical solution. The solution is allowed to drain through a small automated valve at the vertex. The volume $V \text{{ m}}^3$ of solution remaining in the tank at time $t$ minutes is modelled by the differential equation:

$$ \frac{{\text{{d}}V}}{{\text{{d}}t}} = -k \sqrt{{V}} \sin^2\left(\frac{{\pi t}}{{12}}\right) $$

where $k$ is a positive constant. Initially, the tank holds ${sp.latex(V0)} \text{{ m}}^3$ of solution.

\vspace{{0.5cm}}

\noindent \textbf{{(a)}} By solving the differential equation, show that the volume of solution in the tank at time $t$ can be expressed in the form $\sqrt{{V}} = A - B \left( \frac{{\pi t}}{{6}} - \sin\left(\frac{{\pi t}}{{6}}\right) \right)$, where $A$ and $B$ are constants to be found in terms of $k$ and $\pi$. \hfill \textbf{{[7 marks]}}

\vspace{{0.5cm}}

\noindent \textbf{{(b)}} The valve mechanism is designed such that the flow of solution temporarily stops whenever $t$ is a multiple of $12$. Given that the tank first becomes completely empty at exactly $t = 24$ minutes, find the exact value of $k$. \hfill \textbf{{[4 marks]}}

\vspace{{0.5cm}}

\noindent \textbf{{(c)}} The cone has a semi-vertical angle of $30^\circ$. Given that the surface of the solution always forms a horizontal circle, use the chain rule to find the exact rate of change of the depth of the solution, $\frac{{\text{{d}}h}}{{\text{{d}}t}}$, at the instant when $t = 6$. \hfill \textbf{{[8 marks]}}

\newpage

\section*{{Mark Scheme}}

\subsection*{{(a)}}
\begin{{itemize}}
    \item Separates variables correctly: $\int V^{{-\frac{{1}}{{2}}}} \, \text{{d}}V = \int -k \sin^2\left(\frac{{\pi t}}{{12}}\right) \, \text{{d}}t$ \hfill \textbf{{[M1]}}
    \item Integrates LHS correctly to obtain $2\sqrt{{V}}$ \hfill \textbf{{[A1]}}
    \item Uses the double angle formula for the RHS: $\sin^2\left(\frac{{\pi t}}{{12}}\right) = \frac{{1 - \cos\left(\frac{{\pi t}}{{6}}\right)}}{{2}}$ \hfill \textbf{{[M1]}}
    \item Integrates RHS correctly to obtain: $-\frac{{k}}{{2}} \left( t - \frac{{6}}{{\pi}} \sin\left(\frac{{\pi t}}{{6}}\right) \right) + C$ \hfill \textbf{{[A1]}}
    \item Equates both sides and rearranges: $2\sqrt{{V}} = -\frac{{k}}{{2}} \left( t - \frac{{6}}{{\pi}} \sin\left(\frac{{\pi t}}{{6}}\right) \right) + C \implies \sqrt{{V}} = \frac{{C}}{{2}} - \frac{{3k}}{{2\pi}}\left(\frac{{\pi t}}{{6}} - \sin\left(\frac{{\pi t}}{{6}}\right)\right)$ \hfill \textbf{{[M1]}}
    \item Uses boundary conditions ($t=0, V={sp.latex(V0)}$): $\sqrt{{{sp.latex(V0)}}} = \frac{{C}}{{2}} - 0 \implies \frac{{C}}{{2}} = {sp.latex(A_val)}$ \hfill \textbf{{[M1]}}
    \item Completes proof clearly to show the required form explicitly stating $A = {sp.latex(A_val)}$ and $B = \frac{{3k}}{{2\pi}}$. \hfill \textbf{{[A1]}}
\end{{itemize}}

\subsection*{{(b)}}
\begin{{itemize}}
    \item Uses boundary condition ($t=24, V=0$): $0 = {sp.latex(A_val)} - \frac{{3k}}{{2\pi}} \left( \frac{{24\pi}}{{6}} - \sin\left(\frac{{24\pi}}{{6}}\right) \right)$ \hfill \textbf{{[M1]}}
    \item Evaluates the trigonometric term correctly: $\sin(4\pi) = 0$ \hfill \textbf{{[B1]}}
    \item Simplifies the equation: $0 = {sp.latex(A_val)} - \frac{{3k}}{{2\pi}} (4\pi) \implies {sp.latex(A_val)} = 6k$ \hfill \textbf{{[M1]}}
    \item Solves for $k$ to yield exact value: $k = {sp.latex(k_val)}$ \hfill \textbf{{[A1]}}
\end{{itemize}}

\subsection*{{(c)}}
\begin{{itemize}}
    \item Uses $V = \frac{{1}}{{3}}\pi r^2 h$ and relates $r$ to $h$: $\tan(30^\circ) = \frac{{r}}{{h}} \implies r = \frac{{h}}{{\sqrt{{3}}}}$ \hfill \textbf{{[M1]}}
    \item Expresses $V$ in terms of $h$ only: $V = \frac{{1}}{{9}}\pi h^3$ \hfill \textbf{{[A1]}}
    \item Differentiates volume with respect to height: $\frac{{\text{{d}}V}}{{\text{{d}}h}} = \frac{{1}}{{3}}\pi h^2$ \hfill \textbf{{[B1]}}
    \item Finds $\sqrt{{V}}$ at $t=6$: $\sqrt{{V}} = {sp.latex(A_val)} - \frac{{3({sp.latex(k_val)})}}{{2\pi}} \left( \pi - \sin(\pi) \right) = {sp.latex(sqrt_V6)}$ (so $V = {sp.latex(V6)}$) \hfill \textbf{{[M1]}}
    \item Finds $h$ at $t=6$: ${sp.latex(V6)} = \frac{{1}}{{9}}\pi h^3 \implies h^3 = 729 \implies h = {sp.latex(h6)}$ \hfill \textbf{{[M1]}}
    \item Evaluates $\frac{{\text{{d}}V}}{{\text{{d}}t}}$ at $t=6$: $\frac{{\text{{d}}V}}{{\text{{d}}t}} = - ({sp.latex(k_val)}) ({sp.latex(sqrt_V6)}) \sin^2\left(\frac{{\pi}}{{2}}\right) = {sp.latex(dV_dt_expr)}$ \hfill \textbf{{[M1]}}
    \item Applies the chain rule correctly: $\frac{{\text{{d}}V}}{{\text{{d}}t}} = \frac{{\text{{d}}V}}{{\text{{d}}h}} \times \frac{{\text{{d}}h}}{{\text{{d}}t}} \implies {sp.latex(dV_dt_expr)} = {sp.latex(dV_dh_at_6)} \frac{{\text{{d}}h}}{{\text{{d}}t}}$ \hfill \textbf{{[M1]}}
    \item Obtains correct final answer: $\frac{{\text{{d}}h}}{{\text{{d}}t}} = {sp.latex(dh_dt)} \text{{ m/min}}$ \hfill \textbf{{[A1]}}
\end{{itemize}}
"""

    preamble = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\begin{document}
"""

    try:
        preview(latex_document, output='pdf', viewer='file', filename='deterministic_exam.pdf', preamble=preamble)
        print("Rendered deterministic_exam.pdf successfully using sympy.")
        subprocess.run(["powershell", "-Command", 'Invoke-Item "deterministic_exam.pdf"'])
    except Exception as e:
        print("Error rendering:", e)

if __name__ == '__main__':
    solve_and_generate()
