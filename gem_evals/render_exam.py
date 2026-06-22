import sympy as sp
from sympy import preview
import os
import subprocess

def solve_and_generate():
    # Set up symbols
    x, k = sp.symbols('x k', real=True, positive=True)
    
    # PDF function
    f = k * x * sp.log(x)
    
    # (a) Find k
    int_f = sp.integrate(f, (x, 1, sp.E))
    k_val = sp.simplify(sp.solve(int_f - 1, k)[0])
    
    # (b) Find E(X)
    E_X_expr = k_val * x**2 * sp.log(x)
    E_X_val = sp.simplify(sp.integrate(E_X_expr, (x, 1, sp.E)))
    
    # (c) Find E(A)
    E_A_expr = sp.pi * k_val * x**3 * sp.log(x)
    E_A_val = sp.simplify(sp.integrate(E_A_expr, (x, 1, sp.E)))

    # Use f-strings and sp.latex() to dynamically build the LaTeX document
    latex_document = fr"""
\section*{{A-Level Mathematics Question}}

A continuous random variable $X$ has a probability density function $f(x)$ defined by:

$$ f(x) = \begin{{cases}} k x \ln(x) & 1 \le x \le e \\ 0 & \text{{otherwise}} \end{{cases}} $$

where $k$ is a positive constant.

\vspace{{0.5cm}}

\noindent \textbf{{(a)}} Show that $k = {sp.latex(k_val)}$. \hfill \textbf{{[4 marks]}}

\vspace{{0.5cm}}

\noindent \textbf{{(b)}} Find the exact value of the expected value, $\text{{E}}(X)$. \hfill \textbf{{[4 marks]}}

\vspace{{0.5cm}}

\noindent \textbf{{(c)}} The random variable $X$ models the radius, in metres, of a circular oil spill. Find the exact expected area of the oil spill, $\text{{E}}(A)$. \hfill \textbf{{[3 marks]}}

\newpage

\section*{{Mark Scheme}}

\subsection*{{(a)}}
\begin{{itemize}}
    \item Sets up the integral equal to 1: $\int_{{1}}^{{e}} k x \ln(x) \, dx = 1$ \hfill \textbf{{[M1]}}
    \item Uses integration by parts correctly: $\int x \ln(x) \, dx = \frac{{x^2}}{{2}} \ln(x) - \int \frac{{x}}{{2}} \, dx = \frac{{x^2}}{{2}} \ln(x) - \frac{{x^2}}{{4}}$ \hfill \textbf{{[M1]}}
    \item Evaluates limits correctly: $\left[ \frac{{e^2}}{{2}}\ln(e) - \frac{{e^2}}{{4}} \right] - \left[ \frac{{1^2}}{{2}}\ln(1) - \frac{{1^4}}{{4}} \right] = \frac{{e^2}}{{4}} + \frac{{1}}{{4}}$ \hfill \textbf{{[A1]}}
    \item Equates to 1 and solves for $k$: $k \left( \frac{{e^2 + 1}}{{4}} \right) = 1 \implies k = {sp.latex(k_val)}$. \hfill \textbf{{[A1]}}
\end{{itemize}}

\subsection*{{(b)}}
\begin{{itemize}}
    \item Uses the formula for expected value: $\text{{E}}(X) = \int_{{1}}^{{e}} x \cdot f(x) \, dx = \int_{{1}}^{{e}} {sp.latex(k_val)} x^2 \ln(x) \, dx$ \hfill \textbf{{[M1]}}
    \item Integrates $x^2 \ln(x)$ by parts: $\frac{{x^3}}{{3}} \ln(x) - \frac{{x^3}}{{9}}$ \hfill \textbf{{[M1]}}
    \item Evaluates limits: $\left[ \frac{{e^3}}{{3}}\ln(e) - \frac{{e^3}}{{9}} \right] - \left[ \frac{{1^3}}{{3}}\ln(1) - \frac{{1}}{{9}} \right] = \frac{{2e^3}}{{9}} + \frac{{1}}{{9}}$ \hfill \textbf{{[A1]}}
    \item Multiplies by $k$ to obtain final answer: $\text{{E}}(X) = {sp.latex(E_X_val)}$. \hfill \textbf{{[A1]}}
\end{{itemize}}

\subsection*{{(c)}}
\begin{{itemize}}
    \item States that $A = \pi X^2$ and therefore $\text{{E}}(A) = \pi \text{{E}}(X^2)$ \hfill \textbf{{[M1]}}
    \item Sets up the integral for $\text{{E}}(X^2)$: $\int_{{1}}^{{e}} x^2 f(x) \, dx = \int_{{1}}^{{e}} {sp.latex(k_val)} x^3 \ln(x) \, dx$ \hfill \textbf{{[M1]}}
    \item Evaluates integral and multiplies by $\pi$: $\text{{E}}(A) = {sp.latex(E_A_val)} \text{{ m}}^2$. \hfill \textbf{{[A1]}}
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
        preview(latex_document, output='pdf', viewer='file', filename='generated_exam.pdf', preamble=preamble)
        print("Rendered generated_exam.pdf successfully using sympy.")
    except Exception as e:
        print("Error rendering:", e)

if __name__ == '__main__':
    solve_and_generate()
