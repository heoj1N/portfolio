#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_distributions_3d.py

Generates 3D plots (surface or bar) for a variety of common probability
distributions. Each distribution has a dedicated function that saves
a PNG file. Intended for inclusion in a LaTeX document, one distribution
per chapter.

Usage:
  python plot_distributions_3d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb, gamma, exp, pi, sqrt
from mpl_toolkits.mplot3d import Axes3D  # ensures 3D functionality is available
import os  # Add os module for directory operations

###############################################################################
# Utility / Additional functions
###############################################################################
def factorial(n):
    if n <= 1:
        return 1
    return np.math.factorial(n)

def gamma_func(z):
    # We can just use math.gamma(z) or scipy.special.gamma, but let's keep consistent
    return gamma(z)

def beta_func(a, b):
    # Beta(a, b) = Gamma(a)*Gamma(b) / Gamma(a+b)
    return gamma(a)*gamma(b)/gamma(a+b)

def plot_save(fig, filename):
    """ Helper to finalize and save a figure. """
    # Create output directory if it doesn't exist
    output_dir = "probability_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure the filename has .pdf extension
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')
        if not filename.endswith('.pdf'):
            filename += '.pdf'
    
    # Save to the output directory
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close(fig)

###############################################################################
# 1. Normal (Gaussian)
###############################################################################
def pdf_normal(x, mu=0, sigma=1):
    return 1.0/(sqrt(2*pi)*sigma) * np.exp(-((x-mu)**2)/(2*sigma**2))

def plot_normal_3d(filename="normal_3d.pdf", mu=0, sigma=1):
    """ Plot Normal(mu,sigma^2) as 3D surface over x in [mu-4*sigma, mu+4*sigma]. """
    x_min = mu - 4*sigma
    x_max = mu + 4*sigma
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = np.linspace(0, 1, 20)  # dummy axis
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_normal(X, mu, sigma)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Normal(mu={mu}, sigma={sigma})")
    ax.set_xlabel("x")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 2. Exponential
###############################################################################
def pdf_exponential(x, lamb=1.0):
    return lamb*np.exp(-lamb*x) * (x >= 0)

def plot_exponential_3d(filename="exponential_3d.pdf", lamb=1.0):
    """ 3D surface for Exponential(lamb) over x>=0. """
    x_vals = np.linspace(0, 5/lamb, 200)  # e.g. up to ~5/lambda
    y_vals = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_exponential(X, lamb)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
    ax.set_title(f"Exponential(lambda={lamb})")
    ax.set_xlabel("x >= 0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 3. Log-normal
###############################################################################
def pdf_lognormal(x, mu=0, sigma=1):
    # f(x) = 1/(x sigma sqrt{2π}) * exp(-(ln x - mu)^2 / (2 sigma^2)), x>0
    res = np.zeros_like(x)
    mask = (x > 0)
    res[mask] = (1.0 / (x[mask]*sigma*sqrt(2*pi))) * np.exp(-((np.log(x[mask]) - mu)**2)/(2*sigma**2))
    return res

def plot_lognormal_3d(filename="lognormal_3d.pdf", mu=0, sigma=1):
    """ 3D surface for Lognormal(mu, sigma) over x>0. """
    x_vals = np.linspace(0.0001, np.exp(mu+4*sigma), 200)  # from near 0 to ~ e^(mu+4s)
    y_vals = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_lognormal(X, mu, sigma)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='cividis', edgecolor='none')
    ax.set_title(f"Lognormal(mu={mu}, sigma={sigma})")
    ax.set_xlabel("x>0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 4. Pareto
###############################################################################
def pdf_pareto(x, alpha=2.0, x_m=1.0):
    # f(x) = alpha x_m^alpha / x^(alpha+1), x >= x_m
    mask = (x >= x_m)
    res = np.zeros_like(x)
    res[mask] = alpha*(x_m**alpha)/(x[mask]**(alpha+1))
    return res

def plot_pareto_3d(filename="pareto_3d.pdf", alpha=2.0, x_m=1.0):
    """ 3D surface for Pareto(alpha, x_m). """
    x_vals = np.linspace(x_m, x_m*10, 200)
    y_vals = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_pareto(X, alpha, x_m)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none')
    ax.set_title(f"Pareto(alpha={alpha}, x_m={x_m})")
    ax.set_xlabel("x >= x_m")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 5. Weibull
###############################################################################
def pdf_weibull(x, k=1.5, lam=1.0):
    # f(x) = (k/lam)*(x/lam)^(k-1) * exp(-(x/lam)^k), x>=0
    mask = (x >= 0)
    res = np.zeros_like(x)
    xm = x[mask]/lam
    res[mask] = (k/lam)*(xm**(k-1))*np.exp(-(xm**k))
    return res

def plot_weibull_3d(filename="weibull_3d.pdf", k=1.5, lam=1.0):
    x_vals = np.linspace(0, 5*lam, 200)
    y_vals = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_weibull(X, k, lam)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='inferno', edgecolor='none')
    ax.set_title(f"Weibull(k={k}, λ={lam})")
    ax.set_xlabel("x >= 0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 6. Gumbel
###############################################################################
def pdf_gumbel(x, mu=0.0, beta=1.0):
    # f(x) = (1/beta) * exp(-((x-mu)/beta)) * exp(-e^{-((x-mu)/beta)})
    # or equivalently 1/beta * exp( -z - e^-z ), where z = (x-mu)/beta
    z = (x - mu)/beta
    return (1.0/beta)*np.exp(-z - np.exp(-z))

def plot_gumbel_3d(filename="gumbel_3d.pdf", mu=0.0, beta=1.0):
    x_vals = np.linspace(mu-6*beta, mu+6*beta, 200)
    y_vals = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_gumbel(X, mu, beta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='turbo', edgecolor='none')
    ax.set_title(f"Gumbel(mu={mu}, beta={beta})")
    ax.set_xlabel("x")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 7. Beta prime (a.k.a. inverted beta)
###############################################################################
def pdf_beta_prime(x, alpha=2.0, beta_=3.0):
    # f(x) = x^(alpha-1)*(1+x)^(-alpha-beta) / B(alpha,beta)
    # x>0
    mask = (x>0)
    res = np.zeros_like(x)
    res[mask] = (x[mask]**(alpha-1)*((1.0 + x[mask])**(-alpha-beta_))) / beta_func(alpha,beta_)
    return res

def plot_beta_prime_3d(filename="beta_prime_3d.pdf", alpha=2.0, beta_=3.0):
    x_vals = np.linspace(0.0001, 10, 300)
    y_vals = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_beta_prime(X, alpha, beta_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Beta prime(α={alpha}, β={beta_})")
    ax.set_xlabel("x>0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 8. Logistic
###############################################################################
def pdf_logistic(x, mu=0.0, s=1.0):
    # f(x) = exp(-(x-mu)/s) / [ s (1+ exp(-(x-mu)/s))^2 ]
    z = (x - mu)/s
    return np.exp(-z)/( s*(1+np.exp(-z))**2 )

def plot_logistic_3d(filename="logistic_3d.pdf", mu=0.0, s=1.0):
    x_vals = np.linspace(mu-6*s, mu+6*s, 200)
    y_vals = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_logistic(X, mu, s)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
    ax.set_title(f"Logistic(mu={mu}, s={s})")
    ax.set_xlabel("x")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 9. Discrete uniform
###############################################################################
def plot_discrete_uniform_3d(filename="discrete_uniform_3d.pdf", n=6):
    """
    P(X=k) = 1/n for k = 1,...,n
    We'll plot as 3D bars: x-axis = {1..n}, y=0, z= pmf.
    """
    k_vals = np.arange(1, n+1)
    pmf_vals = np.ones(n)/n

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)
    z_positions = np.zeros_like(k_vals)

    dx = 0.6
    dy = 0.5
    dz = pmf_vals

    ax.bar3d(x_positions, y_positions, z_positions, dx, dy, dz, shade=True)
    ax.set_title(f"Discrete uniform(1..{n})")
    ax.set_xlabel("k")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")
    ax.set_xticks(k_vals)

    plot_save(fig, filename)

###############################################################################
# 10. Continuous uniform
###############################################################################
def pdf_uniform(x, a=0.0, b=1.0):
    mask = (x>=a) & (x<=b)
    res = np.zeros_like(x)
    res[mask] = 1.0/(b-a)
    return res

def plot_continuous_uniform_3d(filename="continuous_uniform_3d.pdf", a=0.0, b=1.0):
    x_vals = np.linspace(a, b, 200)
    y_vals = np.linspace(0,1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_uniform(X, a, b)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='cividis', edgecolor='none')
    ax.set_title(f"Continuous Uniform(a={a}, b={b})")
    ax.set_xlabel("x")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 11. Bernoulli
###############################################################################
def plot_bernoulli_3d(filename="bernoulli_3d.pdf", p=0.3):
    """
    P(X=1)=p, P(X=0)=1-p. We'll do 2 bars at x=0,1
    """
    k_vals = np.array([0,1])
    pmf_vals = np.array([1-p, p])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)
    z_positions = np.zeros_like(k_vals)

    dx = 0.5
    dy = 0.5
    dz = pmf_vals

    ax.bar3d(x_positions, y_positions, z_positions, dx, dy, dz, shade=True)
    ax.set_title(f"Bernoulli(p={p})")
    ax.set_xlabel("k=0,1")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")

    plot_save(fig, filename)

###############################################################################
# 12. Binomial
###############################################################################
def pmf_binomial(k, n, p):
    return comb(n,k)*(p**k)*((1-p)**(n-k))

def plot_binomial_3d(filename="binomial_3d.pdf", n=10, p=0.4):
    k_vals = np.arange(0,n+1)
    pmf_vals = np.array([pmf_binomial(k, n, p) for k in k_vals])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)
    z_positions = np.zeros_like(k_vals)

    dx = 0.6
    dy = 0.5
    dz = pmf_vals

    ax.bar3d(x_positions, y_positions, z_positions, dx, dy, dz, shade=True)
    ax.set_title(f"Binomial(n={n}, p={p})")
    ax.set_xlabel("k")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")
    ax.set_xticks(k_vals)

    plot_save(fig, filename)

###############################################################################
# 13. Negative binomial
###############################################################################
def pmf_neg_binomial(k, r=5, p=0.4):
    # P(X=k) = comb(k+r-1, k)* p^r * (1-p)^k,  k=0,1,2,...
    return comb(k+r-1, k)*(p**r)*((1-p)**k)

def plot_neg_binomial_3d(filename="neg_binomial_3d.pdf", r=5, p=0.4):
    k_vals = np.arange(0, 20)  # let's just go up to 20 for illustration
    pmf_vals = np.array([pmf_neg_binomial(k, r, p) for k in k_vals])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)
    z_positions = np.zeros_like(k_vals)

    dx = 0.6
    dy = 0.5
    dz = pmf_vals

    ax.bar3d(x_positions, y_positions, z_positions, dx, dy, dz, shade=True)
    ax.set_title(f"Negative Binomial(r={r}, p={p})")
    ax.set_xlabel("k (# failures)")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")

    plot_save(fig, filename)

###############################################################################
# 14. Geometric
###############################################################################
def pmf_geometric(k, p):
    return (1-p)**k * p

def plot_geometric_3d(filename="geometric_3d.pdf", p=0.3):
    # We'll plot k=0..15
    k_vals = np.arange(0, 16)
    pmf_vals = np.array([pmf_geometric(k, p) for k in k_vals])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)
    z_positions = np.zeros_like(k_vals)
    dx = 0.6
    dy = 0.5
    dz = pmf_vals

    ax.bar3d(x_positions, y_positions, z_positions, dx, dy, dz, shade=True)
    ax.set_title(f"Geometric(p={p})")
    ax.set_xlabel("k (# failures)")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")

    plot_save(fig, filename)

###############################################################################
# 15. Hypergeometric
###############################################################################
def pmf_hypergeometric(k, N, K, n):
    # P(X=k) = (C(K,k)*C(N-K, n-k)) / C(N,n)
    return comb(K,k)*comb(N-K, n-k)/comb(N,n)

def plot_hypergeometric_3d(filename="hypergeometric_3d.pdf", N=20, K=8, n=5):
    """
    We'll plot k=0..n (the possible # of successes drawn).
    """
    k_vals = np.arange(0, n+1)
    pmf_vals = np.array([pmf_hypergeometric(k, N, K, n) for k in k_vals])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dx = 0.6
    dy = 0.5
    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)

    ax.bar3d(x_positions, y_positions, np.zeros_like(k_vals), dx, dy, pmf_vals, shade=True)
    ax.set_title(f"Hypergeometric(N={N}, K={K}, n={n})")
    ax.set_xlabel("k (# successes in draw)")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")

    plot_save(fig, filename)

###############################################################################
# 16. Beta-binomial
###############################################################################
from math import comb
from scipy.special import beta as beta_sp

def pmf_beta_binomial(k, n, alpha, beta_):
    # P(X=k)= comb(n,k)* B(alpha+k, beta_+n-k)/[B(alpha,beta_)* B(alpha+beta_,n)]
    # Using scipy's special.beta for Beta(a,b):
    num = comb(n,k)* beta_sp(alpha + k, beta_ + n - k)
    den = beta_sp(alpha,beta_)*beta_sp(alpha+beta_, n - (alpha+beta_==0)) # or need binomial coefficient?
    # Actually, exact formula for B(a+ b, n) is ambiguous. Let's do standard approach:
    # We'll do an explicit approach with gamma_func
    # B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b)
    # B(a+beta_, n) doesn't look standard, typically it's B(alpha+beta_, n) but n isn't sum of alphas
    # We'll rely on an identity from references or just do the ratio with gamma funcs
    # For simplicity, let's do a direct approach:
    # P(X=k) = comb(n,k)* Gamma(alpha+k)*Gamma(beta_+n-k)/Gamma(alpha+beta_+n)  * Gamma(alpha+beta_)/ [Gamma(alpha)*Gamma(beta_)]
    # but let's keep it simple with scipy's betabinom: from stats we can do a small domain approach
    # For demonstration, let's proceed with a simplified ratio approach:
    pass
    # We'll do it fully numeric:
    # Numerator = comb(n, k)* B(alpha + k, beta_ + n - k)
    # Denominator = B(alpha, beta_)/ B(alpha+beta_, n)? Actually it's B(alpha+beta_, n) = integral of x^(alpha+beta_-1)(1-x)^(n-1) dx over x in [0,1]? This might not be standard.
    # We'll do a simpler approach: we can use the pmf from "scipy.stats import betabinom", but let's do it manually.

def pmf_beta_binomial_simplified(k, n, alpha, beta_):
    """
    We can do a direct formula via gamma functions:
      P(X=k) = comb(n,k)
               * Gamma(alpha + k)*Gamma(beta_ + n - k) / Gamma(alpha + beta_ + n)
               * Gamma(alpha + beta_) / [Gamma(alpha)*Gamma(beta_)]
    for k=0..n.
    """
    num = comb(n,k)*(gamma_func(alpha + k)*gamma_func(beta_ + n - k))
    den = gamma_func(alpha + beta_ + n)
    pref = (gamma_func(alpha+beta_)/(gamma_func(alpha)*gamma_func(beta_)))
    return pref*(num/den)

def plot_beta_binomial_3d(filename="beta_binomial_3d.pdf", n=10, alpha=2, beta_=3):
    k_vals = np.arange(0, n+1)
    pmf_vals = [pmf_beta_binomial_simplified(k, n, alpha, beta_) for k in k_vals]
    pmf_vals = np.array(pmf_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)

    ax.bar3d(x_positions, y_positions, 0, 0.6, 0.5, pmf_vals, shade=True)
    ax.set_title(f"Beta-Binomial(n={n}, alpha={alpha}, beta={beta_})")
    ax.set_xlabel("k")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")

    plot_save(fig, filename)

###############################################################################
# 17. Categorical
###############################################################################
def plot_categorical_3d(filename="categorical_3d.pdf", pvals=[0.2, 0.5, 0.3]):
    k_vals = np.arange(len(pvals))
    pmf_vals = np.array(pvals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)
    z_positions = np.zeros_like(k_vals)

    ax.bar3d(x_positions, y_positions, z_positions, 0.6, 0.5, pmf_vals, shade=True)
    ax.set_title(f"Categorical p={pvals}")
    ax.set_xlabel("category index")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")

    plot_save(fig, filename)

###############################################################################
# 18. Multinomial
###############################################################################
def pmf_multinomial(counts, n, pvals):
    # p(X1=k1,...,XK=kK)= n!/(k1!...kK!) * p1^k1 *...* pK^kK
    # We'll do a function that returns a single value for a given vector of counts
    from math import factorial
    ksum = sum(counts)
    if ksum != n:
        return 0
    num = factorial(n)
    for k, p in zip(counts, pvals):
        num *= (p**k)/factorial(k)
    return num

def plot_multinomial_3d(filename="multinomial_3d.pdf", n=5, pvals=[0.3,0.2,0.5]):
    """
    We'll do a 3D bar for (k1,k2) with k3 = n - k1 - k2,
    i.e. K=3 categories => a 2D domain: k1=0..n, k2=0..(n-k1). 
    z = pmf(k1, k2, k3).
    We'll attempt a bar3d in the plane of k1,k2, with height=pmf.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    k1_vals = np.arange(0, n+1)
    k2_vals = np.arange(0, n+1)

    # We'll collect the x, y, z data
    X = []
    Y = []
    Z = []
    for k1 in k1_vals:
        for k2 in k2_vals:
            if k1 + k2 <= n:
                k3 = n - k1 - k2
                val = pmf_multinomial([k1, k2, k3], n, pvals)
                X.append(k1)
                Y.append(k2)
                Z.append(val)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    dx = 0.5
    dy = 0.5
    dz = Z

    ax.bar3d(X, Y, np.zeros_like(Z), dx, dy, dz, shade=True)
    ax.set_title(f"Multinomial(n={n}, p={pvals})")
    ax.set_xlabel("k1")
    ax.set_ylabel("k2")
    ax.set_zlabel("PMF")
    plot_save(fig, filename)

###############################################################################
# 19. Multivariate hypergeometric
###############################################################################
def pmf_multiv_hyp(kvec, Kvec, N, n):
    """
    p(k1,...,kK) = ( \prod_i comb(Ki, ki) ) / comb(N, n),
    sum(ki)=n, sum(Ki)=N
    """
    from math import comb
    # Check sum(ki)==n, sum(Ki)==N
    if sum(kvec) != n:
        return 0.0
    val_num = 1
    for ki, Ki in zip(kvec, Kvec):
        val_num *= comb(Ki, ki)
    val_den = comb(N, n)
    return val_num/val_den

def plot_multivariate_hypergeometric_3d(filename="multiv_hypergeom_3d.pdf", Kvec=[4,5,6], n=5):
    """
    We'll do K=3 categories => sum(Ki)=15 => N=15
    We'll vary (k1,k2), then k3= n-(k1+k2).
    We'll do bar3d as in the multinomial case, but with pmf_multiv_hyp.
    """
    N = sum(Kvec)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    k1_vals = np.arange(0, n+1)
    k2_vals = np.arange(0, n+1)
    X, Y, Z = [], [], []

    for k1 in k1_vals:
        for k2 in k2_vals:
            if k1+k2 <= n:
                k3 = n - (k1 + k2)
                val = pmf_multiv_hyp([k1, k2, k3], Kvec, N, n)
                X.append(k1)
                Y.append(k2)
                Z.append(val)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    ax.bar3d(X, Y, np.zeros_like(Z), 0.5, 0.5, Z, shade=True)
    ax.set_title(f"Multivariate Hypergeometric(K={Kvec}, n={n})")
    ax.set_xlabel("k1")
    ax.set_ylabel("k2")
    ax.set_zlabel("PMF")
    plot_save(fig, filename)

###############################################################################
# 20. Poisson
###############################################################################
def pmf_poisson(k, lamb=3.0):
    # P(X=k) = e^{-lambda} lambda^k / k!
    return (np.exp(-lamb)* (lamb**k))/ factorial(k)

def plot_poisson_3d(filename="poisson_3d.pdf", lamb=3.0):
    k_vals = np.arange(0, 15)  # up to 14
    pmf_vals = np.array([pmf_poisson(k, lamb) for k in k_vals])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_positions = k_vals
    y_positions = np.zeros_like(k_vals)
    z_positions = np.zeros_like(k_vals)

    ax.bar3d(x_positions, y_positions, z_positions, 0.6, 0.5, pmf_vals, shade=True)
    ax.set_title(f"Poisson(lambda={lamb})")
    ax.set_xlabel("k")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PMF")
    ax.set_xticks(k_vals)

    plot_save(fig, filename)

###############################################################################
# 21. Gamma
###############################################################################
def pdf_gamma(x, alpha=2.0, beta_=1.0):
    # f(x)= beta^alpha / Gamma(alpha) * x^(alpha-1)* e^(-beta*x), x>0
    mask = (x>0)
    res = np.zeros_like(x)
    norm = (beta_**alpha)/gamma_func(alpha)
    res[mask] = norm*(x[mask]**(alpha-1))*np.exp(-beta_*x[mask])
    return res

def plot_gamma_3d(filename="gamma_3d.pdf", alpha=2.0, beta_=1.0):
    x_vals = np.linspace(0, 10, 200)
    y_vals = np.linspace(0,1,20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_gamma(X, alpha, beta_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Gamma(alpha={alpha}, beta={beta_})")
    ax.set_xlabel("x>0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 22. Rayleigh
###############################################################################
def pdf_rayleigh(r, sigma=1.0):
    # f(r)= (r/sigma^2)* exp(-r^2/(2 sigma^2)), r>=0
    mask = (r>=0)
    res = np.zeros_like(r)
    res[mask] = (r[mask]/(sigma**2))* np.exp(-(r[mask]**2)/(2*sigma**2))
    return res

def plot_rayleigh_3d(filename="rayleigh_3d.pdf", sigma=1.0):
    r_vals = np.linspace(0, 5*sigma, 200)
    y_vals = np.linspace(0,1, 20)
    R, Y = np.meshgrid(r_vals, y_vals)
    Z = pdf_rayleigh(R, sigma)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R, Y, Z, cmap='plasma', edgecolor='none')
    ax.set_title(f"Rayleigh(sigma={sigma})")
    ax.set_xlabel("r>=0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 23. Rice (Rician)
###############################################################################
from scipy.special import i0

def pdf_rice(r, sigma=1.0, nu=1.0):
    # f(r)= (r/sigma^2)* exp(-(r^2+nu^2)/(2 sigma^2))* I0(r nu/ sigma^2), r>=0
    mask = (r>=0)
    res = np.zeros_like(r)
    # i0 is the modified Bessel function of the first kind, order 0
    res[mask] = (r[mask]/(sigma**2)) * np.exp(-(r[mask]**2 + nu**2)/(2*sigma**2))* i0((r[mask]*nu)/(sigma**2))
    return res

def plot_rice_3d(filename="rice_3d.pdf", sigma=1.0, nu=1.0):
    r_vals = np.linspace(0, 5*sigma, 200)
    y_vals = np.linspace(0,1, 20)
    R, Y = np.meshgrid(r_vals, y_vals)
    Z = pdf_rice(R, sigma, nu)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Rice(sigma={sigma}, nu={nu})")
    ax.set_xlabel("r>=0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 24. Chi-squared
###############################################################################
def pdf_chi_squared(x, k=3):
    # f(x)= 1/(2^(k/2)*Gamma(k/2)) * x^(k/2 -1) e^{-x/2}, x>0
    mask = (x>0)
    res = np.zeros_like(x)
    coeff = 1.0/(2.0**(k/2)*gamma_func(k/2))
    power = (k/2)-1
    res[mask] = coeff*(x[mask]**power)*np.exp(-x[mask]/2)
    return res

def plot_chi_squared_3d(filename="chi_squared_3d.pdf", k=3):
    x_vals = np.linspace(0, 15, 200)
    y_vals = np.linspace(0,1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_chi_squared(X, k)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='inferno', edgecolor='none')
    ax.set_title(f"Chi-squared(k={k})")
    ax.set_xlabel("x>0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 25. Student's t
###############################################################################
def pdf_student_t(x, nu=3):
    # f(x)= Gamma((nu+1)/2)/ [sqrt(nu*pi)* Gamma(nu/2)] * (1 + x^2/nu)^(-(nu+1)/2)
    from math import gamma
    from math import sqrt
    coeff = gamma_func((nu+1)/2)/(sqrt(nu*pi)* gamma_func(nu/2))
    return coeff*(1.0 + (x**2)/nu)**(-0.5*(nu+1))

def plot_student_t_3d(filename="student_t_3d.pdf", nu=3):
    x_vals = np.linspace(-10, 10, 300)
    y_vals = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_student_t(X, nu)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
    ax.set_title(f"Student's t(nu={nu})")
    ax.set_xlabel("x")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 26. F-distribution
###############################################################################
def pdf_f(x, d1=5, d2=8):
    # f(x)= sqrt( ( (d1 x)^d1 * d2^d2 ) / (d1 x + d2)^(d1 + d2 ) ) / [ x * B(d1/2, d2/2) ]
    # We'll do a stable approach using logs maybe, but let's keep direct for demonstration:
    from math import sqrt
    from scipy.special import beta as B
    num = (d1*x)**(d1)* (d2**(d2))
    den = (d1*x + d2)**(d1 + d2)
    # watch out for x>0
    # Also there's a factor 1/[ x * B(d1/2, d2/2 ) ] * sqrt(num/den)
    return np.where(x>0,
        (1.0/(x*B(d1/2, d2/2))) * np.sqrt(num/den),
        0.0
    )

def plot_f_dist_3d(filename="f_dist_3d.pdf", d1=5, d2=8):
    x_vals = np.linspace(0.001, 5, 300)
    y_vals = np.linspace(0,1, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_f(X, d1, d2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none')
    ax.set_title(f"F-distribution(d1={d1}, d2={d2})")
    ax.set_xlabel("x>0")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 27. Beta
###############################################################################
def pdf_beta(x, alpha=2, beta_=3):
    # f(x)= x^(alpha-1)*(1-x)^(beta-1)/ B(alpha,beta), x in [0,1]
    mask = (x>=0) & (x<=1)
    res = np.zeros_like(x)
    norm = 1.0/beta_func(alpha,beta_)
    res[mask] = norm*(x[mask]**(alpha-1))*((1-x[mask])**(beta_-1))
    return res

def plot_beta_3d(filename="beta_3d.pdf", alpha=2, beta_val=3):
    x_vals = np.linspace(0,1,200)
    y_vals = np.linspace(0,1,20)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = pdf_beta(X, alpha, beta_val)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Beta(alpha={alpha}, beta={beta_val})")
    ax.set_xlabel("x in [0,1]")
    ax.set_ylabel("dummy axis")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 28. Dirichlet (K=3 for 2-simplex)
###############################################################################
from scipy.special import gamma as gamma_sp
def dirichlet_pdf(x, alpha):
    # alpha = [alpha1, alpha2, alpha3], sum x_i=1, x_i>=0
    # f(x1,x2,x3)= 1/B(alpha)* \prod x_i^(alpha_i-1)
    # B(alpha)= prod Gamma(alpha_i)/Gamma(sum alpha_i)
    # We'll assume x is shape(N,3). We'll return the pdf for each row
    alpha0 = sum(alpha)
    denom = 1.0/gamma_sp(alpha0)
    for a in alpha:
        denom *= gamma_sp(a)
    # denom = B(alpha)
    # pdf(x) = 1/B(alpha)* (x1^(a1-1)* x2^(a2-1)* x3^(a3-1))
    # We'll do a version that returns a 2D array for plotting
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]
    fvals = (x1**(alpha[0]-1))*(x2**(alpha[1]-1))*(x3**(alpha[2]-1))/denom
    return fvals

def plot_dirichlet_3d(filename="dirichlet_3d.pdf", alpha=[2,3,4]):
    """
    We'll sample points in the 2-simplex: x1+x2+x3=1, x_i>=0,
    then use x1,x2 as axes => x3=1-x1-x2. We'll 3D-plot (x1,x2, PDF).
    """
    ngrid = 50
    x1_vals = np.linspace(0, 1, ngrid)
    x2_vals = np.linspace(0, 1, ngrid)

    # We'll build arrays for points in the simplex x1+x2<=1
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    X1f = X1.flatten()
    X2f = X2.flatten()
    mask = (X1f+X2f<=1)
    X1r = X1f[mask]
    X2r = X2f[mask]
    X3r = 1 - X1r - X2r
    points = np.stack([X1r, X2r, X3r], axis=1)
    pdf_vals = dirichlet_pdf(points, alpha)

    # Convert back to 2D form for plotting
    # We'll treat Z=pdf, with x axis = x1, y axis= x2
    Z = np.zeros_like(X1f)
    Z[mask] = pdf_vals
    Z = Z.reshape(X1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Dirichlet(alpha={alpha}), 2-simplex")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("PDF")
    plot_save(fig, filename)

###############################################################################
# 29. Wishart
###############################################################################
# Wishart is a distribution on positive-definite matrices. 
# A direct 3D plot is tricky. We provide a conceptual approach:
# For 2x2 Wishart with df=nu and scale=Sigma=I, we might attempt to plot
# density over some constrained space of symmetrical matrices. 
# But let's do something simpler: We'll just pick a parameterization and
# show a 3D "slice" for a fixed off-diagonal = 0, etc. This is advanced,
# so we'll do a placeholder.

def plot_wishart_placeholder(filename="wishart_3d.pdf"):
    """
    Placeholder approach:
    - We'll treat the 2x2 Wishart(I, nu=3) but fix off-diagonal=0,
      so effectively we have W=diag(x,y) with x>0,y>0.
    - Then the density becomes a function of x,y in R^+ x R^+.
    We'll do a surface plot of the PDF(x,y).
    This is not the full Wishart distribution in 3D, but a 2D slice.
    """
    # PDF for Wishart(2x2, Sigma=I, nu) restricted to diag only => 
    # The real formula for general W is more complex.
    # We'll define a function for the diagonal-only slice:
    def wishart_2x2_diag(x, y, nu=3):
        # dimension p=2, Sigma=I => det(I)=1, 
        # f(W)= 1/(2^(nu p/2)Gamma_p(nu/2)|Sigma|^(nu/2)) det(W)^{(nu-p-1)/2} exp(-trace(W)/2)
        # For diag(x,y) with x>0,y>0, det(W)= x*y, trace(W)= x+y
        # Gamma_p(nu/2)= pi^(p(p-1)/4)*\prod_{k=0}^{p-1}Gamma((nu-k)/2)
        # For p=2 => Gamma_2(a)= pi^(1*(2-1)/2)= pi^(1/2)= sqrt(pi)*Gamma(a)Gamma(a-1/2)? 
        # Actually the known formula for p=2 => Gamma_2(a)= pi^(1/2)*Gamma(a)*Gamma(a-1/2).
        from math import sqrt, pi
        # Let's define gamma2(a):
        def gamma2(a):
            return (pi**(1.0/2)) * gamma_func(a)* gamma_func(a-0.5)

        if (x<=0 or y<=0):
            return 0.0
        p=2
        # det(W)=xy, trace(W)=x+y
        c = 1.0/(2.0**(nu*p/2)* gamma2(nu/2))
        # exponent = - (x+y)/2
        return c*(x*y)**((nu-p-1)/2)* np.exp(-(x+y)/2)

        # This is only for the diagonal-only slice W=diag(x,y)!
    
    nu=3
    x_vals = np.linspace(0.01, 5, 100)
    y_vals = np.linspace(0.01, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = wishart_2x2_diag(X[i,j], Y[i,j], nu=nu)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("Wishart(2x2, I, nu=3) [diagonal slice]")
    ax.set_xlabel("W_{11}=x>0")
    ax.set_ylabel("W_{22}=y>0")
    ax.set_zlabel("PDF (slice)")
    plot_save(fig, filename)

###############################################################################
# MAIN to generate all
###############################################################################
def main():
    # Continuous
    plot_normal_3d()
    plot_exponential_3d()
    plot_lognormal_3d()
    plot_pareto_3d()
    plot_weibull_3d()
    plot_gumbel_3d()
    plot_beta_prime_3d()
    plot_logistic_3d()
    plot_continuous_uniform_3d()
    plot_gamma_3d()
    plot_rayleigh_3d()
    plot_rice_3d()
    plot_chi_squared_3d()
    plot_student_t_3d()
    plot_f_dist_3d()
    plot_beta_3d()
    plot_dirichlet_3d()  # K=3 case

    # Discrete
    plot_discrete_uniform_3d()
    plot_bernoulli_3d()
    plot_binomial_3d()
    plot_neg_binomial_3d()
    plot_geometric_3d()
    plot_hypergeometric_3d()
    plot_beta_binomial_3d()
    plot_categorical_3d()
    plot_multinomial_3d()
    plot_multivariate_hypergeometric_3d()
    plot_poisson_3d()

    # Matrix-variate (placeholder)
    plot_wishart_placeholder()

    print("All 3D distribution plots generated!")

if __name__ == "__main__":
    main()
