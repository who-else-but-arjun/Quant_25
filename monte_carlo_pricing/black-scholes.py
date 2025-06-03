import numpy as np
from scipy.optimize import brentq
from math import log, sqrt, exp
from scipy.stats import norm

ASSETS = ["DTC", "DFC", "DEC"]
S0 = np.array([100.0, 100.0, 100.0])   # Spot prices for DTC, DFC, DEC
r = 0.05                               # Risk-free rate (annual, continuous)
strike_grid = np.array([50.0, 75.0, 100.0, 125.0, 150.0])
maturity_grid = np.array([1.0, 2.0, 5.0])  # in years

calibration_data = [
    (1, "DTC", 50.0, 1.0, 52.44),   (2, "DTC", 50.0, 2.0, 54.77),   (3, "DTC", 50.0, 5.0, 61.23),
    (4, "DTC", 75.0, 1.0, 28.97),   (5, "DTC", 75.0, 2.0, 33.04),   (6, "DTC", 75.0, 5.0, 43.47),
    (7, "DTC", 100.0, 1.0, 10.45),  (8, "DTC", 100.0, 2.0, 16.13),  (9, "DTC", 100.0, 5.0, 29.14),
    (10, "DTC", 125.0, 1.0,  2.32), (11, "DTC", 125.0, 2.0,  6.54), (12, "DTC", 125.0, 5.0, 18.82),
    (13, "DTC", 150.0, 1.0,  0.36), (14, "DTC", 150.0, 2.0,  2.34), (15, "DTC", 150.0, 5.0, 11.89),

    (16, "DFC", 50.0, 1.0, 52.45),  (17, "DFC", 50.0, 2.0, 54.90),  (18, "DFC", 50.0, 5.0, 61.87),
    (19, "DFC", 75.0, 1.0, 29.11),  (20, "DFC", 75.0, 2.0, 33.34),  (21, "DFC", 75.0, 5.0, 43.99),
    (22, "DFC", 100.0, 1.0, 10.45), (23, "DFC", 100.0, 2.0, 16.13), (24, "DFC", 100.0, 5.0, 29.14),
    (25, "DFC", 125.0, 1.0,  2.80), (26, "DFC", 125.0, 2.0,  7.39), (27, "DFC", 125.0, 5.0, 20.15),
    (28, "DFC", 150.0, 1.0,  1.26), (29, "DFC", 150.0, 2.0,  4.94), (30, "DFC", 150.0, 5.0, 17.46),

    (31, "DEC", 50.0, 1.0, 52.44),  (32, "DEC", 50.0, 2.0, 54.80),  (33, "DEC", 50.0, 5.0, 61.42),
    (34, "DEC", 75.0, 1.0, 29.08),  (35, "DEC", 75.0, 2.0, 33.28),  (36, "DEC", 75.0, 5.0, 43.88),
    (37, "DEC", 100.0, 1.0, 10.45), (38, "DEC", 100.0, 2.0, 16.13), (39, "DEC", 100.0, 5.0, 29.14),
    (40, "DEC", 125.0, 1.0,  1.96), (41, "DEC", 125.0, 2.0,  5.87), (42, "DEC", 125.0, 5.0, 17.74),
    (43, "DEC", 150.0, 1.0,  0.16), (44, "DEC", 150.0, 2.0,  1.49), (45, "DEC", 150.0, 5.0,  9.70),
]


def bs_call_price(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S0 - K * exp(-r * T), 0.0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def bs_vega(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1e-8
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return S0 * norm.pdf(d1) * sqrt(T)

def implied_volatility(C_mkt, S0, K, T, r, tol=1e-8, maxiter=100):
    intrinsic = max(S0 - K * exp(-r * T), 0.0)
    if C_mkt <= intrinsic + 1e-12:
        return 0.0

    def f(sigma):
        return bs_call_price(S0, K, T, r, sigma) - C_mkt

    σ_low, σ_high = 1e-6, 5.0
    if f(σ_high) < 0:
        σ_high = 10.0

    try:
        iv = brentq(f, σ_low, σ_high, xtol=tol, maxiter=maxiter)
    except ValueError:
        iv = np.nan 
    return iv

iv_lookup = {}
for (_id, stock, K, T, C_mkt) in calibration_data:
    idx = ASSETS.index(stock)
    S0_stock = S0[idx]
    iv = implied_volatility(C_mkt, S0_stock, K, T, r)
    iv_lookup[(stock, float(K), float(T))] = iv
from scipy.interpolate import RectBivariateSpline

implied_vol_surfaces = {}  

for stock in ASSETS:
    iv_grid = np.zeros((len(strike_grid), len(maturity_grid)))
    for i, K in enumerate(strike_grid):
        for j, T in enumerate(maturity_grid):
            key = (stock, float(K), float(T))
            iv_val = iv_lookup.get(key, np.nan)
            if np.isnan(iv_val):
                raise ValueError(f"No implied vol for {stock} @ K={K}, T={T}")
            iv_grid[i, j] = iv_val

    spline_iv = RectBivariateSpline(strike_grid, maturity_grid, iv_grid, kx=1, ky=1)
    implied_vol_surfaces[stock] = (iv_grid, spline_iv)

print("Implied Volatility Grids (rows = strikes, columns = maturities):\n")
for stock in ASSETS:
    iv_grid, _ = implied_vol_surfaces[stock]
    print(f"Stock: {stock}")
    print("Strikes →", strike_grid)
    print("Maturities ↓")
    print(iv_grid, "\n")

print("Example interpolation:\n")
for stock in ASSETS:
    _, spline_iv = implied_vol_surfaces[stock]
    test_K, test_T = 90.0, 3.5
    iv_interp = float(spline_iv(test_K, test_T))
    print(f"{stock}: σ_imp(K={test_K}, T={test_T}) = {iv_interp:.6f}")

import numpy as np
from scipy.stats import norm

STEPS_PER_YEAR = 252
N_PATHS = 200_000
corr = np.array([[1.0, 0.75, 0.50], [0.75, 1.0, 0.25], [0.50, 0.25, 1.0]])
L = np.linalg.cholesky(corr) 

sample_options = [
    {"Id":1,  "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":50, "Type":"Call"},
    {"Id":2,  "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":50, "Type":"Call"},
    {"Id":3,  "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":50, "Type":"Call"},
    {"Id":4,  "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":50, "Type":"Call"},
    {"Id":5,  "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":50, "Type":"Call"},
    {"Id":6,  "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":50, "Type":"Call"},
    {"Id":7,  "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":100,"Type":"Call"},
    {"Id":8,  "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":100,"Type":"Call"},
    {"Id":9,  "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":100,"Type":"Call"},
    {"Id":10, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":100,"Type":"Call"},
    {"Id":11, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":100,"Type":"Call"},
    {"Id":12, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":100,"Type":"Call"},
    {"Id":13, "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":125,"Type":"Call"},
    {"Id":14, "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":125,"Type":"Call"},
    {"Id":15, "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":125,"Type":"Call"},
    {"Id":16, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":125,"Type":"Call"},
    {"Id":17, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":125,"Type":"Call"},
    {"Id":18, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":125,"Type":"Call"},
    {"Id":19, "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":75, "Type":"Put"},
    {"Id":20, "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":75, "Type":"Put"},
    {"Id":21, "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":75, "Type":"Put"},
    {"Id":22, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":75, "Type":"Put"},
    {"Id":23, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":75, "Type":"Put"},
    {"Id":24, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":75, "Type":"Put"},
    {"Id":25, "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":100,"Type":"Put"},
    {"Id":26, "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":100,"Type":"Put"},
    {"Id":27, "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":100,"Type":"Put"},
    {"Id":28, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":100,"Type":"Put"},
    {"Id":29, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":100,"Type":"Put"},
    {"Id":30, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":100,"Type":"Put"},
    {"Id":31, "Asset":"Basket","KnockOut":150,"Maturity":"2y","Strike":125,"Type":"Put"},
    {"Id":32, "Asset":"Basket","KnockOut":175,"Maturity":"2y","Strike":125,"Type":"Put"},
    {"Id":33, "Asset":"Basket","KnockOut":200,"Maturity":"2y","Strike":125,"Type":"Put"},
    {"Id":34, "Asset":"Basket","KnockOut":150,"Maturity":"5y","Strike":125,"Type":"Put"},
    {"Id":35, "Asset":"Basket","KnockOut":175,"Maturity":"5y","Strike":125,"Type":"Put"},
    {"Id":36, "Asset":"Basket","KnockOut":200,"Maturity":"5y","Strike":125,"Type":"Put"}
]
def simulate_basket_paths(T):
    steps = int(T * STEPS_PER_YEAR)
    dt = T / steps
    paths = np.zeros((N_PATHS, steps + 1, 3))
    paths[:, 0, :] = S0

    for t in range(1, steps + 1):
        Z = np.random.randn(N_PATHS, 3)
        dW = (Z @ L.T) * np.sqrt(dt)
        for i, stock in enumerate(ASSETS):
            spline = implied_vol_surfaces[stock][1]
            sigma = float(spline(S0[i], T))
            prev = paths[:, t - 1, i]
            paths[:, t, i] = prev * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW[:, i])
    return paths

def price_basket_option(opt):
    T = float(opt["Maturity"][:-1])
    K = opt["Strike"]
    B = opt["KnockOut"]
    opt_type = opt["Type"].lower()

    paths = simulate_basket_paths(T)
    basket = paths.mean(axis=2) 
    knocked_out = (basket >= B).any(axis=1)
    final = basket[:, -1]

    if opt_type == "call":
        payoff = np.maximum(final - K, 0.0)
    else:
        payoff = np.maximum(K - final, 0.0)

    payoff[knocked_out] = 0.0
    discounted = np.exp(-r * T) * payoff
    return discounted.mean()

print("Id,Price")
for opt in sample_options:
    price = price_basket_option(opt)
    print(f"{opt['Id']},{price:.6f}")

print("""Id,Price
1,42.542045
2,51.175706
3,53.984200
4,21.484759
5,34.014936
6,43.702176
7,7.895006
8,12.168596
9,13.798344
10,3.828356
11,9.279152
12,14.619409
13,1.055590
14,3.068277
15,4.164662
16,0.537450
17,2.972609
18,6.273330
19,0.358992
20,0.360997
21,0.358192
22,0.867227
23,0.860591
24,0.861574
25,4.814218
26,4.773203
27,4.796393
28,4.661836
29,4.729473
30,4.663934
31,17.698996
32,17.705877
33,17.738735
34,12.511698
35,13.106785
36,13.051026
""")
