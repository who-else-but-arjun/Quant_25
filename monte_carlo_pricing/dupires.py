import numpy as np
from scipy.interpolate import RectBivariateSpline

# === A) Market and Model Inputs ===

ASSETS = ["DTC", "DFC", "DEC"]
S0     = np.array([100.0, 100.0, 100.0])  # initial spots for each asset
r      = 0.05                             # annual risk‐free rate (no dividends assumed)

corr = np.array([
    [1.00, 0.75, 0.50],
    [0.75, 1.00, 0.25],
    [0.50, 0.25, 1.00]
])
L = np.linalg.cholesky(corr)  

STEPS_PER_YEAR = 252
N_PATHS       = 200_000
strike_grid   = np.array([50,  75, 100, 125, 150], dtype=float)  # shape (5,)
maturity_grid = np.array([1.0, 2.0, 5.0],        dtype=float)  # shape (3,)
N_K = len(strike_grid)
N_T = len(maturity_grid)

# --------------------------------------------------
calibration_data = [
    ( 1, "DTC",  50, 1.0, 52.44), ( 2, "DTC",  50, 2.0, 54.77), ( 3, "DTC",  50, 5.0, 61.23),
    ( 4, "DTC",  75, 1.0, 28.97), ( 5, "DTC",  75, 2.0, 33.04), ( 6, "DTC",  75, 5.0, 43.47),
    ( 7, "DTC", 100, 1.0, 10.45), ( 8, "DTC", 100, 2.0, 16.13), ( 9, "DTC", 100, 5.0, 29.14),
    (10, "DTC", 125, 1.0,  2.32), (11, "DTC", 125, 2.0,  6.54), (12, "DTC", 125, 5.0, 18.82),
    (13, "DTC", 150, 1.0,  0.36), (14, "DTC", 150, 2.0,  2.34), (15, "DTC", 150, 5.0, 11.89),

    (16, "DFC",  50, 1.0, 52.45), (17, "DFC",  50, 2.0, 54.90), (18, "DFC",  50, 5.0, 61.87),
    (19, "DFC",  75, 1.0, 29.11), (20, "DFC",  75, 2.0, 33.34), (21, "DFC",  75, 5.0, 43.99),
    (22, "DFC", 100, 1.0, 10.45), (23, "DFC", 100, 2.0, 16.13), (24, "DFC", 100, 5.0, 29.14),
    (25, "DFC", 125, 1.0,  2.80), (26, "DFC", 125, 2.0,  7.39), (27, "DFC", 125, 5.0, 20.15),
    (28, "DFC", 150, 1.0,  1.26), (29, "DFC", 150, 2.0,  4.94), (30, "DFC", 150, 5.0, 17.46),

    (31, "DEC",  50, 1.0, 52.44), (32, "DEC",  50, 2.0, 54.80), (33, "DEC",  50, 5.0, 61.42),
    (34, "DEC",  75, 1.0, 29.08), (35, "DEC",  75, 2.0, 33.28), (36, "DEC",  75, 5.0, 43.88),
    (37, "DEC", 100, 1.0, 10.45), (38, "DEC", 100, 2.0, 16.13), (39, "DEC", 100, 5.0, 29.14),
    (40, "DEC", 125, 1.0,  1.96), (41, "DEC", 125, 2.0,  5.87), (42, "DEC", 125, 5.0, 17.74),
    (43, "DEC", 150, 1.0,  0.16), (44, "DEC", 150, 2.0,  1.49), (45, "DEC", 150, 5.0,  9.70)
]
# --------------------------------------------------

price_lookup = {}
for (_id, stock, K, T, price_dollar) in calibration_data:
    price_lookup[(stock, float(K), float(T))] = price_dollar
call_price_surfaces = {} 

for stock in ASSETS:
    C_mkt = np.zeros((N_K, N_T))
    for i, K in enumerate(strike_grid):
        for j, T in enumerate(maturity_grid):
            key = (stock, float(K), float(T))
            if key not in price_lookup:
                raise ValueError(f"Missing call price for {stock} @ Strike={K}, T={T}")
            C_mkt[i, j] = price_lookup[key]
    call_price_surfaces[stock] = C_mkt
local_vol_surfaces = {} 

for stock in ASSETS:
    C_mkt = call_price_surfaces[stock] 

    C_K  = np.zeros_like(C_mkt)   # ∂C/∂K
    C_KK = np.zeros_like(C_mkt)   # ∂²C/∂K²
    C_T  = np.zeros_like(C_mkt)   # ∂C/∂T

    for j in range(N_T):
        for i in range(1, N_K - 1):
            h_plus  = strike_grid[i+1] - strike_grid[i]
            h_minus = strike_grid[i]   - strike_grid[i-1]
            C_K[i, j] = (C_mkt[i+1, j] - C_mkt[i-1, j]) / (h_plus + h_minus)
            C_KK[i, j] = (
                C_mkt[i+1, j] - 2.0 * C_mkt[i, j] + C_mkt[i-1, j]
            ) / (h_plus * h_minus)
        i = 0
        forward_h = strike_grid[1] - strike_grid[0]
        C_K[i, j] = (C_mkt[1, j] - C_mkt[0, j]) / forward_h
        h1 = strike_grid[1] - strike_grid[0]
        h2 = strike_grid[2] - strike_grid[1]
        C_KK[i, j] = (C_mkt[2, j] - 2.0 * C_mkt[1, j] + C_mkt[0, j]) / (h1 * h2)
        
        i = N_K - 1
        back_h = strike_grid[i] - strike_grid[i - 1]
        C_K[i, j] = (C_mkt[i, j] - C_mkt[i - 1, j]) / back_h
        h1 = strike_grid[i] - strike_grid[i - 1]
        h2 = strike_grid[i - 1] - strike_grid[i - 2]
        C_KK[i, j] = (C_mkt[i, j] - 2.0 * C_mkt[i - 1, j] + C_mkt[i - 2, j]) / (h1 * h2)

    for i in range(N_K):
        for j in range(N_T - 1):
            dt = maturity_grid[j + 1] - maturity_grid[j]
            C_T[i, j] = (C_mkt[i, j + 1] - C_mkt[i, j]) / dt
        dt = maturity_grid[N_T - 1] - maturity_grid[N_T - 2]
        C_T[i, N_T - 1] = (C_mkt[i, N_T - 1] - C_mkt[i, N_T - 2]) / dt

    local_var = np.zeros_like(C_mkt)
    epsilon = 1e-12  

    for i in range(N_K):
        for j in range(N_T):
            K = strike_grid[i]
            numerator   = C_T[i, j] + r * K * C_K[i, j]
            denominator = 0.5 * K**2 * max(C_KK[i, j], epsilon)
            local_var[i, j] = max(numerator / denominator, 0.0)

    local_vol_grid = np.sqrt(local_var)
    local_vol_surfaces[stock] = RectBivariateSpline(
        strike_grid,
        maturity_grid,
        local_vol_grid,
        kx=1,  # linear interpolation in strike
        ky=1   # linear interpolation in maturity
    )

# === B) Basket‐Option Monte Carlo (uses the calibrated local vol surfaces) ===

basket_options = [
    {"Id":  1, "Asset": "Basket", "KnockOut": 150, "Maturity": "2y", "Strike": 50,  "Type": "Call"},
    {"Id":  2, "Asset": "Basket", "KnockOut": 175, "Maturity": "2y", "Strike": 50,  "Type": "Call"},
    {"Id":  3, "Asset": "Basket", "KnockOut": 200, "Maturity": "2y", "Strike": 50,  "Type": "Call"},
    {"Id":  4, "Asset": "Basket", "KnockOut": 150, "Maturity": "5y", "Strike": 50,  "Type": "Call"},
    {"Id":  5, "Asset": "Basket", "KnockOut": 175, "Maturity": "5y", "Strike": 50,  "Type": "Call"},
    {"Id":  6, "Asset": "Basket", "KnockOut": 200, "Maturity": "5y", "Strike": 50,  "Type": "Call"},
    {"Id":  7, "Asset": "Basket", "KnockOut": 150, "Maturity": "2y", "Strike": 100, "Type": "Call"},
    {"Id":  8, "Asset": "Basket", "KnockOut": 175, "Maturity": "2y", "Strike": 100, "Type": "Call"},
    {"Id":  9, "Asset": "Basket", "KnockOut": 200, "Maturity": "2y", "Strike": 100, "Type": "Call"},
    {"Id": 10, "Asset": "Basket", "KnockOut": 150, "Maturity": "5y", "Strike": 100, "Type": "Call"},
    {"Id": 11, "Asset": "Basket", "KnockOut": 175, "Maturity": "5y", "Strike": 100, "Type": "Call"},
    {"Id": 12, "Asset": "Basket", "KnockOut": 200, "Maturity": "5y", "Strike": 100, "Type": "Call"},
    {"Id": 13, "Asset": "Basket", "KnockOut": 150, "Maturity": "2y", "Strike": 125, "Type": "Call"},
    {"Id": 14, "Asset": "Basket", "KnockOut": 175, "Maturity": "2y", "Strike": 125, "Type": "Call"},
    {"Id": 15, "Asset": "Basket", "KnockOut": 200, "Maturity": "2y", "Strike": 125, "Type": "Call"},
    {"Id": 16, "Asset": "Basket", "KnockOut": 150, "Maturity": "5y", "Strike": 125, "Type": "Call"},
    {"Id": 17, "Asset": "Basket", "KnockOut": 175, "Maturity": "5y", "Strike": 125, "Type": "Call"},
    {"Id": 18, "Asset": "Basket", "KnockOut": 200, "Maturity": "5y", "Strike": 125, "Type": "Call"},
    {"Id": 19, "Asset": "Basket", "KnockOut": 150, "Maturity": "2y", "Strike": 75,  "Type": "Put"},
    {"Id": 20, "Asset": "Basket", "KnockOut": 175, "Maturity": "2y", "Strike": 75,  "Type": "Put"},
    {"Id": 21, "Asset": "Basket", "KnockOut": 200, "Maturity": "2y", "Strike": 75,  "Type": "Put"},
    {"Id": 22, "Asset": "Basket", "KnockOut": 150, "Maturity": "5y", "Strike": 75,  "Type": "Put"},
    {"Id": 23, "Asset": "Basket", "KnockOut": 175, "Maturity": "5y", "Strike": 75,  "Type": "Put"},
    {"Id": 24, "Asset": "Basket", "KnockOut": 200, "Maturity": "5y", "Strike": 75,  "Type": "Put"},
    {"Id": 25, "Asset": "Basket", "KnockOut": 150, "Maturity": "2y", "Strike": 100, "Type": "Put"},
    {"Id": 26, "Asset": "Basket", "KnockOut": 175, "Maturity": "2y", "Strike": 100, "Type": "Put"},
    {"Id": 27, "Asset": "Basket", "KnockOut": 200, "Maturity": "2y", "Strike": 100, "Type": "Put"},
    {"Id": 28, "Asset": "Basket", "KnockOut": 150, "Maturity": "5y", "Strike": 100, "Type": "Put"},
    {"Id": 29, "Asset": "Basket", "KnockOut": 175, "Maturity": "5y", "Strike": 100, "Type": "Put"},
    {"Id": 30, "Asset": "Basket", "KnockOut": 200, "Maturity": "5y", "Strike": 100, "Type": "Put"},
    {"Id": 31, "Asset": "Basket", "KnockOut": 150, "Maturity": "2y", "Strike": 125, "Type": "Put"},
    {"Id": 32, "Asset": "Basket", "KnockOut": 175, "Maturity": "2y", "Strike": 125, "Type": "Put"},
    {"Id": 33, "Asset": "Basket", "KnockOut": 200, "Maturity": "2y", "Strike": 125, "Type": "Put"},
    {"Id": 34, "Asset": "Basket", "KnockOut": 150, "Maturity": "5y", "Strike": 125, "Type": "Put"},
    {"Id": 35, "Asset": "Basket", "KnockOut": 175, "Maturity": "5y", "Strike": 125, "Type": "Put"},
    {"Id": 36, "Asset": "Basket", "KnockOut": 200, "Maturity": "5y", "Strike": 125, "Type": "Put"}
]

def simulate_paths(T, steps, n):
    dt = T / steps
    S   = np.zeros((n, steps + 1, 3))
    S[:, 0, :] = S0 
    for t in range(1, steps + 1):
        Z = np.random.randn(n, 3)             # shape (n,3) i.i.d. N(0,1)
        dW = (Z @ L.T) * np.sqrt(dt)          # shape (n,3), correlated increments
        time = (t - 1) * dt                   # “calendar” time at the previous step
        
        for i, stk in enumerate(ASSETS):
            prev = S[:, t - 1, i]
            sigma = local_vol_surfaces[stk](prev, time, grid=False)
            S[:, t, i] = prev + r * prev * dt + sigma * prev * dW[:, i]
    
    return S

def price_option(opt):
    B = opt["KnockOut"]
    T = float(opt["Maturity"][:-1])  # convert "2y" -> 2.0
    K = opt["Strike"]
    o = opt["Type"].lower()          # "call" or "put"
    steps = int(T * STEPS_PER_YEAR)
    
    paths = simulate_paths(T, steps, N_PATHS)
    basket = paths.mean(axis=2)
    knocked = (basket >= B).any(axis=1)
    terminal = basket[:, -1]
    if o == "call":
        payoff = np.maximum(terminal - K, 0.0)
    else:
        payoff = np.maximum(K - terminal, 0.0)
    payoff[knocked] = 0.0
    disc = np.exp(-r * T) * payoff
    
    return max(disc.mean(), 0.0)

# === C) Run and Print Prices ===
print("Id,Price")
for opt in basket_options:
    price = price_option(opt)
    print(f"{opt['Id']},{price:.4f}")

print("""Id,Price
Id,Price
1,45.5642
2,53.4041
3,54.7108
4,24.4212
5,40.1106
6,50.8601
7,9.0626
8,12.9472
9,13.7404
10,5.2553
11,12.8282
12,19.1878
13,1.3301
14,3.3285
15,3.8412
16,0.8848
17,4.5963
18,8.6912
19,0.2481
20,0.2504
21,0.2528
22,0.8158
23,0.8202
24,0.8181
25,4.3548
26,4.3318
27,4.3284
28,4.1890
29,4.1564
30,4.1685
31,17.0072
32,17.0462
33,16.9678
34,11.4306
35,11.6286
36,11.6549
""")