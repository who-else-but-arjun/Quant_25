import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog

VAR_CONF_LEVEL = 0.95

#penalty on trading positions to discourage unnecessarily large positions
LAMBDA_COST = 1e-5
#maximum number of shares allowed per stock (long or short)
MAX_QTY = 10000
MIN_ROUND_THRESHOLD = 0.5

def read_input():
    data = sys.stdin.read().strip().split()
    portfolio_id = data[0]
    pnl = np.array(list(map(float, data[1:])))
    return portfolio_id, pnl

def main():
    portfolio_id, pnl = read_input()
    T = len(pnl) #250
    returns = pd.read_csv('stocks_returns.csv', index_col=0) / 100.0  #given
    metadata = pd.read_csv('stocks_metadata.csv')  #given

    R = returns.values   
    costs = metadata['Capital_Cost'].values 
    n = R.shape[1] 

    # we split each stock position into two non-negative parts: one for buying (positive side)
    # and one for selling (negative side). So total position = buy quantity - sell quantity.
    # we also add slack variables to represent possible losses exceeding our threshold in each scenario,
    # plus one free variable representing the VaR threshold.

    total_vars = 2 * n + T + 1  # total number of variables in the optimization

    alpha = VAR_CONF_LEVEL

    # our objective function coefficients:
    # we want to minimize a combination of:
    # 1. the expected shortfall (CVaR),
    # 2. the small cost penalty for holding positions,
    # 3. and the VaR threshold value.
    c = np.zeros(total_vars)

    #small cost penalty for long and short positions
    c[0:n] = LAMBDA_COST * costs
    c[n:2*n] = LAMBDA_COST * costs
    c[2*n:2*n+T] = 1.0 / ((1 - alpha) * T)
    c[-1] = 1.0


    bounds = []
    bounds += [(0, MAX_QTY)] * (2 * n)   # v_plus and v_minus variables
    bounds += [(0, None)] * T           # slack variables u_i
    bounds += [(None, None)]    # VaR threshold t

    # for each historical scenario i, enforce:
    # -u_i - (return in that scenario * position) - t <= pnl_i
    # this ensures that the slack variable and VaR threshold properly bound losses
    A_ub = np.zeros((T, total_vars))
    b_ub = pnl.copy()

    for i in range(T):
        #negative returns times positive position variables
        A_ub[i, :n] = -R[i]
        #returns times negative position variables (since short is negative)
        A_ub[i, n:2*n] = R[i]
        #slack variable coefficient
        A_ub[i, 2*n + i] = -1.0
        #VaR threshold coefficient
        A_ub[i, -1] = -1.0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not res.success:
        print('Optimization failed:', res.message, file=sys.stderr)
        sys.exit(1)

    x = res.x
    v_plus = x[0:n]
    v_minus = x[n:2*n]
    w_opt = v_plus - v_minus

    for idx, qty in enumerate(w_opt):
        if abs(qty) >= MIN_ROUND_THRESHOLD:
            qty_int = int(np.sign(qty) * np.ceil(abs(qty)))
            stock = metadata.loc[idx, 'Stock_Id']
            print(f"{stock} {qty_int}")

if __name__ == '__main__':
    main()
