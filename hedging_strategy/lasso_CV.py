import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

VAR_CONF_LEVEL = 0.95
MIN_ROUND_THRESHOLD = 0.5

warnings.filterwarnings("ignore", category=ConvergenceWarning)
def read_input():
    data = input().strip().split()
    if len(data) < 2:
        sys.exit(1)
    portfolio_id = data[0]
    pnl = np.array(list(map(float, data[1:])))
    return portfolio_id, pnl

def main():
    portfolio_id, pnl = read_input()
    T = len(pnl)

    returns_df = pd.read_csv('stocks_returns.csv', index_col=0) / 100.0
    metadata_df = pd.read_csv('stocks_metadata.csv')

    R = returns_df.values
    stock_ids = metadata_df['Stock_Id'].values
    costs = metadata_df['Capital_Cost'].values

    if R.shape[0] != T:
        sys.exit(1)
    n = R.shape[1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(R)

    lasso = LassoCV(cv=3, max_iter=5000, n_jobs=1).fit(X_scaled, pnl)
    coef_scaled = lasso.coef_

    stds = scaler.scale_
    coef_unscaled = coef_scaled / np.where(stds == 0, 1e-12, stds)
    w_opt = -1*coef_unscaled

    output_positions = {}
    for j in range(n):
        qty = w_opt[j]
        if abs(qty) >= MIN_ROUND_THRESHOLD:
            qty_int = int(np.sign(qty) * np.ceil(abs(qty)))
            output_positions[stock_ids[j]] = qty_int

    hedged_pnl = pnl - (R @ w_opt)
    VaR_95 = np.percentile(hedged_pnl, (1 - VAR_CONF_LEVEL) * 100)
    total_cost = np.sum(np.abs(w_opt) * costs)

    for sid, qty in output_positions.items():
        print(f"{sid} {qty}")

if __name__ == '__main__':
    main()
