# Quantitative Finance Tasks (GSIH 2025)

This repository contains three advanced quantitative finance tasks implemented in Python, covering portfolio hedging, market making, and options pricing with local volatility models.

## Table of Contents

1. [Task 1: Portfolio Hedging with Risk Management](#task-1-portfolio-hedging-with-risk-management)
2. [Task 2: Automated Market Making Strategy](#task-2-automated-market-making-strategy)
3. [Task 3: Options Pricing with Local Volatility Models](#task-3-options-pricing-with-local-volatility-models)
4. [Installation and Setup](#installation-and-setup)

---

## Task 1: Portfolio Hedging with Risk Management

### Overview
Implements two approaches to hedge portfolio risk using historical stock returns data: Linear Programming optimization and LASSO regression with cross-validation.

### Files
- `hedge.py` - Linear programming approach using CVaR optimization
- `lasso_CV.py` - LASSO regression with cross-validation approach

### Input Format
**Standard Input:** `<portfolio_id> <pnl_1> <pnl_2> ... <pnl_T>`

Example:
```
PORTFOLIO_001 -2.5 1.8 -0.9 2.1 -1.2 0.8 ...
```

**Required CSV Files:**
- `stocks_returns.csv` - Historical stock returns (T×N matrix where T=time periods, N=stocks)
- `stocks_metadata.csv` - Stock metadata with columns: `Stock_Id`, `Capital_Cost`

### Output Format
```
<Stock_Id> <Quantity>
<Stock_Id> <Quantity>
...
```

### Strategy Explanation

#### Linear Programming Approach (`hedge.py`)
- **Objective**: Minimize Conditional Value at Risk (CVaR) at 95% confidence level
- **Constraints**: 
  - Maximum position size per stock: 10,000 shares
  - Position cost penalty to discourage unnecessary large positions
- **Method**: Uses scipy's linear programming solver with CVaR optimization
- **Key Parameters**:
  - `VAR_CONF_LEVEL = 0.95` (VaR confidence level)
  - `LAMBDA_COST = 1e-5` (position cost penalty)
  - `MAX_QTY = 10000` (maximum shares per stock)

#### LASSO Regression Approach (`lasso_CV.py`)
- **Objective**: Find sparse hedge portfolio using regularized regression
- **Method**: LASSO with 3-fold cross-validation to select optimal regularization
- **Features**: 
  - Automatic feature standardization
  - Sparse solution (many zero positions)
  - Cross-validation for hyperparameter tuning

### Running the Code

```bash
# Linear Programming Approach
echo "PORTFOLIO_001 -2.5 1.8 -0.9 2.1" | python hedge.py

# LASSO Regression Approach  
echo "PORTFOLIO_001 -2.5 1.8 -0.9 2.1" | python lasso_CV.py
```

---

## Task 2: Automated Market Making Strategy

### Overview
Implements a sophisticated automated market making strategy that dynamically quotes bid/ask prices based on market conditions, inventory management, and risk factors.

### Files
- `amm_strategy.py` - Complete market making implementation with backtesting

### Input Files
- `orderbook_train.csv` - Market depth data with columns:
  - `timestamp`, `bid_1_price`, `bid_1_size`, `ask_1_price`, `ask_1_size`
- `public_trades_train.csv` - Trade execution data with columns:
  - `timestamp`, `price`, `side` (buy/sell)

### Output
- `submission.csv` - Generated quotes with columns: `timestamp`, `bid_price`, `ask_price`

### Strategy Components

#### Core Parameters
- `tick_size = 0.1` - Minimum price increment
- `lot_size = 2` - Base order size
- `max_inventory = 20` - Maximum position limit
- `window = 50` - Historical data window for calculations

#### Key Features

1. **Dynamic Spread Calculation**
   - Base spread: 2 ticks + volatility and imbalance adjustments
   - Volatility factor: Adapts to recent price movements
   - Liquidity factor: Adjusts based on order book depth

2. **Inventory Management**
   - Inventory penalty: Skews quotes when position is large
   - Position limits: Stops quoting when near maximum inventory
   - Adaptive lot sizing: Reduces order size in volatile conditions

3. **Market Microstructure Signals**
   - Order imbalance: Adjusts quotes based on bid/ask size ratio
   - Momentum detection: Uses EMA trend analysis
   - Micro-price calculation: Weighted average of bid/ask

4. **Risk Controls**
   - Maximum inventory limits with gradual and hard stops
   - Volatility-based spread widening
   - Adaptive position sizing

### Running the Code

```bash
python amm_strategy.py
```

The script will:
1. Load orderbook and trade data
2. Run the market making simulation for 3000 timestamps
3. Generate quotes based on the strategy
4. Save results to `submission.csv`

### Performance Metrics
The strategy tracks:
- Total P&L (realized + unrealized)
- Number of trades executed
- Quote refresh frequency
- Final inventory position

---

## Task 3: Options Pricing with Local Volatility Models

### Overview
Implements advanced options pricing using both Black-Scholes with implied volatility surfaces and Dupire local volatility models for pricing exotic basket options with knock-out features.

### Files
- `black-scholes.py` - Black-Scholes implementation with implied volatility calibration
- `dupires.py` - Dupire local volatility model implementation

### Market Setup
- **Assets**: DTC, DFC, DEC (3 correlated stocks)
- **Initial Prices**: All start at $100
- **Risk-free Rate**: 5% annual
- **Correlation Matrix**:
  ```
  DTC  DFC  DEC
  1.00 0.75 0.50  (DTC)
  0.75 1.00 0.25  (DFC)  
  0.50 0.25 1.00  (DEC)
  ```

### Input Data
Market calibration data includes European call option prices for:
- **Strikes**: [50, 75, 100, 125, 150]
- **Maturities**: [1Y, 2Y, 5Y]
- **All three underlying assets**

### Option Types Priced
**Basket Options with Knock-out Features:**
- **Underlying**: Equally-weighted basket of DTC, DFC, DEC
- **Types**: European Call and Put options
- **Knock-out Barriers**: 150, 175, 200
- **Strikes**: 50, 75, 100, 125
- **Maturities**: 2Y, 5Y
- **Feature**: Up-and-out barrier (option expires worthless if basket ever reaches barrier)

### Model Implementations

#### Black-Scholes Approach (`black-scholes.py`)
1. **Implied Volatility Calibration**:
   - Extracts implied volatilities from market call prices
   - Creates volatility surfaces using bivariate spline interpolation
   - Uses constant volatility per asset for simulation

2. **Monte Carlo Simulation**:
   - 200,000 paths with 252 steps per year
   - Correlated Brownian motion using Cholesky decomposition
   - Knock-out monitoring at each time step

#### Dupire Local Volatility Approach (`dupires.py`)
1. **Local Volatility Surface Construction**:
   - Computes partial derivatives of call prices (∂C/∂K, ∂²C/∂K², ∂C/∂T)
   - Applies Dupire formula: σ²(K,T) = (∂C/∂T + rK∂C/∂K) / (½K²∂²C/∂K²)
   - Creates local volatility surfaces for each asset

2. **Enhanced Monte Carlo**:
   - Path-dependent volatility using local vol surfaces
   - More accurate pricing for exotic options
   - State-dependent volatility at each simulation step

### Mathematical Formulations

#### Dupire Formula
```
σ²ₗᵥ(K,T) = (∂C/∂T + rK∂C/∂K) / (½K²∂²C/∂K²)
```

#### Basket Option Payoff
```
Call: max(Basket(T) - K, 0) × 𝟙{max(Basket(t)) < Barrier for all t ∈ [0,T]}
Put:  max(K - Basket(T), 0) × 𝟙{max(Basket(t)) < Barrier for all t ∈ [0,T]}
```

Where `Basket(t) = (S₁(t) + S₂(t) + S₃(t)) / 3`

### Running the Code

```bash
# Black-Scholes with Implied Volatility
python black-scholes.py

# Dupire Local Volatility Model  
python dupires.py
```

Both scripts will:
1. Calibrate volatility surfaces from market data
2. Run Monte Carlo simulations for all 36 basket options
3. Output prices in CSV format: `Id,Price`

### Expected Output Format
```
Id,Price
1,42.542045
2,51.175706
3,53.984200
...
36,13.051026
```

---

## Installation and Setup

### Required Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### Python Version
- Python 3.7+
- All code tested with standard scientific Python stack

### File Structure
```
project/
├── hedging_strategy/
│   ├── hedge.py
│   ├── lasso_CV.py
│   ├── stocks_returns.csv
│   └── stocks_metadata.csv
├── automated_market_making/
│   ├── amm_strategy.py
│   ├── orderbook_train.csv
│   ├── public_trades_train.csv
│   └── submission.csv (generated)
├── monte_carlo_pricing/
│   ├── black-scholes.py
│   └── dupires.py
└── README.md
```

### Performance Considerations

- **Task 1**: Runs in seconds for typical portfolio sizes
- **Task 2**: Processes 3000 market timestamps efficiently  
- **Task 3**: Monte Carlo with 200K paths takes several minutes

### Key Features Across All Tasks

1. **Risk Management**: All implementations include sophisticated risk controls
2. **Market Realism**: Models incorporate real market microstructure effects
3. **Numerical Stability**: Robust handling of edge cases and numerical precision
4. **Scalability**: Efficient implementations suitable for production use

---

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone https://github.com/who-else-but-arjun/Quant_25.git
   cd Quant_25
   pip install -r requirements.txt
   ```

2. **Run individual tasks**:
   ```bash
   # Portfolio Hedging
   echo "PORTFOLIO_001 -2.5 1.8 -0.9" | python hedge.py
   
   # Market Making
   python amm_strategy.py
   
   # Options Pricing
   python black-scholes.py
   ```

3. **Check outputs** in respective directories

Each task is self-contained and can be run independently with the provided sample data or your own datasets following the specified input formats.
