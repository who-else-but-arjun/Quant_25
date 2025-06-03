import pandas as pd
import numpy as np

class AutomatedMarketMaking:
    def __init__(self, tick_size=0.1, lot_size=2, max_inventory=20, window=50):
        self.tick_size = tick_size
        self.base_lot = lot_size
        self.max_inventory = max_inventory
        self.window = window

        self.imbalance_factor = 3
        self.inventory_penalty = 2
        self.momentum_multiplier = 6.0
        self.use_adaptive_lot = True
        self.skew_limit = 0.5

        self.reset_simulator()

    def reset_simulator(self):
        self.inventory = 0
        self.active_bid = None
        self.active_ask = None
        self.valid_from = None
        self.past_mid_prices = []
        self.past_trades = []
        self.pnl = 0.0
        self.trade_count = 0
        self.quote_refresh_count = 0

    def round_to_tick(self, price):
        return np.round(price / self.tick_size) * self.tick_size

    def update_quote(self, timestamp, bid_price, ask_price):
        self.active_bid = bid_price
        self.active_ask = ask_price
        self.valid_from = timestamp + 1
        if bid_price is not None or ask_price is not None:
            self.quote_refresh_count += 1

    def process_trades(self, timestamp, trades_at_time):
        if self.valid_from is None or timestamp < self.valid_from:
            return self.inventory

        filled = False
        current_lot = getattr(self, 'current_lot', self.base_lot)

        if self.active_bid is not None:
            sell_trades = trades_at_time[trades_at_time.side == 'sell']
            if not sell_trades.empty and self.active_bid >= sell_trades.price.max():
                fill_price = self.active_bid
                self.inventory += current_lot
                self.pnl -= fill_price * current_lot
                self.active_bid = None
                filled = True
                self.trade_count += 1
                self.past_trades.append(('buy', timestamp, fill_price))

        if self.active_ask is not None:
            buy_trades = trades_at_time[trades_at_time.side == 'buy']
            if not buy_trades.empty and self.active_ask <= buy_trades.price.min():
                fill_price = self.active_ask
                self.inventory -= current_lot
                self.pnl += fill_price * current_lot
                self.active_ask = None
                filled = True
                self.trade_count += 1
                self.past_trades.append(('sell', timestamp, fill_price))

        if filled:
            self.valid_from = float('inf')

        return self.inventory

    def compute_volatility(self):
        if len(self.past_mid_prices) < 2:
            return 0.0
        return np.std(self.past_mid_prices[-self.window:])

    def compute_ema_trend(self):
        if len(self.past_mid_prices) < self.window:
            return 0.0
        mid_series = pd.Series(self.past_mid_prices[-self.window:])
        ema = mid_series.ewm(span=10).mean().values
        x = np.arange(len(ema))
        slope, _ = np.polyfit(x, ema, 1)
        return slope

    def strategy(self, orderbook_df, trades_df, inventory, timestamp):
        orderbook_row = orderbook_df[orderbook_df.timestamp == timestamp]
        if orderbook_row.empty:
            return None, None

        best_bid_price = orderbook_row['bid_1_price'].values[0]
        best_bid_size = orderbook_row['bid_1_size'].values[0]
        best_ask_price = orderbook_row['ask_1_price'].values[0]
        best_ask_size = orderbook_row['ask_1_size'].values[0]

        micro_price = (best_ask_price * best_bid_size + best_bid_price * best_ask_size) / (best_bid_size + best_ask_size + 1e-6)
        mid_price = 0.5 * ((best_bid_price + best_ask_price) / 2) + 0.5 * micro_price
        self.past_mid_prices.append(mid_price)

        volatility = self.compute_volatility()
        volatility_factor = min(0.005, volatility / (mid_price + 1e-6))

        total_liquidity = best_bid_size + best_ask_size
        liquidity_factor = min(0.5, total_liquidity / 500.0)

        order_imbalance = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size + 1e-6)

        spread_ticks = 2 + (4 * volatility_factor + 2 * abs(order_imbalance))
        spread_ticks = max(1, spread_ticks)
        dynamic_spread = max(1, spread_ticks * (1 + volatility_factor - liquidity_factor))
        half_spread_value = (dynamic_spread * self.tick_size) / 2

        inventory_ratio = inventory / self.max_inventory
        inventory_adjustment = (inventory_ratio * self.inventory_penalty + inventory_ratio**2 * self.inventory_penalty) * half_spread_value
        imbalance_adjustment = order_imbalance * half_spread_value * self.imbalance_factor

        trend = self.compute_ema_trend()
        trend_adjustment = trend * self.momentum_multiplier * self.tick_size

        skew_adjustment = 0
        if abs(inventory_ratio) > self.skew_limit:
            skew_adjustment = np.sign(inventory_ratio) * half_spread_value * 0.3

        lot_size = self.base_lot
        if self.use_adaptive_lot:
            volume_scale = 1 - volatility_factor
            inventory_scale = 1 - min(0.2, abs(inventory) / self.max_inventory)
            lot_size = max(0.1, (self.base_lot * inventory_scale * volume_scale))
        self.current_lot = lot_size

        bid_price = mid_price - half_spread_value - inventory_adjustment - imbalance_adjustment + trend_adjustment - skew_adjustment
        ask_price = mid_price + half_spread_value - inventory_adjustment - imbalance_adjustment + trend_adjustment + skew_adjustment

        bid_price = self.round_to_tick(bid_price)
        ask_price = self.round_to_tick(ask_price)

        if bid_price is not None and ask_price is not None:
            if ask_price - bid_price < self.tick_size:
                ask_price = bid_price + self.tick_size

        if bid_price is not None and ask_price is not None and volatility_factor > 0.3:
            if ask_price - bid_price < 2 * self.tick_size:
                ask_price = bid_price + 2 * self.tick_size

        if inventory >= 0.9 * self.max_inventory:
            bid_price = None
        if inventory <= -0.9 * self.max_inventory:
            ask_price = None

        if inventory >= self.max_inventory:
            bid_price = None
        if inventory <= -self.max_inventory:
            ask_price = None

        return bid_price, ask_price

    def final_unrealized_pnl(self):
        if not self.past_mid_prices:
            return 0.0
        return self.inventory * self.past_mid_prices[-1]

    def run(self, orderbook_df, trades_df):
        self.reset_simulator()
        quotes = []
        timestamps = sorted(orderbook_df.timestamp.unique())[:3000]

        for t in timestamps:
            prev_bid = self.active_bid
            prev_ask = self.active_ask
            prev_inventory = self.inventory

            updated_inventory = self.process_trades(t, trades_df[trades_df.timestamp == t])
            bid, ask = self.strategy(orderbook_df, trades_df, updated_inventory, t)

            quote_changed = (updated_inventory != prev_inventory) or (bid != prev_bid) or (ask != prev_ask)
            if quote_changed:
                self.update_quote(t, bid, ask)
                quotes.append({'timestamp': t, 'bid_price': bid, 'ask_price': ask})

        return pd.DataFrame(quotes)

if __name__ == "__main__":
    orderbook_data = pd.read_csv('orderbook_train.csv').head(3000)
    trade_data = pd.read_csv('public_trades_train.csv').head(3000)

    market_maker = AutomatedMarketMaking()
    quotes_output = market_maker.run(orderbook_data, trade_data)
    quotes_output.to_csv('submission.csv', index=False)