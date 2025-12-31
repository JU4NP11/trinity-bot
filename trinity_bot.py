#!/usr/bin/env python3
import time, json, threading, requests, logging, websocket
import numpy as np
import os
import csv
from datetime import datetime, timedelta, timezone
from collections import deque
from colorama import Fore, Style, init

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                        filename='trinity_bot.log',
                        filemode='w')
    # Suppress verbose logs from libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

setup_logging()
init(autoreset=True)

# ================= CONFIG =================
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')
with open(config_path, 'r') as f:
    CONFIG = json.load(f)

# CAPITAL and CAPITAL_START are now initialized in load_state()


# ================= COLORS =================
CYAN = Fore.CYAN
GREEN = Fore.GREEN
ORANGE = Fore.LIGHTRED_EX
GREY = Fore.LIGHTBLACK_EX
BLOOD = Fore.RED + Style.BRIGHT
YELLOW = Fore.LIGHTYELLOW_EX

# ================= STATE =================
prices = {}
rolling_prices = {}
positions = {}
ohlc = {}
htf_ohlc = {}
dynamic_risk_pct = CONFIG['trading']['risk_pct']

potential_symbols = []
entry_blacklist = set()
dynamic_risk_pct = CONFIG['trading']['risk_pct']
IS_BACKTESTING = False
lock = threading.Lock()
symbols_lock = threading.Lock()
scanner_lock = threading.Lock()
trading_halted = False

# ================= INDICATORS =================
def ema(values, period):
    if len(values) < period: return None
    k = 2/(period+1)
    e = values[0]
    for v in values[1:]:
        e = v*k + e*(1-k)
    return e

def atr(symbol):
    c = ohlc.get(symbol)
    if not c or len(c) < CONFIG['strategy']['atr_period']+1: return None
    tr=[]
    for i in range(1,len(c)):
        h,l,pc = c[i]["h"],c[i]["l"],c[i-1]["c"]
        tr.append(max(h-l, abs(h-pc), abs(l-pc)))
    return sum(tr[-CONFIG['strategy']['atr_period']:])/CONFIG['strategy']['atr_period']

def htf_trend_ok(symbol):
    c = htf_ohlc.get(symbol)
    if not c or len(c) < CONFIG['strategy']['htf_ema_period']: return False
    closes = [x["c"] for x in c]
    return closes[-1] > ema(closes, CONFIG['strategy']['htf_ema_period'])

def volatility_score(symbol):
    dq = rolling_prices[symbol]
    if len(dq)<2: return 0
    return (max(dq)-min(dq))/min(dq)

def signal_confidence(symbol):
    with lock:
        htf_ok = 1 if htf_trend_ok(symbol) else 0
        c = ohlc.get(symbol)
        if not c or len(c)<15:
            ema_ok = 0
        else:
            closes = [x["c"] for x in c]
            short_ema = ema(closes[-5:],5)
            long_ema = ema(closes,15)
            ema_ok = 1 if short_ema and long_ema and short_ema>long_ema else 0
        vol_ok = 1 if volatility_score(symbol) > 0.01 else 0
        conf = (htf_ok + ema_ok + vol_ok)/3*100
        return int(conf)

# ================= AGENTS =================
class BaseAgent(threading.Thread):
    def __init__(self, name):
        super().__init__(name=name, daemon=True)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        while not self.stopped():
            try:
                self.step()
            except Exception as e:
                logging.error(f"Error in {self.name} agent: {e}", exc_info=True)
                if not IS_BACKTESTING: time.sleep(5) # Avoid rapid-fire errors

    def step(self):
        raise NotImplementedError("Each agent must implement the 'step' method.")

class DashboardAgent(BaseAgent):
    def __init__(self):
        super().__init__("Dashboard")

    def step(self):
        if IS_BACKTESTING:
            # During backtesting, log key metrics but don't print to console
            realized = sum(p['pnl'] for p in positions.values())
            unrealized = sum((prices.get(s, 0.0) - p['entry']) * p['qty'] if p.get('entry') else 0 for s, p in positions.items())
            total_pnl = realized + unrealized
            logging.info(f"BACKTEST - Virtual Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Capital: {CAPITAL:.2f}, Drawdown: {(CAPITAL_START - CAPITAL) / CAPITAL_START * 100:.2f}%, Total PnL: {total_pnl:.2f}")
            # No sleep in backtest mode, simulation controls time
            return
        
        # Live mode dashboard rendering
        print("\033c", end="")
        
        with symbols_lock:
            current_symbols = SYMBOLS[:]

        with lock:
            api_status = GREEN + "API CONNECTED"
            # This check needs to be adjusted for live data availability
            if not current_symbols or all(prices.get(s, 0.0) == 0.0 for s in current_symbols):
                api_status = BLOOD + "API DISCONNECTED"

            realized = sum(p['pnl'] for p in positions.values())
            unrealized = sum((prices.get(s, 0.0) - p['entry']) * p['qty'] if p.get('entry') else 0 for s, p in positions.items())
            total_pnl = realized + unrealized

            print(f"{CYAN}TRINITY AUTO-BUY + TRAILING SELL BOT DASHBOARD")
            print(f"{api_status}"
                  f"{GREY} | Capital: ${CAPITAL:>8,.2f}"
                  f" | Drawdown: {(CAPITAL_START - CAPITAL) / CAPITAL_START * 100:>5.2f}%"
                  f" | Total PnL: ${total_pnl:>8.2f}"
                  f" | {datetime.now().strftime('%H:%M:%S')}")
            print(GREY + "-" * 140)
            
            header = f"{CYAN}{'SYMBOL':<10}{'PRICE':>12} CONF {'STATE':<14}{'TRAIL':>10} {'PNL':>8} HISTORY"
            if not current_symbols:
                header += f"\n{YELLOW}Scanning for symbols..."
            
            print(header)
            print(GREY + "-" * 140)

            sorted_syms = sorted(current_symbols, key=lambda s: volatility_score(s), reverse=True)

            for s in sorted_syms:
                price = prices.get(s, 0.0)
                p = positions.get(s)
                if not p: continue 

                a = atr(s)
                trail = f"{p.get('peak', 0.0) - a * CONFIG['strategy']['atr_mult']:.4f}" if a and p.get('peak') else "--"
                price_col = GREEN if p.get('entry') and price >= p['entry'] else ORANGE
                state_col = GREEN if p['state'] == "IN_POSITION" else BLOOD if p['state'] == "SOLD" else GREY
                conf = signal_confidence(s)
                bar = pnl_bar(p['pnl_history'])
                print(f"{CYAN}{s:<10}{price_col}{price:>12,.4f} "
                      f"{YELLOW}{conf:>3}% "
                      f"{state_col}{p['state']:<14}"
                      f"{GREY}{trail:>10} "
                      f"{GREEN if p['pnl'] > 0 else ORANGE if p['pnl'] < 0 else GREY}{p['pnl']:>8.2f} "
                      f"{bar}")

        if not IS_BACKTESTING: time.sleep(1)


class MarketScannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("MarketScanner")

    def step(self):
        global SYMBOLS, potential_symbols
        
        tickers = requests.get(f"{CONFIG['api']['rest_base_url']}/api/v3/ticker/24hr", timeout=10).json()
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT') and 'UP' not in t['symbol'] and 'DOWN' not in t['symbol']]
        top_20_by_volume = sorted(usdt_tickers, key=lambda x: float(x['quoteVolume']), reverse=True)[:20]
        
        with scanner_lock:
            potential_symbols = [t['symbol'] for t in top_20_by_volume]

        time.sleep(45)

        scored_symbols = []
        with lock:
            for symbol in potential_symbols:
                score = signal_confidence(symbol)
                scored_symbols.append({"symbol": symbol, "score": score})
        
        top_3 = sorted(scored_symbols, key=lambda x: x['score'], reverse=True)[:3]
        new_symbols = [s['symbol'] for s in top_3]

        with symbols_lock:
            SYMBOLS = new_symbols
            with lock:
                for s in SYMBOLS:
                    if s not in prices:
                        prices[s] = 0.0
                        rolling_prices[s] = deque(maxlen=CONFIG['display']['rolling_len'])
                        ohlc[s] = []
                        htf_ohlc[s] = []
                        positions[s] = {
                            "state":"IDLE", "entry":None, "peak":None, "qty":0.0,
                            "sold_1":False, "sold_2":False, "last_exit":0, "pnl":0.0,
                            "pnl_history":deque(maxlen=CONFIG['display']['pnl_history_len'])
                        }
        
        if not IS_BACKTESTING: time.sleep(CONFIG['scanner']['scan_interval_sec'])

class RiskManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskManager")
        self.pnl_history = deque(maxlen=10)
        self.win_streak = 0
        self.loss_streak = 0

    def _calculate_correlation_matrix(self, symbols, period):
        returns = {}
        for s in symbols:
            if s in ohlc and len(ohlc[s]) >= period:
                closes = np.array([d['c'] for d in ohlc[s][-period:]])
                # Calculate percentage returns
                returns[s] = (closes[1:] - closes[:-1]) / closes[:-1]
        
        if len(returns) < 2:
            return None

        # Create a DataFrame and calculate correlation
        import pandas as pd # Local import to avoid global dependency if not used
        df = pd.DataFrame(returns)
        return df.corr()

    def _update_blacklist(self):
        global entry_blacklist
        if not CONFIG['correlation_analysis']['enabled']:
            return

        cfg = CONFIG['correlation_analysis']
        
        with lock:
            with symbols_lock:
                current_symbols = SYMBOLS[:]
            
            active_positions = {s for s, p in positions.items() if p.get('state') == 'IN_POSITION'}
            if not active_positions:
                entry_blacklist = set()
                return

            correlation_matrix = self._calculate_correlation_matrix(current_symbols, cfg['period'])
            if correlation_matrix is None:
                return

            new_blacklist = set()
            candidate_symbols = [s for s in current_symbols if s not in active_positions]

            for candidate in candidate_symbols:
                is_correlated = False
                for active in active_positions:
                    if candidate in correlation_matrix and active in correlation_matrix:
                        corr_value = correlation_matrix.loc[candidate, active]
                        if corr_value > cfg['correlation_threshold']:
                            is_correlated = True
                            break
                if is_correlated:
                    new_blacklist.add(candidate)
            
            entry_blacklist = new_blacklist

    def step(self):
        global trading_halted, dynamic_risk_pct, entry_blacklist
        
        with lock:
            # 1. Max Drawdown Check
            drawdown = (CAPITAL_START - CAPITAL) / CAPITAL_START * 100
            if drawdown >= CONFIG['trading']['max_drawdown_pct']:
                if not trading_halted:
                    logging.critical(f"MAX DRAWDOWN REACHED! TRADING HALTED. Drawdown: {drawdown:.2f}%")
                trading_halted = True
            else:
                if trading_halted:
                    logging.info("Trading re-enabled as drawdown is back within limits.")
                trading_halted = False

            # 2. Dynamic Risk Adjustment
            if CONFIG['dynamic_risk']['enabled']:
                realized = sum(p['pnl'] for p in positions.values())
                unrealized = sum((prices.get(s, 0.0) - p['entry']) * p['qty'] if p.get('entry') else 0 for s, p in positions.items())
                total_pnl = realized + unrealized
                self.pnl_history.append(total_pnl)

                if len(self.pnl_history) == self.pnl_history.maxlen:
                    pnl_now = self.pnl_history[-1]
                    pnl_then = self.pnl_history[0]
                    if pnl_now > pnl_then: self.win_streak += 1; self.loss_streak = 0
                    elif pnl_now < pnl_then: self.loss_streak += 1; self.win_streak = 0
                    else: self.win_streak = 0; self.loss_streak = 0
                    
                    risk_cfg = CONFIG['dynamic_risk']
                    if self.win_streak >= risk_cfg['streak_threshold']:
                        new_risk = dynamic_risk_pct + risk_cfg['win_streak_increment']
                        if new_risk > dynamic_risk_pct:
                            dynamic_risk_pct = min(new_risk, risk_cfg['max_risk_pct'])
                            logging.info(f"Performance good, increasing risk to {dynamic_risk_pct*100:.2f}%")
                        self.win_streak = 0
                    elif self.loss_streak >= risk_cfg['streak_threshold']:
                        new_risk = dynamic_risk_pct - risk_cfg['loss_streak_decrement']
                        if new_risk < dynamic_risk_pct:
                            dynamic_risk_pct = max(new_risk, risk_cfg['min_risk_pct'])
                            logging.info(f"Performance poor, decreasing risk to {dynamic_risk_pct*100:.2f}%")
                        self.loss_streak = 0
        
        # 3. Correlation Analysis (outside main lock to avoid long holds)
        self._update_blacklist()

        if not IS_BACKTESTING: time.sleep(10)

class TradingAgent(BaseAgent):
    def __init__(self, symbol):
        super().__init__(f"TradingAgent-{symbol}")
        self.symbol = symbol

    def step(self):
        global CAPITAL
        with lock:
            if trading_halted:
                time.sleep(1)
                return

            p = positions[self.symbol]
            price = prices.get(self.symbol)
            if not price:
                time.sleep(1)
                return

            if p["state"] == "SOLD" and time.time() - p["last_exit"] > CONFIG['trading']['cooldown_sec']:
                p.update({"state": "IDLE", "entry": None, "peak": None,
                          "sold_1": False, "sold_2": False, "pnl": 0.0})
                p['pnl_history'].append(0.0)

            if p["state"] == "IDLE" and self.symbol not in entry_blacklist and signal_confidence(self.symbol) >= CONFIG['strategy']['entry_confidence_threshold']:
                qty = round((CAPITAL * dynamic_risk_pct) / price, 6)
                p.update({"state": "IN_POSITION", "entry": price, "peak": price, "qty": qty})
                logging.info(f"ENTERED position for {self.symbol} at {price} with confidence {signal_confidence(self.symbol)}% and risk {dynamic_risk_pct*100:.2f}%")

            if p["state"] == "IN_POSITION":
                p["peak"] = max(p.get("peak", price), price)
                a = atr(self.symbol)
                if not a:
                    time.sleep(1)
                    return
                
                if not p["sold_1"] and price >= p["entry"] + a:
                    p["sold_1"] = True
                    gain = price * (p["qty"] * 0.3)
                    CAPITAL += gain
                    p["pnl"] += gain
                    p['pnl_history'].append(gain)
                    logging.info(f"TP 1 HIT for {self.symbol} at {price}. PnL: ${gain:.2f}")
                elif not p["sold_2"] and price >= p["entry"] + 2 * a:
                    p["sold_2"] = True
                    gain = price * (p["qty"] * 0.3)
                    CAPITAL += gain
                    p["pnl"] += gain
                    p['pnl_history'].append(gain)
                    logging.info(f"TP 2 HIT for {self.symbol} at {price}. PnL: ${gain:.2f}")
                
                stop = p["peak"] - a * CONFIG['strategy']['atr_mult']
                if price <= stop:
                    gain = price * (p["qty"] * 0.4)
                    CAPITAL += gain
                    p["pnl"] += gain
                    p['pnl_history'].append(gain)
                    p.update({"state": "SOLD", "last_exit": time.time()})
                    logging.info(f"STOP HIT for {self.symbol} at {price}. PnL: ${gain:.2f}")
        
        if not IS_BACKTESTING: time.sleep(0.5)

class TradeManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("TradeManager")
        self.active_agents = {}

    def step(self):
        with symbols_lock:
            current_symbols = SYMBOLS[:]
        
        # Start agents for new symbols
        for s in current_symbols:
            if s not in self.active_agents:
                agent = TradingAgent(s)
                agent.start()
                self.active_agents[s] = agent
                logging.info(f"Started TradingAgent for {s}")
        
        # Stop agents for removed symbols
        stale_symbols = [s for s in self.active_agents if s not in current_symbols]
        for s in stale_symbols:
            self.active_agents[s].stop()
            del self.active_agents[s]
            logging.info(f"Stopped TradingAgent for {s}")
            
        if not IS_BACKTESTING: time.sleep(5)

class WebSocketManager(BaseAgent):
    def __init__(self):
        super().__init__("WebSocketManager")
        self.ws = None
        self.subscribed_symbols = set()

    def _connect(self):
        if self.ws:
            self.ws.close()
        self.ws = websocket.create_connection(CONFIG['api']['ws_base_url'])
        logging.info("WebSocket connection established.")
        self.subscribed_symbols = set()

    def _subscribe(self, symbols_to_add):
        if not symbols_to_add: return
        params = [f"{s.lower()}@ticker" for s in symbols_to_add]
        subscribe_message = {"method": "SUBSCRIBE", "params": params, "id": int(time.time())}
        self.ws.send(json.dumps(subscribe_message))
        self.subscribed_symbols.update(symbols_to_add)
        logging.info(f"Subscribed to WebSocket tickers: {list(symbols_to_add)}")

    def _unsubscribe(self, symbols_to_remove):
        if not symbols_to_remove: return
        params = [f"{s.lower()}@ticker" for s in symbols_to_remove]
        unsubscribe_message = {"method": "UNSUBSCRIBE", "params": params, "id": int(time.time())}
        self.ws.send(json.dumps(unsubscribe_message))
        self.subscribed_symbols.difference_update(symbols_to_remove)
        logging.info(f"Unsubscribed from WebSocket tickers: {list(symbols_to_remove)}")
        
    def run(self):
        while not self.stopped():
            try:
                self._connect()
                self.step() # Initial subscription
                last_sub_check = time.time()
                
                while not self.stopped():
                    if time.time() - last_sub_check > 10:
                        self.step()
                        last_sub_check = time.time()

                    try:
                        message = self.ws.recv()
                        data = json.loads(message)
                        
                        if 's' in data and 'c' in data:
                            symbol = data['s']
                            if symbol in self.subscribed_symbols:
                                price = float(data['c'])
                                with lock:
                                    if symbol in prices:
                                        prices[symbol] = price
                                        if symbol in rolling_prices:
                                            rolling_prices[symbol].append(price)
                    except websocket.WebSocketTimeoutException:
                        if not IS_BACKTESTING: time.sleep(1) # Add a small sleep to prevent busy looping in live mode
                        continue
                    except websocket.WebSocketConnectionClosedException:
                        logging.warning("WebSocket connection closed. Reconnecting...")
                        break
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logging.error(f"WebSocketManager error: {e}", exc_info=True)
                if self.ws: self.ws.close()
                if not IS_BACKTESTING: time.sleep(5)

    def step(self):
        with symbols_lock:
            current_symbols = set(SYMBOLS)
        symbols_to_add = current_symbols - self.subscribed_symbols
        symbols_to_remove = self.subscribed_symbols - current_symbols
        
        if symbols_to_add: self._subscribe(symbols_to_add)
        if symbols_to_remove: self._unsubscribe(symbols_to_remove)





# ================= STATE MANAGEMENT =================
STATE_FILE = os.path.join(script_dir, 'trinity_state.json')

def save_state():
    with lock:
        state = {
            'capital': CAPITAL,
            'capital_start': CAPITAL_START,
            'positions': {s: dict(p) for s, p in positions.items()}
        }
        # Convert deques to lists for JSON serialization
        for s in state['positions']:
            if 'pnl_history' in state['positions'][s]:
                state['positions'][s]['pnl_history'] = list(state['positions'][s]['pnl_history'])
            
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    logging.info(f"State saved to {STATE_FILE}")

def load_state():
    global CAPITAL, CAPITAL_START, positions, SYMBOLS
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        
        CAPITAL = state['capital']
        CAPITAL_START = state.get('capital_start', CAPITAL)

        temp_positions = state['positions']
        loaded_symbols = list(temp_positions.keys())

        with lock:
            for s in loaded_symbols:
                if s not in prices:
                    prices[s] = 0.0
                    rolling_prices[s] = deque(maxlen=CONFIG['display']['rolling_len'])
                    ohlc[s] = []
                    htf_ohlc[s] = []
            
            for s, p_dict in temp_positions.items():
                p_dict['pnl_history'] = deque(p_dict.get('pnl_history', []), maxlen=CONFIG['display']['pnl_history_len'])
            positions.update(temp_positions)
        
        with symbols_lock:
            SYMBOLS = list(set(SYMBOLS + loaded_symbols))

        logging.info(f"State loaded from {STATE_FILE} for {len(loaded_symbols)} symbols.")

    except FileNotFoundError:
        logging.warning("No state file found. Starting with fresh capital.")
        CAPITAL_START = CONFIG['trading']['capital_start']
        CAPITAL = CAPITAL_START
    except Exception as e:
        logging.error(f"Error loading state file: {e}. Starting fresh.")
        CAPITAL_START = CONFIG['trading']['capital_start']
        CAPITAL = CAPITAL_START

# ================= PNL BAR =================
def pnl_bar(pnl_history):
    bar=""
    for v in pnl_history:
        if v>0: bar+=GREEN+"█"
        elif v<0: bar+=ORANGE+"░"
        else: bar+=GREY+"─"
    return bar+Style.RESET_ALL

# ================= FETCH OHLC =================
def ohlc_worker():
    while True:
        with symbols_lock:
            current_symbols = SYMBOLS[:]
        with scanner_lock:
            symbols_to_fetch = list(set(current_symbols + potential_symbols))
        
        for s in symbols_to_fetch:
            for tf, store in [(CONFIG['strategy']['timeframe'], ohlc),(CONFIG['strategy']['htf_timeframe'],htf_ohlc)]:
                try:
                    r=requests.get(f"{CONFIG['api']['rest_base_url']}/api/v3/klines",params={"symbol":s,"interval":tf,"limit":300},timeout=5).json()
                    store[s]=[{"h":float(c[2]),"l":float(c[3]),"c":float(c[4])} for c in r]
                except:
                    continue
        if not IS_BACKTESTING: time.sleep(30)

def run_backtest():
    global IS_BACKTESTING, CAPITAL, CAPITAL_START, SYMBOLS, prices, rolling_prices, ohlc, htf_ohlc, positions, dynamic_risk_pct, entry_blacklist

    IS_BACKTESTING = True
    logging.info("Starting backtesting mode...")

    # Load historical data
    # This is a placeholder, will implement actual loading later
    historical_data = {} # {symbol: {timestamp: {open, high, low, close, volume}}}
    data_folder = CONFIG['backtest']['data_folder']
    start_date_bt = datetime.fromisoformat(CONFIG['backtest']['start_date'].replace('Z', '+00:00'))
    end_date_bt = datetime.fromisoformat(CONFIG['backtest']['end_date'].replace('Z', '+00:00'))

    # Temporary: populate some dummy historical data for now
    # In real implementation, this would load from CSVs
    # For now, let's just make sure the loop runs.
    logging.info(f"Loading historical data from {data_folder} for {start_date_bt} to {end_date_bt}")
    
    # Initialize agents (without starting threads)
    scanner_agent = MarketScannerAgent()
    risk_agent = RiskManagerAgent()
    trade_manager_agent = TradeManagerAgent()
    dashboard_agent = DashboardAgent() # Will only log in backtest mode

    # Reset state for backtest
    CAPITAL = CONFIG['trading']['capital_start']
    CAPITAL_START = CAPITAL
    SYMBOLS = []
    prices.clear()
    rolling_prices.clear()
    ohlc.clear()
    htf_ohlc.clear()
    positions.clear()
    dynamic_risk_pct = CONFIG['trading']['risk_pct']
    entry_blacklist.clear()

    # Backtest simulation loop (placeholder)
    current_time = start_date_bt
    while current_time <= end_date_bt:
        # Simulate time passing
        # Here, load data for current_time and update global state
        
        # In a real backtest, this would load actual historical klines
        # For demonstration: assume some data is available.
        # This part will be filled out once we have loaded CSV data.

        # Manually step agents
        # The agents' step methods should not block or sleep in backtest mode
        scanner_agent.step()
        risk_agent.step()
        trade_manager_agent.step()
        dashboard_agent.step() # Logs current state

        # Advance time by the smallest timeframe (e.g., 1 minute)
        current_time += timedelta(minutes=1)

    logging.info("Backtesting complete. Generating report...")
    # TODO: Generate a proper backtest report here.

    logging.info(f"-----------------------")

def run_live_trading():
    global IS_BACKTESTING
    IS_BACKTESTING = False
    load_state()
    # Data fetchers can remain as simple threads
    threading.Thread(target=ohlc_worker, daemon=True, name="OHLCWorker").start()

    agents = [
        WebSocketManager(),
        MarketScannerAgent(),
        RiskManagerAgent(),
        TradeManagerAgent(),
        DashboardAgent()
    ]

    for agent in agents:
        agent.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Saving state and stopping agents...")
        save_state()
        for agent in agents:
            agent.stop()
        for agent in agents:
            agent.join()
        logging.info("All agents stopped. Goodbye.")


# ================= START =================
if __name__=="__main__":
    if CONFIG['backtest']['enabled']:
        run_backtest()
    else:
        run_live_trading()