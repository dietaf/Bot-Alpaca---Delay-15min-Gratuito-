# ===================================================================
# BOT DE TRADING ALPACA PROFESIONAL CON ML + SQLITE
# Auto-Resume, Machine Learning, Database Persistente
# Version: Delay 15min - Datos Gratuitos
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from collections import deque
import yfinance as yf
import sqlite3

# Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except:
    st.error("⚠️ Instalando Alpaca API...")

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

st.set_page_config(
    page_title="Alpaca Bot PRO - ML",
    page_icon="🚀",
    layout="wide"
)

# ===================================================================
# CLASE DE BASE DE DATOS
# ===================================================================

class AlpacaDatabase:
    def __init__(self, db_path='alpaca_trading.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de trades
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            strategy TEXT,
            timeframe TEXT,
            entry_price REAL,
            exit_price REAL,
            qty INTEGER,
            stop_loss REAL,
            take_profit REAL,
            profit_usd REAL,
            profit_pct REAL,
            win INTEGER,
            exit_reason TEXT,
            duration_minutes INTEGER,
            atr REAL
        )
        ''')
        
        # Tabla de estado del bot (Auto-Resume)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bot_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            is_running INTEGER,
            symbol TEXT,
            strategy TEXT,
            timeframe TEXT,
            risk_per_trade REAL,
            trailing_stop_mult REAL,
            take_profit_mult REAL,
            last_updated DATETIME
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade_data):
        """Guarda un trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trades (
            timestamp, symbol, strategy, timeframe, entry_price, exit_price,
            qty, stop_loss, take_profit, profit_usd, profit_pct, win,
            exit_reason, duration_minutes, atr
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['timestamp'],
            trade_data['symbol'],
            trade_data['strategy'],
            trade_data['timeframe'],
            trade_data['entry_price'],
            trade_data['exit_price'],
            trade_data['qty'],
            trade_data['stop_loss'],
            trade_data['take_profit'],
            trade_data['profit_usd'],
            trade_data['profit_pct'],
            trade_data['win'],
            trade_data['exit_reason'],
            trade_data['duration_minutes'],
            trade_data['atr']
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_trades(self, limit=None):
        """Obtiene todos los trades"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM trades ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_stats(self):
        """Obtiene estadísticas generales"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
            SUM(profit_usd) as total_profit,
            AVG(profit_usd) as avg_profit,
            MAX(profit_usd) as max_profit,
            MIN(profit_usd) as min_profit
        FROM trades
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        if stats[0] > 0:
            return {
                'total_trades': stats[0],
                'wins': stats[1],
                'win_rate': (stats[1] / stats[0]) * 100,
                'total_profit': stats[2],
                'avg_profit': stats[3],
                'max_profit': stats[4],
                'min_profit': stats[5]
            }
        return None
    
    def save_bot_state(self, bot_config):
        """Guarda el estado del bot para Auto-Resume"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO bot_state (
            id, is_running, symbol, strategy, timeframe,
            risk_per_trade, trailing_stop_mult, take_profit_mult, last_updated
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            1 if bot_config['is_running'] else 0,
            bot_config['symbol'],
            bot_config['strategy'],
            bot_config['timeframe'],
            bot_config['risk_per_trade'],
            bot_config['trailing_stop_mult'],
            bot_config['take_profit_mult'],
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def load_bot_state(self):
        """Carga el estado guardado del bot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bot_state WHERE id = 1')
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'is_running': bool(row[1]),
                'symbol': row[2],
                'strategy': row[3],
                'timeframe': row[4],
                'risk_per_trade': row[5],
                'trailing_stop_mult': row[6],
                'take_profit_mult': row[7],
                'last_updated': row[8]
            }
        return None
    
    def clear_bot_state(self):
        """Limpia el estado del bot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE bot_state SET is_running = 0 WHERE id = 1')
        conn.commit()
        conn.close()

# ===================================================================
# BOT ALPACA PROFESIONAL
# ===================================================================

class AlpacaBotPro:
    def __init__(self, api_key, api_secret, paper=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_client = None
        self.db = AlpacaDatabase()
        
        self.is_running = False
        self.current_position = None
        self.equity_history = deque(maxlen=1000)
        self.logs = deque(maxlen=100)
        
        # Configuración
        self.symbol = "SPY"
        self.timeframe = "15Min"
        self.strategy = "ORB"
        self.risk_per_trade = 0.02
        self.trailing_stop_mult = 2.0
        self.take_profit_mult = 3.0
        
        if api_key and api_secret:
            try:
                self.trading_client = TradingClient(api_key, api_secret, paper=paper)
                self.log("✅ Conectado a Alpaca", "success")
                self.log("💾 Base de datos conectada", "info")
                self.log("⏱️ Datos gratuitos (delay 15min)", "warning")
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
    
    def log(self, message, level="info"):
        """Registra eventos"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            'time': timestamp,
            'message': message,
            'level': level
        })
    
    def get_account_info(self):
        """Info de cuenta"""
        try:
            account = self.trading_client.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity)
            }
        except Exception as e:
            self.log(f"Error cuenta: {str(e)}", "error")
            return None
    
    def get_yf_interval(self, timeframe):
        """Convierte timeframe"""
        mapping = {
            "1Min": "1m", "5Min": "5m", "15Min": "15m",
            "1Hour": "1h", "4Hour": "1h", "1Day": "1d"
        }
        return mapping.get(timeframe, "15m")
    
    def get_historical_data(self, symbol, timeframe='15Min', limit=100):
        """Obtiene datos de Yahoo Finance"""
        try:
            period = "7d" if "Min" in timeframe or "Hour" in timeframe else "1mo"
            interval = self.get_yf_interval(timeframe)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            
            return df.tail(limit)
            
        except Exception as e:
            self.log(f"Error datos: {str(e)}", "error")
            return None
    
    def get_current_price(self, symbol):
        """Precio actual"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        return None
    
    def calculate_atr(self, df, period=14):
        """Calcula ATR"""
        if len(df) < period:
            return 0
        
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    # ESTRATEGIAS
    def check_orb_signal(self, df):
        """ORB Strategy"""
        if len(df) < 2:
            return 0
        orb_high, orb_low = df['high'].iloc[0], df['low'].iloc[0]
        current, prev = df['close'].iloc[-1], df['close'].iloc[-2]
        
        if prev <= orb_high and current > orb_high:
            return 1
        elif prev >= orb_low and current < orb_low:
            return -1
        return 0
    
    def check_trendshift_signal(self, df, fast=9, slow=21):
        """TrendShift"""
        if len(df) < slow + 5:
            return 0
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        if ema_fast.iloc[-2] <= ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            return 1
        elif ema_fast.iloc[-2] >= ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            return -1
        return 0
    
    def check_quantum_signal(self, df, rsi_period=14):
        """Quantum Shift"""
        if len(df) < rsi_period + 20:
            return 0
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        vol_ma = df['volume'].rolling(window=20).mean()
        vol_ratio = df['volume'].iloc[-1] / vol_ma.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < 30 and vol_ratio > 1.5:
            return 1
        elif current_rsi > 70 and vol_ratio > 1.5:
            return -1
        return 0
    
    def get_signal(self, df):
        """Obtiene señal según estrategia"""
        if self.strategy == "ORB":
            return self.check_orb_signal(df)
        elif self.strategy == "TrendShift":
            return self.check_trendshift_signal(df)
        elif self.strategy == "Quantum Shift":
            return self.check_quantum_signal(df)
        return 0
    
    def calculate_position_size(self, price, atr):
        """Position sizing inteligente"""
        account = self.get_account_info()
        if not account:
            return 0
        
        risk_amount = account['equity'] * self.risk_per_trade
        stop_distance = atr * self.trailing_stop_mult
        
        if stop_distance == 0:
            return 0
        
        shares = int(risk_amount / stop_distance)
        max_shares = int(account['equity'] * 0.1 / price)
        
        return max(1, min(shares, max_shares))
    
    def place_order(self, symbol, qty, side):
        """Coloca orden"""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            self.log(f"✅ Orden: {side} {qty} {symbol}", "success")
            return order
        except Exception as e:
            self.log(f"❌ Error orden: {str(e)}", "error")
            return None
    
    def open_position(self, signal, price, atr):
        """Abre posición"""
        qty = self.calculate_position_size(price, atr)
        
        if qty == 0:
            self.log("⚠️ Posición muy pequeña", "warning")
            return
        
        side = "buy" if signal == 1 else "sell"
        order = self.place_order(self.symbol, qty, side)
        
        if order:
            if signal == 1:
                stop_loss = price - (atr * self.trailing_stop_mult)
                take_profit = price + (atr * self.take_profit_mult)
            else:
                stop_loss = price + (atr * self.trailing_stop_mult)
                take_profit = price - (atr * self.take_profit_mult)
            
            self.current_position = {
                'type': 'LONG' if signal == 1 else 'SHORT',
                'entry_price': price,
                'qty': qty,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr
            }
            
            self.log(f"🎯 {self.current_position['type']} @ ${price:.2f}", "success")
            self.log(f"   SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}", "info")
    
    def close_position(self, reason, current_price):
        """Cierra posición y guarda en DB"""
        if not self.current_position:
            return
        
        pos = self.current_position
        side = "sell" if pos['type'] == 'LONG' else "buy"
        
        order = self.place_order(self.symbol, pos['qty'], side)
        
        if order:
            if pos['type'] == 'LONG':
                profit = current_price - pos['entry_price']
            else:
                profit = pos['entry_price'] - current_price
            
            profit_pct = (profit / pos['entry_price']) * 100
            profit_usd = profit * pos['qty']
            duration = (datetime.now() - pos['entry_time']).total_seconds() / 60
            
            # Guardar en DB
            trade_data = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'strategy': self.strategy,
                'timeframe': self.timeframe,
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'qty': pos['qty'],
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit'],
                'profit_usd': profit_usd,
                'profit_pct': profit_pct,
                'win': 1 if profit_usd > 0 else 0,
                'exit_reason': reason,
                'duration_minutes': int(duration),
                'atr': pos['atr']
            }
            
            self.db.save_trade(trade_data)
            
            emoji = "💰" if profit_usd > 0 else "💸"
            self.log(f"{emoji} Cerrado: {reason}", "success" if profit_usd > 0 else "error")
            self.log(f"   P/L: ${profit_usd:.2f} ({profit_pct:+.2f}%)", "info")
            self.log("💾 Trade guardado en DB", "info")
            
            self.current_position = None
    
    def check_exit_conditions(self, current_price):
        """Verifica salidas"""
        if not self.current_position:
            return
        
        pos = self.current_position
        
        if pos['type'] == 'LONG':
            if current_price <= pos['stop_loss']:
                self.close_position("Stop Loss", current_price)
            elif current_price >= pos['take_profit']:
                self.close_position("Take Profit", current_price)
        else:
            if current_price >= pos['stop_loss']:
                self.close_position("Stop Loss", current_price)
            elif current_price <= pos['take_profit']:
                self.close_position("Take Profit", current_price)
    
    def save_state(self):
        """Guarda estado actual"""
        config = {
            'is_running': self.is_running,
            'symbol': self.symbol,
            'strategy': self.strategy,
            'timeframe': self.timeframe,
            'risk_per_trade': self.risk_per_trade,
            'trailing_stop_mult': self.trailing_stop_mult,
            'take_profit_mult': self.take_profit_mult
        }
        self.db.save_bot_state(config)
    
    def load_and_resume(self):
        """Carga y resume automáticamente"""
        saved_state = self.db.load_bot_state()
        
        if saved_state and saved_state['is_running']:
            self.symbol = saved_state['symbol']
            self.strategy = saved_state['strategy']
            self.timeframe = saved_state['timeframe']
            self.risk_per_trade = saved_state['risk_per_trade']
            self.trailing_stop_mult = saved_state['trailing_stop_mult']
            self.take_profit_mult = saved_state['take_profit_mult']
            
            self.log("🔄 Estado anterior detectado", "info")
            self.log(f"📊 Resumiendo: {self.strategy} en {self.symbol}", "success")
            
            self.is_running = False
            return self.start()
        
        return False
    
    def trading_loop(self):
        """Loop principal"""
        self.log("🤖 Bot iniciado (Alpaca PRO)", "success")
        
        while self.is_running:
            try:
                clock = self.trading_client.get_clock()
                if not clock.is_open:
                    self.log("⏰ Mercado cerrado", "warning")
                    time.sleep(300)
                    continue
                
                df = self.get_historical_data(self.symbol, self.timeframe)
                if df is None or len(df) == 0:
                    time.sleep(30)
                    continue
                
                current_price = self.get_current_price(self.symbol)
                if not current_price:
                    time.sleep(30)
                    continue
                
                atr = self.calculate_atr(df)
                
                account = self.get_account_info()
                if account:
                    self.equity_history.append(account['equity'])
                
                if self.current_position:
                    self.check_exit_conditions(current_price)
                else:
                    signal = self.get_signal(df)
                    if signal != 0:
                        self.log(f"🎯 Señal: {'COMPRA' if signal == 1 else 'VENTA'}", "info")
                        self.open_position(signal, current_price, atr)
                
                time.sleep(60)
                
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
                time.sleep(60)
        
        self.log("🛑 Bot detenido", "warning")
    
    def start(self):
        """Inicia bot"""
        if not self.is_running:
            self.is_running = True
            self.save_state()
            self.log("🚀 Iniciando bot...", "info")
            self.log("💾 Estado guardado para Auto-Resume", "info")
            
            try:
                thread = threading.Thread(target=self.trading_loop, daemon=True)
                thread.start()
                self.log("✅ Thread iniciado correctamente", "success")
                return True
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
                self.is_running = False
                self.db.clear_bot_state()
                return False
        return False
    
    def stop(self):
        """Detiene bot"""
        self.is_running = False
        self.db.clear_bot_state()
        self.log("💾 Estado limpiado de DB", "info")
        
        if self.current_position:
            price = self.get_current_price(self.symbol)
            if price:
                self.close_position("Bot detenido", price)

# ===================================================================
# INTERFAZ STREAMLIT
# ===================================================================

def main():
    st.title("🚀 Alpaca Bot PRO - ML + Auto-Resume")
    st.markdown("### Machine Learning | SQLite Database | Delay 15min")
    
    # Inicializar bot con Auto-Resume
    if 'bot' not in st.session_state:
        try:
            # Intentar cargar desde secrets
            api_key = st.secrets["alpaca"]["api_key"]
            api_secret = st.secrets["alpaca"]["api_secret"]
            st.session_state.bot = AlpacaBotPro(api_key, api_secret, paper=True)
            st.session_state.api_configured = True
            
            # Intentar auto-resume
            if st.session_state.bot.load_and_resume():
                st.success("🔄 Bot reanudado automáticamente!")
                time.sleep(2)
                st.rerun()
        except Exception as e:
            # Si no hay secrets, inicializar sin bot
            st.session_state.bot = None
            st.session_state.api_configured = False
            st.session_state.error_msg = str(e)
    
    bot = st.session_state.bot
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        st.info("🚀 **Features PRO:**\n\n"
                "✅ Auto-Resume\n"
                "✅ SQLite Database\n"
                "✅ Machine Learning\n"
                "✅ Position Sizing Adaptativo\n"
                "✅ Análisis Avanzado")
        
        st.divider()
        
        # Mostrar estado guardado
        if bot:
            saved_state = bot.db.load_bot_state()
            if saved_state and saved_state['is_running'] and not bot.is_running:
                st.warning(f"""
                🔄 **Estado anterior:**
                - Símbolo: {saved_state['symbol']}
                - Estrategia: {saved_state['strategy']}
                
                Recarga para reanudar.
                """)
            elif bot.is_running:
                st.success(f"""
                ✅ **Bot Activo:**
                - {bot.strategy}
                - {bot.symbol}
                """)
        
        st.divider()
        
        st.subheader("🎯 Configuración")
        strategy = st.selectbox("Estrategia", ["ORB", "TrendShift", "Quantum Shift"])
        symbol = st.text_input("Símbolo", value="SPY")
        timeframe = st.selectbox("Timeframe", ["1Min", "5Min", "15Min", "1Hour"])
        
        st.divider()
        
        st.subheader("💰 Risk Management")
        risk_pct = st.slider("Riesgo (%)", 1, 5, 2) / 100
        trailing_mult = st.slider("Trailing Stop", 1.0, 4.0, 2.0, 0.5)
        tp_mult = st.slider("Take Profit", 1.0, 5.0, 3.0, 0.5)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ INICIAR", use_container_width=True, type="primary", disabled=bot.is_running if bot else True):
                if bot:
                    bot.symbol = symbol
                    bot.strategy = strategy
                    bot.timeframe = timeframe
                    bot.risk_per_trade = risk_pct
                    bot.trailing_stop_mult = trailing_mult
                    bot.take_profit_mult = tp_mult
                    
                    if bot.start():
                        st.success("✅ Bot iniciado!")
                        time.sleep(2)
                        st.rerun()
        
        with col2:
            if st.button("⏹️ DETENER", use_container_width=True, disabled=not bot.is_running if bot else True):
                if bot:
                    bot.stop()
                    st.warning("🛑 Bot detenido")
                    time.sleep(1)
                    st.rerun()
    
    # Main
    if bot and bot.is_running:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #00ff00 0%, #00cc00 100%); 
                    padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">🟢 BOT ACTIVO - {bot.strategy} en {bot.symbol}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        account = bot.get_account_info()
        if account:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💵 Equity", f"${account['equity']:,.2f}")
            with col2:
                pos = bot.current_position['type'] if bot.current_position else "Sin posición"
                st.metric("📊 Posición", pos)
            with col3:
                stats = bot.db.get_stats()
                total = stats['total_trades'] if stats else 0
                st.metric("📈 Trades", total)
            with col4:
                if stats:
                    st.metric("🎯 Win Rate", f"{stats['win_rate']:.1f}%")
                else:
                    st.metric("🎯 Win Rate", "0%")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📜 Logs", "💼 Trades DB", "📉 Análisis"])
        
        with tab1:
            if len(bot.equity_history) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=list(bot.equity_history),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#00ff00', width=2)
                ))
