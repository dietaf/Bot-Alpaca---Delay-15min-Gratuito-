# ===================================================================
# BOT DE TRADING AUTOMATIZADO - ALPACA (DELAY 15 MIN)
# Version: Datos Gratuitos - Sin suscripción requerida
# Max Way Strategies con datos retrasados 15 minutos
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

# Importar Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except:
    st.error("Instalando Alpaca API...")

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

st.set_page_config(
    page_title="Alpaca Bot - Delay 15min",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# CLASE DEL BOT CON DATOS GRATUITOS
# ===================================================================

class AlpacaTradingBotDelayed:
    def __init__(self, api_key, api_secret, paper=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_client = None
        self.is_running = False
        self.current_position = None
        self.trades_history = []
        self.equity_history = deque(maxlen=1000)
        self.logs = deque(maxlen=100)
        
        # Configuración
        self.symbol = "SPY"
        self.timeframe = "15Min"
        self.strategy = "ORB"
        self.capital = 10000
        self.risk_per_trade = 0.02
        self.trailing_stop_mult = 2.0
        self.take_profit_mult = 3.0
        
        # Estado
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.position_size = 0
        
        if api_key and api_secret:
            try:
                self.trading_client = TradingClient(api_key, api_secret, paper=paper)
                self.log("✅ Conectado a Alpaca", "success")
                self.log("⏱️ Usando datos GRATUITOS (delay 15min)", "warning")
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
        """Convierte timeframe a intervalo de yfinance"""
        mapping = {
            "1Min": "1m",
            "5Min": "5m",
            "15Min": "15m",
            "1Hour": "1h",
            "4Hour": "1h",  # yfinance no tiene 4h, usamos 1h
            "1Day": "1d"
        }
        return mapping.get(timeframe, "15m")
    
    def get_historical_data(self, symbol, timeframe='15Min', limit=100):
        """Obtiene datos de Yahoo Finance (GRATIS)"""
        try:
            # Yahoo Finance usa días como periodo
            if "Min" in timeframe or "Hour" in timeframe:
                period = "7d"  # Última semana para intradía
            else:
                period = "1mo"  # Último mes para daily
            
            interval = self.get_yf_interval(timeframe)
            
            # Descargar datos
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                self.log(f"⚠️ No hay datos para {symbol}", "warning")
                return None
            
            # Renombrar columnas para compatibilidad
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Tomar últimas N barras
            df = df.tail(limit)
            
            self.log(f"📊 Datos obtenidos: {len(df)} barras", "info")
            return df
            
        except Exception as e:
            self.log(f"Error obteniendo datos: {str(e)}", "error")
            return None
    
    def get_current_price(self, symbol):
        """Obtiene precio actual de Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            self.log(f"Error precio: {str(e)}", "error")
            return None
    
    def calculate_atr(self, df, period=14):
        """Calcula ATR"""
        if len(df) < period:
            return 0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    # ===============================================================
    # ESTRATEGIAS
    # ===============================================================
    
    def check_orb_signal(self, df):
        """ORB Strategy"""
        if len(df) < 2:
            return 0
        
        orb_high = df['high'].iloc[0]
        orb_low = df['low'].iloc[0]
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        if prev_price <= orb_high and current_price > orb_high:
            return 1
        elif prev_price >= orb_low and current_price < orb_low:
            return -1
        
        return 0
    
    def check_pivot_signal(self, df, window=20):
        """Pivot Hunter"""
        if len(df) < window * 2:
            return 0
        
        pivot_high = df['high'].iloc[-window*2:-window].max()
        pivot_low = df['low'].iloc[-window*2:-window].min()
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        if prev_price <= pivot_high and current_price > pivot_high:
            return 1
        elif prev_price >= pivot_low and current_price < pivot_low:
            return -1
        
        return 0
    
    def check_gap_signal(self, df, threshold=0.5):
        """Gap Trading"""
        if len(df) < 2:
            return 0
        
        prev_close = df['close'].iloc[-2]
        current_open = df['open'].iloc[-1]
        
        gap_pct = ((current_open - prev_close) / prev_close) * 100
        
        if gap_pct > threshold:
            return 1
        elif gap_pct < -threshold:
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
    
    def get_signal(self, df):
        """Obtiene señal según estrategia"""
        if self.strategy == "ORB":
            return self.check_orb_signal(df)
        elif self.strategy == "Pivot Hunter":
            return self.check_pivot_signal(df)
        elif self.strategy == "Gap Trading":
            return self.check_gap_signal(df)
        elif self.strategy == "Quantum Shift":
            return self.check_quantum_signal(df)
        elif self.strategy == "TrendShift":
            return self.check_trendshift_signal(df)
        return 0
    
    # ===============================================================
    # EJECUCIÓN DE TRADES
    # ===============================================================
    
    def calculate_position_size(self, price, atr):
        """Calcula tamaño de posición"""
        account = self.get_account_info()
        if not account:
            return 0
        
        risk_amount = account['equity'] * self.risk_per_trade
        stop_distance = atr * self.trailing_stop_mult
        
        if stop_distance == 0:
            return 0
        
        shares = int(risk_amount / stop_distance)
        max_shares = int(account['equity'] * 0.1 / price)
        
        return min(shares, max_shares, 1)
    
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
            self.current_position = {
                'type': 'LONG' if signal == 1 else 'SHORT',
                'entry_price': price,
                'qty': qty,
                'entry_time': datetime.now()
            }
            
            if signal == 1:
                self.stop_loss = price - (atr * self.trailing_stop_mult)
                self.take_profit = price + (atr * self.take_profit_mult)
            else:
                self.stop_loss = price + (atr * self.trailing_stop_mult)
                self.take_profit = price - (atr * self.take_profit_mult)
            
            self.log(f"🎯 {self.current_position['type']} @ ${price:.2f}", "success")
            self.log(f"   SL: ${self.stop_loss:.2f} | TP: ${self.take_profit:.2f}", "info")
    
    def update_trailing_stop(self, current_price, atr):
        """Actualiza trailing stop"""
        if not self.current_position:
            return
        
        if self.current_position['type'] == 'LONG':
            new_stop = current_price - (atr * self.trailing_stop_mult)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
                self.log(f"📈 Trailing stop: ${self.stop_loss:.2f}", "info")
        else:
            new_stop = current_price + (atr * self.trailing_stop_mult)
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop
                self.log(f"📉 Trailing stop: ${self.stop_loss:.2f}", "info")
    
    def close_position(self, reason, current_price):
        """Cierra posición"""
        if not self.current_position:
            return
        
        side = "sell" if self.current_position['type'] == 'LONG' else "buy"
        qty = self.current_position['qty']
        
        order = self.place_order(self.symbol, qty, side)
        
        if order:
            if self.current_position['type'] == 'LONG':
                profit = current_price - self.current_position['entry_price']
            else:
                profit = self.current_position['entry_price'] - current_price
            
            profit_pct = (profit / self.current_position['entry_price']) * 100
            profit_usd = profit * qty
            
            self.trades_history.append({
                'entry_time': self.current_position['entry_time'],
                'exit_time': datetime.now(),
                'type': self.current_position['type'],
                'entry_price': self.current_position['entry_price'],
                'exit_price': current_price,
                'qty': qty,
                'profit': profit_usd,
                'profit_pct': profit_pct,
                'reason': reason
            })
            
            emoji = "💰" if profit_usd > 0 else "💸"
            self.log(f"{emoji} Cerrado: {reason}", "success" if profit_usd > 0 else "error")
            self.log(f"   P/L: ${profit_usd:.2f} ({profit_pct:+.2f}%)", "info")
            
            self.current_position = None
    
    def check_exit_conditions(self, current_price):
        """Verifica salidas"""
        if not self.current_position:
            return
        
        if self.current_position['type'] == 'LONG':
            if current_price <= self.stop_loss:
                self.close_position("Stop Loss", current_price)
            elif current_price >= self.take_profit:
                self.close_position("Take Profit", current_price)
        else:
            if current_price >= self.stop_loss:
                self.close_position("Stop Loss", current_price)
            elif current_price <= self.take_profit:
                self.close_position("Take Profit", current_price)
    
    # ===============================================================
    # LOOP PRINCIPAL
    # ===============================================================
    
    def trading_loop(self):
        """Loop principal"""
        self.log("🤖 Bot iniciado (Delay 15min)", "success")
        
        while self.is_running:
            try:
                clock = self.trading_client.get_clock()
                if not clock.is_open:
                    self.log("⏰ Mercado cerrado", "warning")
                    time.sleep(60)
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
                    self.update_trailing_stop(current_price, atr)
                else:
                    signal = self.get_signal(df)
                    if signal != 0:
                        self.log(f"🎯 Señal: {'COMPRA' if signal == 1 else 'VENTA'}", "info")
                        self.open_position(signal, current_price, atr)
                
                time.sleep(30)
                
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
                time.sleep(60)
        
        self.log("🛑 Bot detenido", "warning")
    
    def start(self):
        """Inicia bot"""
        if not self.is_running:
            self.is_running = True
            thread = threading.Thread(target=self.trading_loop, daemon=True)
            thread.start()
            return True
        return False
    
    def stop(self):
        """Detiene bot"""
        self.is_running = False
        if self.current_position:
            price = self.get_current_price(self.symbol)
            if price:
                self.close_position("Bot detenido", price)

# ===================================================================
# INTERFAZ STREAMLIT
# ===================================================================

def main():
    st.title("⏱️ Alpaca Bot - DELAY 15 MIN")
    st.markdown("### Versión GRATUITA con datos retrasados")
    
    # Warning prominente
    st.warning("""
    ⏱️ **IMPORTANTE:** Este bot usa datos con **delay de 15 minutos** (gratuitos).
    Las señales y trades se ejecutan con retraso. Ideal para backtesting y práctica.
    """)
    
    if 'bot' not in st.session_state:
        st.session_state.bot = None
        st.session_state.bot_running = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        st.subheader("🔑 Alpaca API")
        try:
            api_key = st.secrets["alpaca"]["api_key"]
            api_secret = st.secrets["alpaca"]["api_secret"]
            st.success("✅ Keys desde secrets")
        except:
            api_key = st.text_input("API Key", type="password")
            api_secret = st.text_input("API Secret", type="password")
        
        paper_trading = st.checkbox("Paper Trading", value=True)
        
        st.divider()
        
        st.subheader("🎯 Estrategia")
        strategy = st.selectbox(
            "Estrategia",
            ["ORB", "Pivot Hunter", "Gap Trading", "Quantum Shift", "TrendShift"]
        )
        
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
            if st.button("▶️ INICIAR", use_container_width=True, type="primary"):
                if api_key and api_secret:
                    st.session_state.bot = AlpacaTradingBotDelayed(api_key, api_secret, paper_trading)
                    st.session_state.bot.strategy = strategy
                    st.session_state.bot.symbol = symbol
                    st.session_state.bot.timeframe = timeframe
                    st.session_state.bot.risk_per_trade = risk_pct
                    st.session_state.bot.trailing_stop_mult = trailing_mult
                    st.session_state.bot.take_profit_mult = tp_mult
                    
                    if st.session_state.bot.start():
                        st.session_state.bot_running = True
                        st.success("✅ Bot iniciado!")
                        st.rerun()
                else:
                    st.error("⚠️ Ingresa API keys")
        
        with col2:
            if st.button("⏹️ DETENER", use_container_width=True):
                if st.session_state.bot:
                    st.session_state.bot.stop()
                    st.session_state.bot_running = False
                    st.warning("🛑 Bot detenido")
                    st.rerun()
    
    # Main
    if st.session_state.bot and st.session_state.bot_running:
        bot = st.session_state.bot
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #ffa500 0%, #ff8c00 100%); 
                    padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">⏱️ BOT ACTIVO (Delay 15min) - {bot.strategy} en {bot.symbol}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        account = bot.get_account_info()
        if account:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💵 Balance", f"${account['equity']:,.2f}")
            with col2:
                pos = "LONG" if bot.current_position and bot.current_position['type'] == 'LONG' else "SHORT" if bot.current_position else "Sin posición"
                st.metric("📊 Posición", pos)
            with col3:
                st.metric("📈 Trades", len(bot.trades_history))
            with col4:
                if bot.trades_history:
                    wins = len([t for t in bot.trades_history if t['profit'] > 0])
                    wr = (wins / len(bot.trades_history) * 100)
                    st.metric("🎯 Win Rate", f"{wr:.1f}%")
                else:
                    st.metric("🎯 Win Rate", "0%")
        
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📜 Logs", "💼 Trades"])
        
        with tab1:
            if len(bot.equity_history) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=list(bot.equity_history),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#ffa500', width=2)
                ))
                fig.update_layout(height=300, template="plotly_dark", title="Equity Curve")
                st.plotly_chart(fig, use_container_width=True)
            
            if bot.current_position:
                st.subheader("📍 Posición Actual")
                pos = bot.current_position
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"""
                    **Tipo:** {pos['type']}  
                    **Entrada:** ${pos['entry_price']:.2f}  
                    **Cantidad:** {pos['qty']}
                    """)
                
                with col2:
                    cp = bot.get_current_price(bot.symbol)
                    if cp:
                        if pos['type'] == 'LONG':
                            pnl = (cp - pos['entry_price']) * pos['qty']
                        else:
                            pnl = (pos['entry_price'] - cp) * pos['qty']
                        
                        color = "green" if pnl > 0 else "red"
                        st.markdown(f"""
                        **Precio:** ${cp:.2f}  
                        **P/L:** <span style="color: {color};">${pnl:.2f}</span>
                        """, unsafe_allow_html=True)
                
                with col3:
                    st.warning(f"""
                    **Stop:** ${pos['stop_loss']:.2f}  
                    **TP:** ${pos['take_profit']:.2f}
                    """)
        
        with tab2:
            st.subheader("📜 Logs")
            for log in reversed(list(bot.logs)):
                color = {'success': 'green', 'error': 'red', 'warning': 'orange', 'info': 'blue'}.get(log['level'], 'white')
                st.markdown(f"""
                <div style="background: rgba(0,0,0,0.3); padding: 8px; margin: 4px 0; border-radius: 5px; border-left: 3px solid {color};">
                    [{log['time']}] {log['message']}
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("💼 Historial")
            if bot.trades_history:
                df = pd.DataFrame(bot.trades_history)
                total = df['profit'].sum()
                avg = df['profit'].mean()
                wins = len(df[df['profit'] > 0])
                losses = len(df[df['profit'] < 0])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total P/L", f"${total:.2f}")
                with col2:
                    st.metric("Avg P/L", f"${avg:.2f}")
                with col3:
                    st.metric("✅ Wins", wins)
                with col4:
                    st.metric("❌ Losses", losses)
                
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No hay trades")
        
        time.sleep(5)
        st.rerun()
    
    else:
        st.info("""
        ### ⏱️ Bot Alpaca con Datos Gratuitos (Delay 15min)
        
        **Características:**
        - ✅ Datos completamente GRATUITOS
        - ⏱️ Delay de 15 minutos (suficiente para práctica)
        - ✅ Todas las estrategias funcionan
        - ✅ Paper trading ilimitado
        - ✅ Sin costo de suscripción
        
        **Ideal para:**
        - 🎓 Aprender y practicar
        - 📊 Backtesting de estrategias
        - 🧪 Probar el bot sin riesgo
        - 📈 Ver resultados con datos reales (retrasados)
        
        **Para comenzar:**
        1. Ingresa tus API keys de Alpaca
        2. Selecciona estrategia (ORB recomendada)
        3. Click ▶️ INICIAR
        
        **Nota:** Para datos en tiempo real necesitas suscripción de Alpaca (~$9/mes)
        """)

if __name__ == "__main__":
    main()
