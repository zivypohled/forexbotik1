import MetaTrader5 as mt5
import pandas as pd
import time, json, os, threading
from flask import Flask, render_template_string
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ========================
# 🔑 NASTAVENÍ
# ========================
SYMBOL = "EURUSD"
LOG_FILE = "trades.csv"
SIGNAL_FILE = "signals.json"

model = None

# ========================
# MT5 INIT
# ========================
if not mt5.initialize():
    print("MT5 ERROR")
    quit()

# ========================
# 📊 DATA
# ========================
def fetch(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 300)
    df = pd.DataFrame(rates)
    return df

# ========================
# SESSION FILTER
# ========================
def session():
    h = datetime.utcnow().hour
    if 8 <= h < 12: return "LONDON"
    if 13 <= h < 17: return "NY"
    return None

# ========================
# KORELACE
# ========================
def correlation():
    eur = fetch("EURUSD")
    dxy = fetch("USDX")
    gold = fetch("XAUUSD")
    
    eur_t = eur['close'].iloc[-1] - eur['close'].iloc[-10]
    dxy_t = dxy['close'].iloc[-1] - dxy['close'].iloc[-10]
    gold_t = gold['close'].iloc[-1] - gold['close'].iloc[-10]
    
    return eur_t, dxy_t, gold_t

# ========================
# FEATURES
# ========================
def features(df, entry, sl, tp):
    r = df.iloc[-1]['high'] - df.iloc[-1]['low']
    vol = df['high'].iloc[-10:].max() - df['low'].iloc[-10:].min()
    rr = abs(tp-entry)/abs(entry-sl)
    return [r, vol, rr]

# ========================
# AI MODEL
# ========================
def train():
    global model
    if not os.path.exists(LOG_FILE): return
    
    df = pd.read_csv(LOG_FILE)
    df = df[df['status']!="OPEN"]
    
    if len(df)<30: return
    
    X = df[['range','vol','rr']]
    y = df['status'].apply(lambda x:1 if x=="WIN" else 0)
    
    model = RandomForestClassifier()
    model.fit(X,y)

def ai(feat):
    if model is None: return 0.5
    return model.predict_proba([feat])[0][1]

# ========================
# SAVE SIGNAL
# ========================
def save_signal(sig):
    try:
        with open(SIGNAL_FILE) as f:
            data=json.load(f)
    except:
        data=[]
    
    data.insert(0,sig)
    
    with open(SIGNAL_FILE,"w") as f:
        json.dump(data[:20],f,indent=2)

# ========================
# BOT LOOP
# ========================
def bot():
    print("BOT STARTED")
    train()
    
    while True:
        try:
            if session() is None:
                time.sleep(60)
                continue
            
            df = fetch(SYMBOL)
            entry = df['close'].iloc[-1]
            sl = entry - 0.001
            tp = entry + 0.002
            
            feat = features(df,entry,sl,tp)
            prob = ai(feat)
            
            eur,dxy,gold = correlation()
            
            signal="HOLD"
            
            if eur>0 and dxy<0 and gold>0:
                signal="BUY"
            if eur<0 and dxy>0 and gold<0:
                signal="SELL"
            
            if signal!="HOLD" and prob>0.65:
                sig = {
                    "pair":"EURUSD",
                    "signal":signal,
                    "entry":round(entry,5),
                    "sl":round(sl,5),
                    "tp":round(tp,5),
                    "ai":round(prob*100,2),
                    "time":str(datetime.now())
                }
                
                save_signal(sig)
                print("SIGNAL:",sig)
            
            time.sleep(60)
        
        except Exception as e:
            print("ERROR:",e)
            time.sleep(60)

# ========================
# WEB APP
# ========================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Trading App</title>
<style>
body {background:#111;color:white;font-family:Arial}
.card {background:#1e1e1e;padding:15px;margin:10px;border-radius:10px}
.buy {color:lime}
.sell {color:red}
</style>
</head>
<body>
<h2>📊 Trading Signals</h2>

{% for s in signals %}
<div class="card">
<b>{{s.pair}}</b><br>
<span class="{{'buy' if s.signal=='BUY' else 'sell'}}">{{s.signal}}</span><br>
Entry: {{s.entry}}<br>
SL: {{s.sl}}<br>
TP: {{s.tp}}<br>
AI: {{s.ai}}%<br>
{{s.time}}
</div>
{% endfor %}

</body>
</html>
"""

@app.route("/")
def home():
    try:
        with open(SIGNAL_FILE) as f:
            signals=json.load(f)
    except:
        signals=[]
    return render_template_string(HTML,signals=signals)

# ========================
# RUN
# ========================
if __name__ == "__main__":
    threading.Thread(target=bot).start()
    app.run(host="0.0.0.0",port=5000)
