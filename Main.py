import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from textblob import TextBlob
import telepot
from keep_alive import keep_alive
keep_alive()

# تنظیمات لاگ‌گیری
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# اطلاعات بات تلگرام
TOKEN = '7480455484:AAG12ja2RXY67rsrDPtBEQZUQGdx3eH3uNc'
bot = telepot.Bot(TOKEN)

# تایم‌فریم‌ها
VALID_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
selected_asset = None
selected_timeframe = None

# دریافت داده‌های قیمت از Binance
def get_historical_data(symbol, timeframe='1h'):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval={timeframe}&limit=500"
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=[
            'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore'
        ])
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        df.set_index('Time', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return df
    except Exception as e:
        logger.error(f"خطا در دریافت داده‌ها: {e}")
        return None

# مدل LSTM برای پیش‌بینی قیمت
def train_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    return model, scaler

# پیش‌بینی قیمت آینده
def predict_trend(model, scaler, data):
    last_60_days = data['Close'].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    prediction = model.predict(X_test)
    return scaler.inverse_transform(prediction)[0][0]

# تحلیل تکنیکال: MACD, RSI, Bollinger Bands
def analyze_technical(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    macd = df['EMA12'][-1] - df['EMA26'][-1]

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))[-1]

    sma = df['Close'].rolling(window=20).mean()
    stddev = df['Close'].rolling(window=20).std()
    bb_high = sma[-1] + 2 * stddev[-1]
    bb_low = sma[-1] - 2 * stddev[-1]

    trend = "صعودی 🚀" if macd > 0 and rsi < 70 else "نزولی 📉" if macd < 0 and rsi > 30 else "خنثی ⚪"
    return trend, macd, rsi, bb_high, bb_low

# گرفتن اخبار کریپتو
def get_crypto_news():
    url = "https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_API_KEY&filter=hot"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        news_list = [f"📰 {item['title']}\n🔗 {item['url']}" for item in news_data['results'][:5]]
        return "\n\n".join(news_list)
    except Exception as e:
        logger.error(f"خطا در دریافت اخبار: {e}")
        return "❌ خطا در دریافت اخبار!"

# تحلیل احساسی (Sentiment Analysis)
def analyze_sentiment(news_text):
    analysis = TextBlob(news_text)
    sentiment = analysis.sentiment.polarity
    return "🔵 مثبت" if sentiment > 0 else "🔴 منفی" if sentiment < 0 else "⚪ خنثی"

# مدیریت پیام‌ها و تحلیل کلی
def handle(msg):
    global selected_asset, selected_timeframe

    chat_id = msg['chat']['id']
    command = msg['text'].strip()

    if command == '/start':
        bot.sendMessage(chat_id, 'سلام! ارز مورد نظر را وارد کنید (BTC, ETH, BNB)')
    elif command.upper() in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']:
        selected_asset = command.upper()
        bot.sendMessage(chat_id, f"تایم‌فریم را انتخاب کن: {', '.join(VALID_TIMEFRAMES)}")
    elif command in VALID_TIMEFRAMES:
        selected_timeframe = command
        df = get_historical_data(selected_asset, selected_timeframe)
        if df is not None:
            model, scaler = train_model(df)
            future_price = predict_trend(model, scaler, df)
            trend, macd, rsi, bb_high, bb_low = analyze_technical(df)
            news = get_crypto_news()
            sentiment = analyze_sentiment(news)

            bot.sendMessage(chat_id, f'''
                🔥 **سیگنال {selected_asset}/{selected_timeframe}** 🔥
                📈 **روند:** {trend}
                💰 **قیمت پیش‌بینی‌شده:** {future_price:.2f} USDT
                🎯 **MACD:** {macd:.2f}
                💪 **RSI:** {rsi:.2f}
                🎯 **باند بولینگر:** بالا {bb_high:.2f} / پایین {bb_low:.2f}
                🌍 **تحلیل فاندامنتال:** {sentiment}
                📰 **اخبار داغ بازار:**  
                {news}
            ''')
        else:
            bot.sendMessage(chat_id, "❌ خطا در دریافت داده‌ها!")

bot.message_loop(handle)

# نگه داشتن بات زنده
while True:
    time.sleep(5)
