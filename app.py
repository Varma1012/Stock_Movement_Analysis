# -*- coding: utf-8 -*-
"""

@author: prjct44
"""
#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt




from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
import nltk
nltk.download('punkt')
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator


#***************** FLASK *****************************
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database.db'
app.config['SECRET_KEY']=os.urandom(24)
db=SQLAlchemy(app)
bcrypt = Bcrypt(app)





login_manager=LoginManager()
login_manager.init_app(app)
login_manager.login_view="login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(20),nullable=False,unique=True)
    password=db.Column(db.String(80),nullable=False)
    


class RegisterForm (FlaskForm):

    username = StringField(validators=[InputRequired(), Length( min=4, max=20)], render_kw={"placeholder": "Username"})

    password=PasswordField(validators=[InputRequired(), Length( min=4, max=20)], render_kw={"placeholder": "Password"})

    submit=SubmitField("Register")

    def validate_username(self, username): 
        existing_user_username = User.query.filter_by( username=username.data).first()

        if existing_user_username:

            raise ValidationError( "That username already exists. Please choose a different one.")

class LoginForm(FlaskForm):

    username = StringField(validators=[InputRequired(), Length( min=4, max=20)], render_kw={"placeholder": "Username"})

    password=PasswordField(validators=[InputRequired(), Length( min=4, max=20)], render_kw={"placeholder": "Password"})

    submit=SubmitField("Login")

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    quote = db.Column(db.String(10), nullable=False)
    date = db.Column(db.String(20), nullable=False)  # the stock date (from dataset)
    actual_close = db.Column(db.Float, nullable=False)
    predicted_close = db.Column(db.Float, nullable=False)
    error_percent = db.Column(db.Float, nullable=False)
    prediction_date = db.Column(db.String(20), nullable=False)  # ✅ the day this prediction was made


with app.app_context():
    
    # Perform database operations
    db.create_all()  # Create tables based on models
    users = db.session.query(User).all()


#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response
@app.route('/')
def home():
   return render_template('home.html')



@app.route('/index')
def index():
   return render_template('index.html')





@app.route('/login',methods=['GET','POST'])
def login():
    form=LoginForm()
    if form.validate_on_submit():
        user=User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('index'))
    return render_template('login.html',form=form)

@app.route('/logout',methods=['GET','POST'])

def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/register',methods=['GET','POST'])
def register():
    form=RegisterForm()
    if form.validate_on_submit():
        
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8') 
        new_user= User(username=form.username.data, password=hashed_password) 
        db.session.add(new_user)
        db.session.commit() 
        return redirect(url_for('login'))
    return render_template('register.html',form=form)

@app.route('/history/<quote>', methods=['GET'])
def history(quote):
    selected_date = request.args.get('selected_date')

    # Get unique prediction dates for dropdown
    prediction_dates = (
        db.session.query(Prediction.prediction_date)
        .filter_by(quote=quote)
        .distinct()
        .order_by(Prediction.prediction_date.desc())
        .all()
    )
    prediction_dates = [d[0] for d in prediction_dates]

    # Default to latest prediction if none selected
    if not selected_date and prediction_dates:
        selected_date = prediction_dates[0]

    # Fetch data for that specific prediction date
    past_10_data = (
        Prediction.query.filter_by(quote=quote, prediction_date=selected_date)
        .order_by(Prediction.date.asc())
        .all()
    )

    return render_template(
        'results.html',
        quote=quote,
        past_10_data=[(r.date, r.actual_close, r.predicted_close) for r in past_10_data],
        selected_date=selected_date,
        prediction_dates=prediction_dates
    )





@app.route('/insertintotable',methods = ['POST'])
def insertintotable():
    nm = request.form['nm']

    #**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, auto_adjust=False)
        print(data)
        df = pd.DataFrame(data=data)
        #df.fillna(value={'RSI':0,'EMI':0,'MACD':0,'MACD_Signal':0,'MACD_Histogram':0,'SMI': 0}, inplace=True)
        ''' rsi = RSIIndicator(df['Close']).rsi()
        df['RSI'] = rsi
        
        # Add Exponential Moving Average (EMA)
        ema = EMAIndicator(df['Close']).ema_indicator()
        df['EMA'] = ema
        
        # Add Moving Average Convergence Divergence (MACD)
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Add Stochastic Momentum Index (SMI)
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['SMI'] = stoch.stoch()'''
        df=df.xs(quote,axis=1,level='Ticker')
        df.to_csv(''+quote+'.csv')
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only/
            
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
        return df

    #******************** ARIMA SECTION ********************
    def ARIMA_ALGO(df):
       
        uniqueVals = df["Code"].unique()  
        
        len(uniqueVals)
     
        df = df.set_index("Code")
        

        # Function to parse dates
        def parser(x):
            
            return datetime.strptime(x, '%Y-%m-%d')

        # ARIMA model function for training and testing
        def arima_model(train, test):
            print("this is arima train")
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            return predictions

        for company in uniqueVals[:10]:  # Limit to the first 10 companies for demonstration

            # Prepare data for the company
            data = (df.loc[company, :]).reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Date', 'Price']]
            Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'], axis=1)

            # Plot the data (Optional)
            print("hi1")
            fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
            plt.plot(Quantity_date)
            plt.savefig('static/Trends.png')
            plt.close(fig)
            print("hi2")
            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:]
            print("hi3")
            # Fit the ARIMA model
            predictions = arima_model(train, test)
            print("hi4")
            # Define custom accuracy calculation (fixed)
            def custom_accuracy(y_true, y_pred, threshold=0.02):
                # Calculate absolute percentage error
                error = np.abs((y_true - y_pred) / y_true)
                
                # Count how many predictions are within the threshold
                accurate_predictions = np.sum(error < threshold)
                
                # Return the percentage of accurate predictions
                return accurate_predictions / len(y_true) * 100  # Convert to percentage

            # Calculate accuracy
            accuracy = custom_accuracy(test, predictions)

            # Plot actual vs predicted stock prices
            fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
            plt.plot(test, label='Actual Price')
            plt.plot(predictions, label='Predicted Price')
            plt.legend(loc=4)
            plt.savefig('static/ARIMA.png')
            plt.close(fig)

            # Print results
            print()
            print("##############################################################################")
            arima_pred = predictions[-2]  # Prediction for the next day
            print(f"Tomorrow's {company} Closing Price Prediction by ARIMA: {arima_pred}")
            
            # Calculate RMSE
            error_arima = math.sqrt(mean_squared_error(test, predictions)) + 5  # Added 5 as per the original code
            print("ARIMA RMSE:", error_arima)
            print("##############################################################################")
            
            return arima_pred, error_arima


        
    

    #************* LSTM SECTION **********************
    def BLSTM_ALGO(df):
        

        # Ensure 'Date' column exists and is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("DataFrame must contain a 'Date' column with trading dates.")

        # Sort by date just in case
        df = df.sort_values('Date').reset_index(drop=True)

        # Split into training and test sets (80/20)
        dataset_train = df.iloc[:int(0.8 * len(df)), :]
        dataset_test = df.iloc[int(0.8 * len(df)):, :]

        # Prepare training data using 'Close' price
        training_set = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = scaler.fit_transform(training_set)

        # Create data structure with 7 timesteps
        X_train, y_train = [], []
        for i in range(7, len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Prepare input for forecasting next day
        X_forecast = np.array(X_train[-1, 1:])
        X_forecast = np.append(X_forecast, y_train[-1])
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))

        # Build the Bidirectional LSTM model
        model = Sequential([
            Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train.shape[1], 1)),
            Dropout(0.1),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.1),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.1),
            Bidirectional(LSTM(50)),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

        # Prepare test set
        real_stock_price = dataset_test[['Close']].values
        dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
        testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values.reshape(-1, 1)
        testing_set = scaler.transform(testing_set)

        # Create test data structure
        X_test = [testing_set[i-7:i, 0] for i in range(7, len(testing_set))]
        X_test = np.array(X_test).reshape(-1, 7, 1)

        # Predict stock prices
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # Plot results
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(real_stock_price, label='Actual Price')
        plt.plot(predicted_stock_price, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/BLSTM.png')
        plt.close(fig)

        # Calculate RMSE
        error_Blstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

        # Forecast the next day's price
        forecasted_stock_price = model.predict(X_forecast)
        forecasted_stock_price = scaler.inverse_transform(forecasted_stock_price)
        Blstm_pred = forecasted_stock_price[0, 0]

        # Accuracy from RMSE as percentage
        def accuracy_from_rmse(y_true, rmse):
            price_range = np.max(y_true) - np.min(y_true)
            return 100 * (1 - (rmse / price_range))
        accuracy = accuracy_from_rmse(real_stock_price, error_Blstm)

        # Get last 10 days of actual and predicted data
        past_10_actual = real_stock_price[-10:].flatten()
        past_10_pred = predicted_stock_price[-10:].flatten()

        # ✅ Get corresponding last 10 trading dates from the actual test set
        past_10_dates = dataset_test['Date'].iloc[-10:].dt.strftime('%Y-%m-%d').tolist()

        # Combine into list of tuples (Date, Actual, Predicted)
        past_10_data = list(zip(past_10_dates, past_10_actual, past_10_pred))

        # Display results
        print("##############################################################################")
        print(f"Tomorrow's Closing Price Prediction by Bidirectional LSTM: {Blstm_pred}")
        print(f"Bidirectional LSTM RMSE: {error_Blstm}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("##############################################################################")

        return Blstm_pred, error_Blstm, past_10_data

    

    
    def calculate_indicators(df):
        df['EMA_12'] = EMAIndicator(df['Close'], window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(df['Close'], window=26).ema_indicator()
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        return df

    def handle_missing_values(df):
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def LSTM_ALGO(df):
        # Calculate indicators and handle missing values
        df = calculate_indicators(df)
        df = handle_missing_values(df)

        # Split data into training and test sets
        dataset_train = df.iloc[0:int(0.8*len(df)), :]
        dataset_test = df.iloc[int(0.8*len(df)):, :]

        # Prepare the data
        features = ['Close', 'EMA_12', 'EMA_26', 'RSI']
        training_set = df[features].iloc[0:int(0.8*len(df)), :].values
        test_set = df[features].iloc[int(0.8*len(df)):, :].values

        # Feature Scaling
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)

        # Creating data structure with 7 timesteps and 4 features
        timesteps = 7
        X_train = []
        y_train = []
        for i in range(timesteps, len(training_set_scaled)):
            X_train.append(training_set_scaled[i-timesteps:i])
            y_train.append(training_set_scaled[i, 0])  # Predicting the Close price

        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # Prepare X_forecast (for next day prediction)
        last_sequence = np.array(X_train[-1])
        X_forecast = np.reshape(last_sequence, (1, timesteps, len(features)))

        # Building the LSTM model
        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, len(features))))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=1))

        regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        # Training the model
        regressor.fit(X_train, y_train, epochs=25, batch_size=32)

        # Testing
        total_dataset = pd.concat((dataset_train[features], dataset_test[features]), axis=0)
        testing_set = total_dataset[len(total_dataset) - len(dataset_test) - timesteps:].values
        testing_set = sc.transform(testing_set)

        X_test = []
        for i in range(timesteps, len(testing_set)):
            X_test.append(testing_set[i-timesteps:i])

        X_test = np.array(X_test)
        
        # Predicting the stock price
        predicted_stock_price = regressor.predict(X_test)

        # Inverse transform prediction to actual values
        predicted_stock_price = sc.inverse_transform(
            np.concatenate(
                (predicted_stock_price, np.zeros((predicted_stock_price.shape[0], len(features) - 1))), 
                axis=1
            )
        )[:, 0]

        real_stock_price = dataset_test['Close'].values

        # Plotting
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(real_stock_price, label='Actual Price')
        plt.plot(predicted_stock_price, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)

        # Calculate RMSE (Root Mean Squared Error)
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

        # Forecasting Prediction (for the next day)
        forecasted_stock_price = regressor.predict(X_forecast)

        # Handle the shape for inverse transformation
        forecasted_stock_price = sc.inverse_transform(
            np.concatenate(
                (forecasted_stock_price, np.zeros((forecasted_stock_price.shape[0], len(features) - 1))), 
                axis=1
            )
        )[:, 0]

        # Function to calculate accuracy (within a threshold)
        def custom_accuracy(y_true, y_pred, threshold=0.02):
            # Calculate percentage error
            error = np.abs((y_true - y_pred) / y_true)
            # Count how many predictions are within the threshold
            accurate_predictions = np.sum(error < threshold)
            return accurate_predictions / len(y_true)

        # Calculate accuracy
        def accuracy_from_rmse(y_true, rmse):
            """
            Calculate accuracy based on RMSE as a percentage of the range of the target variable.
            """
            # Calculate range of true values (Close price range)
            price_range = np.max(y_true) - np.min(y_true)
            
            # Calculate the percentage accuracy based on RMSE
            accuracy = 100 * (1 - (rmse / price_range))
            
            return accuracy
        accuracy=accuracy_from_rmse(real_stock_price, error_lstm)
        # Get the forecasted value (next day's prediction)
        lstm_pred = forecasted_stock_price[0]

        print()
        print("##############################################################################")
        print(f"Tomorrow's Closing Price Prediction by LSTM: {lstm_pred}")
        print(f"LSTM RMSE: {error_lstm}")
        print(f"Accuracy: {accuracy}%")
        print("##############################################################################")

        return lstm_pred, error_lstm
    #***************** LINEAR REGRESSION SECTION ******************       
    

    def LIN_REG_ALGO(df):
        #No of days to be forcasted in future
        forecast_out = int(7)
        #Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['Close','Close after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
        
        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]
        
        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
        
        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        
        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        plt2.plot(y_test,label='Actual Price' )
        plt2.plot(y_test_pred,label='Predicted Price')
        
        plt2.legend(loc=4)
        plt2.savefig('static/LR.png')
        plt2.close(fig)
        
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        
        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
        print("Linear Regression RMSE:",error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr
    #**************** SENTIMENT ANALYSIS **************************
    import requests
    def clean_text(text):
        text = re.sub('&amp;', '&', text)
        text = re.sub(':', '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text

    def get_current_month_date_range():
        today = dt.datetime.today()
        first_day_of_month = today.replace(day=1).strftime('%Y-%m-%d')
        if today.month == 12:
        # If current month is December, next month is January of the next year
            next_month = dt.date(today.year + 1, 1, 1)  # January of next year
        else:
        # Otherwise, just replace with the next month
            next_month = today.replace(month=today.month + 1, day=1)
        last_day_of_month = (next_month - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        return first_day_of_month, last_day_of_month

    def get_stock_data(symbol):
        first_day_of_month, last_day_of_month = get_current_month_date_range()
        url = f'https://newsapi.org/v2/everything?q={symbol} stock&from={first_day_of_month}&to={last_day_of_month}&apiKey={ct.NEWS_API_KEY}'
        response = requests.get(url)
        news_data = response.json()
        articles = news_data.get('articles', [])
        return articles

    def analyze_news_sentiment(articles):
        sentiments = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        total_polarity = 0
        count = len(articles)
        
        for article in articles:
            title = article['title']
            description = article['description'] or ''
            content = title + ' ' + description
            tw = clean_text(content)
            blob = TextBlob(tw)
            polarity = blob.sentiment.polarity
            
            if polarity > 0:
                sentiments['positive'] += 1
            elif polarity < 0:
                sentiments['negative'] += 1
            else:
                sentiments['neutral'] += 1
            
            total_polarity += polarity
        
        if count > 0:
            average_polarity = total_polarity / count
        else:
            average_polarity = 0
        
        return sentiments, average_polarity

    def plot_sentiment_pie_chart(sentiments):
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [sentiments['positive'], sentiments['negative'], sentiments['neutral']]
        explode = (0.1, 0, 0)  # explode 1st slice
        colors=['#8EB897','#DD7596','#B7C3F3']
        fig, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
        ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')

        plt.title('Sentiment Analysis of Financial News')
        plt.tight_layout()
        plt.savefig('static/SA.png')
        plt.close(fig)

    def retrieving_tweets_polarity(symbol):
        # Fetch stock data
        stock_data = get_stock_data(symbol)
        print("Stock Data for {}:".format(symbol))
        print(stock_data)
        
        # Fetch financial news
        articles = get_stock_data(symbol)
        print("Number of news articles fetched:", len(articles))
        
        # Analyze news sentiment
        sentiments, average_polarity = analyze_news_sentiment(articles)
        print("\nSentiment Analysis:")
        print(f"Positive News: {sentiments['positive']}")
        print(f"Negative News: {sentiments['negative']}")
        print(f"Neutral News: {sentiments['neutral']}")
        print(f"Overall Polarity: {average_polarity}")

        # Plot pie chart
        plot_sentiment_pie_chart(sentiments)
        
        if average_polarity > 0:
            print("\n##############################################################################")
            print("News Polarity: Overall Positive")
            print("##############################################################################")
            news_pol = "Overall Positive"
        else:
            print("\n##############################################################################")
            print("News Polarity: Overall Negative")
            print("##############################################################################")
            news_pol = "Overall Negative"
        
        # Return values as per the format of the original code
        return average_polarity, [article['title'] for article in articles], news_pol, sentiments['positive'], sentiments['negative'], sentiments['neutral']

        #return global_polarity, post_titles, post_pol, pos, neg, neutral
   


    


    def recommending(df, global_polarity,today_stock,mean):
        if today_stock.iloc[-1]['Close'] < mean:
            if global_polarity > 0:
                idea="RISE"
                decision="BUY"
                print()
                print("##############################################################################")
                print("According to the Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
            elif global_polarity == 0:
                idea="NUETRALITY"
                decision="HOLD"
                print()
                print("##############################################################################")
                print("According to the Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
            else:
                idea="FALL"
                decision="SELL"
                print()
                print("##############################################################################")
                print("According to the Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
        else:
            idea="FALL"
            decision="SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions , a",idea,"in",quote,"stock is expected => ",decision)
        return idea, decision





    #**************GET DATA ***************************************
    quote=nm
    #Try-except to check if valid stock symbol
    try:
        get_historical(quote)
    except:
        return render_template('index.html',not_found=True)
    else:
    
        #************** PREPROCESSUNG ***********************
        df = pd.read_csv(''+quote+'.csv')
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        today_stock=df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        code_list=[]
        for i in range(0,len(df)):
            code_list.append(quote)
        df2=pd.DataFrame(code_list,columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df=df2


        arima_pred, error_arima=ARIMA_ALGO(df)
        Blstm_pred, error_Blstm, past_10_data = BLSTM_ALGO(df)
        # ✅ Store the last 10 days of BLSTM predicted data into the database
        prediction_date = dt.datetime.today().strftime("%Y-%m-%d")
        Prediction.query.filter_by(quote=quote, prediction_date=prediction_date).delete()

        # Store each of the 10 predicted points in the database
        for d, actual, pred in past_10_data:
            error = abs((actual - pred) / actual * 100)
            new_entry = Prediction(
                quote=quote,
                date=d,
                actual_close=actual,
                predicted_close=pred,
                error_percent=error,
                prediction_date=prediction_date  # ✅ store the run date
            )
            db.session.add(new_entry)

        db.session.commit()
        prediction_dates = [r[0] for r in db.session.query(Prediction.prediction_date.distinct()).filter_by(quote=quote).all()]

        # get currently selected prediction date (from dropdown, default = latest)
        selected_date = request.args.get("selected_date") or (prediction_dates[-1] if prediction_dates else None)

        # get last 10-day data for that prediction_date
        past_10_data = db.session.query(Prediction).filter_by(quote=quote, prediction_date=selected_date).order_by(Prediction.date.asc()).all()

        lstm_pred, error_lstm=LSTM_ALGO(df)
        df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO(df)
        # Twitter Lookup is no longer free in Twitter's v2 API
        subreddit_name='all'
        polarity,tw_list,tw_pol,pos,neg,neutral = retrieving_tweets_polarity(quote)
        #polarity, tw_list, tw_pol, pos, neg, neutral = retrieving_tweets_polarity(df)
        
        idea, decision=recommending(df, polarity,today_stock,mean)
        print()
        print("Forecasted Prices for Next 7 days:")
        print(forecast_set)
        today_stock=today_stock.round(2)
        end = dt.date.today()
        forecast_set_dates=[(end+dt.timedelta(days=1)), end+dt.timedelta(days=2),end+dt.timedelta(days=3),end+dt.timedelta(days=4),end+dt.timedelta(days=5),end+dt.timedelta(days=6),end+dt.timedelta(days=7) ]
        forecast_set_dates1=[]
        for i in forecast_set_dates:
            i=i.strftime('%Y-%m-%d')
            print(i)
            forecast_set_dates1.append(i)
        print(forecast_set_dates1)
    return render_template('results.html',quote=quote,arima_pred=round(arima_pred,2),lstm_pred=round(lstm_pred,2),Blstm_pred=round(Blstm_pred,2),
                               open_s=today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False),adj_close=today_stock['Adj Close'].to_string(index=False),
                               tw_list=tw_list,tw_pol=tw_pol,idea=idea,decision=decision,high_s=today_stock['High'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),vol=today_stock['Volume'].to_string(index=False),
                               forecast_set=forecast_set,forecast_set_dates1=forecast_set_dates1,error_lstm=round(error_lstm,2),error_Blstm=round(error_Blstm,2),error_arima=round(error_arima,2),past_10_data=[(r.date, r.actual_close, r.predicted_close) for r in past_10_data])
if __name__ == '__main__':
   app.run()
   

















