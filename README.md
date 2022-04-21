# -*- coding: utf-8 -*-
pred_price = 0
epochs = 2
pi = 3.14

import time
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yf
import pandas as pd
from termcolor import colored
from binance.client import Client
import password

client = Client(password.binance_api_key, password.binance_secret_key)
info = client.get_all_tickers()

start_time = time.time()


while True:
    #Место обновляемых данных
    info_BTCUSDT = info[11]
    info_BTCUSDT_symbol, info_BTCUSDT_price = info_BTCUSDT["symbol"], info_BTCUSDT["price"]
    for i in range(22):
        #Место использования циклических операций
        real_time = time.time() - start_time
        print(colored([i,int(real_time),time.strftime("%Y-%m-%d : %H-%M-%S")],"green"))
        if i == 1:
            info_get_exchange = client.get_exchange_info()
            info_get_exchange_order_types = info_get_exchange
            info_get_exchange_order_types_rateLimits = pd.DataFrame(info_get_exchange_order_types["rateLimits"])
            info_get_exchange_order_types_symbols = pd.DataFrame(info_get_exchange_order_types["symbols"])
            #info_get_exchange_order_types_rateLimits_rateLimitType = info_get_exchange_order_types_rateLimits["rateLimitType"]

            file =  open("text_info_get_exchange_oreder_types.txt","w")
            file.write(str(info_get_exchange_order_types))
            file.close()

            #print(colored((str(info_get_exchange_order_types)),"cyan"))
            #print(colored((str(info_get_exchange_order_types_rateLimits)), "cyan"))
            print(colored(info_get_exchange_order_types_symbols, "cyan"))

            info = client.get_all_tickers()
            #BTC-USDT
            info_BTCUSDT = info[11]
            info_BTCUSDT_symbol,info_BTCUSDT_price = info_BTCUSDT["symbol"],info_BTCUSDT["price"]

            #BTC-RUB
            info_BTCRUB= info[666]
            info_BTCRUB_symbol, info_BTCRUB_price = info_BTCRUB["symbol"], info_BTCRUB["price"]

            #print(colored([time.strftime("%Y-%m-%d : ") + str(real_time), i],"yellow"))
            #print(colored(info,"white"))
            print(colored([info_BTCUSDT_symbol,info_BTCUSDT_price], "blue"))
            print(colored([info_BTCRUB_symbol, info_BTCRUB_price], "red"))

            time.sleep(0)
        elif i == 2:
            trades = client.get_recent_trades(symbol='BTCUSDT')
            trades_0 = trades[i]
            trades_id = trades_0['id']
            trades_price = trades_0['price']
            #if str(trades_price) >= str(40000):
                #print(trades_price + " Больше 40000")
            trades_qty = trades_0['qty']
            trades_quoteQty = trades_0['quoteQty']
            trades_time = trades_0['time']
            trades_isBuyerMaker = trades_0['isBuyerMaker']
            if trades_isBuyerMaker == True:
                trades_isBuyerMaker = "покупатель"
            elif trades_isBuyerMaker == False:
                trades_isBuyerMaker = "не покупатель"
            trades_isBestMatch = trades_0['isBestMatch']
            if trades_isBestMatch == True:
                trades_isBestMatch = "лучшее совпадение"
            elif trades_isBestMatch == False:
                trades_isBestMatch = "не лучшее совпадение"
            print([trades_id, trades_price, trades_qty, trades_quoteQty, trades_time, trades_isBuyerMaker,trades_isBestMatch])

        elif i == 3:
            info = client.get_all_tickers()
            symbol_BTCUSDT = info[11]
            symbol_ETHUSDT = info[12]
            symbol_BNBUSD = info[98]
            symbol_LTCUSD = info[190]
            symbol_XRPUSD = info[306]
            symbol_BTCRUB = info[666]
            symbol_DOTUSD = info[954]


            print("Запуск программы анализа баланса")
            balance_btc = client.get_asset_balance(asset='BTC')
            # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
            balance_usd_btc_usd_free_locked_sum = float(balance_btc["free"]) + float(balance_btc["locked"])
            balance_usd_btc_usd_present = balance_usd_btc_usd_free_locked_sum * float(symbol_BTCUSDT["price"])

            balance_usdt = client.get_asset_balance(asset='USDT')
            # print(balance_USDT["asset"],balance_USDT["free"],balance_USDT["locked"])
            balance_usd_usd_usd_free_locked_sum = float(balance_usdt["free"]) + float(balance_usdt["locked"])
            balance_usd_usd_usd_present = balance_usd_usd_usd_free_locked_sum

            all_balance_usd = balance_usd_btc_usd_present + balance_usd_usd_usd_present

            data = pd.DataFrame([["asset", "free", "locked", "converter USD", "USD"],
                                 [balance_btc["asset"], balance_btc["free"], balance_btc["locked"],
                                  symbol_BTCUSDT['price'], balance_usd_btc_usd_present],
                                 [balance_usdt["asset"], balance_usdt["free"], balance_usdt["locked"],
                                  "{1:1}", balance_usd_usd_usd_present],
                                 [" ", " ", " ", " ", all_balance_usd]])
            print(data)
            real_date = time.strftime("%Y-%m-%d")
            data_btcusdt = yf.download("BTC-USD", start="2018-02-01", end=real_date, interval='1d')
            print(data_btcusdt)
            # create dATEFRAME CLOSE
            data = data_btcusdt.filter(['Close'])
            print(data)
            print(data.shape)
            # convert dataframe
            dataset = data.values
            print(dataset)
            # get the number rows to train the model
            training_data_len = math.ceil(len(dataset) * .8)
            print(training_data_len)
            # scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            print(scaled_data)
            # create the training dataset
            train_data = scaled_data[0:training_data_len, :]
            print(train_data)
            # split the data into x_train and y_train data sets
            x_train = []
            y_train = []
            for rar in range(60, len(train_data)):
                x_train.append(train_data[rar - 60:rar, 0])
                y_train.append(train_data[rar, 0])
                if rar <= 61:
                    print(x_train)
                    print(y_train)
                    print()
            # conver the x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)
            # reshape the data
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            print(x_train.shape)
            # biuld to LST model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(101, return_sequences=False))
            model.add(Dense(50))
            model.add(Dense(25))
            model.add(Dense(1))
            # cmopale th emodel
            model.compile(optimizer='adam', loss='mean_squared_error')
            # train_the_model
            model.summary()
            print("Fit model on training data")
            # Evaluate the model on the test data using `evaluate`
            print("Evaluate on test data")
            results = model.evaluate(x_train, y_train, batch_size=1)
            print("test loss, test acc:", results)
            filename = "BTC-USDT_exterminate 2018-02-01_1D.h5"
            model = tf.keras.models.load_model(filename)
            model.fit(x_train, y_train, batch_size=1, epochs=2)
            model.save(filename)
            reconstructed_model = tf.keras.models.load_model(filename)
            np.testing.assert_allclose(model.predict(x_train), reconstructed_model.predict(x_train))
            reconstructed_model.fit(x_train, y_train)
            # create the testing data set
            # create a new array containing scaled values from index 1713 to 2216
            test_data = scaled_data[training_data_len - 60:, :]
            # create the fata sets x_test and y_test
            x_test = []
            y_test = dataset[training_data_len:, :]
            for resr in range(60, len(test_data)):
                x_test.append(test_data[resr - 60:resr, 0])
            # conert the data to numpy array
            x_test = np.array(x_test)
            # reshape the data
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            # get the model predicted price values
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            # get the root squared error (RMSE)
            rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
            print(rmse)
            # get the quate
            btc_quote = yf.download("BTC-USD", start="2018-02-01", end=real_date, interval='1d')
            # btc_quote = pd.read_csv(str(balance_btc["asset"]) + ".csv", delimiter=",")
            # new_df = btc_quote.filter(["Well"])
            new_df = btc_quote.filter(['Close'])

            # get teh last 60 days closing price values and convert the dataframe to an array
            last_60_days = new_df[-60:].values
            # scale the data to be values beatwet 0 and 1

            last_60_days_scaled = scaler.transform(last_60_days)

            # creAte an enemy list
            X_test = []
            # Append past 60 days
            X_test.append(last_60_days_scaled)

            # convert the x tesst dataset to numpy
            X_test = np.array(X_test)

            # Reshape the dataframe
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            # get predict scaled

            pred_price = model.predict(X_test)
            # undo the scaling
            pred_price = scaler.inverse_transform(pred_price)
            print(pred_price)
            import numpy as np
            pred_price_a = pred_price[0]
            pred_price_aa = pred_price_a[0]
            preset_pred_price = int(pred_price_aa)
            print(pred_price)
            print(preset_pred_price)
            a = float(symbol_BTCUSDT['price'])
            b = float(balance_btc["free"])
            ab_sum = a * b
            data_coin = float(ab_sum) - 12
            print(data_coin)
            if pred_price <= float(symbol_BTCUSDT['price']):
                info = client.get_all_tickers()
                symbol_BTCUSDT = info[11]
                a = float(1)
                b = float(balance_usdt["free"])
                ab_sum = a * b
                data_coin = float(ab_sum) - 12
                print(data_coin)
                if data_coin <= 0:
                    print([data_coin, a, b])
                    print(ab_sum)
                    quantity = float(12 / float(symbol_BTCUSDT['price']))
                    print(quantity)
                    print("Недостаточно USD для покупки BTC")
                elif data_coin >= 0:
                    print([data_coin, a, b])
                    print("\n" + "BUY -USDT  " + str(preset_pred_price))
                    print(a)
                    quantity = float(12 / float(symbol_BTCUSDT['price']))
                    quantity_start = round(quantity, 5)
                    print(quantity_start)
                    order = client.order_limit_sell(symbol='BTCUSDT',
                                                    quantity=quantity_start,
                                                    price=preset_pred_price)
            elif pred_price >= float(symbol_BTCUSDT['price']):
                info = client.get_all_tickers()
                symbol_BTCUSDT = info[11]
                a = float(symbol_BTCUSDT['price'])
                b = float(balance_btc["free"])
                ab_sum = a * b
                data_coin = float(ab_sum) - 12
                print(data_coin)
                if data_coin <= 0:
                    print([data_coin, a, b])
                    print(ab_sum)
                    quantity = float(12 / float(symbol_BTCUSDT['price']))
                    print(quantity)
                    print("Недостаточно BTC для продажи USDT")
                elif data_coin >= 0:
                    print([data_coin, a, b])
                    print("\n" + "SELL -BTC  " + str(preset_pred_price))
                    print(a)
                    quantity = float(12 / float(symbol_BTCUSDT['price']))
                    quantity_start = round(quantity, 5)
                    print(quantity_start)
                    order = client.order_limit_buy(symbol='BTCUSDT',
                                                   quantity=quantity_start,
                                                   price=preset_pred_price)
        elif i == 21:
            old_time = time.time() - start_time
            print("Время на расчеты :" + str(old_time))


        ollo_time = time.time() - start_time
        print(colored(pred_price,"cyan"))
        print(colored(str(info_BTCUSDT_price), "magenta"))
        print("Время работы программы :" + str(ollo_time))
        data_time = time.strftime("%Y-%m-%d")
        for i in range(1):
            if ollo_time <= 60:
                print("Меньше минуты" + str(ollo_time))
            elif ollo_time >= 60:
                print("Больше минуты " + str(ollo_time))
                if ollo_time <= 3600:
                    print("Меньше часа " + str(ollo_time))
                elif ollo_time >= 3600:
                    print("Больше часа " + str(ollo_time))
        else:
            print(" ")
            time.sleep(1)
