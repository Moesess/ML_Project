import mysql.connector
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split

from LogisticReggression import LogisticRegression, accuracy
from LinearReggresion import LinearRegression, mean_squared_error
from PMV import calculate_pmv, find_optimal_temperature

DB_CONFIG = {
    'user': 'admin',              # Nazwa użytkownika
    'password': 'admin',          # Hasło
    'host': 'localhost',          # Host
    'port': 3306,                 # Port
    'database': 'MLDB'            # Nazwa bazy danych
}
MIN_TS = '2023-01-01 00:00:00'
MAX_TS = '2023-11-30 23:59:59'
ALL_TIMESTAMPS = pd.date_range(start=MIN_TS, end=MAX_TS, freq='min')

linearModel = LinearRegression(learning_rate=0.0001)

def get_temperatures():
    sql_temp = "SELECT DISTINCT shared_attrs, last_updated_ts FROM states as t1 \
        INNER JOIN state_attributes as t2 on t1.attributes_id=t2.attributes_id \
        WHERE JSON_EXTRACT(shared_attrs, '$.friendly_name') = 'Dom' AND \
        JSON_EXTRACT(shared_attrs, '$.pressure') IS NOT NULL \
        ORDER BY last_updated_ts;"
    
    try:       
        # Połączenie z bazą i pobranie danych
        conn = mysql.connector.connect(**DB_CONFIG)
        df = pd.read_sql(sql_temp, conn)

        # Znormalizowanie danych do postaci dataframe
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        df['date'] = pd.to_datetime(df['last_updated_ts'], unit='s').dt.floor('min')
        result = pd.concat([df['date'], normalized[["temperature", "pressure"]]], axis=1)

        # Pogrupuj po datach i usuń powtórki
        result = result.groupby('date').first().reset_index()
        result.set_index('date', inplace=True)

        # Uzupełnij resztę wierszy jako spekulację
        result = result.reindex(ALL_TIMESTAMPS, method='ffill').reset_index()
        result.rename(columns={'index': 'date'}, inplace=True)
        result.to_csv('CSV/temperatures_out.csv', index=False)

    except mysql.connector.Error as e:
        print(f"Błąd połączenia: {e}")
    finally:
        if conn.is_connected():
            conn.close()    

def get_temperatures_local():
    sql_temp = "SELECT DISTINCT shared_attrs, last_updated_ts FROM states as t1 \
            INNER JOIN state_attributes as t2 on t1.attributes_id=t2.attributes_id \
            WHERE JSON_EXTRACT(shared_attrs, '$.device_class') = 'temperature' AND \
            JSON_EXTRACT(shared_attrs, '$.friendly_name') like '%tomek local%' AND \
            JSON_EXTRACT(shared_attrs, '$.local_temperature') is not null \
            ORDER BY last_updated_ts;"
    
    try:       
        # Połączenie z bazą i pobranie danych
        conn = mysql.connector.connect(**DB_CONFIG)
        df = pd.read_sql(sql_temp, conn)

        # Znormalizowanie danych do postaci dataframe
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        df['date'] = pd.to_datetime(df['last_updated_ts'], unit='s').dt.floor('min')
        result = pd.concat([df['date'], normalized["local_temperature"]], axis=1)

        # Pogrupuj po datach i usuń powtórki
        result = result.groupby('date').first().reset_index()
        result.set_index('date', inplace=True)

        # Uzupełnij resztę wierszy jako spekulację
        result["humidity"] = np.round(np.random.uniform(45, 55, size=len(result)), 0)
        result = result.reindex(ALL_TIMESTAMPS, method='ffill').reset_index()
        result.rename(columns={'index': 'date'}, inplace=True)
        result.to_csv('CSV/temperatures_local.csv', index=False)

    except mysql.connector.Error as e:
        print(f"Błąd połączenia: {e}")
    finally:
        if conn.is_connected():
            conn.close()    


def create_training_data_sets():
    out = pd.read_csv("CSV/temperatures_out.csv")
    local = pd.read_csv("CSV/temperatures_local.csv")
    presence = pd.read_csv("CSV/power_predict.csv")

    temperatures = pd.concat([local[['date', 'local_temperature', 'humidity']], out[['temperature', 'pressure']]], axis=1)
    temperatures.to_csv("CSV/temperatures.csv", index=False)
    
    # stworzenie danych testowych
    temperatures["date"] = pd.to_datetime(temperatures['date'])
    temperatures_train = temperatures
    temperatures_train["hour"] = temperatures_train['date'].dt.hour
    temperatures_train["month"] = temperatures_train['date'].dt.month
    
    # dołącz obecność
    if True:
        # Przeliczanie PMV i optymalnych danych do treningu długo trwa więc pobieram z pliku
        temperatures_train["PMV"] = temperatures_train.apply(lambda row: np.round(calculate_pmv(row["local_temperature"], row["humidity"]), 2), axis=1)
        temperatures_train["optimal_temperature"] = temperatures_train.apply(lambda row: np.round(find_optimal_temperature(row["humidity"]), 2), axis=1)
    else:
        temperatures_train = pd.read_csv("CSV/temperatures_train.csv")

    X = temperatures_train[['PMV', 'humidity', 'local_temperature']]
    y = temperatures_train['optimal_temperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1234)
    linearModel.fit(X_train, y_train)

    # Predykcja na danych testowych
    predictions = linearModel.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("Błąd średniokwadratowy:", mse)
    
    temperatures_train.sort_values(by="date", inplace=True)
    temperatures_train.to_csv("CSV/temperatures_train.csv", index=False)

    # Dodanie obliczonych temperatur do DataFrame
    temperatures_train['optimal_temperature'] =(linearModel.predict(temperatures_train[['PMV', 'humidity', 'local_temperature']]))
    temperatures_train = pd.concat([temperatures_train, presence["presence"]], axis=1)
    temperatures_train["PMV_after"] = temperatures_train.apply(lambda row: np.round(calculate_pmv(row["optimal_temperature"], row["humidity"]), 2), axis=1)

    # Zapis do CSV
    temperatures_train.to_csv('CSV/optimal_temperatures.csv', index=False)

def get_powers():
    sql_power = "SELECT DISTINCT shared_attrs, last_updated_ts FROM states as t1 \
                INNER JOIN state_attributes as t2 on t1.attributes_id=t2.attributes_id \
                WHERE JSON_EXTRACT(shared_attrs, '$.device_class') = 'power' AND \
                JSON_EXTRACT(shared_attrs, '$.friendly_name') like '%Tomek %' AND \
                JSON_EXTRACT(shared_attrs, '$.power') is not null \
                ORDER BY last_updated_ts;"
    try:
        # Połączenie z bazą i pobranie danych
        conn = mysql.connector.connect(**DB_CONFIG)
        df = pd.read_sql(sql_power, conn)

        # Znormalizowanie danych do postaci dataframe
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        df['date'] = pd.to_datetime(df['last_updated_ts'], unit='s').dt.floor('min')
        result = pd.concat([df['date'], normalized["power"]], axis=1)
        
        # Wstaw dzien tygodnia
        result["weekday"] = result["date"].dt.weekday

        # Pogrupuj po datach i usuń powtórki
        result = result.groupby('date').first().reset_index()
        result.set_index('date', inplace=True)

        # Uzupełnij resztę wierszy jako spekulację
        result = result.reindex(ALL_TIMESTAMPS, method='ffill').reset_index()
        result.rename(columns={'index': 'date'}, inplace=True)
        result.to_csv('CSV/power.csv', index=False)

    except mysql.connector.Error as e:
        print(f"Błąd połączenia: {e}")
    finally:
        if conn.is_connected():
            conn.close()     

def create_presence_data_set():
    df = pd.read_csv("CSV/power.csv")
    df['date'] = pd.to_datetime(df['date'])

    # Weź dane styczniowe
    df = df[((df['date']).dt.date >= pd.to_datetime("2023-01-01 00:00:00").date()) & 
            ((df['date']).dt.date <= pd.to_datetime("2023-01-31 23:59:59").date())]
    
    # Obecność gdy moc większa od 60W i w przedziale 00:00 - 08:00
    df['presence'] = ((df['power'] > 60) | ((df['date'].dt.hour >= 0) & (df['date'].dt.hour <= 8))).astype(int)
    df.to_csv('CSV/power_train.csv', index=False)

def presence_prediction():
    # Pobierz dane treningowe i dostosuj do nauki
    dane_wejsciowe = pd.read_csv("CSV/power_train.csv")
    dane_wejsciowe_X = dane_wejsciowe[['power', 'weekday']]
    dane_wejsciowe_X['hour'] = pd.to_datetime(dane_wejsciowe['date']).dt.hour
    dane_wejsciowe_Y = dane_wejsciowe['presence']

    X, y = dane_wejsciowe_X, dane_wejsciowe_Y

    classifier = LogisticRegression(learning_rate=0.01) # Chyba najlepsze ratio
    y = y.squeeze().values
    classifier.fit(X, y)

    # Pobierz dane testowe po styczniu i przeprowadź predykcję
    dane_testowe = pd.read_csv("CSV/power.csv")
    dane_testowe = dane_testowe[(pd.to_datetime(dane_testowe['date']).dt.date > pd.to_datetime("2023-01-31 23:59:59").date())]
    dane_testowe_X = dane_testowe[['power', 'weekday']]
    dane_testowe_X['hour'] = pd.to_datetime(dane_testowe['date']).dt.hour
    results = classifier.predict(dane_testowe_X)

    # Dopisz predykcje do danych i zapisz dane treningowe wraz z testowymi
    dane_testowe['presence'] = results
    dane_testowe = pd.concat([dane_wejsciowe, dane_testowe], ignore_index=True)
    dane_testowe.to_csv("CSV/power_predict.csv")

if __name__ == '__main__':
    # Wymagane połączenie do bazy sql
    # get_powers()
    # create_presence_data_set()
    # presence_prediction()

    # get_temperatures_local()
    # get_temperatures()
    create_training_data_sets()