import mysql.connector
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def get_csv():
    db_config = {
        'user': 'admin',              # Nazwa użytkownika
        'password': 'admin',          # Hasło
        'host': 'localhost',          # Host
        'port': 3306,                 # Port
        'database': 'MLDB'            # Nazwa bazy danych
    }

    sql_power = "SELECT DISTINCT shared_attrs, last_updated_ts FROM states as t1 \
                INNER JOIN state_attributes as t2 on t1.attributes_id=t2.attributes_id \
                WHERE JSON_EXTRACT(shared_attrs, '$.device_class') = 'power' AND \
                JSON_EXTRACT(shared_attrs, '$.friendly_name') like '%Tomek %' AND \
                JSON_EXTRACT(shared_attrs, '$.power') is not null \
                ORDER BY last_updated_ts;"
    
    sql_temp = "SELECT DISTINCT shared_attrs, last_updated_ts FROM states as t1 \
                INNER JOIN state_attributes as t2 on t1.attributes_id=t2.attributes_id \
                WHERE JSON_EXTRACT(shared_attrs, '$.device_class') = 'temperature' AND \
                JSON_EXTRACT(shared_attrs, '$.friendly_name') like '%tomek local%' AND \
                JSON_EXTRACT(shared_attrs, '$.local_temperature') is not null \
                ORDER BY last_updated_ts;"
    
    sql_temperature = "SELECT shared_attrs FROM state_attributes WHERE \
             JSON_EXTRACT(shared_attrs, '$.device_class') = 'temperature' \
             AND JSON_EXTRACT(shared_attrs, '$.friendly_name') like '%tomek local%' \
             ;"
    
    sql_other = "SELECT shared_attrs FROM state_attributes WHERE \
            JSON_EXTRACT(shared_attrs, '$.device_class') NOT IN ('battery', 'window', 'problem', 'update', 'frequency', 'plug', 'wind_speed', 'temperature', 'power') \
            AND JSON_EXTRACT(shared_attrs, '$.friendly_name') like '%Tomek %' \
            ;"

    try:
        conn = mysql.connector.connect(**db_config)

        min_ts = '2023-04-01 00:00:00'
        max_ts = '2023-04-30 23:59:59'
        all_timestamps = pd.date_range(start=min_ts, end=max_ts, freq='S')

        df = pd.read_sql(sql_temp, conn)
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        df['last_updated_ts'] = pd.to_datetime(df['last_updated_ts'], unit='s').dt.floor('S')
        result = pd.concat([df['last_updated_ts'], normalized], axis=1)
        result.drop(['auto_lock', 'away_mode', 'away_preset_days', 'away_preset_temperature',
                     'battery_low', 'boost_time', 'child_lock', 'eco_temperature','force',
                     'holidays','holidays_schedule','linkquality', 'local_temperature_calibration',
                     'max_temperature','min_temperature','position','preset','running_state',
                     'system_mode','update_available', 'valve_detection','week','window_detection',
                     'window_open','workdays','workdays_schedule','unit_of_measurement','device_class',
                     'icon','friendly_name', 'update.installed_version','update.state',
                     'window_detection_params.minutes','window_detection_params.temperature',
                     'update.latest_version','programming_mode', 'comfort_temperature','current_heating_setpoint'
                     ], axis='columns', inplace=True)
        
        result = result.groupby('last_updated_ts').first().reset_index()
        result.set_index('last_updated_ts', inplace=True)
        result = result.reindex(all_timestamps, method='ffill').reset_index()
        result.rename(columns={'index': 'last_updated_ts'}, inplace=True)
        result.to_csv('CSV/t4.csv', index=False)

        df = pd.read_sql(sql_power, conn)
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        df['last_updated_ts'] = pd.to_datetime(df['last_updated_ts'], unit='s').dt.floor('S')
        result = pd.concat([df['last_updated_ts'], normalized], axis=1)
        result.drop(['state_class', 'indicator_mode', 'linkquality', 'power_outage_memory', 'energy', 'unit_of_measurement',
                     'device_class', 'friendly_name'], axis='columns', inplace=True)
        
        result = result.groupby('last_updated_ts').first().reset_index()
        result.set_index('last_updated_ts', inplace=True)
        result = result.reindex(all_timestamps, method='ffill').reset_index()
        result.rename(columns={'index': 'last_updated_ts'}, inplace=True)
        result.to_csv('CSV/p4.csv', index=False)

    except mysql.connector.Error as e:
        print(f"Błąd połączenia: {e}")
    finally:
        if conn.is_connected():
            conn.close()    

def create_presence_data_set():
    df = pd.read_csv("CSV/p1.csv")
    df['last_updated_ts'] = pd.to_datetime(df['last_updated_ts'])
    df['presence'] = ((df['power'] > 60) | ((df['last_updated_ts'].dt.hour >= 0) & (df['last_updated_ts'].dt.hour <= 8))).astype(int)
    df.to_csv('CSV/p1_presence.csv', index=False)

def avg_power_to_temp():
    # Wczytanie danych
    power_data = pd.read_csv('CSV/power.csv')
    temperature_data = pd.read_csv('CSV/temperatures.csv')

    # Agregacja danych o zużyciu energii
    segment_length = len(power_data) // len(temperature_data)
    aggregated_power = []

    for i in range(len(temperature_data)):
        segment = power_data[i*segment_length:(i+1)*segment_length]
        avg_power = segment['power'].mean()  # Średnia dla segmentu
        aggregated_power.append(avg_power)

    # Tworzenie DataFrame z agregowanymi danymi
    aggregated_power_data = pd.DataFrame({
        'average_power': aggregated_power
    })

    print(len(aggregated_power))
    
    # Dołączenie agregowanych danych o zużyciu energii do danych o temperaturze
    merged_data = pd.concat([aggregated_power_data.reset_index(drop=True), temperature_data['local_temperature'].reset_index(drop=True)], axis=1)

    # Obliczenie współczynnika korelacji Pearsona
    correlation = merged_data.corr()

    print("Macierz korelacji:")
    print(correlation)

    plt.scatter(merged_data['average_power'], merged_data['local_temperature'])
    plt.title('Zależność między zużyciem energii a temperaturą')
    plt.xlabel('Średnie zużycie energii (W)')
    plt.ylabel('Lokalna temperatura (°C)')
    plt.show()

def predict_power_consumption():
    print("Predict power consumption")
    # Przygotowanie danych
    power_data = pd.read_csv('CSV/power.csv')
    temperature_data = pd.read_csv('CSV/temperatures.csv')

    min_length = min(len(power_data), len(temperature_data))
    power_data = power_data.head(min_length)
    temperature_data = temperature_data.head(min_length)

    # Usuwanie wierszy z NaN
    power_features = power_data[['current', 'energy', 'power']].dropna().reset_index(drop=True)
    temperature_features = temperature_data[['local_temperature']].dropna().reset_index(drop=True)

    # Synchronizacja indeksów
    common_indices = power_features.index.intersection(temperature_features.index)
    power_features = power_features.loc[common_indices]
    temperature_features = temperature_features.loc[common_indices]

    if not power_features.index.equals(temperature_features.index):
        print("Indeksy nie są zsynchronizowane!")
    else:
        # Połączenie cech z obu zestawów danych
        features = pd.concat([power_features.reset_index(drop=True), temperature_features.reset_index(drop=True)], axis=1)
        target = power_data['power'].reset_index(drop=True)

        # Wspólny indeks
        common_index = features.index.intersection(target.index)

        # Użycie wspólnego indeksu do synchronizacji
        features = features.loc[common_index]
        target = target.loc[common_index]

        if len(features) == len(target):
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
            
            # Usunięcie rekordów z NaN w target (y)
            non_nan_indices = ~y_train.isna()
            X_train = X_train[non_nan_indices]
            y_train = y_train[non_nan_indices]

            # Podobnie dla zestawu testowego
            non_nan_indices_test = ~y_test.isna()
            X_test = X_test[non_nan_indices_test]
            y_test = y_test[non_nan_indices_test]

            if len(X_train) != len(y_train):
                raise ValueError("Liczba wierszy w X_train i y_train nie jest równa. Upewnij się, że dane są poprawnie przetworzone.")

            # Trenowanie modelu
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Predykcja i ocena
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f'MSE: {mse}')

            # Analiza ważności cech
            feature_importances = pd.Series(model.feature_importances_, index=features.columns)
            feature_importances.plot(kind='barh')
            plt.title('Ważność cech')
            plt.show()
        else:
            print("Błąd: liczba wierszy w features i target nie jest taka sama.")
            print("Liczba wierszy w features:", len(features))
            print("Liczba wierszy w target:", len(target))


if __name__ == '__main__':
    # avg_power_to_temp()
    # predict_power_consumption()
    # get_csv()
    create_presence_data_set()