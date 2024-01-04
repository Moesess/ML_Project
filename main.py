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

    sql_power = "SELECT shared_attrs FROM state_attributes WHERE \
             JSON_EXTRACT(shared_attrs, '$.device_class') = 'power' \
             AND JSON_EXTRACT(shared_attrs, '$.friendly_name') like '%Tomek %' \
             ;"
    
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
        df = pd.read_sql(sql_temperature, conn)
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        normalized.to_csv('CSV/temperatures.csv', index=False)

        df = pd.read_sql(sql_power, conn)
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        normalized.to_csv('CSV/power.csv', index=False)

        df = pd.read_sql(sql_other, conn)
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        normalized.to_csv('CSV/other.csv', index=False)
    except mysql.connector.Error as e:
        print(f"Błąd połączenia: {e}")
    finally:
        if conn.is_connected():
            conn.close()    

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
    avg_power_to_temp()
    predict_power_consumption()
    # get_csv()