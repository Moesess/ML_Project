import mysql.connector
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_csv():
    db_config = {
        'user': 'admin',              # Nazwa użytkownika
        'password': 'admin',          # Hasło
        'host': 'localhost',          # Host (użyj nazwy usługi Docker Compose, jeśli łączysz się z innego kontenera)
        'port': 3306,                 # Port
        'database': 'MLDB'            # Nazwa bazy danych
    }

    query = "SELECT shared_attrs FROM state_attributes                          \
             WHERE JSON_EXTRACT(shared_attrs, '$.device_class') = 'temperature' \
             AND JSON_EXTRACT(shared_attrs, '$.local_temperature') IS NOT NULL  \
             LIMIT 1000;"

    try:
        conn = mysql.connector.connect(**db_config)
        df = pd.read_sql(query, conn)
        normalized = pd.json_normalize(df['shared_attrs'].apply(json.loads))
        normalized.to_csv('temperatures.csv', index=False)
    except mysql.connector.Error as e:
        print(f"Błąd połączenia: {e}")
    finally:
        if conn.is_connected():
            conn.close()    

def learn():
    # Wczytanie danych
    data = pd.read_csv('temperatures.csv')

    # Wstępne przetwarzanie danych
    # Czyszczenie danych
    data.dropna(subset=['local_temperature'], inplace=True)  # Usuwanie wierszy z brakującą temperaturą

    # Przekształcanie cech kategorycznych
    data = pd.get_dummies(data, columns=['system_mode', 'running_state'])
    data = data[data['friendly_name'] == 'Regulator temperatury - tomek local temperature calibration']
    data = data.drop(columns=['away_preset_days', 'comfort_temperature', 'auto_lock', 'away_mode', 'child_lock', 
                                  'force', 'holidays', 'holidays_schedule', 'preset', 'week', 
                                  'window_detection', 'workdays', 'workdays_schedule', 'unit_of_measurement', 
                                  'device_class', 'icon', 'friendly_name', 'update.installed_version', 'update.state'])

    # Podział danych
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Zapis przetworzonych danych
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

    features = data[['local_temperature', 'current_heating_setpoint']]
    target = data['comfort_temperature']

    # Podział na zestawy treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


    # Budowa modelu regresji
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ocena modelu
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Błąd średniokwadratowy (MSE): {mse}')

    # Przewidywanie temperatury komfortu
    comfort_prediction = model.predict(X_test)
    print(comfort_prediction)


if __name__ == '__main__':
    learn()
    # get_csv()