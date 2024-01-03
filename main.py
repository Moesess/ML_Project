import mysql.connector
import pandas as pd
import json


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

if __name__ == '__main__':
    get_csv()