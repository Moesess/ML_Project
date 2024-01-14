import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
import dash_bootstrap_components as dbc
from LogisticReggression import LogisticRegression
from dash_bootstrap_templates import load_figure_template


filePath = 'CSV/Zeszyt1.xlsx'
nazwy_kolumn = ['data','dzień tygodnia','godzina','zużycie energii']
xls = pd.ExcelFile(filePath)
dane_wejsciowe_X = xls.parse('styczeń', usecols=nazwy_kolumn)

#Przekształcamy pola w macierzy na odpowiednie type
dane_wejsciowe_X['data'] = pd.to_datetime(dane_wejsciowe_X['data'], format='%Y-%m-%d')
# Przekształć kolumnę 'godzina' na format string
dane_wejsciowe_X['godzina_str'] = dane_wejsciowe_X['godzina'].astype(str)


# Przekształć kolumnę 'data' na cechy liczbowe
dane_wejsciowe_X['rok'] = dane_wejsciowe_X['data'].dt.year
dane_wejsciowe_X['miesiac'] = dane_wejsciowe_X['data'].dt.month
dane_wejsciowe_X['dzien'] = dane_wejsciowe_X['data'].dt.day
dane_wejsciowe_X['godzina'] = dane_wejsciowe_X['godzina_str'].str.split(':').str[0].astype(int)
dane_wejsciowe_X['minuta'] = dane_wejsciowe_X['godzina_str'].str.split(':').str[1].astype(int)

# Usuń kolumnę 'data', ponieważ już ją przekształciliśmy
dane_wejsciowe_X.drop(['data','godzina_str'], axis=1, inplace=True)

#dane_wejsciowe_X = pd.read_excel(filePath, sheet_name = ['styczeń'], usecols = nazwy_kolumn)

#robimy to samo dla danych wejściowych Y
xls2 = pd.ExcelFile(filePath)
dane_wejsciowe_Y =xls.parse('styczeń', usecols = ['obecnosc'])

X, y = dane_wejsciowe_X, dane_wejsciowe_Y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1234)

classifier = LogisticRegression(learning_rate=0.00001)
y_train = y_train.squeeze().values
classifier.fit(X_train, y_train)

X_chart = X
Y_chart = y

results = classifier.predict(X_chart)


X_chart["obecność"] = results
X_chart['data'] = pd.to_datetime(X_chart['rok'].astype(str) + '-' + 
                                 X_chart['miesiac'].astype(str) + '-' + 
                                 X_chart['dzien'].astype(str) + ' ' + 
                                 X_chart['godzina'].astype(str) + ':' + 
                                 X_chart['minuta'].astype(str))
X_chart.sort_values(by="data", inplace=True)

X_chart_filtered = X_chart[
    (X_chart['data'].dt.date >= pd.to_datetime('2023-01-01').date()) & 
    (X_chart['data'].dt.date <= pd.to_datetime('2023-01-07').date())]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
template = "DARKLY"
load_figure_template(template)

app.layout = html.Div(children=[
    html.H1(children='Wykres Zużycia Energii'),
    
    html.Div([
        html.Label('Wybierz datę początkową:'),
        dcc.DatePickerSingle(
            id='date-picker-start',
            date='2023-01-01'
        ),
        html.Label('Wybierz datę końcową:'),
        dcc.DatePickerSingle(
            id='date-picker-end',
            date='2023-01-01'
        )
    ]),

    dcc.Graph(
        id='graph',
        # figure=fig
    )
])

# Callback do aktualizacji wykresu
@app.callback(
    Output('graph', 'figure'),
    [Input('date-picker-start', 'date'),
     Input('date-picker-end', 'date')]
)
def update_graph(start_date, end_date):
    # Filtrowanie danych na podstawie wybranego zakresu dat    
    filtered_data = X_chart[(X_chart['data'].dt.date >= pd.to_datetime(start_date).date()) &
                            (X_chart['data'].dt.date <= pd.to_datetime(end_date).date())]
        
    # Tworzenie wykresu
    traces = []
    obecnosc_x = []
    obecnosc_y = []
    for i, row in filtered_data.iterrows():
        if row['obecność'] == 1:
            obecnosc_x.extend([row['data'], row['data'] + pd.Timedelta(minutes=1), None])
            obecnosc_y.extend([0, max(filtered_data['zużycie energii']), None])
    
    traces.append(
        go.Scatter(
            x=obecnosc_x,
            y=obecnosc_y,
            mode='lines',
            name='Obecność',
            opacity=0.3
        )
    )
    traces.append(
        go.Scatter(
            x=filtered_data['data'],
            y=filtered_data['zużycie energii'],
            mode='lines', 
            name='Zużycie energii',
        )
    )

    # Layout wykresu z dodanymi przedziałami obecności
    layout = go.Layout(
        title='Zużycie energii a wykrywanie obecności',
        xaxis=dict(title='Data'),
        yaxis=dict(title='Zużycie energii'),
        showlegend=True,
        template="DARKLY"
    )

    return {'data': traces, 'layout': layout}


if __name__ == '__main__':
    app.run_server(debug=True)
