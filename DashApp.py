import dash
from dash import dcc
from dash import html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template


filePath = 'CSV/power_predict.csv'

dane_wejsciowe = pd.read_csv(filePath)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
template = "DARKLY"
load_figure_template(template)

app.layout = html.Div(children=[
    html.H1(children='Wykres Zużycia Energii'),
    
    html.Div([
        html.Label('Wybierz datę początkową:'),
        dcc.DatePickerSingle(
            id='date-picker-start',
            date='2023-02-01'
        ),
        html.Label('Wybierz datę końcową:'),
        dcc.DatePickerSingle(
            id='date-picker-end',
            date='2023-02-01'
        )
    ]),

    dcc.Graph(
        id='graph',
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
    filtered_data = dane_wejsciowe[(pd.to_datetime(dane_wejsciowe['date']).dt.date >= pd.to_datetime(start_date).date()) &
                                   (pd.to_datetime(dane_wejsciowe['date']).dt.date <= pd.to_datetime(end_date).date())]
        
    # Tworzenie danych dla wykresu obecności
    presence_data = filtered_data[filtered_data['presence'] == 1]
    presence_data['date'] = pd.to_datetime(presence_data['date'])
    max_power = filtered_data['power'].max() + 10

    # Współrzędne X i Y dla linii pionowych
    obecnosc_x = []
    obecnosc_y = []
    for ts in presence_data['date']:
        obecnosc_x.extend([ts, ts, None])
        obecnosc_y.extend([0, max_power, None])

    traces = [
        go.Scatter(
            x=obecnosc_x,
            y=obecnosc_y,
            mode='lines',
            name='Obecność',
            line=dict(color='pink', width=2),
            opacity=0.3
        ),
        go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['power'],
            mode='lines', 
            line=dict(color='yellow', width=2),
            name='Zużycie energii'
        ),
    ]

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
