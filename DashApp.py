import dash
from dash import dcc
from dash import html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template


PowerFile = 'CSV/power_predict.csv'
TemperaturesFile = 'CSV/optimal_temperatures.csv'

power = pd.read_csv(PowerFile)
temperatures = pd.read_csv(TemperaturesFile)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
template = "DARKLY"
load_figure_template(template)

cardPower = dbc.Card(
    [
        dbc.CardHeader("Wykres Zużycia Energii"),
        dbc.CardBody(
            [
                html.Div([
                    html.P(
                        dcc.Graph(
                            id='graphPower',
                        )
                    ),
                ]),
            ]
        ),
        dbc.CardFooter(""),
    ],
)

cardTemperatures = dbc.Card(
    [
        dbc.CardHeader("Wykres Temperatur"),
        dbc.CardBody(
            [
                html.Div([
                    html.P(
                        dcc.Graph(
                            id='graphTemp',
                        )
                    ),
                ], style={'padding': 10, 'flex': 1}),
                html.Div([
                    html.P(
                        dcc.Graph(
                            id='graphPMV',
                        )
                    ),
                ], style={'padding': 10, 'flex': 1}),
            ], style={'display': 'flex', 'flexDirection': 'row'}
        ),
        dbc.CardFooter(""),
    ],
)

app.layout = html.Div(
    children=[
        html.P([
            html.Label('Wybierz datę początkową:'),
            dcc.DatePickerSingle(
                id='date-picker-start',
                date='2023-02-01',
                style={"margin-left": "5px"}
            ),
            
            html.Label('Wybierz datę końcową:', style={"margin-left": "10px"}),
            dcc.DatePickerSingle(
                id='date-picker-end',
                date='2023-02-01',
                style={"margin-left": "5px"}
            ),
        ], style={"margin": "5px"}),
        html.Div([
            cardPower,
            cardTemperatures
        ], style={"margin": "10px"}),
    ],
)

# Callback do aktualizacji wykresu energii
@app.callback(
    Output('graphPower', 'figure'),
    [Input('date-picker-start', 'date'),
     Input('date-picker-end', 'date')]
)
def update_graph_power(start_date, end_date):
    # Filtrowanie danych na podstawie wybranego zakresu dat    
    filtered_data = power[(pd.to_datetime(power['date']).dt.date >= pd.to_datetime(start_date).date()) &
                                   (pd.to_datetime(power['date']).dt.date <= pd.to_datetime(end_date).date())]
        
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
        yaxis=dict(title='Zużycie energii (W)'),
        showlegend=True,
        template="DARKLY"
    )

    return {'data': traces, 'layout': layout}

# Callback do aktualizacji wykresu temperatur
@app.callback(
    Output('graphTemp', 'figure'),
    [Input('date-picker-start', 'date'),
     Input('date-picker-end', 'date')]
)
def update_graph_temp(start_date, end_date):
    # Filtrowanie danych na podstawie wybranego zakresu dat    
    filtered_data = temperatures[(pd.to_datetime(temperatures['date']).dt.date >= pd.to_datetime(start_date).date()) &
                                   (pd.to_datetime(temperatures['date']).dt.date <= pd.to_datetime(end_date).date())]
    avgT= filtered_data['local_temperature'].mean()
    avgOT = filtered_data['optimal_temperature'].mean()
    # presence_data = filtered_data[filtered_data['presence'] == 1]
    # presence_data['date'] = pd.to_datetime(presence_data['date'])
    # max_temp = max(filtered_data['local_temperature'].max(), filtered_data['optimal_temperature'].max()) + 5
    # min_temp = min(filtered_data['local_temperature'].min(), filtered_data['optimal_temperature'].min()) - 5

    # # Współrzędne X i Y dla linii pionowych
    # obecnosc_x = []
    # obecnosc_y = []
    # for ts in presence_data['date']:
    #     obecnosc_x.extend([ts, ts, None])
    #     obecnosc_y.extend([min_temp, max_temp, None])

    traces = [
        go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['local_temperature'],
            mode='lines', 
            line=dict(color='red', width=2),
            name='LT',
            opacity=0.3
        ),
        go.Scatter(
            x=filtered_data['date'], 
            y=[avgT]*len(filtered_data['date']), 
            mode='lines', 
            name='Avg T', 
            line=dict(color='red', dash='dash'),
            opacity=1
        ),
        go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['optimal_temperature'],
            mode='lines', 
            line=dict(color='yellow', width=2),
            name='OT',
            opacity=0.3
        ),
        go.Scatter(
            x=filtered_data['date'], 
            y=[avgOT]*len(filtered_data['date']), 
            mode='lines', 
            name='Avg OT', 
            line=dict(color='yellow', dash='dash'),
            opacity=1
        ),
    ]

    # Layout wykresu z dodanymi przedziałami obecności
    layout = go.Layout(
        title='Predykcja optymalnej temperatury ( OT ) i porównanie z lokalną temperaturą ( LT )',
        xaxis=dict(title='Data'),
        yaxis=dict(title='Temperatury (C)'),
        showlegend=True,
        template="DARKLY"
    )

    return {'data': traces, 'layout': layout}

# Callback do aktualizacji wykresu PMV
@app.callback(
    Output('graphPMV', 'figure'),
    [Input('date-picker-start', 'date'),
     Input('date-picker-end', 'date')]
)
def update_graph_temp(start_date, end_date):
    # Filtrowanie danych na podstawie wybranego zakresu dat    
    filtered_data = temperatures[(pd.to_datetime(temperatures['date']).dt.date >= pd.to_datetime(start_date).date()) &
                                   (pd.to_datetime(temperatures['date']).dt.date <= pd.to_datetime(end_date).date())]

    avgPMV = filtered_data['PMV'].mean()
    avgOptimalPMV = filtered_data['PMV_after'].mean()

    traces = [
        go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['PMV'],
            mode='lines', 
            line=dict(color='red', width=2),
            name='PMV',
            opacity=0.3
        ),
        go.Scatter(
            x=filtered_data['date'], 
            y=[avgPMV]*len(filtered_data['date']), 
            mode='lines', 
            name='Avg PMV', 
            line=dict(color='red', dash='dash'),
            opacity=1
        ),
        go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['PMV_after'],
            mode='lines', 
            line=dict(color='yellow', width=2),
            name='OPMV',
            opacity=0.3
        ),
        go.Scatter(
            x=filtered_data['date'], 
            y=[avgOptimalPMV]*len(filtered_data['date']), 
            mode='lines', 
            name='Avg OPMV', 
            line=dict(color='yellow', dash='dash'),
            opacity=1
        ),
    ]

    # Layout wykresu z dodanymi przedziałami obecności
    layout = go.Layout(
        title='Wykres zmiany PMV na zoptymalizowane PMV ( OPMV )',
        xaxis=dict(title='Data'),
        yaxis=dict(title='PMV'),
        showlegend=True,
        template="DARKLY"
    )

    return {'data': traces, 'layout': layout}


if __name__ == '__main__':
    app.run_server(debug=True)
