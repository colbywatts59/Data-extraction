# Dash app to manually label peaks in power traces
#
# Author: Colby Watts
#
# To use:
#   1. Input the name of the button 
button_name = "bottom left"

#   2. Input the path to the folder where the CSVs are located
path_to_buttons = "train_data"

#   3. Input the path to the folder where labeled JSON files will be saved  
label_folder = "labeled_jsons"

#   4. Run the script
#   5. Click on either end of the peaks
#   6. Press "Save Peaks" to save the labeled peaks
#       * If you make a mistake labeling a peak, press save peak and then it will allow you to restart labeling it
#   7. Press "Next Trace" to move to the next trace
#   8. Once all traces are labeled, the "Next Trace" button won't do anything
#
# Code explanation:
#  - Each CSV file is read and plotted
#  - Clicking on the graph will store the x-coordinates of the clicks
#  - The clicks are stored in a list and saved as json (it stores the beginning and end of each peak)
#  - The json file is named the same as the CSV file but with a .json extension to the specified directory
#  - json is used in train_peak_extractor.py to train the peak extractor model


import os
import json
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import dash_bootstrap_components as dbc


full_path = os.path.join(path_to_buttons, button_name)

# Create the label folder if it doesn't exist
os.makedirs(label_folder, exist_ok=True)

# List CSV files
csv_files = sorted([f for f in os.listdir(full_path) if f.endswith(".csv")])


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Peak Labeling Tool"

# App layout
app.layout = dbc.Container([
    html.H2("Super Awesome Peak Labeling Tool"),
    dcc.Store(id='click-store', data=[]),  # For storing click data
    dcc.Store(id='saved-peaks-store', data=[]),  # For storing saved peak data
    dcc.Store(id='current-file-index', data=0),
    dcc.Graph(id='voltage-graph'),
    html.Div(id='file-name', style={'marginTop': '10px'}),
    dbc.Button("Save Peaks", id='save-button', color='primary', n_clicks=0, style={'marginTop': '10px'}),
    html.Div(id='save-status', style={'marginTop': '10px'}),
    dbc.Button("Next Trace", id='next-button', color='secondary', n_clicks=0, style={'marginTop': '10px'}),
], fluid=True)

# Callback to update graph
@app.callback(
    Output('voltage-graph', 'figure'),
    Output('file-name', 'children'),
    Input('current-file-index', 'data'),
    Input('click-store', 'data') 
)
def update_graph(file_index, clicks):
    if file_index >= len(csv_files):
        return go.Figure(), "All files labeled."

    file_path = os.path.join(full_path, csv_files[file_index])
    df = pd.read_csv(file_path, skiprows=21)
    df = df.rename(columns={df.columns[2]: 'Voltage'}) 
    
    df = df.drop(columns=[df.columns[3], df.columns[4]], errors='ignore')
    df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
    df = df.dropna(subset=['Voltage'])
    
    df['Moving_Avg_Voltage'] = df['Voltage'].rolling(window=30).mean()
    df = df.dropna().reset_index(drop=True)

    # Create the base figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Sample Number'], y=df['Moving_Avg_Voltage'], mode='lines', name='Voltage'))

    # Plot vertical lines for the peaks
    if clicks:
        for i in range(0, len(clicks), 2):
            if i+1 < len(clicks):
                start, end = clicks[i], clicks[i+1]
                fig.add_vline(x=start, line=dict(color='red', dash='dash'), annotation_text="Start", annotation_position="top right")
                fig.add_vline(x=end, line=dict(color='green', dash='dash'), annotation_text="End", annotation_position="top left")
            else:
                # In case there's an unmatched click, still draw it
                fig.add_vline(x=clicks[i], line=dict(color='orange', dash='dot'), annotation_text="Click", annotation_position="top right")

    fig.update_layout(title="Click to mark start and end of peaks", clickmode='event+select')
    return fig, f"Labeling file: {csv_files[file_index]}"

# Callback to store click data
@app.callback(
    Output('click-store', 'data'),
    Output('save-status', 'children'),
    Output('current-file-index', 'data'),
    Input('voltage-graph', 'clickData'),
    Input('save-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    State('click-store', 'data'),
    State('current-file-index', 'data'),
    prevent_initial_call=True
)
def unified_callback(click_data, save_clicks, next_clicks, stored_clicks, file_index):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'voltage-graph':
        if click_data:
            x_val = click_data['points'][0]['x']
            return stored_clicks + [x_val], dash.no_update, dash.no_update

    elif triggered_id == 'save-button':
        if len(stored_clicks) % 2 != 0:
            return stored_clicks, "Please select an even number of points.", file_index

        peaks = [{"range": [stored_clicks[i], stored_clicks[i + 1]]} for i in range(0, len(stored_clicks), 2)]
        file_name = os.path.splitext(button_name + "/" + csv_files[file_index])[0] + ".json"
        with open(os.path.join(label_folder, file_name), "w") as f:
            json.dump(peaks, f)

        return [], f"Saved peaks to {file_name}", file_index

    elif triggered_id == 'next-button':
        next_index = file_index + 1 if file_index + 1 < len(csv_files) else file_index
        return [], "", next_index

    return dash.no_update, dash.no_update, dash.no_update
 

if __name__ == '__main__':
    app.run(debug=True, port=8050)
