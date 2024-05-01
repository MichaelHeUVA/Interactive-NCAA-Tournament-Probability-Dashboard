import pandas as pd
from dash import Dash, dcc, html

# import the stylesheet
stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# read the cleaned data
df = pd.read_csv("./data/match_data_cleaned.csv")

# create the Dash app
app = Dash(
    __name__, external_stylesheets=stylesheets, suppress_callback_exceptions=True
)

# get the server
server = app.server
