import pandas as pd
from dash import Dash, dcc, html, callback, Input, Output, dcc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# import the stylesheet
stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# read the cleaned data
data = pd.read_csv("./data/match_data_cleaned.csv")

# create the Dash app
app = Dash(
    __name__, external_stylesheets=stylesheets, suppress_callback_exceptions=True
)

# get the server
server = app.server

# build the model
features = [
    "Assists Turnover Ratio",
    "Bench Points",
    "Biggest Lead",
    "Blocks",
    "Blocks Received",
    "Defensive Points Per Possession",
    "Fast Break Points Percentage",
    "Field Goals Percentage",
    "Fouls Total",
    "Free Throws Percentage",
    "Lead Changes",
    "Offensive Points Per Possession",
    "Points Second Chance",
    "Second Chance Points Percentage",
    "Steals",
    "Three Pointers Percentage",
    "Turnovers",
    "Two Pointers Percentage",
]
target = "Win"
data[features] = data[features].fillna(data[features].mean())
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
)

# initialize the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# making predictions
y_pred = model.predict(X_test)

# since we're using Linear Regression inappropriately for classification, we can round predictions
# to retrieve binary outcomes: 0 for lose, 1 for win
y_pred_binary = np.round(y_pred)

# Calculate the accuracy manually
accuracy = np.mean(y_pred_binary == y_test)

# mean squared error
mse = mean_squared_error(y_test, y_pred_binary)

# map the team name to the team ID
team_mapping = dict(zip(data["Team ID"], data["Team Name"]))

team_avg = data.groupby("Team ID")[features].mean()

team_avg["Team Name"] = team_avg.index.map(team_mapping)

team_avg = team_avg.sort_values(by="Team Name")

teams = team_avg["Team Name"].values

# create the dropdown options
teams_options = [{"label": team, "value": team} for team in teams]

# app layout
app.layout = html.Div(
    [
        html.H1(
            "Interactive NCAA Tournament Probability Dashboard", className="center"
        ),
        html.Div(
            [
                html.Div(
                    "The model is a Linear Regression model trained on the cleaned data and fine tuned to have the most accurate predictions."
                ),
                html.Div(f"Model Accuracy: {accuracy}, Mean Squared Error: {mse}"),
            ],
            className="border",
        ),
        html.Div(
            [
                html.H3("Select two teams to predict the outcome of a match"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Team 1"),
                                dcc.Dropdown(
                                    id="team1",
                                    value=teams[0],
                                    options=teams_options,
                                ),
                            ],
                            className="w-48",
                        ),
                        html.Div(
                            [
                                html.Div("Team 2"),
                                dcc.Dropdown(
                                    id="team2",
                                    value=teams[1],
                                    options=teams_options,
                                ),
                            ],
                            className="w-48",
                        ),
                    ],
                    className="flex justify-between",
                ),
                html.Div(id="1v1output"),
                dcc.Graph(id="1v1graph"),
            ],
            className="border",
        ),
        html.Div(
            [
                html.H3(
                    "Enter a team name to predict the probability of making it to the Championship"
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="championship-name",
                            value=teams[0],
                            options=teams_options,
                        )
                    ]
                ),
                html.Div(id="championship-output"),
                dcc.Graph(id="championship-graph"),
            ],
            className="border",
        ),
        html.Div(
            [
                html.H3("Predict the Champion from the Sweet 16"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Team 1"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-1",
                                    value=teams[np.where(teams == "UConn")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 2"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-2",
                                    value=teams[
                                        np.where(teams == "San Diego St.")[0][0]
                                    ],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 3"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-3",
                                    value=teams[np.where(teams == "Illinois")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 4"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-4",
                                    value=teams[np.where(teams == "Iowa St.")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 5"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-5",
                                    value=teams[
                                        np.where(teams == "North Carolina")[0][0]
                                    ],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 6"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-6",
                                    value=teams[np.where(teams == "Alabama")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 7"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-7",
                                    value=teams[np.where(teams == "Clemson")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 8"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-8",
                                    value=teams[np.where(teams == "Arizona")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 9"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-9",
                                    value=teams[np.where(teams == "Houston")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 10"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-10",
                                    value=teams[np.where(teams == "Duke")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 11"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-11",
                                    value=teams[np.where(teams == "NC State")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 12"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-12",
                                    value=teams[np.where(teams == "Marquette")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 13"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-13",
                                    value=teams[np.where(teams == "Purdue")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 14"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-14",
                                    value=teams[np.where(teams == "Gonzaga")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 15"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-15",
                                    value=teams[np.where(teams == "Creighton")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                        html.Div(
                            [
                                html.Div("Team 16"),
                                dcc.Dropdown(
                                    id="sweet-sixteen-16",
                                    value=teams[np.where(teams == "Tennessee")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-6",
                        ),
                    ],
                    className="flex justify-between",
                ),
                html.Div(id="sweet-sixteen-output"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Team 1"),
                                dcc.Dropdown(
                                    id="elite-eight-1",
                                    value=teams[np.where(teams == "UConn")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-12",
                        ),
                        html.Div(
                            [
                                html.Div("Team 2"),
                                dcc.Dropdown(
                                    id="elite-eight-2",
                                    value=teams[np.where(teams == "Illinois")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-12",
                        ),
                        html.Div(
                            [
                                html.Div("Team 3"),
                                dcc.Dropdown(
                                    id="elite-eight-3",
                                    value=teams[np.where(teams == "Alabama")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-12",
                        ),
                        html.Div(
                            [
                                html.Div("Team 4"),
                                dcc.Dropdown(
                                    id="elite-eight-4",
                                    value=teams[np.where(teams == "Clemson")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-12",
                        ),
                        html.Div(
                            [
                                html.Div("Team 5"),
                                dcc.Dropdown(
                                    id="elite-eight-5",
                                    value=teams[np.where(teams == "Duke")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-12",
                        ),
                        html.Div(
                            [
                                html.Div("Team 6"),
                                dcc.Dropdown(
                                    id="elite-eight-6",
                                    value=teams[np.where(teams == "NC State")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-12",
                        ),
                        html.Div(
                            [
                                html.Div("Team 7"),
                                dcc.Dropdown(
                                    id="elite-eight-7",
                                    value=teams[np.where(teams == "Purdue")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-12",
                        ),
                        html.Div(
                            [
                                html.Div("Team 8"),
                                dcc.Dropdown(
                                    id="elite-eight-8",
                                    value=teams[np.where(teams == "Tennessee")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-12",
                        ),
                    ],
                    className="flex justify-between",
                ),
                html.Div(id="elite-eight-output"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Final Four 1"),
                                dcc.Dropdown(
                                    id="final-four-1",
                                    value=teams[np.where(teams == "UConn")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-24",
                        ),
                        html.Div(
                            [
                                html.Div("Final Four 2"),
                                dcc.Dropdown(
                                    id="final-four-2",
                                    value=teams[np.where(teams == "Alabama")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-24",
                        ),
                        html.Div(
                            [
                                html.Div("Final Four 3"),
                                dcc.Dropdown(
                                    id="final-four-3",
                                    value=teams[np.where(teams == "NC State")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-24",
                        ),
                        html.Div(
                            [
                                html.Div("Final Four 4"),
                                dcc.Dropdown(
                                    id="final-four-4",
                                    value=teams[np.where(teams == "Purdue")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-24",
                        ),
                    ],
                    className="flex justify-between",
                ),
                html.Div(id="final-four-output"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Championship 1"),
                                dcc.Dropdown(
                                    id="championship-1",
                                    value=teams[np.where(teams == "UConn")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-48",
                        ),
                        html.Div(
                            [
                                html.Div("Championship 2"),
                                dcc.Dropdown(
                                    id="championship-2",
                                    value=teams[np.where(teams == "Purdue")[0][0]],
                                    options=teams_options,
                                ),
                            ],
                            className="w-48",
                        ),
                    ],
                    className="flex justify-between",
                ),
                html.Div(id="championship-game-output"),
                dcc.Graph(id="championship-game-graph"),
            ],
            className="border",
        ),
    ]
)


@callback(
    Output("1v1output", "children"),
    Input("team1", "value"),
    Input("team2", "value"),
)
def predict1v1(team1, team2):
    # Predict the outcome of a match between two teams
    input_row1 = team_avg[team_avg["Team Name"] == team1]
    prediction1 = model.predict(input_row1[features])
    input_row2 = team_avg[team_avg["Team Name"] == team2]
    prediction2 = model.predict(input_row2[features])
    total = prediction1[0] + prediction2[0]
    if prediction1[0] > prediction2[0]:
        output = [
            f"{team1} is more likely to win. With a win percentage of {prediction1[0]/total}"
        ]
        return [html.P(out, className="center") for out in output]
    else:
        output = [
            f"{team2} is more likely to win. With a win percentage of {prediction2[0]/total}"
        ]
        return [html.P(out, className="center") for out in output]


@callback(
    Output("1v1graph", "figure"),
    Input("team1", "value"),
    Input("team2", "value"),
)
def update_graph(team1, team2):
    # Predict the outcome of a match between two teams
    input_row1 = team_avg[team_avg["Team Name"] == team1]
    prediction1 = model.predict(input_row1[features])
    input_row2 = team_avg[team_avg["Team Name"] == team2]
    prediction2 = model.predict(input_row2[features])
    total = prediction1[0] + prediction2[0]
    return {
        "data": [
            {
                "x": [team1, team2],
                "y": [prediction1[0] / total, prediction2[0] / total],
                "type": "bar",
            }
        ],
        "layout": {
            "title": f"{team1} vs {team2} Prediction",
        },
    }


@callback(
    Output("championship-output", "children"),
    Input("championship-name", "value"),
)
def predict_championship(team_name):
    # Calculate the probability of a team making it to the next round
    input_row = team_avg[team_avg["Team Name"] == team_name]
    prediction = model.predict(input_row[features])
    prediction = prediction[0] / (prediction[0] + 0.5)
    probabilities = [
        f"Probability of {team_name} winning Round of 64 is {prediction}",
        f"Probability of {team_name} winning Round of 32 is {pow(prediction, 2)}",
        f"Probability of {team_name} winning Sweet 16 is {pow(prediction, 3)}",
        f"Probability of {team_name} winning Elite 8 is {pow(prediction, 4)}",
        f"Probability of {team_name} winning Final 4 is {pow(prediction, 5)}",
        f"Probability of {team_name} winning Championship is {pow(prediction, 6)}",
    ]

    return [html.P(prob, className="center") for prob in probabilities]


@callback(
    Output("championship-graph", "figure"),
    Input("championship-name", "value"),
)
def update_championship_graph(team_name):
    # Calculate the probability of a team making it to the next round
    input_row = team_avg[team_avg["Team Name"] == team_name]
    prediction = model.predict(input_row[features])
    prediction = prediction[0] / (prediction[0] + 0.5)
    probabilities = [
        prediction,
        pow(prediction, 2),
        pow(prediction, 3),
        pow(prediction, 4),
        pow(prediction, 5),
        pow(prediction, 6),
    ]
    return {
        "data": [
            {
                "x": [
                    "Round of 64",
                    "Round of 32",
                    "Sweet 16",
                    "Elite 8",
                    "Final 4",
                    "Championship",
                ],
                "y": probabilities,
                "type": "bar",
            }
        ],
        "layout": {
            "title": f"Probability of {team_name} Winning Each Round",
        },
    }


@callback(
    Output("sweet-sixteen-output", "children"),
    Input("sweet-sixteen-1", "value"),
    Input("sweet-sixteen-2", "value"),
    Input("sweet-sixteen-3", "value"),
    Input("sweet-sixteen-4", "value"),
    Input("sweet-sixteen-5", "value"),
    Input("sweet-sixteen-6", "value"),
    Input("sweet-sixteen-7", "value"),
    Input("sweet-sixteen-8", "value"),
    Input("sweet-sixteen-9", "value"),
    Input("sweet-sixteen-10", "value"),
    Input("sweet-sixteen-11", "value"),
    Input("sweet-sixteen-12", "value"),
    Input("sweet-sixteen-13", "value"),
    Input("sweet-sixteen-14", "value"),
    Input("sweet-sixteen-15", "value"),
    Input("sweet-sixteen-16", "value"),
)
def predict_sweet_sixteen(
    team_name1,
    team_name2,
    team_name3,
    team_name4,
    team_name5,
    team_name6,
    team_name7,
    team_name8,
    team_name9,
    team_name10,
    team_name11,
    team_name12,
    team_name13,
    team_name14,
    team_name15,
    team_name16,
):
    input_row1 = team_avg[team_avg["Team Name"] == team_name1]
    prediction1 = model.predict(input_row1[features])[0]

    input_row2 = team_avg[team_avg["Team Name"] == team_name2]
    prediction2 = model.predict(input_row2[features])[0]

    input_row3 = team_avg[team_avg["Team Name"] == team_name3]
    prediction3 = model.predict(input_row3[features])[0]

    input_row4 = team_avg[team_avg["Team Name"] == team_name4]
    prediction4 = model.predict(input_row4[features])[0]

    input_row5 = team_avg[team_avg["Team Name"] == team_name5]
    prediction5 = model.predict(input_row5[features])[0]

    input_row6 = team_avg[team_avg["Team Name"] == team_name6]
    prediction6 = model.predict(input_row6[features])[0]

    input_row7 = team_avg[team_avg["Team Name"] == team_name7]
    prediction7 = model.predict(input_row7[features])[0]

    input_row8 = team_avg[team_avg["Team Name"] == team_name8]
    prediction8 = model.predict(input_row8[features])[0]

    input_row9 = team_avg[team_avg["Team Name"] == team_name9]
    prediction9 = model.predict(input_row9[features])[0]

    input_row10 = team_avg[team_avg["Team Name"] == team_name10]
    prediction10 = model.predict(input_row10[features])[0]

    input_row11 = team_avg[team_avg["Team Name"] == team_name11]
    prediction11 = model.predict(input_row11[features])[0]

    input_row12 = team_avg[team_avg["Team Name"] == team_name12]
    prediction12 = model.predict(input_row12[features])[0]

    input_row13 = team_avg[team_avg["Team Name"] == team_name13]
    prediction13 = model.predict(input_row13[features])[0]

    input_row14 = team_avg[team_avg["Team Name"] == team_name14]
    prediction14 = model.predict(input_row14[features])[0]

    input_row15 = team_avg[team_avg["Team Name"] == team_name15]
    prediction15 = model.predict(input_row15[features])[0]

    input_row16 = team_avg[team_avg["Team Name"] == team_name16]
    prediction16 = model.predict(input_row16[features])[0]

    output = [
        f"Probability of {team_name1} winning against {team_name2} is {prediction1/(prediction1+prediction2)}",
        f"Probability of {team_name3} winning against {team_name4} is {prediction3/(prediction3+prediction4)}",
        f"Probability of {team_name5} winning against {team_name6} is {prediction5/(prediction5+prediction6)}",
        f"Probability of {team_name7} winning against {team_name8} is {prediction7/(prediction7+prediction8)}",
        f"Probability of {team_name9} winning against {team_name10} is {prediction9/(prediction9+prediction10)}",
        f"Probability of {team_name11} winning against {team_name12} is {prediction11/(prediction11+prediction12)}",
        f"Probability of {team_name13} winning against {team_name14} is {prediction13/(prediction13+prediction14)}",
        f"Probability of {team_name15} winning against {team_name16} is {prediction15/(prediction15+prediction16)}",
    ]
    return [html.P(out, className="center") for out in output]


@callback(
    Output("elite-eight-output", "children"),
    Input("elite-eight-1", "value"),
    Input("elite-eight-2", "value"),
    Input("elite-eight-3", "value"),
    Input("elite-eight-4", "value"),
    Input("elite-eight-5", "value"),
    Input("elite-eight-6", "value"),
    Input("elite-eight-7", "value"),
    Input("elite-eight-8", "value"),
)
def predict_elite_eight(
    team_name1,
    team_name2,
    team_name3,
    team_name4,
    team_name5,
    team_name6,
    team_name7,
    team_name8,
):
    input_row1 = team_avg[team_avg["Team Name"] == team_name1]
    prediction1 = model.predict(input_row1[features])[0]

    input_row2 = team_avg[team_avg["Team Name"] == team_name2]
    prediction2 = model.predict(input_row2[features])[0]

    input_row3 = team_avg[team_avg["Team Name"] == team_name3]
    prediction3 = model.predict(input_row3[features])[0]

    input_row4 = team_avg[team_avg["Team Name"] == team_name4]
    prediction4 = model.predict(input_row4[features])[0]

    input_row5 = team_avg[team_avg["Team Name"] == team_name5]
    prediction5 = model.predict(input_row5[features])[0]

    input_row6 = team_avg[team_avg["Team Name"] == team_name6]
    prediction6 = model.predict(input_row6[features])[0]

    input_row7 = team_avg[team_avg["Team Name"] == team_name7]
    prediction7 = model.predict(input_row7[features])[0]

    input_row8 = team_avg[team_avg["Team Name"] == team_name8]
    prediction8 = model.predict(input_row8[features])[0]

    output = [
        f"Probability of {team_name1} winning against {team_name2} {prediction1/(prediction1+prediction2)}",
        f"Probability of {team_name3} winning against {team_name4} {prediction3/(prediction3+prediction4)}",
        f"Probability of {team_name5} (seed 3) winning against {team_name6} (seed 6) {prediction5/(prediction5+prediction6)}",
        f"Probability of {team_name7} (seed 4) winning against {team_name8} (seed 5) {prediction7/(prediction7+prediction8)}",
    ]
    return [html.P(out, className="center") for out in output]


@callback(
    Output("final-four-output", "children"),
    Input("final-four-1", "value"),
    Input("final-four-2", "value"),
    Input("final-four-3", "value"),
    Input("final-four-4", "value"),
)
def predict_final_four(
    team_name1,
    team_name2,
    team_name3,
    team_name4,
):
    input_row1 = team_avg[team_avg["Team Name"] == team_name1]
    prediction1 = model.predict(input_row1[features])[0]

    input_row2 = team_avg[team_avg["Team Name"] == team_name2]
    prediction2 = model.predict(input_row2[features])[0]

    input_row3 = team_avg[team_avg["Team Name"] == team_name3]
    prediction3 = model.predict(input_row3[features])[0]

    input_row4 = team_avg[team_avg["Team Name"] == team_name4]
    prediction4 = model.predict(input_row4[features])[0]

    # input_row14 = team_avg[team_avg["Team Name"] == team_winner14]
    # prediction14 = model.predict(input_row14[features])[0]

    # team_predictions[team_winner23] = (
    #     prediction3 if team_winner23 == team_name2 else prediction2
    # )
    # input_row23 = team_avg[team_avg["Team Name"] == team_winner23]
    # prediction23 = model.predict(input_row23[features])[0]

    # input_row_final = team_avg[team_avg["Team Name"] == champion]
    # prediction_final = model.predict(input_row_final[features])[0]

    output = [
        f"Probability of {team_name1} winning against {team_name2} {prediction1/(prediction1+prediction2)}",
        f"Probability of {team_name3} winning against {team_name4} {prediction3/(prediction3+prediction4)}",
    ]
    return [html.P(out, className="center") for out in output]


@callback(
    Output("championship-game-output", "children"),
    Input("championship-1", "value"),
    Input("championship-2", "value"),
)
def predict_championship_game(team_name1, team_name2):
    input_row1 = team_avg[team_avg["Team Name"] == team_name1]
    prediction1 = model.predict(input_row1[features])[0]

    input_row2 = team_avg[team_avg["Team Name"] == team_name2]
    prediction2 = model.predict(input_row2[features])[0]

    output = [
        f"Probability of {team_name1} winning against {team_name2} {prediction1/(prediction1+prediction2)}",
    ]
    return [html.P(out, className="center") for out in output]


@callback(
    Output("championship-game-graph", "figure"),
    Input("championship-1", "value"),
    Input("championship-2", "value"),
)
def update_championship(team1, team2):
    # Predict the outcome of a match between two teams
    input_row1 = team_avg[team_avg["Team Name"] == team1]
    prediction1 = model.predict(input_row1[features])
    input_row2 = team_avg[team_avg["Team Name"] == team2]
    prediction2 = model.predict(input_row2[features])
    total = prediction1[0] + prediction2[0]
    return {
        "data": [
            {
                "x": [team1, team2],
                "y": [prediction1[0] / total, prediction2[0] / total],
                "type": "bar",
            }
        ],
        "layout": {
            "title": f"{team1} vs {team2} Prediction",
        },
    }


# run the app
if __name__ == "__main__":
    app.run_server(debug=True)
