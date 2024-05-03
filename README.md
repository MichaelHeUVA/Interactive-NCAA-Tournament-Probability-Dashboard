# Interactive-NCAA-Tournament-Probability-Dashboard

This repository contains the code for an interactive NCAA Tournament probability dashboard. The dashboard is built using the Dash web application framework in Python. The app is hosted on Render and can be accessed [here](https://interactive-ncaa-tournament-probability.onrender.com).

# Overview

The purpose of this app is to predict the outcomes of NCAA basketball games based on historical data and to provide insights into the probability of teams progressing through the tournament rounds. The app uses a linear regression model to predict game outcomes and provides visualizations to help users understand the predictions better.

The app has the following functionalities:

-   **Matchup Prediction**: Users can input two team IDs and see the predicted outcome of a hypothetical match between the two teams.
-   **Team Comparison**: Users can compare the predicted outcomes of multiple team matchups using bar graphs.
-   **Championship Prediction**: Users can select a team and see the probability of the team winning the championship based on the tournament rounds.
-   **Championship Visualization**: Users can visualize the progression probabilities of a team through the tournament rounds.

# Data

The data used in this app is a cleaned version of the NCAA March Madness historical data from the assignment description (match_data.csv). It is not included in the repository because Github doesn't allow for files larger than 100 MB. The data contains information about team matchups, game statistics, and game outcomes. The data is stored in a CSV file and is read into the app for model building and predictions.

# Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install -r requirements.txt
```

# Usage

Run app.py in the terminal to start the Dash app.

```bash
python3 app.py
```

# Data Reading and Preprocessing

-   My app reads cleaned matchup data from a CSV file.
-   The missing values for selected features are filled with the mean values of their respective columns.

# Model Building

-   My app employs a linear regression model to learn the relationship between various basketball game statistics and the outcome (Win/Loss).
-   Data is split into training and testing sets.
-   Accuracy and mean squared error are calculated for model evaluation.
-   The model is evaulated using OLS regression and the R-squared value is calculated.
-   Multiple models were built and tested to determine the best model for predicting game outcomes.
-   Full model evaluation is available in model.ipynb

# Callbacks for Interactivity:

-   **predict1v1**: Predicts the likelihood of one team winning against another in a hypothetical match based on their average historical stats.
-   **update_graph**: Updates bar graphs depicting the comparative predictions of team matchups.
-   **predict_championship**: Estimates the probability of a selected team progressing through successive match rounds up to winning the championship.
-   **update_championship_graph**: Visualizes the progression probabilities for a team through the tournament rounds.
-   Multistage tournament predictions brandishing further predictions based on team matchups, progressing from Sweet Sixteen to the Championship round.
