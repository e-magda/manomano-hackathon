from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

pd.set_option("display.max_colwidth", 255)


# DATA
df = pd.read_csv("datasets/manomano-dataset-nps.csv")
df_transaction = pd.read_csv("datasets/nouvelle_date.csv")

df_trustpilot = pd.read_csv("datasets/trustpilot_sentiment_final.csv")
df_trustpilot_polarity = df_trustpilot.groupby(["date", "polarity"], as_index=False)[
    "score"
].mean()
trustpilot_comments_sort_bad = df_trustpilot.sort_values(by="score")[
    "text"
].reset_index(drop=True)

df_twitter = pd.read_csv("datasets/twitter_sentiment_final.csv")
df_twitter_polarity = df_twitter.groupby(["created_at", "polarity"], as_index=False)[
    "score"
].mean()
twitter_comments_sort_bad = df_twitter.sort_values(by="score")["text"].reset_index(
    drop=True
)

df_manomano = pd.read_csv("datasets/dataset_sentiment_final.csv")
df_mano_polarity = df_manomano.groupby(["date", "polarity"], as_index=False)[
    "score"
].mean()
manomano_comments_sort_bad = df_manomano.sort_values(by="score")["comment"].reset_index(
    drop=True
)

fig_bar = px.bar(
    df_transaction.sort_values(by="semaine_mois"),
    x="family",
    y="bv_transaction",
    color="nps_respondent",
    animation_frame="semaine_mois",
    color_discrete_map={
        "Promoter": "#488A99",
        "Passive": "#DBAE58",
        "Detractor": "#AC3E31",
    },
    labels={"nps_respondent": "Customer category"},
    height=600,
)
fig_bar.update_yaxes(showgrid=False)
fig_bar.update_xaxes(
    {
        "categoryorder": "array",
        "categoryarray": [
            "Jardin piscine",
            "Outillage",
            "Mobilier d'intérieur",
            "Plomberie chauffage",
            "Salle de bain, WC",
            "Quincaillerie",
            "Electricité",
            "Luminaire",
            "Animalerie",
            "Revêtement sol et mur",
            "Cuisine",
            "Construction matériaux",
        ],
    }
)
fig_bar.update_traces(hovertemplate=None)
fig_bar.update_layout(
    margin=dict(t=70, b=70, l=70, r=40),
    hovermode="x",
    xaxis_tickangle=45,
    xaxis_title="Family",
    yaxis_title="Business volume (total)",
    plot_bgcolor="white",
    paper_bgcolor="white",
    title_font=dict(size=25, color="#a5a7ab", family="Lato, sans-serif"),
    font=dict(color="#8a8d93"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=16),
    ),
    xaxis=dict(tickfont=dict(size=15)),
    yaxis=dict(tickfont=dict(size=15)),
)
fig_bar["layout"]["updatemenus"][0]["pad"] = dict(r=10, t=150)
fig_bar["layout"]["sliders"][0]["pad"] = dict(
    r=10,
    t=150,
)
fig_bar.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 3000
fig_bar.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 450

fig_nps = go.Figure(
    go.Indicator(
        domain={"x": [0, 1], "y": [0, 1]},
        value=66.03,
        mode="gauge+number+delta",
        title={"text": "NPS Total"},
        delta={"reference": 66.03},
        gauge={
            "axis": {"range": [-100, 100]},
            "bar": {"color": "#DADADA"},
            "steps": [
                {"range": [-100, 0], "color": "#AC3E31"},
                {"range": [0, 50], "color": "#DBAE58"},
                {"range": [50, 100], "color": "#488A99"},
            ],
        },
    )
)

hist_mano = px.histogram(
    df_manomano,
    x="polarity",
    color="polarity",
    barmode="group",
    color_discrete_map={
        "positive": "#488A99",
        "neutral": "#DBAE58",
        "negative": "#AC3E31",
    },
)

scatter_mano = px.scatter(
    df_mano_polarity,
    x="date",
    y="score",
    color="polarity",
    color_discrete_map={
        "positive": "#488A99",
        "neutral": "#DBAE58",
        "negative": "#AC3E31",
    },
    labels={"date": "Date [month]", "score": "Sentiment score"},
)

hist_trustpilot = px.histogram(
    df_trustpilot,
    x="polarity",
    color="polarity",
    barmode="group",
    color_discrete_map={
        "positive": "#488A99",
        "neutral": "#DBAE58",
        "negative": "#AC3E31",
    },
)

scatter_trustpilot = px.scatter(
    df_trustpilot_polarity,
    x="date",
    y="score",
    color="polarity",
    color_discrete_map={
        "positive": "#488A99",
        "neutral": "#DBAE58",
        "negative": "#AC3E31",
    },
    labels={"date": "Date [month]", "score": "Sentiment score"},
)

hist_twitter = px.histogram(
    df_twitter,
    x="polarity",
    color="polarity",
    barmode="group",
    color_discrete_map={
        "positive": "#488A99",
        "neutral": "#DBAE58",
        "negative": "#AC3E31",
    },
)

scatter_twitter = px.scatter(
    df_twitter_polarity,
    x="created_at",
    y="score",
    color="polarity",
    color_discrete_map={
        "positive": "#488A99",
        "neutral": "#DBAE58",
        "negative": "#AC3E31",
    },
    labels={"created_at": "Date [month]", "score": "Sentiment score"},
)

# FRONTEND
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.title = "ManoMano"
app._favicon = "icon.png"

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Img(src="assets/logo_2.png", width="auto", height=100),
                    style={"textAlign": "center", "marginTop": "5px"},
                    width=3,
                ),
                dbc.Col(
                    html.H1(
                        "ManoMano - Boosting your customer engagement",
                        className="page-title",
                    )
                ),
            ]
        ),
        dcc.Tabs(
            id="tabs-graphs",
            value="tab-1-content",
            children=[
                dcc.Tab(
                    label="ManoMano Customer Satisfaction Data",
                    value="tab-1-content",
                    children=[
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(width=1),
                                        dbc.Col(
                                            html.H4(
                                                "Overview of ManoMano's Net Promoter Score (NPS) from August to November 2021",
                                                className="tab-title",
                                            )
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(width=1),
                                        dbc.Col(
                                            [
                                                html.H3(
                                                    "Customer group by NPS score",
                                                    style={"marginBottom": "30px"},
                                                ),
                                                dcc.Graph(
                                                    figure=fig_nps,
                                                    config={"displayModeBar": False},
                                                ),
                                                html.P(
                                                    "ManoMano's NPS score over the 4 month-period is 66. NPS is a customer satisfaction and loyalty metric \
                            ranging from -100 to 100. Scores above 50 are considered as excellent.",
                                                    style={"marginTop": "60px"},
                                                    className="simpleText",
                                                ),
                                            ],
                                            width=5,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Embed(
                                                    src="https://chart-studio.plotly.com/~emi-magda/78.embed",
                                                    width=800,
                                                    height=600,
                                                ),
                                                html.P(
                                                    "Scores by country platform are similar. The average British customer experience \
                                   has the best score while the German one is slightly below other ManoMano's markets.",
                                                    className="simpleText",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(width=1),
                                        dbc.Col(
                                            [
                                                html.H3(
                                                    "Business volume by product family and customer group over time",
                                                    style={"marginTop": "30px"},
                                                ),
                                                dcc.Graph(
                                                    id="graph-1-tabs",
                                                    figure=fig_bar,
                                                    config={"displayModeBar": False},
                                                ),
                                                html.P(
                                                    "The figure shows customer satisfaction after placing orders on ManoMano's marketplace. Data is shown by category of products and the sum of transactions \
                            placed by the customers who answered the survey. Customer scores are similar across categories of products.",
                                                    className="simpleText",
                                                ),
                                            ],
                                            width=10,
                                        ),
                                        dbc.Col(width=1),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="ManoMano Survey Comment Analysis",
                    value="tab-3-content",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    html.H4(
                                        "Analysis of the comments provided in ManoMano's customer survey",
                                        className="tab-title",
                                    )
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    [
                                        html.H3(
                                            "Sentiment score on ManoMano's customer survey comments"
                                        ),
                                        dcc.Graph(
                                            figure=scatter_mano,
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.H3(
                                            "Polarity of ManoMano's customer survey comments"
                                        ),
                                        dcc.Graph(
                                            figure=hist_mano,
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(width=1),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    [
                                        html.H3(
                                            "Most frequent words in negative comments",
                                            style={
                                                "marginTop": "30px",
                                                "marginBottom": "30px",
                                            },
                                        ),
                                        html.Img(
                                            src="assets/word_key.png",
                                            className="center",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            children=[
                                                html.H3(
                                                    "Selection of comments identified as negative",
                                                    style={
                                                        "marginTop": "30px",
                                                        "marginBottom": "60px",
                                                    },
                                                ),
                                                html.P(
                                                    manomano_comments_sort_bad[0],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    manomano_comments_sort_bad[1],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    manomano_comments_sort_bad[4],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    manomano_comments_sort_bad[5],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    manomano_comments_sort_bad[7],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    manomano_comments_sort_bad[10],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    manomano_comments_sort_bad[12],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    manomano_comments_sort_bad[14],
                                                    className="defaultTextBox",
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(width=1),
                            ]
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Trustpilot Comment Analysis",
                    value="tab-4-content",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    [
                                        html.H4(
                                            "Analysis of the comments left by ManoMano customers on the consumer review website Trustpilot",
                                            className="tab-title",
                                        ),
                                        html.A(
                                            "Trustpilot",
                                            href="https://www.trustpilot.com/review/manomano.fr",
                                            target="_blank",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    [
                                        html.H3(
                                            "Sentiment score on Trustpilot comments"
                                        ),
                                        dcc.Graph(
                                            figure=scatter_trustpilot,
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.H3("Polarity of Trustpilot comments"),
                                        dcc.Graph(
                                            figure=hist_trustpilot,
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(width=1),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    [
                                        html.H3(
                                            "Most frequent words in negative comments",
                                            style={
                                                "marginTop": "30px",
                                                "marginBottom": "30px",
                                            },
                                        ),
                                        html.Img(
                                            src="assets/word_perceuse_trustpilot.png",
                                            className="center",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            children=[
                                                html.H3(
                                                    "Selection of comments identified as negative",
                                                    style={
                                                        "marginTop": "30px",
                                                        "marginBottom": "60px",
                                                    },
                                                ),
                                                html.P(
                                                    trustpilot_comments_sort_bad[2],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    trustpilot_comments_sort_bad[3],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    trustpilot_comments_sort_bad[4],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    trustpilot_comments_sort_bad[5],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    trustpilot_comments_sort_bad[14],
                                                    className="defaultTextBox",
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(width=1),
                            ]
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Twitter Comment Analysis",
                    value="tab-5-content",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    [
                                        html.H4(
                                            "Analysis of tweets mentioning ManoMano's French Twitter handle",
                                            className="tab-title",
                                        ),
                                        html.A(
                                            "Twitter",
                                            href="https://twitter.com/search?q=manomano_FR&src=typed_query&f=live",
                                            target="_blank",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    [
                                        html.H3(
                                            "Sentiment score on ManoMano related tweets"
                                        ),
                                        dcc.Graph(
                                            figure=scatter_twitter,
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.H3("Polarity of ManoMano related tweets"),
                                        dcc.Graph(
                                            figure=hist_twitter,
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(width=1),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(width=1),
                                dbc.Col(
                                    [
                                        html.H3(
                                            "Most frequent words in negative comments",
                                            style={
                                                "marginTop": "30px",
                                                "marginBottom": "30px",
                                            },
                                        ),
                                        html.Img(
                                            src="assets/word_secateur_twitter.png",
                                            className="center",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            children=[
                                                html.H3(
                                                    "Selection of comments identified as negative",
                                                    style={
                                                        "marginTop": "30px",
                                                        "marginBottom": "60px",
                                                    },
                                                ),
                                                html.P(
                                                    twitter_comments_sort_bad[0],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    twitter_comments_sort_bad[2],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    twitter_comments_sort_bad[3],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    twitter_comments_sort_bad[5],
                                                    className="defaultTextBox",
                                                ),
                                                html.P(
                                                    twitter_comments_sort_bad[8],
                                                    className="defaultTextBox",
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(width=1),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)
