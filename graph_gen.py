import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

def get_logistic_curve():
    x = np.linspace(-10, 10, 100)
    y_logistic = 1 / (1 + np.exp(-x))

    np.random.seed(42)
    n_points = 30
    x_binary = np.random.uniform(-8, 8, n_points)
    prob_1 = 1 / (1 + np.exp(-x_binary * 0.8)) 
    y_binary = np.random.binomial(1, prob_1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y_logistic,
        mode='lines',
        name='Fonction Logistique',
        line=dict(color='royalblue', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=x_binary,
        y=y_binary,
        mode='markers',
        name='Données binaires (0/1)',
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Seuil de décision (0.5)")

    fig.update_layout(
        title='Courbe Logistique avec Données Binaires',
        xaxis_title='Variable explicative (x)',
        yaxis_title='Probabilité / Classe',
        showlegend=False,
        template='plotly_white'
        )
    
    return fig.show()


def get_exp_log_curves():

    x_exp = np.linspace(-5, 5, 100)
    y_exp = np.exp(x_exp)

    x_log = np.linspace(0.01, 5, 100)  
    y_log = np.log(x_log)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Exponentielle", "Logarithme naturel"),
        horizontal_spacing=0.15
    )

    fig.add_trace(
        go.Scatter(x=x_exp, y=y_exp, mode='lines', line=dict(color='royalblue', width=3)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=x_log, y=y_log, mode='lines', line=dict(color='darkorange', width=3)),
        row=1, col=2
    )

    fig.update_layout(
        title='Exponentielle vs Logarithme naturel',
        template='plotly_white',
        showlegend=False
    )

    return fig.show()


def get_roc_auc():
    x_perfect = [0, 0, 1]
    y_perfect = [0, 1, 1]

    x_random = [0, 1]
    y_random = [0, 1]

    x_normal = [0, 0.06, 0.12, 0.19, 0.33, 0.45, 0.66, 0.81, 1.0]
    y_normal = [0, 0.53, 0.68, 0.81, 0.87, 0.91, 0.96, 0.99, 1.0]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["ROC parfaite (AUC = 1)", "ROC aléatoire (AUC = 0.5)", "ROC normale (AUC ≈ 0.85)"],
        horizontal_spacing=0.08
    )

    fig.add_trace(go.Scatter(x=x_perfect, y=y_perfect, mode='lines+markers',
                            name='Parfaite', line=dict(color='green', width=4)),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            name='Aléatoire', line=dict(color='gray', dash='dash')),
                row=1, col=1)

    fig.add_trace(go.Scatter(x=x_random, y=y_random, mode='lines+markers',
                            name='Aléatoire', line=dict(color='red', width=4)),
                row=1, col=2)

    fig.add_trace(go.Scatter(x=x_normal, y=y_normal, mode='lines+markers',
                            name='Normale', line=dict(color='blue', width=4)),
                row=1, col=3)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            showlegend=False, line=dict(color='gray', dash='dash')),
                row=1, col=3)

    for i in range(1, 4):
        fig.update_xaxes(title_text="Faux positifs (FPR)", range=[0, 1], row=1, col=i)
        fig.update_yaxes(title_text="Vrais positifs (TPR)", range=[0, 1], row=1, col=i)

    fig.update_layout(
        title="Illustration des trois cas possibles pour la courbe ROC et l'AUC",
        template="plotly_white",
        showlegend=False
    )

    return fig.show()
