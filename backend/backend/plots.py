import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def plot_score_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['habitability_score'], kde=True, color='teal', ax=ax)
    ax.set_title("Distribution of Habitability Scores")
    ax.set_xlabel("Habitability Score (0–1)")
    ax.set_ylabel("Number of Planets")
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def plot_correlations(df):
    numeric = ['planet_mass_earth', 'orbital_period_days', 'orbit_distance_au',
               'star_temperature_k', 'star_radius_solar', 'habitability_score']
    corr = df[numeric].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax)
    ax.set_title("Feature Correlation Matrix")
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')