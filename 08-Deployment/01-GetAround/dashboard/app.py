import os

import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GetAround — Analyse des retards",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx"


# ---------------------------------------------------------------------------
# Chargement des données (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_URL)
    return df


@st.cache_data
def enrich_data(df):
    """Joint le dataset avec lui-même pour récupérer le retard de la location précédente."""
    prev_delay = df[['rental_id', 'delay_at_checkout_in_minutes']].rename(
        columns={
            'rental_id': 'previous_ended_rental_id',
            'delay_at_checkout_in_minutes': 'prev_delay_at_checkout',
        }
    )
    enriched = df[df['previous_ended_rental_id'].notna()].merge(
        prev_delay, on='previous_ended_rental_id', how='left'
    )
    enriched['is_problematic'] = (
        (enriched['prev_delay_at_checkout'] > 0)
        & (enriched['prev_delay_at_checkout'] > enriched['time_delta_with_previous_rental_in_minutes'])
    )
    return enriched


def simulate_threshold(df_all, df_with_prev, threshold, scope):
    """Calcule les métriques pour un threshold et un scope donnés."""
    if scope == 'Connect uniquement':
        mask = df_with_prev['checkin_type'] == 'connect'
    else:
        mask = pd.Series([True] * len(df_with_prev), index=df_with_prev.index)

    affected = df_with_prev[mask]
    total_rentals = len(df_all)
    total_problematic = df_with_prev['is_problematic'].sum()

    blocked = affected[affected['time_delta_with_previous_rental_in_minutes'] < threshold]
    solved = affected[
        affected['is_problematic']
        & (affected['time_delta_with_previous_rental_in_minutes'] < threshold)
    ]

    return {
        'blocked': len(blocked),
        'blocked_pct': round(len(blocked) / total_rentals * 100, 1),
        'solved': len(solved),
        'solve_rate': round(len(solved) / total_problematic * 100, 1) if total_problematic > 0 else 0,
        'total_problematic': int(total_problematic),
    }


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
with st.spinner("Chargement des données..."):
    df = load_data()
    df_with_prev = enrich_data(df)

# ---------------------------------------------------------------------------
# Sidebar — Contrôles
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://lever-client-logos.s3.amazonaws.com/2bd4cdf9-37f2-497f-9096-c2793296a75f-1568844229943.png",
    width="stretch",
)
st.sidebar.title("Paramètres de simulation")

threshold = st.sidebar.slider(
    "Threshold (délai minimum entre deux locations, en minutes)",
    min_value=0,
    max_value=720,
    value=60,
    step=15,
)

scope = st.sidebar.radio(
    "Scope de la feature",
    options=["Toutes les voitures", "Connect uniquement"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **À propos**

    Ce dashboard aide le Product Manager à décider :
    - **Quel threshold** (délai minimum) appliquer
    - **Quel scope** (toutes les voitures ou seulement Connect)

    pour minimiser les frictions liées aux retards tout en préservant les revenus.
    """
)

# ---------------------------------------------------------------------------
# Header principal
# ---------------------------------------------------------------------------
st.title("🚗 GetAround — Analyse des retards au checkout")
st.markdown(
    """
    Lorsqu'une voiture est réservée deux fois dans la même journée, le retard du premier conducteur
    peut impacter le second. Cette analyse aide à définir un **délai minimum entre deux locations**
    pour réduire ces frictions.
    """
)

# ---------------------------------------------------------------------------
# KPIs principaux
# ---------------------------------------------------------------------------
metrics = simulate_threshold(df, df_with_prev, threshold, scope)
ended = df[df['state'] == 'ended']
ended_with_delay = ended[ended['delay_at_checkout_in_minutes'].notna()]
late = ended_with_delay[ended_with_delay['delay_at_checkout_in_minutes'] > 0]

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Locations totales",
    f"{len(df):,}",
    help="Nombre total de locations dans le dataset"
)
col2.metric(
    "Taux de retard",
    f"{len(late)/len(ended_with_delay)*100:.1f}%",
    help="% de locations terminées avec un retard au checkout"
)
col3.metric(
    "Cas problématiques",
    f"{metrics['total_problematic']:,}",
    help="Locations où le retard précédent a impacté le prochain conducteur"
)
col4.metric(
    "Locations bloquées par le threshold",
    f"{metrics['blocked']:,} ({metrics['blocked_pct']}%)",
    help="Locations qui auraient été bloquées avec ce threshold"
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 1 — Vue d'ensemble des données
# ---------------------------------------------------------------------------
st.header("1. Vue d'ensemble des données")

tab1, tab2 = st.tabs(["Distribution des locations", "Types de check-in"])

with tab1:
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.pie(
            df,
            names='state',
            title="État des locations",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, width="stretch")

    with col_r:
        fig = px.pie(
            df,
            names='checkin_type',
            title="Type de check-in",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig, width="stretch")

with tab2:
    ct = pd.crosstab(df['checkin_type'], df['state'])
    ct['total'] = ct.sum(axis=1)
    ct['ended_pct'] = (ct.get('ended', 0) / ct['total'] * 100).round(1)
    ct['canceled_pct'] = (ct.get('canceled', 0) / ct['total'] * 100).round(1)
    st.dataframe(ct, width="stretch")

# ---------------------------------------------------------------------------
# Section 2 — Analyse des retards
# ---------------------------------------------------------------------------
st.header("2. Analyse des retards au checkout")

col_l, col_r = st.columns(2)

with col_l:
    fig = px.histogram(
        ended_with_delay,
        x='delay_at_checkout_in_minutes',
        color='checkin_type',
        nbins=100,
        title='Distribution des retards au checkout',
        labels={
            'delay_at_checkout_in_minutes': 'Retard (minutes)',
            'count': 'Nombre de locations',
        },
        barmode='overlay',
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_xaxes(range=[-300, 600])
    fig.add_vline(x=0, line_dash='dash', line_color='red', annotation_text="À l'heure")
    st.plotly_chart(fig, width="stretch")

with col_r:
    def _late_stats_by_checkin(g: pd.DataFrame) -> pd.Series:
        d = g["delay_at_checkout_in_minutes"]
        late = d > 0
        return pd.Series(
            {
                "Total": len(g),
                "En retard": int(late.sum()),
                "Taux retard (%)": round(late.mean() * 100, 1),
                "Retard moyen (min)": round(d[late].mean(), 0) if late.any() else 0.0,
            }
        )

    late_stats = ended_with_delay.groupby("checkin_type").apply(
        _late_stats_by_checkin,
        include_groups=False,
    ).reset_index()

    fig = px.bar(
        late_stats,
        x='checkin_type',
        y='Taux retard (%)',
        color='checkin_type',
        title='Taux de retard par type de check-in',
        labels={'checkin_type': 'Type de check-in'},
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, yaxis_ticksuffix='%')
    st.plotly_chart(fig, width="stretch")

st.dataframe(late_stats.set_index('checkin_type'), width="stretch")

# ---------------------------------------------------------------------------
# Section 3 — Simulation du threshold
# ---------------------------------------------------------------------------
st.header(f"3. Simulation — Threshold : {threshold} min | Scope : {scope}")

col1, col2 = st.columns(2)
col1.metric(
    "Locations bloquées",
    f"{metrics['blocked']:,}",
    f"{metrics['blocked_pct']}% de toutes les locations",
    delta_color="inverse",
)
col2.metric(
    "Cas problématiques résolus",
    f"{metrics['solved']:,}",
    f"{metrics['solve_rate']}% des cas problématiques",
)

# Courbes pour tous les thresholds
thresholds_range = list(range(0, 721, 15))
scopes_list = ["Toutes les voitures", "Connect uniquement"]

sim_results = []
for s in scopes_list:
    for t in thresholds_range:
        m = simulate_threshold(df, df_with_prev, t, s)
        sim_results.append({
            'threshold': t,
            'scope': s,
            'blocked_pct': m['blocked_pct'],
            'solve_rate': m['solve_rate'],
        })
sim_df = pd.DataFrame(sim_results)

col_l, col_r = st.columns(2)

with col_l:
    fig = px.line(
        sim_df,
        x='threshold',
        y='blocked_pct',
        color='scope',
        title='% de locations bloquées (impact revenus)',
        labels={
            'threshold': 'Threshold (minutes)',
            'blocked_pct': '% de locations bloquées',
        },
        markers=False,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_vline(
        x=threshold,
        line_dash='dash',
        line_color='red',
        annotation_text=f"Threshold actuel : {threshold} min",
    )
    fig.update_layout(yaxis_ticksuffix='%')
    st.plotly_chart(fig, width="stretch")

with col_r:
    fig = px.line(
        sim_df,
        x='threshold',
        y='solve_rate',
        color='scope',
        title='% de cas problématiques résolus',
        labels={
            'threshold': 'Threshold (minutes)',
            'solve_rate': '% de cas résolus',
        },
        markers=False,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_vline(
        x=threshold,
        line_dash='dash',
        line_color='red',
        annotation_text=f"Threshold actuel : {threshold} min",
    )
    fig.update_layout(yaxis_ticksuffix='%')
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Section 4 — Recommandation
# ---------------------------------------------------------------------------
st.header("4. Synthèse et recommandation")

summary_data = []
for s in scopes_list:
    for t in [30, 60, 120, 180, 240]:
        m = simulate_threshold(df, df_with_prev, t, s)
        summary_data.append({
            'Threshold (min)': t,
            'Scope': s,
            'Locations bloquées': m['blocked'],
            '% bloquées': f"{m['blocked_pct']}%",
            'Cas résolus': m['solved'],
            'Taux résolution': f"{m['solve_rate']}%",
        })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, width="stretch", hide_index=True)

st.info(
    """
    **Recommandation** : Un threshold de **60 à 120 minutes** avec le scope **Connect uniquement**
    offre le meilleur compromis :
    - Résout une part significative des cas problématiques
    - Limite l'impact sur les revenus des propriétaires
    - Cibler les voitures Connect est pertinent car leur checkout est entièrement digital
      (moins de flexibilité pour rattraper un retard)
    """
)

# ---------------------------------------------------------------------------
# Section 5 — Test du prix via l'API (module séparé : pricing)
# ---------------------------------------------------------------------------
st.header("5. Estimation de prix (API de pricing)")

st.markdown(
    """
    Le dashboard principal porte sur les **retards**. Le modèle de **prix optimal** est servi par
    une API FastAPI déployée séparément. Vous pouvez tester ici un appel `POST /predict` avec le
    format `input` (liste de lignes) décrit dans le README du projet.
    """
)

api_base = st.text_input(
    "URL de base de l'API (sans `/predict`, ex. Space Hugging Face)",
    value=os.environ.get("GETAROUND_API_URL", "https://VOTRE_USERNAME-getaround-api.hf.space"),
)

example_payload = {
    "input": [
        [
            "Citroën",
            "diesel",
            "black",
            "convertible",
            140411,
            100,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
        ]
    ]
}

if st.button("Envoyer une requête d'exemple"):
    url = f"{api_base.rstrip('/')}/predict"
    try:
        resp = requests.post(url, json=example_payload, timeout=60)
        st.write(f"**Statut HTTP :** {resp.status_code}")
        st.json(resp.json())
    except requests.RequestException as exc:
        st.error(f"Erreur réseau ou timeout : {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Dashboard GetAround — Jedha DSFS (Deployment) | Données : get_around_delay_analysis.xlsx")
