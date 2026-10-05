"""World Cup 2026 player clustering dashboard. Run: streamlit run app.py"""
import plotly.express as px
import streamlit as st

import clustering as cl

st.set_page_config(page_title="Who plays like whom?", page_icon="⚽", layout="wide")

COLORS = ["#ff7a63", "#4cc9f0", "#c5e84a", "#b388ff", "#ffb81c", "#3ddc84", "#f06292", "#90a4ae"]

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;800&display=swap');
.stApp{background:repeating-linear-gradient(90deg,#0d2418 0,#0d2418 90px,#0f2b1c 90px,#0f2b1c 180px);}
h1,h2,h3{font-family:'Barlow Condensed',Impact,sans-serif!important;letter-spacing:.5px;text-transform:uppercase}
[data-testid="stMetric"]{background:#12281c;border:1px solid #21402e;border-left:5px solid #3ddc84;border-radius:10px;padding:10px 14px}
[data-testid="stMetricValue"]{font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:2.4rem}
.stTabs [data-baseweb="tab"]{font-weight:600;border-radius:999px;padding:6px 18px}
.stTabs [aria-selected="true"]{background:#3ddc84;color:#0b1a12!important}
.stTabs [data-baseweb="tab-highlight"]{display:none}
section[data-testid="stSidebar"]{background:#0b1a12;border-right:3px solid #3ddc84}
.hero{border:2px solid rgba(255,255,255,.35);border-radius:14px;padding:22px 24px 8px;margin-bottom:18px;
 background:radial-gradient(circle at 85% 50%,rgba(255,255,255,.08) 0 60px,transparent 61px),linear-gradient(180deg,#1d6b3f,#14502f)}
.hero h1{font-size:clamp(38px,6vw,64px);line-height:.95;margin:0;color:#fff}
.hero p{color:#d6efe0;margin:6px 0 10px}
.lane{position:relative;height:38px;border-bottom:3px dashed rgba(255,255,255,.4)}
.roll{position:absolute;bottom:-2px;left:0;width:32px;height:32px;animation:run 7s ease-in-out infinite alternate}
.ball{width:100%;height:100%;border-radius:50%;border:1.5px solid #222;
 background:radial-gradient(circle at 50% 50%,#1a1a1a 0 22%,transparent 23%),radial-gradient(circle at 30% 30%,#fff,#d9d9d9);
 animation:spin 7s ease-in-out infinite alternate}
@keyframes run{to{left:calc(100% - 32px)}}@keyframes spin{to{transform:rotate(900deg)}}
@media(prefers-reduced-motion:reduce){.roll,.ball{animation:none}}
</style>
"""

HERO = """
<div class="hero"><h1>&#9917; Who plays like whom?</h1>
<p>K-Means groups every outfield player at World Cup 2026 by what they do on the pitch.</p>
<div class="lane"><div class="roll"><div class="ball"></div></div></div></div>
"""


def pitch_shapes():
    """Pitch stripes and markings drawn behind the scatter plot (paper coordinates)."""
    shapes = []
    for i in range(10):
        shapes.append(dict(type="rect", xref="paper", yref="paper", x0=i / 10, x1=(i + 1) / 10, y0=0, y1=1,
                           fillcolor="#217748" if i % 2 == 0 else "#1d6b3f", line_width=0, layer="below"))
    line = dict(color="rgba(255,255,255,.55)", width=2)
    shapes += [
        dict(type="rect", xref="paper", yref="paper", x0=0.01, x1=0.99, y0=0.02, y1=0.98, line=line, layer="below"),
        dict(type="line", xref="paper", yref="paper", x0=0.5, x1=0.5, y0=0.02, y1=0.98, line=line, layer="below"),
        dict(type="circle", xref="paper", yref="paper", x0=0.44, x1=0.56, y0=0.38, y1=0.62, line=line, layer="below"),
        dict(type="rect", xref="paper", yref="paper", x0=0.01, x1=0.14, y0=0.28, y1=0.72, line=line, layer="below"),
        dict(type="rect", xref="paper", yref="paper", x0=0.86, x1=0.99, y0=0.28, y1=0.72, line=line, layer="below"),
    ]
    return shapes


@st.cache_data
def get_data():
    df = cl.load_data()
    feats = cl.build_player_features(df)
    X = cl.scale(feats)
    return df, feats, X


@st.cache_data
def get_silhouette():
    _, feats, X = get_data()
    return cl.silhouette_by_k(X)


df, feats, X = get_data()

# ---- Sidebar controls (change these and the clustering re-runs live)
st.sidebar.header("⚽ Controls")
k = st.sidebar.slider("Number of groups (K)", 2, 8, 4)
teams = st.sidebar.multiselect("Filter by team", sorted(feats["team"].unique()))
st.sidebar.caption("Clusters are always computed on all players; the team filter only changes what is shown.")

clustered = cl.run_kmeans(feats, X, k)
shown = clustered[clustered["team"].isin(teams)] if teams else clustered

# ---- Header
st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Players", df["player_id"].nunique())
c2.metric("Matches", df["match_id"].nunique())
c3.metric("Teams", df["team"].nunique())
c4.metric("Match records", f"{len(df):,}")

tab1, tab2, tab3, tab4 = st.tabs(["Player map", "Group profiles", "Best K", "Explore players"])

with tab1:
    fig = px.scatter(
        shown, x="PC1", y="PC2", color="cluster", color_discrete_sequence=COLORS,
        hover_data=["player_name", "team", "position", "goals", "assists"],
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color="#0b1a12")))
    fig.update_layout(height=560, shapes=pitch_shapes(), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title_text="Group")
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Each dot is a player, projected to 2D with PCA.")

with tab2:
    sizes = clustered["cluster"].value_counts().rename("players")
    st.subheader("Group sizes")
    st.bar_chart(sizes)
    st.subheader("Average stats per match")
    st.dataframe(
        cl.cluster_profiles(clustered).join(sizes).style.background_gradient(cmap="Greens", axis=0),
        width="stretch",
    )

with tab3:
    sil = get_silhouette()
    best_k = int(sil.idxmax())
    st.subheader("Silhouette score by K")
    st.bar_chart(sil)
    st.success(f"Best K by silhouette score: {best_k} ({sil.max():.3f}). Current K: {k} ({sil[k]:.3f}).")

with tab4:
    cluster_pick = st.selectbox("Group", ["All"] + sorted(shown["cluster"].unique()))
    table = shown if cluster_pick == "All" else shown[shown["cluster"] == cluster_pick]
    cols = ["player_name", "team", "position", "cluster"] + cl.FEATURES
    st.dataframe(table[cols].round(2).sort_values("goals", ascending=False), width="stretch")
