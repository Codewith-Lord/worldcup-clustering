"""World Cup 2026 player clustering dashboard. Run: streamlit run app.py

Stack: Streamlit + Pandas + scikit-learn (K-Means, PCA) + Plotly.
All clustering / PCA maths lives in clustering.py; this file is presentation + interactivity.
"""
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import clustering as cl

ROOT = Path(__file__).parent
# ---- Project title: change these three lines to rename the whole dashboard
TITLE_MAIN, TITLE_ACCENT = "Football", "Mania"      # big hero heading (accent part is gradient-coloured)
BRAND_NAME, BRAND_SUB = "FootballMania", "Player clustering lab"   # sidebar logo text
st.set_page_config(page_title=f"{TITLE_MAIN} {TITLE_ACCENT} · World Cup 2026", page_icon="⚽", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- palette
# Distinct, high-contrast cluster colours (consistent: clusters are ordered
# defensive -> attacking and always take the colours in this order).
PALETTE = ["#22d3ee", "#ff8a3d", "#a3e635", "#f472b6", "#facc15", "#a78bfa", "#f43f5e", "#60a5fa"]
POS_COLORS = {"Defender": "#38bdf8", "Midfielder": "#34d399", "Forward": "#fbbf24"}
INK, MUTED, GRID = "#e8eef8", "#8fa3bf", "rgba(150,180,230,.10)"
FEAT_LABEL = {
    "goals": "Goals", "assists": "Assists", "shots": "Shots", "key_passes": "Key passes",
    "tackles": "Tackles", "interceptions": "Interceptions", "clearances": "Clearances",
    "successful_dribbles": "Dribbles", "pass_accuracy": "Pass acc.",
}


def clean(s: str) -> str:
    """Strip indentation / blank lines so Streamlit's markdown never turns HTML into a code block."""
    return "\n".join(l.strip() for l in s.splitlines() if l.strip())


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


@st.cache_resource
def stadium_b64() -> str:
    return base64.b64encode((ROOT / "assets" / "stadium_hero.jpg").read_bytes()).decode()


# --------------------------------------------------------------------------- CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
:root{--stadium:url(data:image/jpeg;base64,__IMG__);--bg:#050b14;--card:rgba(11,22,40,.66);--line:rgba(120,170,255,.16);
--green:#22c55e;--mint:#2ee6a6;--blue:#38bdf8;--ink:#e8eef8;--muted:#8fa3bf}
html,body,.stApp,[class*="css"]{font-family:'Inter',system-ui,sans-serif;color:var(--ink)}
.stApp{background:linear-gradient(180deg,rgba(5,11,20,.62) 0,rgba(5,11,20,.90) 520px,#050b14 860px),var(--stadium) top center/100% auto no-repeat,#050b14;background-attachment:scroll}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none!important}
.block-container{max-width:1720px;padding:1.1rem 1.6rem 3rem}
h1,h2,h3,h4{font-family:'Barlow Condensed',Impact,sans-serif!important;letter-spacing:.02em;color:#fff}
/* ---------- animated background (behind everything) */
[data-testid="stMain"]{position:relative;z-index:1}
.bgfx{position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden}
.bgfx .beam{position:absolute;top:-25%;width:34vw;height:150vh;transform-origin:50% 0;filter:blur(18px);opacity:1;
background:linear-gradient(180deg,rgba(150,215,255,.55),rgba(120,200,255,.16) 55%,transparent 80%);clip-path:polygon(46% 0,54% 0,100% 100%,0 100%);animation:sway 16s ease-in-out infinite alternate}
.bgfx .b1{left:2%;--a:-16deg;--b:10deg}.bgfx .b2{left:30%;--a:12deg;--b:-14deg;animation-duration:20s;animation-delay:-6s}
.bgfx .b3{left:58%;--a:-10deg;--b:16deg;animation-duration:18s;animation-delay:-3s}.bgfx .b4{left:80%;--a:14deg;--b:-12deg;animation-duration:22s;animation-delay:-9s}
@keyframes sway{from{transform:rotate(var(--a))}to{transform:rotate(var(--b))}}
.bgfx .dots{position:absolute;inset:-10% 0;background-repeat:repeat;animation:rise linear infinite}
.bgfx .d1{background-image:radial-gradient(circle,rgba(120,230,255,.9) 0 2px,transparent 2.6px),radial-gradient(circle,rgba(46,230,166,.8) 0 2px,transparent 2.6px);background-size:210px 210px,330px 330px;background-position:0 0,90px 140px;animation-duration:40s;opacity:.9}
.bgfx .d2{background-image:radial-gradient(circle,rgba(255,255,255,.5) 0 1px,transparent 1.6px);background-size:130px 130px;animation-duration:70s;opacity:.55}
@keyframes rise{from{transform:translateY(0)}to{transform:translateY(-330px)}}
.bgfx .ball{position:absolute;border-radius:50%;opacity:.28;filter:blur(1px);
background:radial-gradient(circle at 50% 50%,#101820 0 20%,transparent 21%),radial-gradient(circle at 32% 30%,#fff,#9fb4c8);animation:floaty 26s ease-in-out infinite alternate}
.bgfx .k1{width:90px;height:90px;left:6%;top:62%}.bgfx .k2{width:150px;height:150px;left:84%;top:18%;animation-duration:34s;opacity:.2}
.bgfx .k3{width:60px;height:60px;left:48%;top:80%;animation-duration:22s}.bgfx .k4{width:110px;height:110px;left:70%;top:68%;animation-duration:30s;opacity:.22}.bgfx .k5{width:70px;height:70px;left:20%;top:12%;animation-duration:28s}
@keyframes floaty{0%{transform:translate(0,0) rotate(0)}50%{transform:translate(40px,-70px) rotate(180deg)}100%{transform:translate(-30px,40px) rotate(360deg)}}
@media(prefers-reduced-motion:reduce){.bgfx *{animation:none!important}}
/* ---------- sidebar */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(6,13,24,.96),rgba(5,10,18,.98));border-right:1px solid var(--line);backdrop-filter:blur(18px)}
section[data-testid="stSidebar"][aria-expanded="true"]{width:300px!important;min-width:300px!important}
section[data-testid="stSidebar"] .block-container,section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:.6rem}
.brand{display:flex;align-items:center;gap:12px;padding:6px 2px 14px}
.brand .logo{width:44px;height:44px;border-radius:14px;display:grid;place-items:center;font-size:22px;background:linear-gradient(135deg,#16a34a,#0ea5e9);box-shadow:0 8px 24px rgba(14,165,233,.35)}
.brand b{display:block;font:800 22px/1 'Barlow Condensed',sans-serif;letter-spacing:.06em;color:#fff;text-transform:uppercase}
.brand span{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
.navgroup{font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:#5f7694;margin:14px 2px 6px;font-weight:700}
.nav a{display:flex;align-items:center;gap:11px;padding:10px 12px;margin:3px 0;border-radius:12px;color:#b8c7dc!important;text-decoration:none!important;font-weight:500;font-size:14px;border:1px solid transparent;transition:.18s}
.nav a svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.nav a:hover{background:rgba(56,189,248,.08);border-color:rgba(56,189,248,.22);color:#fff!important}
.nav a.on{background:linear-gradient(90deg,rgba(34,197,94,.22),rgba(14,165,233,.12));border-color:rgba(46,230,166,.35);color:#fff!important;box-shadow:inset 3px 0 0 var(--mint)}
.dstat{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:4px 0 10px}
.dstat div{background:rgba(255,255,255,.035);border:1px solid var(--line);border-radius:12px;padding:9px 11px}
.dstat em{display:block;font-style:normal;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.dstat b{font:700 20px/1.25 'Barlow Condensed',sans-serif;color:#fff}
.dstat b small{font:500 11px 'Inter';color:var(--muted)}
/* ---------- widgets */
[data-testid="stWidgetLabel"] p{color:#9db0ca!important;font-size:11px!important;letter-spacing:.1em;text-transform:uppercase;font-weight:600}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"]{background:rgba(255,255,255,.045)!important;border:1px solid rgba(140,180,255,.20)!important;border-radius:12px!important;color:var(--ink)}
[data-baseweb="select"]>div:hover,[data-baseweb="input"]:focus-within{border-color:rgba(46,230,166,.6)!important}
input{color:var(--ink)!important}
[data-baseweb="tag"]{background:rgba(34,197,94,.16)!important;border:1px solid rgba(46,230,166,.45);border-radius:8px!important;color:#d9ffe9!important}
div[data-baseweb="popover"]>div,ul[role="listbox"]{background:#0a1526!important;border:1px solid var(--line);border-radius:12px!important}
li[role="option"]:hover,li[aria-selected="true"]{background:rgba(56,189,248,.14)!important}
[data-testid="stSlider"] [role="slider"]{background:var(--mint)!important;box-shadow:0 0 0 5px rgba(46,230,166,.22);border:2px solid #04210f}
[data-testid="stSlider"] [data-testid="stTickBarMin"],[data-testid="stSlider"] [data-testid="stTickBarMax"]{color:var(--muted)}
[data-testid="stSliderThumbValue"]{color:#fff!important;font-weight:700}
.stButton>button,.stDownloadButton>button{width:100%;background:linear-gradient(135deg,#16a34a,#0ea5e9);color:#fff;border:0;border-radius:12px;font-weight:600;padding:.5rem .9rem;box-shadow:0 8px 22px rgba(14,165,233,.22);transition:.18s}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);box-shadow:0 10px 28px rgba(46,230,166,.35);color:#fff;border:0}
.stButton>button p,.stDownloadButton>button p{color:#fff}
.st-key-reset button{background:transparent!important;border:1px solid rgba(140,180,255,.3)!important;box-shadow:none!important}
.st-key-reset button:hover{border-color:var(--mint)!important}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:rgba(255,255,255,.035);padding:5px;border-radius:999px;border:1px solid var(--line);width:fit-content}
.stTabs [data-baseweb="tab"]{height:36px;border-radius:999px;padding:0 20px!important;color:#b8c7dc;font-weight:600}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#16a34a,#0ea5e9);color:#fff!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}
[data-testid="stAlert"]{background:rgba(250,204,21,.08);border:1px solid rgba(250,204,21,.3);border-radius:14px}
/* ---------- glass cards */
[class*="st-key-card-"]{background:var(--card);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid var(--line);border-radius:22px;padding:1.05rem 1.15rem 1rem;box-shadow:0 24px 60px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,255,255,.05)}
.ctitle{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin:0 0 6px}
.ctitle h3{margin:0;font-size:24px;text-transform:uppercase;line-height:1}
.ctitle small{color:var(--muted);font-size:12px}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600;background:rgba(46,230,166,.12);border:1px solid rgba(46,230,166,.4);color:#b9ffe0}
.sect{margin:30px 0 12px}
.sect h2{margin:0;font-size:34px;text-transform:uppercase}
.sect p{margin:2px 0 0;color:var(--muted);font-size:13.5px}
.sect:before{content:"";display:block;width:54px;height:4px;border-radius:4px;background:linear-gradient(90deg,var(--mint),var(--blue));margin-bottom:10px}
/* ---------- hero */
.hero{position:relative;overflow:hidden;border-radius:24px;padding:30px 34px 28px;min-height:218px;border:1px solid rgba(140,190,255,.22);
background:linear-gradient(90deg,rgba(3,8,16,.88) 0,rgba(3,8,16,.55) 48%,rgba(3,8,16,.15) 100%),var(--stadium) center 58%/cover;box-shadow:0 24px 70px rgba(0,0,0,.5)}
.hero:after{content:"";position:absolute;inset:0;background:radial-gradient(circle at 82% 20%,rgba(255,255,255,.16),transparent 36%);pointer-events:none}
.kicker{display:inline-flex;align-items:center;gap:9px;font-size:11.5px;letter-spacing:.22em;text-transform:uppercase;color:#bfe9ff;font-weight:600}
.kicker i{width:8px;height:8px;border-radius:50%;background:var(--mint);box-shadow:0 0 0 4px rgba(46,230,166,.25),0 0 14px var(--mint)}
.hero h1{font-size:clamp(40px,4.6vw,66px);line-height:.95;margin:12px 0 10px;text-transform:uppercase;text-shadow:0 4px 30px rgba(0,0,0,.6)}
.hero h1 span{background:linear-gradient(90deg,#2ee6a6,#38bdf8);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{max-width:560px;color:#c9d8ec;font-size:14.5px;margin:0 0 14px;line-height:1.5}
.chips span{display:inline-block;margin:0 8px 6px 0;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:600;color:#e8f6ff;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.18);backdrop-filter:blur(8px)}
/* ---------- KPI */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:14px 0 14px}
.kpi{display:flex;align-items:center;gap:14px;background:var(--card);backdrop-filter:blur(16px);border:1px solid var(--line);border-radius:20px;padding:15px 16px;box-shadow:0 14px 40px rgba(0,0,0,.32)}
.kpi .ic{flex:0 0 46px;height:46px;border-radius:14px;display:grid;place-items:center;background:linear-gradient(135deg,rgba(34,197,94,.25),rgba(14,165,233,.22));border:1px solid rgba(46,230,166,.35)}
.kpi .ic svg{width:23px;height:23px;stroke:#bdfbe0;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.kpi em{display:block;font-style:normal;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
.kpi b{display:block;font:700 34px/1.05 'Barlow Condensed',sans-serif;color:#fff}
.kpi small{color:#7f94b1;font-size:11.5px}
/* ---------- cluster cards */
.cscroll{max-height:480px;overflow-y:auto;padding-right:6px;margin-top:6px}
.cscroll::-webkit-scrollbar{width:6px}.cscroll::-webkit-scrollbar-thumb{background:rgba(140,180,255,.25);border-radius:6px}
.ccard{position:relative;margin:0 0 10px;padding:11px 13px 11px 17px;border-radius:16px;background:linear-gradient(135deg,color-mix(in srgb,var(--c) 14%,rgba(10,20,36,.9)),rgba(10,20,36,.72));border:1px solid color-mix(in srgb,var(--c) 38%,transparent);overflow:hidden}
.ccard:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--c);box-shadow:0 0 18px var(--c)}
.ccard.dim{opacity:.4}
.ctop{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.ctop b{font:700 19px/1 'Barlow Condensed',sans-serif;letter-spacing:.03em;text-transform:uppercase;color:#fff}
.ctop span{font:700 20px/1 'Barlow Condensed',sans-serif;color:var(--c)}
.ctop span small{font:500 10.5px 'Inter';color:var(--muted);margin-left:4px}
.cbar{height:5px;border-radius:5px;background:rgba(255,255,255,.07);margin:8px 0 9px}
.cbar i{display:block;height:100%;border-radius:5px;background:var(--c)}
.cstats{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}
.cstats div{text-align:center}
.cstats em{display:block;font-style:normal;font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.cstats b{font:700 15px 'Barlow Condensed',sans-serif;color:#fff}
/* ---------- selected player */
.pl-head{display:flex;gap:12px;align-items:center;margin:4px 0 10px}
.avatar{flex:0 0 72px;height:72px;overflow:hidden;border-radius:50%;display:grid;place-items:center;font:700 20px 'Barlow Condensed',sans-serif;color:#04121f;background:var(--c);box-shadow:0 0 0 4px color-mix(in srgb,var(--c) 30%,transparent),0 0 24px color-mix(in srgb,var(--c) 55%,transparent)}
.pl-head b{display:block;font:700 22px/1.1 'Barlow Condensed',sans-serif;color:#fff;text-transform:uppercase}
.pl-head small{color:var(--muted);font-size:12px}
.badge{display:inline-block;margin-top:3px;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700;color:#06121f;background:var(--c)}
.pl-meta{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px}
.pl-meta div{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:10px;padding:6px 8px}
.pl-meta em{display:block;font-style:normal;font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.pl-meta b{font-size:12px;color:#fff;font-weight:600}
.pl-stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
.pl-stats div{padding:6px 8px;border-radius:10px;background:rgba(255,255,255,.035)}
.pl-stats em{display:block;font-style:normal;font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.pl-stats b{font:700 17px 'Barlow Condensed',sans-serif;color:#fff}
.pl-stats i{font-style:normal;font-size:10px;margin-left:4px}
.up{color:#4ade80}.down{color:#fb7185}
.empty{padding:18px 10px;text-align:center;color:var(--muted);font-size:13px;border:1px dashed rgba(140,180,255,.25);border-radius:14px;margin-top:6px}
/* ---------- plotly text halo so centroid labels stay readable on dots */
.js-plotly-plot .scatterlayer .textpoint text{paint-order:stroke;stroke:#050b14;stroke-width:4.5px;stroke-linejoin:round}
.js-plotly-plot .plotly .modebar{background:transparent!important}
@media(max-width:1100px){.kpis{grid-template-columns:repeat(2,1fr)}}
</style>
"""

ICON = {
    "players": '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/></svg>',
    "teams": '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></svg>',
    "matches": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7l4 3-1.5 5h-5L8 10z"/></svg>',
    "clusters": '<svg viewBox="0 0 24 24"><circle cx="7" cy="8" r="3"/><circle cx="17" cy="8" r="3"/><circle cx="12" cy="17" r="3"/></svg>',
    "dash": '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="9" rx="2"/><rect x="14" y="3" width="7" height="5" rx="2"/><rect x="14" y="12" width="7" height="9" rx="2"/><rect x="3" y="16" width="7" height="5" rx="2"/></svg>',
    "radar": '<svg viewBox="0 0 24 24"><path d="M12 3l8 6-3 10H7L4 9z"/><path d="M12 8l4 3-1.5 5h-5L8 11z"/></svg>',
    "sil": '<svg viewBox="0 0 24 24"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>',
    "table": '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/></svg>',
}


# --------------------------------------------------------------------------- data (cached)
@st.cache_data
def get_data():
    df = cl.load_data(str(ROOT / cl.DATA_PATH))
    feats = cl.build_player_features(df)
    X = cl.scale(feats)
    meta = df.drop_duplicates("player_id").set_index("player_id")[
        ["age", "nationality", "club_name", "jersey_number", "market_value_eur"]]
    meta["appearances"] = df.groupby("player_id").size()
    return df, feats, X, meta


@st.cache_data
def get_clustered(k: int):
    _, feats, X, _ = get_data()
    return cl.run_kmeans(feats, X, k)


@st.cache_data
def get_silhouette():
    _, _, X, _ = get_data()
    return cl.silhouette_by_k(X)


@st.cache_data
def get_pca_var():
    _, _, X, _ = get_data()
    return cl.pca_variance(X)


df, feats, X, meta = get_data()
sil = get_silhouette()
best_k = int(sil.idxmax())
pvar = get_pca_var()

st.markdown(CSS.replace("__IMG__", stadium_b64()), unsafe_allow_html=True)
st.markdown(clean('''<div class="bgfx"><div class="beam b1"></div><div class="beam b2"></div><div class="beam b3"></div><div class="beam b4"></div>
<div class="dots d1"></div><div class="dots d2"></div><div class="ball k1"></div><div class="ball k2"></div><div class="ball k3"></div><div class="ball k4"></div><div class="ball k5"></div></div>'''), unsafe_allow_html=True)

# --------------------------------------------------------------------------- sidebar
if "k" not in st.session_state:
    st.session_state["k"] = 4


def _use_best_k():
    st.session_state["k"] = best_k


def _reset_filters():
    for key in list(st.session_state.keys()):
        if key.startswith(("f_team", "f_pos", "f_cl_")):
            del st.session_state[key]


with st.sidebar:
    st.markdown(clean(f"""
    <div class="brand"><div class="logo">⚽</div><div><b>{BRAND_NAME}</b><span>{BRAND_SUB}</span></div></div>
    <div class="navgroup">Navigation</div>
    <div class="nav">
      <a class="on" href="#top">{ICON['dash']} Dashboard</a>
      <a href="#clusters">{ICON['radar']} Cluster analysis</a>
      <a href="#silhouette">{ICON['sil']} Silhouette · best K</a>
      <a href="#players">{ICON['table']} Player explorer</a>
    </div>
    <div class="navgroup">Clustering</div>"""), unsafe_allow_html=True)
    k = st.slider("Number of clusters (K)", 2, 8, key="k")
    st.button(f"✦ Use best K  ({best_k} · silhouette {sil.max():.3f})", on_click=_use_best_k, key="bestk")

    clustered = get_clustered(k).join(meta)
    ordered = list(clustered.groupby("cluster")["shots"].mean().sort_values().index)
    CMAP = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(ordered)}

    st.markdown('<div class="navgroup">Filters</div>', unsafe_allow_html=True)
    teams = st.multiselect("Team", sorted(clustered["team"].unique()), key="f_team", placeholder="All teams")
    poss = st.multiselect("Position", ["Defender", "Midfielder", "Forward"], key="f_pos", placeholder="All positions")
    cls_sel = st.multiselect("Cluster", ordered, key=f"f_cl_{k}", placeholder="All clusters")
    st.caption("Clusters are always computed on every outfield player; filters only change what is shown.")

shown = clustered
if teams:
    shown = shown[shown["team"].isin(teams)]
if poss:
    shown = shown[shown["position"].isin(poss)]
if cls_sel:
    shown = shown[shown["cluster"].isin(cls_sel)]

with st.sidebar:
    st.markdown('<div class="navgroup">Dataset</div>', unsafe_allow_html=True)
    st.markdown(clean(f"""
    <div class="dstat">
      <div><em>Outfield players</em><b>{len(clustered):,}</b></div>
      <div><em>Showing</em><b>{len(shown):,}</b></div>
      <div><em>Match records</em><b>{len(df):,}</b></div>
      <div><em>Features</em><b>{len(cl.FEATURES)}</b></div>
      <div><em>PCA variance</em><b>{(pvar[0] + pvar[1]) * 100:.1f}<small>%</small></b></div>
      <div><em>Silhouette @K</em><b>{sil[k]:.3f}</b></div>
    </div>"""), unsafe_allow_html=True)
    st.button("↺ Reset filters", on_click=_reset_filters, key="reset")
    st.download_button("⬇ Export clustered CSV", clustered.reset_index().to_csv(index=False).encode(),
                       file_name=f"worldcup2026_clusters_k{k}.csv", mime="text/csv")


# --------------------------------------------------------------------------- plotly helpers
def base_layout(**kw):
    lay = dict(
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#0a1526", bordercolor="rgba(140,180,255,.4)", font=dict(color="#fff", size=12)),
        margin=dict(l=10, r=10, t=10, b=10),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="rgba(200,215,240,.55)", activecolor="#2ee6a6"),
    )
    lay.update(kw)
    return lay


def style_axes(fig, **kw):
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="rgba(150,180,230,.22)", linecolor="rgba(150,180,230,.2)",
                     tickfont=dict(color=MUTED), automargin=True, title_standoff=10, **kw)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(150,180,230,.22)", linecolor="rgba(150,180,230,.2)",
                     tickfont=dict(color=MUTED), automargin=True, title_standoff=10)


PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"], "scrollZoom": True}


def title_html(title, sub="", pill=""):
    p = f'<span class="pill">{pill}</span>' if pill else ""
    return f'<div class="ctitle"><div><h3>{title}</h3><small>{sub}</small></div>{p}</div>'


def hero_and_kpis(n_players, n_teams, n_matches):
    sub_p = f"of {len(clustered):,} outfield players" if n_players != len(clustered) else "outfield players clustered"
    st.markdown(clean(f"""
    <div class="hero">
      <div class="kicker"><i></i>FIFA World Cup 2026 · Player analytics</div>
      <h1>{TITLE_MAIN} <span>{TITLE_ACCENT}</span></h1>
      <p>K-Means groups every outfield player by what they actually do on the pitch, and PCA flattens nine performance
      features into one map — so players who play alike sit together.</p>
      <div class="chips"><span>K = {k} clusters</span><span>{len(cl.FEATURES)} features</span>
      <span>PCA {(pvar[0] + pvar[1]) * 100:.1f}% variance</span><span>Silhouette {sil[k]:.3f}</span></div>
    </div>
    <div class="kpis">
      <div class="kpi"><div class="ic">{ICON['players']}</div><div><em>Players</em><b>{n_players:,}</b><small>{sub_p}</small></div></div>
      <div class="kpi"><div class="ic">{ICON['teams']}</div><div><em>Teams</em><b>{n_teams}</b><small>nations represented</small></div></div>
      <div class="kpi"><div class="ic">{ICON['matches']}</div><div><em>Matches</em><b>{n_matches:,}</b><small>{len(df):,} records</small></div></div>
      <div class="kpi"><div class="ic">{ICON['clusters']}</div><div><em>Clusters</em><b>{k}</b><small>K-Means · standardised</small></div></div>
    </div>"""), unsafe_allow_html=True)


st.markdown('<div id="top"></div>', unsafe_allow_html=True)

if shown.empty:
    hero_and_kpis(0, 0, 0)
    st.warning("No players match the current filters. Use “Reset filters” in the sidebar.")
    st.stop()

n_matches = df.loc[df["player_id"].isin(shown.index), "match_id"].nunique()
centroids = cl.cluster_centroids(clustered)

# --------------------------------------------------------------------------- selection state
st.session_state.setdefault("sel_pid", None)
st.session_state.setdefault("last_ev", None)

left, right = st.columns([3.15, 1.25], gap="medium")

# =========================================================================== CENTRE
with left:
    hero_and_kpis(len(shown), shown["team"].nunique(), n_matches)

    with st.container(key="card-scatter"):
        st.markdown(title_html(
            "PCA Cluster Map",
            f"PC1 {pvar[0] * 100:.1f}% · PC2 {pvar[1] * 100:.1f}% of variance &nbsp;·&nbsp; click a player · scroll to zoom · click a legend item to hide a cluster",
            f"{len(shown):,} players"), unsafe_allow_html=True)

        fig = go.Figure()
        present = [n for n in ordered if (shown["cluster"] == n).any()]
        trace_ids = []
        cols = ["player_name", "team", "position", "goals", "assists", "shots", "tackles"]
        for n in present:
            d = shown[shown["cluster"] == n]
            trace_ids.append(list(d.index))
            fig.add_trace(go.Scatter(
                x=d["PC1"], y=d["PC2"], mode="markers", name=f"{n} · {len(d)}", legendgroup=n,
                marker=dict(size=9, color=CMAP[n], opacity=0.88, line=dict(width=0.8, color="#050b14")),
                selected=dict(marker=dict(opacity=1, size=9)), unselected=dict(marker=dict(opacity=0.88)),
                customdata=d[cols].values.tolist(),
                hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>"
                               "Goals %{customdata[3]:.2f} · Assists %{customdata[4]:.2f}<br>"
                               "Shots %{customdata[5]:.2f} · Tackles %{customdata[6]:.2f}"
                               f"<extra>{n}</extra>"),
                hoverlabel=dict(bgcolor="#0a1526", bordercolor=CMAP[n], font=dict(color="#fff", size=12)),
            ))
        c = centroids.loc[present]
        fig.add_trace(go.Scatter(
            x=c["PC1"], y=c["PC2"], mode="markers+text", name="Cluster centroid", text=list(c.index),
            textposition="top center", textfont=dict(color="#ffffff", size=14, family="Barlow Condensed, Inter, sans-serif"),
            marker=dict(symbol="diamond", size=19, color=[CMAP[n] for n in present], line=dict(width=2.6, color="#ffffff")),
            customdata=c["players"], hoverinfo="text",
            hovertext=[f"<b>{n}</b> centroid<br>{int(centroids.loc[n, 'players'])} players" for n in present],
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
        ))
        sp = st.session_state["sel_pid"]
        if sp in shown.index:
            r = shown.loc[sp]
            fig.add_trace(go.Scatter(
                x=[r["PC1"]], y=[r["PC2"]], mode="markers+text", text=[r["player_name"]], textposition="bottom center",
                textfont=dict(color="#fff", size=12), showlegend=False, hoverinfo="skip",
                marker=dict(symbol="circle-open", size=24, color="#ffffff", line=dict(width=3, color="#ffffff")),
            ))
        fig.update_layout(**base_layout(
            height=650, margin=dict(l=8, r=8, t=6, b=6), clickmode="event+select", dragmode="pan",
            legend=dict(orientation="h", yanchor="top", y=-0.09, xanchor="center", x=0.5, itemsizing="constant",
                        font=dict(size=13, color="#fff"), bgcolor="rgba(255,255,255,.04)", bordercolor="rgba(140,180,255,.2)",
                        borderwidth=1, itemclick="toggle", itemdoubleclick="toggleothers"),
        ))
        style_axes(fig)
        fig.update_xaxes(title=dict(text=f"PC1 · {pvar[0] * 100:.1f}% variance — overall activity & attacking output", font=dict(color=MUTED, size=12)))
        fig.update_yaxes(title=dict(text=f"PC2 · {pvar[1] * 100:.1f}% variance", font=dict(color=MUTED, size=12)))
        ev = st.plotly_chart(fig, theme=None, key=f"map_{k}", on_select="rerun", selection_mode="points",
                             config=PLOT_CFG, width="stretch")

    pts = (ev.selection.points if ev and ev.selection else []) if ev is not None else []
    hit = None
    for p in pts:
        cn, pn = p.get("curve_number"), p.get("point_index", p.get("point_number"))
        if cn is not None and pn is not None and cn < len(trace_ids):
            hit = trace_ids[cn][pn]
            break
    if hit is not None:
        if hit != st.session_state["last_ev"]:
            st.session_state["last_ev"] = hit
            st.session_state["sel_pid"] = hit
            st.rerun()
    else:
        st.session_state["last_ev"] = None

# =========================================================================== RIGHT
with right:
    with st.container(key="card-clusters"):
        st.markdown(title_html("Cluster Summary", "avg. per match", f"K = {k}"), unsafe_allow_html=True)
        prof = cl.cluster_profiles(clustered)
        cards = []
        for n in ordered:
            sz = int(centroids.loc[n, "players"])
            shown_n = int((shown["cluster"] == n).sum())
            p = prof.loc[n]
            dim = " dim" if shown_n == 0 else ""
            cards.append(f"""<div class="ccard{dim}" style="--c:{CMAP[n]}">
            <div class="ctop"><b>{n}</b><span>{sz}<small>players · {sz / len(clustered) * 100:.0f}%</small></span></div>
            <div class="cbar"><i style="width:{sz / len(clustered) * 100:.1f}%"></i></div>
            <div class="cstats"><div><em>Goals</em><b>{p['goals']:.2f}</b></div><div><em>Ast</em><b>{p['assists']:.2f}</b></div>
            <div><em>Shots</em><b>{p['shots']:.2f}</b></div><div><em>Tckl</em><b>{p['tackles']:.2f}</b></div>
            <div><em>Pass%</em><b>{p['pass_accuracy'] * 100:.0f}</b></div></div></div>""")
        st.markdown(clean('<div class="cscroll">' + "".join(cards) + "</div>"), unsafe_allow_html=True)

    with st.container(key="card-compare"):
        st.markdown(title_html("Quick Compare", "one stat, every cluster"), unsafe_allow_html=True)
        stat = st.selectbox("Stat", cl.FEATURES, index=2, format_func=lambda f: FEAT_LABEL[f], label_visibility="collapsed")
        vals = prof.loc[ordered, stat]
        scale_ = 100 if stat == "pass_accuracy" else 1
        fmt = "{:.0f}%" if stat == "pass_accuracy" else "{:.2f}"
        f2 = go.Figure(go.Bar(
            y=ordered, x=vals * scale_, orientation="h", marker=dict(color=[CMAP[n] for n in ordered], line=dict(width=0)),
            text=[fmt.format(v * scale_) for v in vals], textposition="outside", textfont=dict(color="#fff", size=11),
            hovertemplate="<b>%{y}</b><br>" + FEAT_LABEL[stat] + " %{text}<extra></extra>", cliponaxis=False))
        f2.update_layout(**base_layout(height=48 + 30 * len(ordered), margin=dict(l=4, r=44, t=4, b=4), showlegend=False, bargap=.35))
        style_axes(f2)
        f2.update_xaxes(visible=False)
        f2.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color="#dbe6f5", size=11))
        st.plotly_chart(f2, theme=None, config={"displayModeBar": False}, width="stretch", key="compare")

    with st.container(key="card-player"):
        st.markdown(title_html("Player Details", "click a dot on the map, or search"), unsafe_allow_html=True)

        def _pick():
            v = st.session_state.get("pl_search")
            if v is not None:
                st.session_state["sel_pid"] = v

        st.selectbox("Search player", list(shown.sort_values("player_name").index), index=None, key="pl_search",
                     format_func=lambda pid: f"{shown.loc[pid, 'player_name']} · {shown.loc[pid, 'team']}",
                     placeholder="Search a player…", on_change=_pick, label_visibility="collapsed")
        pid = st.session_state["sel_pid"]
        if pid is None or pid not in clustered.index:
            st.markdown('<div class="empty">No player selected.<br>Click any dot on the cluster map<br>to see their profile here.</div>',
                        unsafe_allow_html=True)
        else:
            r = clustered.loc[pid]
            col = CMAP[r["cluster"]]
            ini = "".join(w[0] for w in r["player_name"].split()[:2]).upper()
            mv = r["market_value_eur"] / 1e6
            cavg = prof.loc[r["cluster"]]
            cells = ""
            for f in ["goals", "assists", "shots", "key_passes", "tackles", "pass_accuracy"]:
                v, a = r[f], cavg[f]
                pv = f"{v * 100:.0f}%" if f == "pass_accuracy" else f"{v:.2f}"
                diff = v - a
                cls_, arr = ("up", "▲") if diff >= 0 else ("down", "▼")
                cells += f'<div><em>{FEAT_LABEL[f]}</em><b>{pv}</b><i class="{cls_}">{arr}</i></div>'
            st.markdown(clean(f"""
            <div class="pl-head" style="--c:{col}"><div class="avatar">{ini}</div>
              <div><b>{r['player_name']}</b><small>{r['team']} · {r['position']} · #{int(r['jersey_number'])}</small><br>
              <span class="badge">{r['cluster']}</span></div></div>
            <div class="pl-meta"><div><em>Age</em><b>{int(r['age'])}</b></div><div><em>Club</em><b>{r['club_name']}</b></div>
              <div><em>Value</em><b>€{mv:.1f}M</b></div></div>
            <div class="pl-stats">{cells}</div>
            <small style="color:#7f94b1;font-size:10.5px">▲▼ vs. cluster average · {int(r['appearances'])} appearances</small>"""),
                unsafe_allow_html=True)
            pct = clustered[cl.FEATURES].rank(pct=True) * 100
            theta = [FEAT_LABEL[f] for f in cl.FEATURES]
            mine = pct.loc[pid].tolist()
            avg = pct[clustered["cluster"] == r["cluster"]].mean().tolist()
            f3 = go.Figure()
            f3.add_trace(go.Scatterpolar(r=avg + avg[:1], theta=theta + theta[:1], name="Cluster avg",
                                         line=dict(color="rgba(255,255,255,.55)", dash="dot", width=1.5), hoverinfo="skip"))
            f3.add_trace(go.Scatterpolar(r=mine + mine[:1], theta=theta + theta[:1], name="Player", fill="toself",
                                         fillcolor=rgba(col, .3), line=dict(color=col, width=2.5),
                                         hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>"))
            f3.update_layout(**base_layout(
                height=250, margin=dict(l=34, r=34, t=14, b=14), showlegend=False,
                polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor=GRID, linecolor=GRID),
                           angularaxis=dict(tickfont=dict(size=9.5, color="#b8c7dc"), gridcolor=GRID, linecolor=GRID))))
            st.plotly_chart(f3, theme=None, config={"displayModeBar": False}, width="stretch", key="player_radar")
            st.caption("Percentile among all outfield players · dotted line = cluster average")

# =========================================================================== BELOW
st.markdown('<div id="clusters"></div>', unsafe_allow_html=True)
st.markdown('<div class="sect"><h2>Cluster comparison</h2><p>How the groups differ across the nine features used for clustering.</p></div>',
            unsafe_allow_html=True)

with st.container(key="card-comparison"):
    t1, t2, t3 = st.tabs(["Radar profiles", "Z-score heatmap", "Position mix"])
    with t1:
        mat = prof.loc[ordered, cl.FEATURES]
        norm = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, 1)
        theta = [FEAT_LABEL[f] for f in cl.FEATURES]
        fr = go.Figure()
        for n in ordered:
            rr = norm.loc[n].tolist()
            fr.add_trace(go.Scatterpolar(
                r=rr + rr[:1], theta=theta + theta[:1], name=n, fill="toself", fillcolor=rgba(CMAP[n], .16),
                line=dict(color=CMAP[n], width=2.6), customdata=mat.loc[n].round(3).tolist() + [mat.loc[n].iloc[0].round(3)],
                hovertemplate="<b>" + n + "</b><br>%{theta}: %{customdata:.2f} per match<extra></extra>"))
        fr.update_layout(**base_layout(
            height=520, margin=dict(l=60, r=60, t=20, b=20),
            legend=dict(font=dict(size=13, color="#fff"), bgcolor="rgba(255,255,255,.04)", bordercolor="rgba(140,180,255,.2)", borderwidth=1),
            polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1.05], showticklabels=False, gridcolor=GRID, linecolor=GRID),
                       angularaxis=dict(tickfont=dict(size=12, color="#dbe6f5"), gridcolor=GRID, linecolor=GRID))))
        st.plotly_chart(fr, theme=None, config={"displayModeBar": False}, width="stretch", key="radar_all")
        st.caption("Each axis is scaled 0–1 across clusters so shapes are comparable; hover for the real per-match averages.")
    with t2:
        z = (mat - feats[cl.FEATURES].mean()) / feats[cl.FEATURES].std()
        fh = go.Figure(go.Heatmap(
            z=z.values, x=[FEAT_LABEL[f] for f in cl.FEATURES], y=ordered, zmid=0, zmin=-2.5, zmax=2.5,
            colorscale=[[0, "#1d4ed8"], [.5, "#0b1626"], [1, "#22c55e"]], xgap=4, ygap=4,
            text=np.round(z.values, 2), texttemplate="%{text}", textfont=dict(color="#fff", size=13),
            hovertemplate="<b>%{y}</b> · %{x}<br>z = %{z:.2f}<extra></extra>",
            colorbar=dict(title=dict(text="z", font=dict(color=MUTED)), tickfont=dict(color=MUTED), thickness=12, outlinewidth=0)))
        fh.update_layout(**base_layout(height=110 + 58 * len(ordered), margin=dict(l=10, r=10, t=10, b=10)))
        fh.update_yaxes(autorange="reversed", tickfont=dict(color="#fff", size=13))
        fh.update_xaxes(side="top", tickfont=dict(color="#dbe6f5", size=12))
        st.plotly_chart(fh, theme=None, config={"displayModeBar": False}, width="stretch", key="heat")
        st.caption("Cluster mean vs. all outfield players, in standard deviations. Green = above average, blue = below.")
    with t3:
        mix = pd.crosstab(clustered["cluster"], clustered["position"]).reindex(ordered)
        fm = go.Figure()
        for pos in ["Defender", "Midfielder", "Forward"]:
            if pos in mix:
                fm.add_trace(go.Bar(y=mix.index, x=mix[pos], name=pos, orientation="h", marker=dict(color=POS_COLORS[pos]),
                                    hovertemplate="<b>%{y}</b><br>" + pos + "s: %{x}<extra></extra>"))
        fm.update_layout(**base_layout(
            height=120 + 52 * len(ordered), barmode="stack", margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.08, font=dict(color="#fff", size=12))))
        style_axes(fm)
        fm.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color="#fff", size=13))
        st.plotly_chart(fm, theme=None, config={"displayModeBar": False}, width="stretch", key="mix")
        st.caption("Number of players of each listed position inside every cluster.")

st.markdown('<div id="silhouette"></div>', unsafe_allow_html=True)
st.markdown('<div class="sect"><h2>Silhouette analysis</h2><p>How well-separated the clusters are for each K (higher is better, max 1.0).</p></div>',
            unsafe_allow_html=True)
s1, s2 = st.columns([2.3, 1], gap="medium")
with s1:
    with st.container(key="card-sil"):
        ks = [int(i) for i in sil.index]
        fs = go.Figure()
        for label, color, pick in [("Best K", "#22c55e", lambda q: q == best_k),
                                   ("Selected K", "#38bdf8", lambda q: q == k and q != best_k),
                                   ("Other K", "#33496b", lambda q: q != best_k and q != k)]:
            ys = [float(sil[q]) if pick(q) else None for q in ks]
            if any(y is not None for y in ys):
                fs.add_trace(go.Bar(x=ks, y=ys, name=label, marker=dict(color=color, line=dict(width=0)),
                                    text=[f"{y:.3f}" if y is not None else "" for y in ys], textposition="outside",
                                    textfont=dict(color="#fff", size=12), hovertemplate="K = %{x}<br>silhouette %{y:.3f}<extra></extra>"))
        fs.add_trace(go.Scatter(x=ks, y=sil.values, mode="lines+markers", showlegend=False, hoverinfo="skip",
                                line=dict(color="rgba(255,255,255,.5)", dash="dot", width=1.6), marker=dict(size=6, color="#fff")))
        fs.update_layout(**base_layout(
            height=400, barmode="overlay", bargap=.35, margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.1, x=0, font=dict(color="#fff", size=12))))
        style_axes(fs)
        fs.update_xaxes(title=dict(text="Number of clusters (K)", font=dict(color=MUTED)), tickvals=ks, showgrid=False)
        fs.update_yaxes(title=dict(text="Silhouette score", font=dict(color=MUTED)), range=[0, float(sil.max()) * 1.22])
        st.plotly_chart(fs, theme=None, config={"displayModeBar": False}, width="stretch", key="sil_chart")
with s2:
    with st.container(key="card-silinfo"):
        st.markdown(title_html("Reading the score", "silhouette interpretation"), unsafe_allow_html=True)
        verdict = "strong" if sil[k] > .5 else "reasonable" if sil[k] > .25 else "weak"
        st.markdown(clean(f"""
        <div class="dstat">
          <div><em>Best K</em><b style="color:#4ade80">{best_k}</b></div>
          <div><em>Best score</em><b>{sil.max():.3f}</b></div>
          <div><em>Selected K</em><b style="color:#38bdf8">{k}</b></div>
          <div><em>Selected score</em><b>{sil[k]:.3f}</b></div>
        </div>
        <p style="color:#b8c7dc;font-size:13px;line-height:1.55;margin:6px 0 10px">
        K = {k} gives a <b style="color:#fff">{verdict}</b> structure
        ({sil[k] - sil.max():+.3f} vs. the best K). Scores above 0.5 mean clearly separated groups, 0.25–0.5 mean
        reasonable but overlapping groups, below 0.25 means little structure.</p>"""), unsafe_allow_html=True)
        st.button(f"Switch to K = {best_k}", on_click=_use_best_k, key="bestk2", disabled=(k == best_k))

st.markdown('<div id="players"></div>', unsafe_allow_html=True)
st.markdown('<div class="sect"><h2>Player explorer</h2><p>Search, filter and sort every player in the current view (sidebar filters apply). Click a column header to sort.</p></div>',
            unsafe_allow_html=True)
with st.container(key="card-table"):
    q1, q2, q3, q4 = st.columns([2, 1.5, 1.3, 1])
    query = q1.text_input("Search", placeholder="Player or team…")
    cl_pick = q2.multiselect("Cluster", [n for n in ordered if n in set(shown["cluster"])], placeholder="All clusters", key=f"t_cl_{k}")
    sort_by = q3.selectbox("Sort by", ["goals", "assists", "shots", "key_passes", "tackles", "interceptions", "clearances",
                                       "successful_dribbles", "pass_accuracy", "player_name", "team"],
                           format_func=lambda f: FEAT_LABEL.get(f, f.replace("_", " ").title()))
    desc = q4.selectbox("Order", ["Descending", "Ascending"]) == "Descending"
    tbl = shown
    if query:
        tbl = tbl[tbl["player_name"].str.contains(query, case=False, regex=False) | tbl["team"].str.contains(query, case=False, regex=False)]
    if cl_pick:
        tbl = tbl[tbl["cluster"].isin(cl_pick)]
    tbl = tbl.sort_values(sort_by, ascending=not desc)
    out = tbl[["player_name", "team", "position", "cluster"] + cl.FEATURES].copy()
    out["pass_accuracy"] *= 100
    st.markdown(f'<small style="color:#8fa3bf">{len(out):,} players match</small>', unsafe_allow_html=True)
    styled = out.style.map(lambda v: f"color:{CMAP.get(v, '#fff')};font-weight:700", subset=["cluster"])
    colcfg = {"player_name": st.column_config.TextColumn("Player", width="medium"),
              "team": st.column_config.TextColumn("Team"), "position": st.column_config.TextColumn("Position"),
              "cluster": st.column_config.TextColumn("Cluster"),
              "pass_accuracy": st.column_config.ProgressColumn("Pass acc.", format="%.0f%%", min_value=0, max_value=100)}
    for f in cl.FEATURES:
        if f != "pass_accuracy":
            colcfg[f] = st.column_config.NumberColumn(FEAT_LABEL[f], format="%.2f")
    st.dataframe(styled, column_config=colcfg, hide_index=True, width="stretch", height=430)
    st.download_button("⬇ Download this table", out.to_csv(index=False).encode(), file_name="players_filtered.csv", mime="text/csv", key="dl_tbl")
