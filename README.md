# World Cup 2026: Player Clustering Dashboard

Groups outfield players by playing style using K-Means, shown in a Streamlit dashboard.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` - dashboard (sidebar K slider, PCA player map, profiles, silhouette chart, player table)
- `clustering.py` - data loading, feature building, K-Means, PCA, silhouette scores
- `data/` - the player performance CSV
- `.streamlit/config.toml` - green football theme

## Method
1. Drop goalkeepers, average each player's stats over all matches.
2. Standardise features (StandardScaler).
3. Run K-Means (K chosen with the slider; silhouette score helps pick the best K).
4. Name each group by its most common position; project to 2D with PCA for the map.
