# FootballMania · World Cup 2026 Player Clustering

Groups outfield players by playing style using K-Means, shown in a Streamlit dashboard.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` - dark stadium-themed dashboard: left sidebar (nav, K, team/position/cluster filters, dataset stats), centre (hero, KPIs, PCA cluster map with centroids), right (cluster cards, quick compare, clickable player details), below (radar / z-score heatmap / position mix, silhouette analysis, searchable player table)
- `assets/stadium_hero.jpg` - generated stadium header image
- `clustering.py` - data loading, feature building, K-Means, PCA, silhouette scores
- `data/` - the player performance CSV
- `.streamlit/config.toml` - dark theme base (custom CSS in app.py does the rest)

## Method
1. Drop goalkeepers, average each player's stats over all matches.
2. Standardise features (StandardScaler).
3. Run K-Means (K chosen with the slider; silhouette score helps pick the best K).
4. Name each group by its most common position; project to 2D with PCA for the map.
