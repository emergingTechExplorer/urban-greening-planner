import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import geopandas as gpd
import folium
from streamlit_folium import st_folium

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Urban Greening Explorer", layout="wide")
PRIMARY = "#2E7D32"

# -----------------------------
# Loaders
# -----------------------------


@st.cache_data
def load_trees_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
    df.columns = [c.upper() for c in df.columns]
    if "GEO_POINT_2D" in df.columns:
        parts = df["GEO_POINT_2D"].astype(str).str.split(",", n=1, expand=True)
        if parts.shape[1] == 2:
            df["LATITUDE"] = pd.to_numeric(parts[0], errors="coerce")
            df["LONGITUDE"] = pd.to_numeric(parts[1], errors="coerce")
    if "DATE_PLANTED" in df.columns:
        df["DATE_PLANTED"] = pd.to_datetime(
            df["DATE_PLANTED"], errors="coerce")
    if "DIAMETER" in df.columns:
        df["DIAMETER"] = pd.to_numeric(df["DIAMETER"], errors="coerce")
    return df


@st.cache_data
def load_local_areas(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "name" in gdf.columns and "NAME" not in gdf.columns:
        gdf = gdf.rename(columns={"name": "NAME"})
    return gdf.to_crs(4326)

# -----------------------------
# Spatial Join & Metrics
# -----------------------------


def attach_neighbourhood(trees_df: pd.DataFrame, areas_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gtrees = gpd.GeoDataFrame(
        trees_df.copy(),
        geometry=gpd.points_from_xy(
            trees_df["LONGITUDE"], trees_df["LATITUDE"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        gtrees, areas_gdf[["NAME", "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={"NAME": "LOCAL_AREA"})
    return pd.DataFrame(joined.drop(columns=["index_right"]))


def compute_area_km2(areas_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    metric = areas_gdf.to_crs(26910)  # NAD83 / UTM zone 10N
    areas_gdf["AREA_KM2"] = metric.geometry.area / 1e6
    return areas_gdf


def summarize_neighbourhoods(trees_with_area: pd.DataFrame, areas_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    grp = trees_with_area.groupby("LOCAL_AREA", dropna=False).agg(
        trees=("LOCAL_AREA", "count"),
        unique_species=("SPECIES_NAME", lambda s: s.dropna().nunique(
        ) if "SPECIES_NAME" in trees_with_area.columns else 0),
        avg_diameter=("DIAMETER", "mean") if "DIAMETER" in trees_with_area.columns else (
            "LOCAL_AREA", "size"),
        oldest_year=("DATE_PLANTED", lambda s: int(pd.to_datetime(s, errors="coerce").dt.year.min())
                     if pd.to_datetime(s, errors="coerce").notna().any() else np.nan)
                     if "DATE_PLANTED" in trees_with_area.columns else ("LOCAL_AREA", "size"),
        newest_year=("DATE_PLANTED", lambda s: int(pd.to_datetime(s, errors="coerce").dt.year.max())
                     if pd.to_datetime(s, errors="coerce").notna().any() else np.nan)
                     if "DATE_PLANTED" in trees_with_area.columns else ("LOCAL_AREA", "size"),
    ).reset_index()

    if "avg_diameter" not in grp.columns:
        grp["avg_diameter"] = np.nan
    if "oldest_year" not in grp.columns:
        grp["oldest_year"] = np.nan
    if "newest_year" not in grp.columns:
        grp["newest_year"] = np.nan

    areas_small = areas_gdf[["NAME", "AREA_KM2"]].rename(
        columns={"NAME": "LOCAL_AREA"})
    out = grp.merge(areas_small, on="LOCAL_AREA", how="left")
    out["trees_per_km2"] = out["trees"] / out["AREA_KM2"]
    return out

# -----------------------------
# Green Comfort Zone helpers
# -----------------------------


def compute_green_comfort_zones(summary_df: pd.DataFrame, q: float = 0.8):
    ser = summary_df["trees_per_km2"].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if ser.empty:
        return np.nan, pd.Series(False, index=summary_df.index)
    thr = ser.quantile(q)
    mask = summary_df["trees_per_km2"] >= thr
    return thr, mask


def make_green_comfort_map(areas_gdf: gpd.GeoDataFrame, summary_df: pd.DataFrame, mask: pd.Series) -> folium.Map:
    winners = summary_df.loc[mask, "LOCAL_AREA"].dropna().unique().tolist()
    m = folium.Map(location=[49.2827, -123.1207],
                   zoom_start=12, tiles="CartoDB Positron")

    if not winners:
        return m

    sel = areas_gdf[areas_gdf["NAME"].isin(winners)]
    if sel.empty:
        return m

    def style_function(_):
        return {
            "color": PRIMARY,
            "weight": 2,
            "fillColor": PRIMARY,
            "fillOpacity": 0.35,
        }

    gmerged = sel.merge(
        summary_df[["LOCAL_AREA", "trees_per_km2", "trees", "unique_species"]],
        left_on="NAME", right_on="LOCAL_AREA", how="left",
    )

    for _, r in gmerged.iterrows():
        dens = r.get("trees_per_km2", np.nan)
        trees = r.get("trees", np.nan)
        sp = r.get("unique_species", np.nan)
        html = f"<b>{r.get('NAME', 'Unknown')}</b><br>"
        html += f"Trees/km²: {dens:.0f}<br>" if pd.notnull(
            dens) else "Trees/km²: n/a<br>"
        html += f"Total trees: {int(trees) if pd.notnull(trees) else 'n/a'}<br>"
        html += f"Unique species: {int(sp) if pd.notnull(sp) else 'n/a'}"

        gj = folium.GeoJson(
            r["geometry"].__geo_interface__,
            style_function=style_function,
            tooltip=folium.Tooltip(html),
            name=r.get("NAME", "Comfort zone"),
        )
        gj.add_to(m)

    folium.LayerControl().add_to(m)
    return m

# -----------------------------
# Map helpers
# -----------------------------


def make_point_map(df_points: pd.DataFrame, areas_gdf: gpd.GeoDataFrame, highlight_area: str | None, max_points: int = 6000):
    # Fallback location = Vancouver
    loc = [49.2827, -123.1207]
    if not df_points.empty and df_points["LATITUDE"].notna().any() and df_points["LONGITUDE"].notna().any():
        loc = [df_points["LATITUDE"].mean(), df_points["LONGITUDE"].mean()]

    m = folium.Map(location=loc, zoom_start=12, tiles="CartoDB Positron")

    # Draw neighbourhood boundaries
    if areas_gdf is not None and not areas_gdf.empty:
        folium.GeoJson(
            areas_gdf.to_json(),
            name="Local Areas",
            style_function=lambda _: dict(color="#666", weight=1, fill=False),
        ).add_to(m)

        if highlight_area:
            sel = areas_gdf[areas_gdf["NAME"] == highlight_area]
            if not sel.empty:
                folium.GeoJson(
                    sel.to_json(),
                    name=f"Selected: {highlight_area}",
                    style_function=lambda _: dict(
                        color=PRIMARY, weight=3, fill=False),
                ).add_to(m)

    # Add tree points with sampling for performance
    pts = df_points
    if len(pts) > max_points:
        pts = pts.sample(max_points, random_state=42)

    for _, r in pts.iterrows():
        lat, lon = r.get("LATITUDE"), r.get("LONGITUDE")
        if pd.isna(lat) or pd.isna(lon):
            continue
        tip = (
            f"<b>{r.get('COMMON_NAME', 'Unknown') or 'Unknown'}</b><br>"
            f"Species: {r.get('SPECIES_NAME', 'Unknown') or 'Unknown'}<br>"
            f"Planted: {r['DATE_PLANTED'].date() if pd.notnull(r.get('DATE_PLANTED')) else 'Unknown'}<br>"
            f"Diameter: {r.get('DIAMETER') if pd.notnull(r.get('DIAMETER')) else 'Unknown'} in<br>"
            f"Local Area: {r.get('LOCAL_AREA', 'Unknown') or 'Unknown'}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=PRIMARY,
            fill=True,
            fill_opacity=0.7,
            tooltip=tip
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _safe_bins(series: pd.Series, k: int = 6) -> list:
    ser = series.replace([np.inf, -np.inf], np.nan).dropna()
    if ser.empty:
        return [0, 1]
    qs = np.unique(np.quantile(ser, np.linspace(0, 1, k + 1)))
    if len(qs) < 2:
        qs = np.array([ser.min(), ser.max() + 1e-9])
    return qs.tolist()


def make_area_choropleth(areas_gdf: gpd.GeoDataFrame, summary_df: pd.DataFrame,
                         metric_col: str, legend_name: str) -> folium.Map:
    g = areas_gdf.merge(
        summary_df[["LOCAL_AREA", metric_col, "trees", "unique_species"]],
        left_on="NAME", right_on="LOCAL_AREA", how="left"
    )

    m = folium.Map(location=[49.2827, -123.1207],
                   zoom_start=12, tiles="CartoDB Positron")

    bins = _safe_bins(g[metric_col], k=6)
    folium.Choropleth(
        geo_data=g.to_json(),
        data=g,
        columns=["NAME", metric_col],
        key_on="feature.properties.NAME",
        fill_color="YlGn",
        fill_opacity=0.85,
        line_opacity=0.6,
        nan_fill_color="#e5e7eb",
        bins=bins,
        legend_name=legend_name,
        name=legend_name,
    ).add_to(m)

    # Add hover tooltips
    for _, r in g.iterrows():
        val = r.get(metric_col, np.nan)
        name = r.get("NAME", "Unknown")
        trees = r.get("trees", np.nan)
        sp = r.get("unique_species", np.nan)

        # Use legend_name instead of hardcoded label
        if pd.notnull(val):
            html = f"<b>{name}</b><br>{legend_name}: {val:.0f}"
        else:
            html = f"<b>{name}</b><br>{legend_name}: n/a"

        html += f"<br>Total trees: {int(trees) if pd.notnull(trees) else 'n/a'}"
        html += f"<br>Unique species: {int(sp) if pd.notnull(sp) else 'n/a'}"

        gj = folium.GeoJson(
            r["geometry"].__geo_interface__,
            style_function=lambda _: {
                "color": "#444", "weight": 0.5, "fillOpacity": 0},
            tooltip=folium.Tooltip(html),
        )
        gj.add_to(m)

    folium.LayerControl().add_to(m)
    return m


# -----------------------------
# Main App
# -----------------------------
DATA_TREES = "data/public-trees.csv"
DATA_AREAS = "data/local_area_boundaries.geojson"

st.title("🌳 Urban Greening Explorer – Vancouver")

with st.spinner("Loading datasets..."):
    trees_raw = load_trees_from_csv(DATA_TREES)
    areas = compute_area_km2(load_local_areas(DATA_AREAS))

    # Keep only rows with valid coordinates for mapping/join
    trees_raw = trees_raw[trees_raw["LATITUDE"].between(
        -90, 90) & trees_raw["LONGITUDE"].between(-180, 180)]
    trees = attach_neighbourhood(trees_raw, areas)

# Precompute neighbourhood summary for Neighbourhood Overview + Comfort Zones
summary = summarize_neighbourhoods(trees, areas)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["Explore Trees", "Neighbourhood Overview", "Green Comfort Zones"])

# Explore Trees tab
with tab1:
    st.subheader("Interactive Tree Map")

    # Layout: left filter panel, right content
    filter_col, main_col = st.columns([1, 3])

    with filter_col:
        st.markdown("### Filters")

        neighbourhoods = [
            "(All)"] + (sorted(areas["NAME"].dropna().unique().tolist()) if "NAME" in areas.columns else [])
        nb = st.selectbox("Local Area", neighbourhoods, index=0)

        species_opts = ["(All)"] + (
            sorted(trees["SPECIES_NAME"].dropna().unique().tolist()
                   ) if "SPECIES_NAME" in trees.columns else []
        )
        sp = st.selectbox("Species (Latin)", species_opts, index=0)

        common_opts = ["(All)"] + (
            sorted(trees["COMMON_NAME"].dropna().unique().tolist()
                   ) if "COMMON_NAME" in trees.columns else []
        )
        cn = st.selectbox("Common Name", common_opts, index=0)

        if "DATE_PLANTED" in trees.columns and pd.to_datetime(trees["DATE_PLANTED"], errors="coerce").notna().any():
            yser = pd.to_datetime(
                trees["DATE_PLANTED"], errors="coerce").dt.year.dropna().astype(int)
            min_year = int(yser.min())
            max_year = int(yser.max())
        else:
            min_year = 1900
            max_year = 2100
        yr_min = st.slider("Planted Year (minimum)",
                           min_year, max_year, min_year)

        if "DIAMETER" in trees.columns and trees["DIAMETER"].notna().any():
            dmin_all = int(np.floor(trees["DIAMETER"].min()))
            dmax_all = int(np.ceil(trees["DIAMETER"].max()))
        else:
            dmin_all, dmax_all = 0, 100
        min_diam = st.slider("Minimum Diameter (inches)",
                             dmin_all, dmax_all, dmin_all)

        # Apply filters
        df = trees.copy()
        if nb != "(All)":
            df = df[df["LOCAL_AREA"] == nb]
        if sp != "(All)" and "SPECIES_NAME" in df.columns:
            df = df[df["SPECIES_NAME"] == sp]
        if cn != "(All)" and "COMMON_NAME" in df.columns:
            df = df[df["COMMON_NAME"] == cn]
        if "DATE_PLANTED" in df.columns:
            years = pd.to_datetime(df["DATE_PLANTED"], errors="coerce").dt.year
            df = df[years >= yr_min]
        if "DIAMETER" in df.columns:
            df = df[df["DIAMETER"].fillna(0) >= min_diam]

    with main_col:
        st.caption(
            "Points are sampled if the dataset is large to keep the map responsive.")
        m = make_point_map(
            df, areas_gdf=areas, highlight_area=None if nb == "(All)" else nb, max_points=6000)
        st_folium(m, height=560, width=None)

        st.markdown("### Key Stats (Filtered)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trees shown", f"{len(df):,}")
        c2.metric("Unique species",
                  f"{df['SPECIES_NAME'].dropna().nunique():,}" if "SPECIES_NAME" in df.columns else "n/a")
        c3.metric("Avg diameter (in)", f"{df['DIAMETER'].mean():.1f}" if "DIAMETER" in df.columns and pd.notnull(
            df["DIAMETER"].mean()) else "n/a")
        if "DATE_PLANTED" in df.columns:
            yrs = pd.to_datetime(df["DATE_PLANTED"], errors="coerce").dt.year
            yr_min_f = int(yrs.min()) if yrs.notna().any() else None
            yr_max_f = int(yrs.max()) if yrs.notna().any() else None
            c4.metric("Planting year range",
                      f"{yr_min_f if yr_min_f is not None else '–'} – {yr_max_f if yr_max_f is not None else '–'}")
        else:
            c4.metric("Planting year range", "n/a")

        st.markdown("### Distributions (Filtered)")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Top 15 Common Names**")
            if "COMMON_NAME" in df.columns and not df.empty:
                top_common = df["COMMON_NAME"].fillna(
                    "Unknown").value_counts().nlargest(15).reset_index()
                top_common.columns = ["Common Name", "Count"]
                chart = alt.Chart(top_common).mark_bar().encode(
                    x="Count:Q", y=alt.Y("Common Name:N", sort="-x")
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No common-name data to plot.")

        with colB:
            st.markdown("**Planting Year Histogram**")
            if "DATE_PLANTED" in df.columns:
                year_series = pd.to_datetime(
                    df["DATE_PLANTED"], errors="coerce").dt.year.dropna().astype(int)
                if not year_series.empty:
                    chart2 = alt.Chart(pd.DataFrame({"Year": year_series})).mark_bar().encode(
                        x="Year:O", y="count()"
                    ).properties(height=350)
                    st.altair_chart(chart2, use_container_width=True)
                else:
                    st.info("No valid planting years to plot.")
            else:
                st.info("No planting year field available.")

# Neighbourhood Overview tab
with tab2:
    st.subheader("Density Choropleth & Ranking")

    st.markdown("**Tree Density (trees per km²)**")
    choromap = make_area_choropleth(
        areas, summary, metric_col="trees_per_km2", legend_name="Trees per km²")
    st_folium(choromap, height=560, width=None)

    st.markdown("**Neighbourhood Ranking (by Density)**")
    rank_df = summary[["LOCAL_AREA", "trees", "unique_species", "avg_diameter",
                       "oldest_year", "newest_year", "AREA_KM2", "trees_per_km2"]].copy()
    rank_df = rank_df.sort_values(
        "trees_per_km2", ascending=False).reset_index(drop=True)
    rank_df_display = rank_df.rename(columns={
        "LOCAL_AREA": "Local Area",
        "trees": "Trees",
        "unique_species": "Unique Species",
        "avg_diameter": "Avg Diameter (in)",
        "oldest_year": "Oldest Planting Year",
        "newest_year": "Newest Planting Year",
        "AREA_KM2": "Area (km²)",
        "trees_per_km2": "Trees/km²",
    })
    st.dataframe(rank_df_display, use_container_width=True, hide_index=True)

# Green Comfort Zones tab
with tab3:
    st.subheader("Green Comfort Zones (High-Density Neighbourhoods)")

    q = st.slider(
        "Select the percentile threshold for defining Green Comfort Zones",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help="Neighbourhoods above this percentile of tree density " \
        "are labeled as Green Comfort Zones.",
    )

    st.caption(
        f"Current threshold: Top {int((1 - q) * 100)}% of neighbourhoods by tree density.")

    thr, mask = compute_green_comfort_zones(summary, q=q)

    if not np.isnan(thr):
        total_areas = summary["LOCAL_AREA"].nunique()
        winners = summary.loc[mask].copy()
        num_winners = winners["LOCAL_AREA"].nunique()

        c1, c2, c3 = st.columns(3)
        c1.metric("Comfort zones found", f"{num_winners} / {total_areas}")
        c2.metric("Density threshold (80th percentile)",
                  f"{thr:.0f} trees/km²")
        c3.metric("Median density (context)",
                  f"{summary['trees_per_km2'].median():.0f} trees/km²")

        st.markdown("**Map of Green Comfort Zones**")
        comfort_map = make_green_comfort_map(areas, summary, mask)
        st_folium(comfort_map, height=560, width=None)

        st.markdown("**High-Density Neighbourhoods (Green Comfort Zones)**")
        winners_display = winners[[
            "LOCAL_AREA", "trees", "unique_species", "AREA_KM2", "trees_per_km2"]].copy()
        winners_display = winners_display.sort_values(
            "trees_per_km2", ascending=False)
        winners_display = winners_display.rename(columns={
            "LOCAL_AREA": "Local Area",
            "trees": "Trees",
            "unique_species": "Unique Species",
            "AREA_KM2": "Area (km²)",
            "trees_per_km2": "Trees/km²",
        })
        st.dataframe(winners_display,
                     use_container_width=True, hide_index=True)

        # -----------------------------
        # AI Summary Section
        # -----------------------------
        st.markdown("### AI Summary of Green Comfort Zones")

        st.caption(
            "The AI summary uses OpenAI's `gpt-5-mini` model to interpret the Green Comfort Zones "
            "and describe patterns in density and species richness in plain language."
        )

        if not OPENAI_AVAILABLE:
            st.info("To enable AI summaries, install the `openai` Python package.")
        elif "OPENAI_API_KEY" not in st.secrets:
            st.info(
                "To enable AI summaries, add your OpenAI API key to `.streamlit/secrets.toml` "
                "as `OPENAI_API_KEY = \"sk-...\"`."
            )
        else:
            if st.button("Generate AI Summary"):
                summary_rows = winners_display.to_dict(orient="records")
                stats_text_lines = []
                for row in summary_rows:
                    stats_text_lines.append(
                        f"{row['Local Area']}: {row['Trees']} trees, "
                        f"{row['Unique Species']} species, "
                        f"{row['Area (km²)']:.2f} km², "
                        f"{row['Trees/km²']:.0f} trees/km²"
                    )
                stats_text = "\n".join(stats_text_lines)

                prompt = (
                    "You are helping analyze urban tree data for Vancouver.\n\n"
                    "The following neighbourhoods have been identified as 'Green Comfort Zones' "
                    "because they have high tree density. For each neighbourhood, you are given "
                    "trees, unique species, area and trees per km².\n\n"
                    "Data:\n"
                    f"{stats_text}\n\n"
                    "Write a short, clear summary (2–3 paragraphs) explaining:\n"
                    "- Which neighbourhoods stand out and why\n"
                    "- Any patterns you see in density and species richness\n"
                    "- How this could be useful for urban planning or public communication\n\n"
                    "Avoid repeating the raw numbers; focus on interpretation and insights."
                )

                try:
                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    model_name = st.secrets.get("OPENAI_MODEL", "gpt-5-mini")

                    with st.spinner("Generating AI summary..."):
                        resp = client.responses.create(
                            model=model_name,
                            input=prompt,
                            max_output_tokens=300,
                            reasoning={"effort": "minimal"},
                        )

                    summary_text = getattr(resp, "output_text", None)

                    if not summary_text:
                        summary_text = "AI summary generated, but no text output was returned."

                    st.markdown(summary_text)

                except Exception as e:
                    st.caption(
                        f"AI summary could not be generated (error: {e}).")

    else:
        st.info("Not enough valid density data to compute Green Comfort Zones.")

st.caption(
    "Data: City of Vancouver Open Data (Public Trees & Local Area Boundaries).")
