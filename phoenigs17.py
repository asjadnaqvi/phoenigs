import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import openpyxl  # Ensure openpyxl is available
import math
from scipy import stats  # Add scipy dependency

def load_data():
    df = pd.read_csv("baci_hs22_2023.csv")
    df.rename(columns={
        "ex_iso3": "from",
        "im_iso3": "to",
        "value": "value",
        "code": "product",
        "code_name": "product_name",
        "ex_name": "ex_name",
        "ex_region": "ex_region",
        "im_name": "im_name",
        "im_region": "im_region"
    }, inplace=True)
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)

    # Load WGI and merge
    try:
        wgi = pd.read_excel("wgirisk_2023.xlsx", engine="openpyxl")
        wgi = wgi[["iso3", "risk"]].dropna()
        wgi["risk"] = pd.to_numeric(wgi["risk"], errors="coerce")
        min_risk, max_risk = wgi["risk"].min(), wgi["risk"].max()
        # wgi["ps_norm"] = 1 - ((wgi["risk"] + 2.7) / 5) ## philips formula
        wgi["ps_norm"] = 1 - ((wgi["risk"] - min_risk) / (max_risk - min_risk))
        df = df.merge(wgi[["iso3", "ps_norm"]], left_on="from", right_on="iso3", how="left")
    except Exception as e:
        st.warning(f"WGI data not found or failed to merge: {e}")
        df["ps_norm"] = None

    # Manually generate the 'eu' column
    eu_countries = [
        "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "ESP", "EST", "FIN", "FRA", 
        "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", 
        "PRT", "ROU", "SVK", "SVN", "SWE"
    ]
    df["eu"] = df["to"].isin(eu_countries)

    # Debug: Check the 'eu' column
    # st.write("EU Column:", df[["to", "eu"]].drop_duplicates())

    return df

# --- Page layout ---
st.set_page_config(layout="wide")

# Add a placeholder for the title
header_col1, header_col2 = st.columns([5, 1])

# with header_col2:
    # Placeholder for the logo
    # st.image("path/to/logo.png", width=100)  # Replace "path/to/logo.png" with the actual path to your logo file

with header_col1:
    st.title("Placeholder for title + Logo")   ## dashboard title


col1, col2, col3 = st.columns([1, 4, 1])

# with col1:
    # st.title("Product network")

# Load data
df = load_data()
if df.empty:
    st.stop()

with col1:
    if "product" not in df.columns:
        st.error("The column 'product' is missing from the dataset.")
        st.stop()
    product_options = df.drop_duplicates(subset=["product"]).sort_values("product")
    product_labels = [f"{row['code_deu']} ({row['product']})" for _, row in product_options.iterrows()]
    product_lookup = dict(zip(product_labels, product_options["product"]))
    selected_label = st.selectbox("Produkt auswählen:", product_labels)
    selected_product = product_lookup[selected_label]
    # Remove raw flow option and set metric type directly
metric_type = col1.radio("Select metric", ["Ohne risiko", "Mit risiko"], index=0, label_visibility="hidden")

# Filter data
df_product = df[df["product"] == selected_product].copy()

# Compute flow_weight and build graph before filtering by nodes for dropdown
if metric_type == "Mit risiko" and "ps_norm" in df_product.columns:
    df_product["flow_weight"] = df_product["value"] * df_product["ps_norm"]
else:
    df_product["flow_weight"] = df_product["value"]

edge_columns = ["flow_weight", "value"]
if "ps_norm" in df_product.columns:
    edge_columns.append("ps_norm")

# Drop zero-flow edges
df_product = df_product[df_product["flow_weight"] > 0]

G = nx.from_pandas_edgelist(
    df_product,
    source="from",
    target="to",
    edge_attr=edge_columns,
    create_using=nx.DiGraph()
)

center_country = "AUT"  # Fixed focus country

# Recenter and relayer nodes around selected country
if center_country in G:

    center = center_country

    # Top N partners
    top_n = col1.slider("Anzahl der Top-Nachbarn", min_value=2, max_value=10, value=5)
    # Removed risk threshold slider
    min_width = 0.3  # Set in code
    max_width = 8  # Set in code
    dim_opacity = 0.1  # Set in code

    top_flows_df = df_product[df_product['from'] == center].nlargest(top_n, 'value')
    inner_circle = top_flows_df['to'].tolist()

    # Outer ring: all remaining nodes with positive exports, sorted
    outer_min_val, outer_max_val = int(df_product['value'].min()), int(df_product['value'].max())
    outer_value_threshold = 1000  # Set in code

    outer_df = df_product[(~df_product['from'].isin([center] + inner_circle))]
    outer_df = outer_df.groupby("from")["value"].sum().reset_index()
    outer_df = outer_df[outer_df["value"] >= outer_value_threshold]
    outer_df = outer_df.merge(df_product[["from", "ex_region"]].drop_duplicates(), on="from", how="left")
    outer_df = outer_df.sort_values(by=["ex_region", "value"], ascending=[True, False])

    # Limit the number of outer circle nodes to 40
    outer_circle = outer_df["from"].tolist()[:30]

    def polar_to_cartesian(radius, angle_deg):
        angle_rad = math.radians(angle_deg)
        return radius * math.cos(angle_rad), radius * math.sin(angle_rad)

    pos = {}
    pos[center] = (0, 0)
    for i, node in enumerate(inner_circle):
        angle = 360 * i / max(1, len(inner_circle))
        pos[node] = polar_to_cartesian(1.5, angle)
    for i, node in enumerate(outer_circle):
        angle = 360 * i / max(1, len(outer_circle))
        pos[node] = polar_to_cartesian(3.0, angle)
    keep_nodes = set(df_product["from"]).union(df_product["to"])
    df_product = df_product[df_product["from"].isin(keep_nodes) & df_product["to"].isin(keep_nodes)]

    def polar_to_cartesian(radius, angle_deg):
        angle_rad = math.radians(angle_deg)
        return radius * math.cos(angle_rad), radius * math.sin(angle_rad)

    pos = {}
    pos[center] = (0, 0)
    for i, node in enumerate(inner_circle):
        angle = 360 * i / max(1, len(inner_circle))
        pos[node] = polar_to_cartesian(1.5, angle)
    for i, node in enumerate(outer_circle):
        angle = 360 * i / max(1, len(outer_circle))
        pos[node] = polar_to_cartesian(3.0, angle)
    

    # No additional ring — outer_circle now holds all remaining nodes
    # Remaining ring removed to simplify layout

# Edge data
edge_x, edge_y, edge_width, edge_color, edge_hover = [], [], [], [], []
visible_weights = [edge[2]["flow_weight"] for edge in G.edges(data=True) if edge[0] in pos and edge[1] in pos]
max_edge_weight = max(visible_weights) if visible_weights else 1
for edge in G.edges(data=True):
    if edge[0] not in pos or edge[1] not in pos:
        continue
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    weight = edge[2]["flow_weight"]

    scaled_weight = np.log1p(weight)
    scaled_max = np.log1p(max_edge_weight)
    
    # Edge thickness scaling bounds
    # Use sliders for min_width and max_width above
    width = (scaled_weight / scaled_max) * (max_width - min_width) + min_width
    edge_width.append(width)
    # edge_width.append((weight / max_edge_weight * 15) if max_edge_weight and not pd.isna(weight) else 1)
    
    risk = edge[2].get("ps_norm")
    if risk is None:
        risk = 0
    important_nodes = set([center] + inner_circle)
    is_relevant = edge[0] in important_nodes or edge[1] in important_nodes


    if metric_type == "Ohne risiko" and is_relevant and edge[0] == center:
        edge_color.append("darkblue")  # Dark blue for Austria raw flows

    elif metric_type == "Mit risiko" and isinstance(risk, (float, int)) and not pd.isna(risk) and is_relevant:
        if risk < 0.25:
            edge_color.append("rgba(0, 200, 0, 0.7)")
        elif risk < 0.6:
            edge_color.append("rgba(255, 215, 0, 0.7)")
        else:
            edge_color.append("rgba(255, 0, 0, 0.7)")
    elif is_relevant:
        edge_color.append("rgba(170, 170, 170, 0.7)")
    else:
        edge_color.append(f"rgba(170, 170, 170, {dim_opacity})")  # dimmed edges for unrelated flows
    try:
        risk_display = f"{float(risk):.2f}"
    except:
        risk_display = "N/A"
    edge_hover.append(f"{edge[0]} → {edge[1]}<br>Flow: {weight:,.0f}<br>Risk: {risk_display}")

aut_edge_traces = []
dimmed_edge_traces = []

for i in range(0, len(edge_x), 3):
    trace = go.Scatter(
        x=edge_x[i:i+3],
        y=edge_y[i:i+3],
        line=dict(width=edge_width[i // 3], color=edge_color[i // 3]),
        mode='lines+markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        hoverinfo='text',
        text=[edge_hover[i // 3]] * 3,
        showlegend=False  # Disable legend for edge traces
    )
    if edge_color[i // 3] in ["rgba(0, 51, 153, 0.7)", "rgba(0, 200, 0, 0.7)", "rgba(255, 215, 0, 0.7)", "rgba(255, 0, 0, 0.7)"]:
        aut_edge_traces.append(trace)
    else:
        dimmed_edge_traces.append(trace)
# edge_traces.append(trace)

# Define region colors
region_colors = {
    "Europe": "#1f77b4",
    "Asia": "#ff7f0e",
    "Africa": "#2ca02c",
    "Oceania": "#d62728",
    "Americas": "#9467bd",
    "Other": "#8c564b"
}

# Node trace with actual country names and ISO3 labels
node_x, node_y, node_text, node_size, node_label, node_color, node_region = [], [], [], [], [], [], []
for node in G.nodes():
    if node not in pos:
        continue
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    exports = df_product[df_product['from'] == node]['value'].sum()
    subset = df_product[df_product['from'] == node]
    if not subset.empty:
        row = subset.iloc[0]
        name = row["ex_name"] if "ex_name" in row else node
        iso3_code = row["ex_iso3"] if "ex_iso3" in row else node
        region = row["ex_region"] if "ex_region" in row else "Other"
    else:
        name = node
        iso3_code = node
        region = "Other"
    node_label.append(iso3_code)
    node_region.append(region)
    node_color.append(region_colors.get(region, "lightgray"))
    label = f"{name} ({iso3_code})<br>Exports: {exports:,.0f}"
    node_text.append(label)

    scaled_exports = np.log1p(exports)
    size = (scaled_exports / np.log1p(df_product['value'].max())) * 40 + 10
    node_size.append(size)

# Highlight Austria
highlight_color = "tomato"
node_color = [highlight_color if lbl.startswith("Austria") or "AUT" in lbl else region_colors.get(region, "lightgray") for lbl, region in zip(node_label, node_region)]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_label,
    textfont=dict(size=12, color="black"),
    textposition="middle center",
    hoverinfo='text',
    hovertext=node_text,
    marker=dict(
        showscale=False,
        color=node_color,
        opacity=0.9,
        size=node_size,
        line=dict(color="black", width=0.8)
    ),
    showlegend=False  # Disable legend for node trace
)


# Filter regions based on active nodes in the graph
active_regions = set(node_region)  # Get unique regions from the active graph
filtered_region_colors = {region: color for region, color in region_colors.items() if region in active_regions}

# Add region legend
region_traces = []
for region, color in filtered_region_colors.items():
    region_traces.append(
        go.Scatter(
            x=[None],  # Dummy point for legend
            y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            name=region,  # Explicitly set the region name
            showlegend=True  # Ensure the legend is shown for active regions
        )
    )



# --- Calculate and display hub scores ---
with col3:
    # st.subheader("Hub Scores for EU")
    
    # Debug: Check the graph before applying HITS
    # st.write("Graph Nodes:", list(G.nodes))
    # st.write("Graph Edges:", list(G.edges(data=True)))

    # Create raw and risk-weighted graphs
    H_raw = G.copy()
    for u, v, d in H_raw.edges(data=True):
        d["weight"] = 1  # Raw Flow: All weights set to 1
    hubs_raw = nx.hits(H_raw, normalized=True)[0]

    # Convert values to numpy array for min-max normalization
    scores = np.array(list(hubs_raw.values()))
    min_score = scores.min()
    max_score = scores.max()

    if max_score != min_score:
        hubs_raw = {k: 100 * (v - min_score) / (max_score - min_score) for k, v in hubs_raw.items()}
    else:
        hubs_raw = {k: 0 for k in hubs_raw}  # All values are the same    

    H_risk = G.copy()
    for u, v, d in H_risk.edges(data=True):
        d["weight"] = d.get("flow_weight", 1)  # Risk-Weighted: Use flow_weight
    hubs_risk = nx.hits(H_risk, normalized=True)[0]

    # Convert values to numpy array for min-max normalization
    scores = np.array(list(hubs_risk.values()))
    min_score = scores.min()
    max_score = scores.max()

    if max_score != min_score:
        hubs_risk = {k: 100 * (v - min_score) / (max_score - min_score) for k, v in hubs_risk.items()}
    else:
        hubs_risk = {k: 0 for k in hubs_risk}  # All values are the same    


    # Filter EU nodes
    eu_nodes = [n for n in G.nodes if df[df["to"] == n]["eu"].any()]
    # st.write("EU Nodes:", eu_nodes)  # Debug: Check EU nodes

    # Extract hub scores for EU nodes
    hub_raw_vals  = [hubs_raw.get(n, np.nan) for n in eu_nodes]
    hub_risk_vals = [hubs_risk.get(n, np.nan) for n in eu_nodes]

    # Debug: Check extracted hub scores
    #st.write("Hub Raw Values:", hub_raw_vals)
    #st.write("Hub Risk Values:", hub_risk_vals)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        "Country": eu_nodes,
        "Raw Hub Score": hub_raw_vals,
        "Risk-weighted Hub Score": hub_risk_vals
    }).sort_values("Country")

    # Debug: Check the DataFrame
    # st.write("Plot DataFrame:", plot_df)

    # Debug: Check Austria's hub score
    # st.write("Austria Row in DataFrame:", plot_df[plot_df["Country"] == "AUT"])
    # st.write("Extracted Austria Hub Score (Raw):", plot_df.loc[plot_df["Country"] == "AUT", "Raw Hub Score"].values)
    # st.write("Extracted Austria Hub Score (Risk-Weighted):", plot_df.loc[plot_df["Country"] == "AUT", "Risk-weighted Hub Score"].values)
    # st.write("Is Austria in EU Nodes:", "AUT" in eu_nodes)
    # st.write("Hub Values List:", hub_values)
    
    # Handle empty hub values
    if not hub_raw_vals:  # Check if the list is empty
        st.warning("No valid hub scores available for the selected metric.")
        hub_raw_vals = [0]  # Placeholder to avoid errors

    # st.write("Average Hub Score:", avg)  # Debug: Check average hub score
    # st.write("Austria Hub Score:", aut)  # Debug: Check Austria's hub score

    fig1 = go.Figure()

    # Add box plot for hub values
    if metric_type == "Ohne risiko":
        hub_values = plot_df["Raw Hub Score"].dropna().tolist()
        # Extract Austria's hub score using .loc[]
        if "AUT" in plot_df["Country"].values:
            aut = plot_df.loc[plot_df["Country"] == "AUT", "Raw Hub Score"].values[0]
        else:
            aut = np.nan

        avg = np.nanmean(hub_raw_vals)

        fig1.add_trace(
            go.Box(
                y=hub_raw_vals,
                name="",
                boxpoints=False,  # Disable outliers
                #boxpoints='outliers',
                marker_color='lightblue',
                showlegend=False
            )
        )
    else:
        hub_values = plot_df["Risk-weighted Hub Score"].dropna().tolist()

        if "AUT" in plot_df["Country"].values:
            aut = plot_df.loc[plot_df["Country"] == "AUT", "Risk-weighted Hub Score"].values[0]
        else:
            aut = np.nan

        avg = np.nanmean(hub_risk_vals)

        fig1.add_trace(
            go.Box(
                y=hub_risk_vals,
                name="",
                boxpoints=False,  # Disable outliers
                # boxpoints='outliers',
                marker_color='lightblue',
                showlegend=False
            )
        )    

    # Add scatter point for average hub score
    fig1.add_trace(
        go.Scatter(
            y=[avg],
            x=[""],
            mode='markers',
            name='EU',
            marker=dict(color='darkblue', symbol='circle', size=15)
        )
    )

    # Add scatter point for Austria's hub score
    if not np.isnan(aut):  # Ensure Austria's hub score is valid
        fig1.add_trace(
            go.Scatter(
                y=[aut],
                x=[""],
                mode='markers',
                name='Österreich',
                marker=dict(color='tomato', symbol='circle', size=15)
            )
        )

    # Update layout
    fig1.update_layout(
        title="Diversifikationsindex",
        yaxis_title="Werte",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.2,  # Position below the plot
            xanchor="center",
            x=0.5
        )
    )

    # Render the plot
    st.plotly_chart(fig1, use_container_width=True)

        # Build and render figure
    with col2:
        fig = go.Figure(
            data=dimmed_edge_traces + aut_edge_traces + [node_trace] + region_traces,
            layout=go.Layout(
                showlegend=True,  # Enable legend
                hovermode='closest',
                height=800,
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False)
            )
        )

        st.plotly_chart(fig, use_container_width=True)
