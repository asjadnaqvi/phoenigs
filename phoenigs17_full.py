import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import openpyxl  # Ensure openpyxl is available
import math
from scipy import stats  # Add scipy dependency


# --- Load data ---
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


    return df

# --- Page layout ---
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 4])

with col1:
    st.title("Product network")

# Load data
df = load_data()
if df.empty:
    st.stop()

with col1:
    if "product" not in df.columns:
        st.error("The column 'product' is missing from the dataset.")
        st.stop()
    product_options = df.drop_duplicates(subset=["product"]).sort_values("product")
    product_labels = [f"{row['product_name']} ({row['product']})" for _, row in product_options.iterrows()]
    product_lookup = dict(zip(product_labels, product_options["product"]))
    selected_label = st.selectbox("Select Product:", product_labels)
    selected_product = product_lookup[selected_label]
    # Remove raw flow option and set metric type directly
metric_type = col1.radio("Select network weighting:", ["Raw Flow", "Risk-Weighted"], index=1)

# Filter data
df_product = df[df["product"] == selected_product].copy()

# Compute flow_weight and build graph before filtering by nodes for dropdown
if metric_type == "Risk-Weighted" and "ps_norm" in df_product.columns:
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

with col1:
    available_nodes = sorted(set(df_product["from"]).union(df_product["to"]).intersection(set(G.nodes)))
    default_index = available_nodes.index("AUT") if "AUT" in available_nodes else 0
    center_country = st.selectbox("Select country:", available_nodes, index=default_index)

# Recenter and relayer nodes around selected country
if center_country in G:
    import math
    center = center_country

    # Top N partners
    top_n = col1.slider("Number of top neighbors", min_value=2, max_value=10, value=5)
    
    #risk_threshold = col1.slider("Risk threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Risk threshold range
    risk_threshold = col1.slider(
        "Risk threshold range", 
        min_value=0.0, 
        max_value=1.0, 
        value=(0.25, 0.6), 
        step=0.05
    )
    min_risk_threshold, max_risk_threshold = risk_threshold

    line_thickness = col1.slider(
        "Line thickness", 
        min_value=0.1, 
        max_value=30.0, 
        value=(0.3, 14.0), 
        step=0.1
    )
    min_width, max_width = line_thickness


    # min_width = col1.slider("Minimum edge width", min_value=0.1, max_value=2.0, value=0.3, step=0.1)
    # max_width = col1.slider("Maximum edge width", min_value=3.0, max_value=10.0, value=8.0, step=0.5)
    dim_opacity = col1.slider("Dimmed edge opacity", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    top_flows_df = df_product[df_product['from'] == center].nlargest(top_n, 'value')
    inner_circle = top_flows_df['to'].tolist()

    # Outer ring: all remaining nodes with positive exports, sorted
    outer_min_val, outer_max_val = int(df_product['value'].min()), int(df_product['value'].max())
    # outer_value_threshold = col1.slider("Minimum export value for outer ring", min_value=1, max_value=outer_max_val, value=1)
    outer_value_threshold = 100

    outer_df = df_product[(~df_product['from'].isin([center] + inner_circle))]
    outer_df = outer_df.groupby("from")["value"].sum().reset_index()
    outer_df = outer_df[outer_df["value"] >= outer_value_threshold]
    outer_df = outer_df.merge(df_product[["from", "ex_region"]].drop_duplicates(), on="from", how="left")
    outer_df = outer_df.sort_values(by=["ex_region", "value"], ascending=[True, False])  # Sort by region, then by export value (descending)

    # outer_circle = outer_df["from"].tolist()
    # Slider for outer_circle limit
    max_outer_circle = col1.slider(
        "Maximum nodes in the outer circle", 
        min_value=10, 
        max_value=len(available_nodes), 
        value=30, 
        step=1
    )

    outer_circle = outer_df["from"].tolist()[:max_outer_circle]  # Select top nodes after sorting by region and value

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
    import numpy as np
    scaled_weight = np.log1p(weight)
    scaled_max = np.log1p(max_edge_weight)
    
    # Edge thickness scaling bounds
    width = (scaled_weight / scaled_max) * (max_width - min_width) + min_width
    edge_width.append(width)
    
    risk = edge[2].get("ps_norm", 0)
    important_nodes = set([center] + inner_circle)
    is_relevant = edge[0] in important_nodes or edge[1] in important_nodes

    if is_relevant and edge[0] == center:
        edge_color.append("rgba(0, 150, 0, 0.7)")  # direct edges from center
    elif metric_type == "Risk-Weighted" and isinstance(risk, (float, int)) and not pd.isna(risk) and is_relevant:
        if risk < min_risk_threshold:
            edge_color.append("rgba(0, 200, 0, 0.7)")  # green
        elif min_risk_threshold <= risk < max_risk_threshold:
            edge_color.append("rgba(255, 215, 0, 0.7)")  # yellow
        elif risk >= max_risk_threshold:
            edge_color.append("rgba(255, 0, 0, 0.7)")  # red
    elif is_relevant:
        edge_color.append("rgba(170, 170, 170, 0.6)")
    else:
        edge_color.append(f"rgba(170, 170, 170, {dim_opacity})")  # dimmed edges for unrelated flows
    
    try:
        risk_display = f"{float(risk):.2f}"
    except:
        risk_display = "N/A"
    edge_hover.append(f"{edge[0]} → {edge[1]}<br>Flow: {weight:,.0f}<br>Risk: {risk_display}")


edge_traces = []
for i in range(0, len(edge_x), 3):
    trace = go.Scatter(
        x=edge_x[i:i+3],
        y=edge_y[i:i+3],
        line=dict(width=edge_width[i // 3], color=edge_color[i // 3]),
        mode='lines+markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        hoverinfo='text',
        text=[edge_hover[i // 3]] * 3
    )
    edge_traces.append(trace)

# Define region colors (colorblind-friendly)
region_colors = {
    "Europe": "#1f77b4",
    "Asia": "#ff7f0e",
    "Africa": "#2ca02c",
    "Oceania": "#d62728",
    "Americas": "#9467bd",
    "Other": "#8c564b"
}

# Define minimum and maximum node sizes
node_size = col1.slider(
    "Node size (linear scaling)", 
    min_value=10, 
    max_value=100, 
    value=(25, 50), 
    step=1
)
min_node_size, max_node_size = node_size    

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


    # Scale node sizes linearly based on export values
    max_exports = df_product['value'].max()
    min_exports = df_product['value'].min()


    # Linear scaling of node size
    if max_exports > min_exports:
        size = ((exports - min_exports) / (max_exports - min_exports)) * (max_node_size - min_node_size) + min_node_size
    else:
        size = min_node_size  # Default size if all exports are the same
    node_size.append(size)

    # scaled_exports = np.log1p(exports)
    # size = (scaled_exports / np.log1p(df_product['value'].max())) * 40 + 10
    # node_size.append(size)

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
    )
)

# Build and render figure
fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title=dict(text=f"Trade Network for: {selected_label}", font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    height=800,              
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False)
                )
)

with col2:
    st.plotly_chart(fig, use_container_width=True)
