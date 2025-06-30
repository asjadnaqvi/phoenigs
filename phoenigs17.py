import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
# import plotly.figure_factory as ff
import openpyxl  
import math
from scipy import stats 

# import locale # does not work on streamlit cloud
# locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

## ADD PRODUCT: 251010, 310540  

# Custom HITS algorithm implementation

def hits_iterative(
    adj_matrix,
    country_names=None,
    max_iter=10,
    tol=1e-6,
    node_weights=None,
    return_all=True
):
    """
    Manual HITS algorithm, with optional node (country) weights applied at each iteration.
    
    Parameters:
        adj_matrix (np.ndarray or pd.DataFrame): Adjacency matrix (import dependencies).
        country_names (list): Optional list of country names for labeling.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        node_weights (pd.Series or array): Optional vector of length n for node-specific multipliers.
        return_all (bool): If True, returns per-iteration DataFrames.
        
    Returns:
        history (list): List of DataFrames with scores at each iteration.
        or
        final_hub, final_authority (pd.Series): Final scores if return_all=False.
    """
    if isinstance(adj_matrix, pd.DataFrame):
        A = adj_matrix.values
        if country_names is None:
            country_names = adj_matrix.index.tolist()
    else:
        A = adj_matrix
        if country_names is None:
            country_names = [f"Country {i}" for i in range(A.shape[0])]
    n = len(country_names)
    h = np.ones(n)
    a = np.ones(n)
    # Prepare node_weights vector
    if node_weights is not None:
        node_weights = np.array(node_weights)
        if len(node_weights) != n:
            raise ValueError("Length of node_weights must match number of countries.")
    else:
        node_weights = np.ones(n)
    history = []
    for i in range(max_iter):
        a_new = A.T @ h
        a_new = a_new * node_weights  # Node weights applied to authority
        a_new = a_new / np.linalg.norm(a_new) if np.linalg.norm(a_new) else a_new
        h_new = A @ a_new
        h_new = h_new * node_weights  # Node weights applied to hub
        h_new = h_new / np.linalg.norm(h_new) if np.linalg.norm(h_new) else h_new
        # Store in DataFrame for this iteration
        df = pd.DataFrame({
            "Hub": h_new,
            "Authority": a_new
        }, index=country_names)
        df["Iteration"] = i + 1
        history.append(df)
        # Convergence check
        if np.linalg.norm(h - h_new) < tol and np.linalg.norm(a - a_new) < tol:
            break
        h, a = h_new, a_new
    if return_all:
        return history  # List of DataFrames for each iteration
    else:
        return pd.Series(h, index=country_names, name="Hub"), pd.Series(a, index=country_names, name="Authority")

# Example usage:
# node_weights = pd.Series({'Germany': 1.7, 'Austria': 1.1, 'Italy': 1.3})
# history = hits_iterative(trade_matrix, country_names=countries, node_weights=node_weights, max_iter=5)


def create_trade_network(countries, trade_matrix):
    G = nx.from_numpy_array(trade_matrix, create_using=nx.DiGraph())
    mapping = {i: country for i, country in enumerate(countries)}
    G = nx.relabel_nodes(G, mapping)
    return G

def country_to_country_risk(G, countries, max_iter=5):
    A = nx.to_numpy_array(G, nodelist=countries)
    n = A.shape[0]
    h = np.ones(n)
    risk_matrices = []

    for _ in range(max_iter):
        h = A @ h
        h /= np.linalg.norm(h, 2)
        risk_matrix = A * h[np.newaxis, :]
        risk_matrices.append(pd.DataFrame(risk_matrix, columns=countries, index=countries))

    return risk_matrices

def get_risk_for_country(risk_matrices, country_from=None, country_to=None):
    results = []
    for i, df in enumerate(risk_matrices):
        if country_from and country_to:
            value = df.loc[country_from, country_to]
        elif country_from:
            value = df.loc[country_from]
        elif country_to:
            value = df[country_to]
        else:
            value = df
        results.append((f"Iter {i+1}", value))
    return results



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
        wgi = pd.read_excel("gew_riskindi.xlsx", engine="openpyxl")
        wgi = wgi[["iso3", "risk"]].dropna()
        wgi["risk"] = pd.to_numeric(wgi["risk"], errors="coerce")
        min_risk, max_risk = wgi["risk"].min(), wgi["risk"].max()
        # wgi["ps_norm"] = 1 - ((wgi["risk"] + 2.7) / 5) ## philips formula
        wgi["ps_norm"] = wgi["risk"] / 100 # take the values as they are from Irene's file.
        # wgi["ps_norm"] = 1 - ((wgi["risk"] - min_risk) / (max_risk - min_risk))
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

# Title + image
header_col1, header_col2 = st.columns([8, 1])

with header_col1:
    st.header("Dashboard zu den Abhängigkeiten Österreichs im internationalen Handelsnetzwerk")   ## dashboard title

with header_col2:
    st.image("Logo.jpg", width=100)



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
metric_type = col1.radio("Handelsverflechtungen", ["Ströme", "Länderrisiko-Ampel"], index=0)

# Filter data
df_product = df[df["product"] == selected_product].copy()





# Compute flow_weight and build graph before filtering by nodes for dropdown
if metric_type == "Länderrisiko-Ampel" and "ps_norm" in df_product.columns:
    df_product["flow_weight"] = df_product["value"] * df_product["ps_norm"]
else:
    df_product["flow_weight"] = df_product["value"]

df_product["flow_weight"] = df_product["flow_weight"].fillna(0)  # ensure flow_weight is numeric and fill NaNs with 0

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
    top_n = col1.slider("Anzahl der Top-Importländer", min_value=2, max_value=10, value=5)

    min_width   = 0.3   # Set in code
    max_width   = 8     # Set in code
    dim_opacity = 0.1  # Set in code

    # new code
    top_flows_df = df_product[df_product['to'] == center].nlargest(top_n, 'value')
    inner_circle = top_flows_df['from'].tolist()

    # st.write(f"Top {top_n} partners for {center_country}:", top_flows_df)

    # Outer ring: all remaining nodes with positive exports, sorted
    outer_min_val, outer_max_val = int(df_product['value'].min()), int(df_product['value'].max())
    outer_value_threshold = 100  # Set in code

    outer_df = df_product[(~df_product['from'].isin([center] + inner_circle))]
    outer_df = outer_df.groupby("from")["value"].sum().reset_index()
    outer_df = outer_df[outer_df["value"] >= outer_value_threshold]
    outer_df = outer_df.merge(df_product[["from", "ex_region"]].drop_duplicates(), on="from", how="left")

    # Limit the number of outer circle nodes to 25 per region
    outer_df = outer_df.nlargest(25, "value")
    outer_df = outer_df.sort_values(by=["ex_region", "value"], ascending=[True, False])
    outer_circle = outer_df["from"].tolist()


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
    


risk_threshold = col1.slider(
    "Risk thresholds", 
    min_value=0.0, 
    max_value=1.0, 
    value=(0.4, 0.65), 
    step=0.05
)


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
    width = (scaled_weight / scaled_max) * (max_width - min_width) + min_width
    edge_width.append(width)
    # edge_width.append((weight / max_edge_weight * 15) if max_edge_weight and not pd.isna(weight) else 1)
    
    risk = edge[2].get("ps_norm")
    if risk is None:
        risk = 0
    important_nodes = set([center] + inner_circle)
    is_relevant = edge[0] in important_nodes or edge[1] in important_nodes


    min_risk, max_risk = risk_threshold


    if metric_type == "Ströme" and is_relevant and edge[1] == center:
        edge_color.append("rgba(2, 8, 186, 0.8)")  # Dark blue for Austria raw flows
    elif metric_type == "Ströme" and is_relevant and edge[0] == center:
        edge_color.append("rgba(217, 2, 117, 0.65)")  #  Magenta/pink for Austria raw flows
    
    elif metric_type == "Länderrisiko-Ampel" and isinstance(risk, (float, int)) and not pd.isna(risk) and is_relevant:
        if risk < min_risk: # min risk
            edge_color.append("rgba(0, 200, 0, 0.7)") # green
        elif risk < max_risk: # max risk
            edge_color.append("rgba(255, 215, 0, 0.7)") # yellow
        else:
            edge_color.append("rgba(255, 0, 0, 0.7)") # red
    elif is_relevant:
        edge_color.append("rgba(110, 197, 245, 0.65)") ## light blue for related flows
    else:
        edge_color.append(f"rgba(170, 170, 170, {dim_opacity})")  # dimmed edges for unrelated flows
    try:
        risk_display = f"{float(risk):,.2f}"
    except:
        risk_display = "N/A"
    edge_hover.append(f"{edge[0]} → {edge[1]}<br>Werte: {weight:,.0f}<br>Risiko: {risk_display}")

dimmed_edge_traces = []
relevant_edges = []
center_edges = []

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
    # Append trace to the appropriate category
    if "rgba(2, 8, 186, 0.8)" in edge_color[i // 3] or "rgba(217, 2, 117, 0.65)" in edge_color[i // 3]:
        center_edges.append(trace)  # Center edges (to/from AUT)
    elif "170, 170, 170" in edge_color[i // 3]:  # Dimmed edges
        dimmed_edge_traces.append(trace)
    else:  # Relevant edges
        relevant_edges.append(trace)

# Define region colors
region_colors = {
    "Asien"         : "#ff7f0e",
    "Afrika"        : "#2ca02c",
    "Ozeanien"      : "#f25a78",
    "Amerikas"      : "#9467bd",
    "Europa - EU"   : "#057ef7",
    "Europa - Rest" : "#abd0f5",
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
    exports_to_center = df_product[(df_product['from'] == node) & (df_product['to'] == center_country)]['value'].sum()
    exports_share = (exports_to_center / exports * 100) if exports > 0 else 0
    
    subset = df_product[df_product['from'] == node]
    if not subset.empty:
        row = subset.iloc[0]
        name = row["ex_name"] if "ex_name" in row else node
        iso3_code = row["ex_iso3"] if "ex_iso3" in row else node
        region = row["ex_region"] if "ex_region" in row else "Other"
        risk_display = f"{row['ps_norm']:.2f}" if "ps_norm" in row and not pd.isna(row["ps_norm"]) else "N/A"
        # risk_display = locale.format_string('%.2f', row["ps_norm"]) if "ps_norm" in row and not pd.isna(row["ps_norm"]) else "N/A"
    else:
        name = node
        iso3_code = node
        region = "Other"
        risk_display = "N/A"
    node_label.append(iso3_code)
    node_region.append(region)
    node_color.append(region_colors.get(region, "lightgray"))
    label = (
        f"{name} ({iso3_code})<br>"
        f"Exporte - Gesamt: Tsd. EUR {exports:,.0f}<br>"
        f"Exporte - {center_country}: Tsd. EUR {exports_to_center:,.0f} ({exports_share:.2f}%)<br>"
        f"Risiko: {risk_display}"
    )
    node_text.append(label)


    scaled_exports = np.log1p(exports)
    size = (scaled_exports / np.log1p(df_product['value'].max())) * 60 + 8
    node_size.append(size)

# Highlight Austria
highlight_color = "red"
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

sorted_regions = sorted(filtered_region_colors.items(), key=lambda x: x[0])

# Add region legend
region_traces = []
for region, color in sorted_regions:
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

# Add risk threshold legend entries
if metric_type == "Länderrisiko-Ampel":
    region_traces += [
        go.Scatter(     # Blank entry for spacing
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=0, color='rgba(0,0,0,0)'),
            name=" ", 
            showlegend=True
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="rgba(0, 200, 0, 0.7)", width=2.5),
            name=f"0 – {min_risk:.2f}",
            showlegend=True
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="rgba(255, 215, 0, 0.7)", width=2.5),
            name=f"{min_risk:.2f} – {max_risk:.2f}",
            showlegend=True
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="rgba(255, 0, 0, 0.7)", width=2.5),
            name=f"{max_risk:.2f} – 1",
            showlegend=True
        ),
    ]


# Build and render figure
with col2:
    # Add traces in the desired order: dimmed -> relevant -> center
    fig = go.Figure(
        data=dimmed_edge_traces + relevant_edges + center_edges + [node_trace] + region_traces,
        layout=go.Layout(
            title=dict(text=f"Direkte und indirekte Handelsverflechtungen ({metric_type}): {selected_label}", font=dict(size=14)),
            showlegend=True,  # Enable legend
            hovermode='closest',
            height=800,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# --- Calculate and display hub scores ---
with col3:

    edge_columns_present = [col for col in edge_columns if col in df_product.columns]

    G_full = nx.from_pandas_edgelist(
        df_product,
        source="from",
        target="to",
        edge_attr=edge_columns_present,
        create_using=nx.DiGraph()
    )    


    adj_df = nx.to_pandas_adjacency(G_full, weight="weight")
    country_names = adj_df.index.tolist()

    # Unweighted HITS
    hits_history_raw = hits_iterative(
        adj_matrix=adj_df,
        country_names=country_names,
        max_iter=5,
        tol=1e-6,
        node_weights=None,
        return_all=True
    )
    hubs_direct_raw = hits_history_raw[0]["Hub"].to_dict()
    hubs_indirect_raw = hits_history_raw[4]["Hub"].to_dict()

    # Risk-weighted HITS
    # Prepare node weights (risk) if available
    node_weights = None
    if "ps_norm" in df_product.columns:
        node_weights_series = df_product.groupby("from")["ps_norm"].mean()
        node_weights = node_weights_series.reindex(country_names).fillna(1).values

    hits_history_risk = hits_iterative(
        adj_matrix=adj_df,
        country_names=country_names,
        max_iter=5,
        tol=1e-6,
        node_weights=node_weights,
        return_all=True
    )
    hubs_direct_risk = hits_history_risk[0]["Hub"].to_dict()
    hubs_indirect_risk = hits_history_risk[4]["Hub"].to_dict()

    # Multiply hub scores by 10 for better readability
    hubs_direct_raw    = {k: v * 10 for k, v in hubs_direct_raw.items()}
    hubs_direct_risk   = {k: v * 10 for k, v in hubs_direct_risk.items()}
    hubs_indirect_raw  = {k: v * 10 for k, v in hubs_indirect_raw.items()}
    hubs_indirect_risk = {k: v * 10 for k, v in hubs_indirect_risk.items()}


    if metric_type == "Ströme":
        hubs_direct = hubs_direct_raw
        hubs_indirect = hubs_indirect_raw
    else:
        hubs_direct = hubs_direct_risk
        hubs_indirect = hubs_indirect_risk


    # Get top_n countries by direct hub score
    top_countries = [c for c in inner_circle + outer_circle if c != "AUT"]
    top_countries = sorted(top_countries, key=lambda c: hubs_direct.get(c, 0), reverse=True)[:10]
    top_countries = top_countries[::-1]

    # Prepare data for heatmap
    z = [
        [hubs_direct.get(c, 0), hubs_indirect.get(c, 0)] for c in top_countries
    ]
    x = ["Direkt", "Indirekt"]
    y = top_countries
    

    # Create heatmap
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Sunsetdark',
            zmin=1,
            zmax=6,
            colorbar=dict(
                title="Index",
                tickvals=[1, 2, 3, 4, 5, 6],
                ticktext=["1", "2", "3", "4", "5", "6"],  
                thickness=15  # Adjust thickness as needed
            ),
            showscale=True,
            text=[[f"{val:.2f}" for val in row] for row in z],
            hoverinfo="text"
        )
    )

    # Add values as annotations in each heatmap box
    for i, country in enumerate(y):
        for j, col in enumerate(x):
            fig_heatmap.add_annotation(
                x=col,
                y=country,
                text=f"{z[i][j]:.2f}",
                showarrow=False,
                font=dict(color="black", size=14),
                xanchor="center",
                yanchor="middle"
            )

    fig_heatmap.update_layout(
        title=dict(text=f"Importabhängigkeitindex (Top 10)", font=dict(size=14)),
        height=600
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)
        
      
