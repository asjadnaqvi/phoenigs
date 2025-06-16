import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import openpyxl  
import math
from scipy import stats 

# import locale # does not work on streamlit cloud
# locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

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



    if metric_type == "Ströme" and is_relevant and edge[1] == center:
        edge_color.append("rgba(2, 8, 186, 0.8)")  # Dark blue for Austria raw flows
    elif metric_type == "Ströme" and is_relevant and edge[0] == center:
        edge_color.append("rgba(217, 2, 117, 0.65)")  #  Magenta/pink for Austria raw flows
    
    elif metric_type == "Länderrisiko-Ampel" and isinstance(risk, (float, int)) and not pd.isna(risk) and is_relevant:
        if risk < 0.4: # min risk
            edge_color.append("rgba(0, 200, 0, 0.7)") # green
        elif risk < 0.65: # max risk
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
        # f"{name} ({iso3_code})<br>"
        # f"Exporte - Gesamt: Tsd. EUR {locale.format_string('%.0f', exports, grouping=True)}<br>"
        # f"Exporte - {center_country}: Tsd. EUR {locale.format_string('%.0f', exports_to_center, grouping=True)} ({locale.format_string('%.2f', exports_share, grouping=True)}%)<br>"
        # f"Risiko: {risk_display}"
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
            name="0 - 0.40",
            showlegend=True
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="rgba(255, 215, 0, 0.7)", width=2.5),
            name="0.40 – 0.65",
            showlegend=True
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="rgba(255, 0, 0, 0.7)", width=2.5),
            name="0.65 - 1",
            showlegend=True
        ),
    ]



# --- Calculate and display hub scores ---
with col3:
    # st.subheader("Hub Scores for EU")
    
    # Debug: Check the graph before applying HITS
    # st.write("Graph Nodes:", list(G.nodes))
    # st.write("Graph Edges:", list(G.edges(data=True)))

    # Calculate HITS on the full (unfiltered) network for normalization
    

    edge_columns_present = [col for col in edge_columns if col in df_product.columns]

    G_full = nx.from_pandas_edgelist(
        df_product,
        source="from",
        target="to",
        edge_attr=edge_columns_present,
        create_using=nx.DiGraph()
    )    


    # Raw HITS on full network
    H_raw_full = G_full.copy()
    for u, v, d in H_raw_full.edges(data=True):
        d["weight"] = 1
    hubs_raw_full = nx.hits(H_raw_full, normalized=True)[0]


    # Risk-weighted HITS on full network
    H_risk_full = G_full.copy()

    for u, v, d in H_risk_full.edges(data=True):
        value = d.get("value", 1)
        ps_norm = d.get("ps_norm", 1)
        if pd.isna(ps_norm):
            ps_norm = 0
        d["weight"] = value * ps_norm

    #for u, v, d in H_risk_full.edges(data=True):
    #    d["weight"] = d["flow_weight"] if "flow_weight" in d else 1
    
    hubs_risk_full = nx.hits(H_risk_full, normalized=True)[0]
    # hubs_risk_full = nx.hits(H_raw_full, normalized=True, weight="ps_norm")[0]


    #for u, v, d in H_risk_full.edges(data=True):
    #    print(f"{u} -> {v}: weight={d.get('weight')}, flow_weight={d.get('flow_weight')}")

    #for u, v, d in H_risk_full.edges(data=True):
    #    print(f"{u} -> {v}: flow_weight={d.get('flow_weight')}, wgi_risk={d.get('wgi_risk')}")

    #for u, v, d in H_risk_full.edges(data=True):
    #    print(f"{u} -> {v}: ps_norm={d.get('ps_norm')}")


    # Import perspective (reverse) on full network
    H_raw_import_full = H_raw_full.reverse(copy=True)
    hubs_raw_import_full = nx.hits(H_raw_import_full, normalized=True)[0]

    H_risk_import_full = H_risk_full.reverse(copy=True)
    hubs_risk_import_full = nx.hits(H_risk_import_full, normalized=True)[0]


    # Normalize on the full network
    def normalize_dict(d):
        vals = np.array(list(d.values()))
        min_score = 0 #vals.min()
        max_score = vals.max()
        if max_score != min_score:
            # return {k: 100 * (v - min_score) / (max_score - min_score) for k, v in d.items()}

            return {k: 100 * (v) for k, v in d.items()}
        else:
            return {k: 0 for k in d}    
        

    #hubs_raw_full = normalize_dict(hubs_raw_full)
    #hubs_risk_full = normalize_dict(hubs_risk_full)
    hubs_raw_import_full = normalize_dict(hubs_raw_import_full)
    hubs_risk_import_full = normalize_dict(hubs_risk_import_full)


    # Filter EU nodes
    eu_nodes = [n for n in G.nodes if df[df["to"] == n]["eu"].any()]
    # st.write("EU Nodes:", eu_nodes)  # Debug: Check EU nodes

    
    # Add radio button for direction selection
    #direction = col1.radio(
    #    "Perspektive für HITS-Berechnung:",
    #    ["Export-Sicht (Österreich als Exporteur)", "Import-Sicht (Österreich als Importeur)"],
    #    index=0
    #)

    # Extract hub scores for EU nodes based on direction, but normalized on the full network
    #if direction == "Export-Sicht (Österreich als Exporteur)":
    #    hub_raw_vals  = [hubs_raw_full.get(n, np.nan) for n in eu_nodes]
    #    hub_risk_vals = [hubs_risk_full.get(n, np.nan) for n in eu_nodes]
    #else:
    hub_raw_vals  = [hubs_raw_import_full.get(n, np.nan) for n in eu_nodes]
    hub_risk_vals = [hubs_risk_import_full.get(n, np.nan) for n in eu_nodes]




    # Debug: Check extracted hub scores
    #st.write("Hub Raw Values:", hub_raw_vals)
    #st.write("Hub Risk Values:", hub_risk_vals)

    #for n, v in zip(eu_nodes, hub_raw_vals):
    #    print(f"{n}: {v}")
    
    #for n, v in zip(eu_nodes, hub_risk_vals):
    #    print(f"{n}: {v}")   

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
    if metric_type == "Ströme":

        #eu_hub_raw_vals = [hub_raw_vals[eu_nodes.index(n)] for n in eu_nodes if n in eu_nodes and not pd.isna(hub_raw_vals[eu_nodes.index(n)])]

        aut = hub_raw_vals[eu_nodes.index("AUT")] if "AUT" in eu_nodes else np.nan
        # hub_values = plot_df[plot_df["Country"].isin(eu_nodes)]["Raw Hub Score"].dropna().tolist()

        # Extract Austria's hub score using .loc[]
        #if "AUT" in eu_plot_df["Country"].values:
        #    aut = eu_plot_df.loc[eu_plot_df["Country"] == "AUT", "Raw Hub Score"].values[0]
        #else:
        #    aut = np.nan


        avg = np.nanmean(hub_raw_vals)

        #print("Average Hub Score (Un-weighted):", avg)  # Debug: Check average hub score
        #print("AUT Hub Score (Un-weighted):", aut)  # Debug: Check average hub score


        # Set y-axis range based only on EU node values
        y_min = np.nanmin(hub_raw_vals)
        y_max = np.nanmax(hub_raw_vals)

        fig1.add_trace(
            go.Box(
                y=hub_raw_vals,
                name="",
                boxpoints=False,  # Disable outliers
                #boxpoints='outliers',
                marker_color='#aaacad',
                showlegend=False,
                hoverinfo='skip'
            )
        )
    else:
        hub_values = plot_df["Risk-weighted Hub Score"].dropna().tolist()

        if "AUT" in plot_df["Country"].values:
            aut = plot_df.loc[plot_df["Country"] == "AUT", "Risk-weighted Hub Score"].values[0]
        else:
            aut = np.nan

        avg = np.nanmean(hub_risk_vals)

        #print("Average Hub Score (Risk-weighted):", avg)  # Debug: Check average hub score
        #print("AUT Hub Score (Risk-weighted):", aut)  # Debug: Check average hub score

        #y_min = np.nanmin(hub_values)
        #y_max = np.nanmax(hub_values)


        fig1.add_trace(
            go.Box(
                y=hub_risk_vals,
                name="",
                boxpoints=False,  # Disable outliers
                #boxpoints='outliers',
                marker_color='#aaacad',
                showlegend=False,
                #hovertext=[f"{val:.3f}" for val in hub_risk_vals],  # Values with 3 decimals
                #hoverinfo='text'
                #hovertext=[f"Custom text: {val}" for val in hub_risk_vals],  # Custom hover text
                hoverinfo='skip'  # Show only custom hover text
            )
        )    

    # Add scatter point for average hub score
    fig1.add_trace(
        go.Scatter(
            y=[avg],
            x=[""],
            mode='markers',
            name='EU',
            marker=dict(color='#057ef7', symbol='circle', size=15),
            hovertemplate="%.3f" % avg  # Custom hover text with 3 decimals
        )
    )

    # Add scatter point for Austria's hub score
    if not np.isnan(aut):
        fig1.add_trace(
            go.Scatter(
                y=[aut],
                x=[""],
                mode='markers',
                name='Österreich',
                marker=dict(color='red', symbol='circle', size=15),
                hovertemplate="%.3f" % aut  # Custom hover text with 3 decimals
            )
        )

    # Update layout
    fig1.update_layout(
        title=dict(text=f"Abhängigkeitsindex ({metric_type}): <br>{selected_label}", font=dict(size=14)),
        # yaxis=dict(range=[y_min, y_max]),
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


        
        # --- Add DataSheet at the bottom ---

        displayed_countries = sorted(list(G.nodes))

        df_datasheet = pd.DataFrame({
            "Country": displayed_countries,
            "EU27": ["✅" if n in eu_nodes else "" for n in displayed_countries],
            "Normal Value": [df_product[df_product["to"] == n]["value"].sum() if n in df_product["to"].values else np.nan for n in displayed_countries],
            "Risk-weighted Value": [
                (df_product[df_product["to"] == n].apply(lambda row: row["value"] * (row["ps_norm"] if not pd.isna(row["ps_norm"]) else 0), axis=1).sum())
                if "ps_norm" in df_product.columns and n in df_product["to"].values else np.nan
                for n in displayed_countries
            ],
            "Raw Hub Score": [hubs_raw_import_full.get(n, np.nan) for n in displayed_countries],
            "Risk-weighted Hub Score": [hubs_risk_import_full.get(n, np.nan) for n in displayed_countries],
            #"ps_norm risk value": [
            #    df_product[df_product["from"] == n]["ps_norm"].iloc[0] if "ps_norm" in df_product.columns and n in df_product["from"].values and len(df_product[df_product["from"] == n]["ps_norm"]) > 0 else np.nan
            #    for n in displayed_countries
            #],

        })


        st.markdown("### Übersicht aller Länder (normalisiert am Gesamtnetz)")
        st.dataframe(df_datasheet, hide_index=True)


# eu_hub_raw_vals = [hub_raw_vals[eu_nodes.index(n)] for n in eu_nodes if n in eu_nodes and not pd.isna(hub_raw_vals[eu_nodes.index(n)])]
# print("Min (EU, unweighted):", np.nanmin(eu_hub_raw_vals), "Max (EU, unweighted):", np.nanmax(eu_hub_raw_vals))