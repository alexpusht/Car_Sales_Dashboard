import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Car Sales Dashboard", page_icon="📊")

# Data Laden
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Car Sales.xlsx - car_data.csv')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        # Aannemen quantity_sold = 1 per verkoop
        df['quantity_sold'] = 1
        # Converteer prijs naar €
        df['Price (€)'] = df['Price ($)'] * 0.85
        return df
    except FileNotFoundError:
        st.error("Bestand niet gevonden.")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# Sidebar Filters
st.sidebar.title("Filters")
date_range = st.sidebar.date_input("Selecteer datumrange", [df['Date'].min(), df['Date'].max()])
companies = st.sidebar.multiselect("Selecteer merk(en)", df['Company'].unique())
regions = st.sidebar.multiselect("Selecteer regio('s)", df['Dealer_Region'].unique())
genders = st.sidebar.multiselect("Selecteer geslacht(en)", df['Gender'].unique())

# Filter Data
filtered_df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]
if companies:
    filtered_df = filtered_df[filtered_df['Company'].isin(companies)]
if regions:
    filtered_df = filtered_df[filtered_df['Dealer_Region'].isin(regions)]
if genders:
    filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_revenue = filtered_df['Price (€)'].sum()
    st.metric("Totale Omzet", f"€{total_revenue:,.2f}")
with col2:
    total_transactions = len(filtered_df)
    st.metric("Aantal Transacties", total_transactions)
with col3:
    total_items = filtered_df['quantity_sold'].sum()
    st.metric("Totale Verkochte Items", total_items)
with col4:
    avg_transaction = filtered_df['Price (€)'].mean()
    st.metric("Gemiddelde Transactiewaarde", f"€{avg_transaction:,.2f}")

# Vergelijking met Vorige Periode (30 dagen)
prev_start = pd.to_datetime(date_range[0]) - pd.Timedelta(days=30)
prev_end = pd.to_datetime(date_range[0]) - pd.Timedelta(days=1)
prev_df = df[(df['Date'] >= prev_start) & (df['Date'] <= prev_end)]
if not prev_df.empty:
    prev_revenue = prev_df['Price (€)'].sum()
    growth = ((total_revenue - prev_revenue) / prev_revenue) * 100 if prev_revenue > 0 else 0
    st.metric("Omzet Groei vs Vorige Periode", f"{growth:.2f}%")

# Visualisaties
st.header("Visualisaties")
col1, col2 = st.columns(2)
with col1:
    # Bar chart verkoop per bedrijf
    company_sales = filtered_df.groupby('Company')['Price (€)'].sum().sort_values(ascending=False)
    fig = px.bar(company_sales, x=company_sales.index, y=company_sales.values, title="Verkoop per Bedrijf", color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(fig)
with col2:
    # Bar chart verkoop per regio
    region_sales = filtered_df.groupby('Dealer_Region')['Price (€)'].sum().sort_values(ascending=False)
    fig = px.bar(region_sales, x=region_sales.index, y=region_sales.values, title="Verkoop per Regio", color_discrete_sequence=px.colors.sequential.Greens)
    st.plotly_chart(fig)

# Line chart verkoop over tijd
daily_sales = filtered_df.groupby(filtered_df['Date'].dt.date)['Price (€)'].sum()
fig = px.line(daily_sales, x=daily_sales.index, y=daily_sales.values, title="Verkoop over Tijd")
st.plotly_chart(fig)

col1, col2 = st.columns(2)
with col1:
    # Pie chart per geslacht
    gender_sales = filtered_df.groupby('Gender')['Price (€)'].sum()
    fig = px.pie(gender_sales, values=gender_sales.values, names=gender_sales.index, title="Verkoop per Geslacht")
    st.plotly_chart(fig)
with col2:
    # Pie chart per transmissie
    trans_sales = filtered_df.groupby('Transmission')['Price (€)'].sum()
    fig = px.pie(trans_sales, values=trans_sales.values, names=trans_sales.index, title="Verkoop per Transmissie")
    st.plotly_chart(fig)

# Bar chart verkoop per kleur (in plaats van loyaliteit)
color_sales = filtered_df.groupby('Color')['Price (€)'].sum().sort_values(ascending=False)
fig = px.bar(color_sales, x=color_sales.index, y=color_sales.values, title="Verkoop per Kleur", color_discrete_sequence=px.colors.sequential.Reds)
st.plotly_chart(fig)

# Top 10 modellen
top_models = filtered_df.groupby('Model')['Price (€)'].sum().sort_values(ascending=False).head(10)
fig = px.bar(top_models, x=top_models.values, y=top_models.index, orientation='h', title="Top 10 Modellen op Omzet")
st.plotly_chart(fig)

# Business Aanbevelingen
st.header("Business Aanbevelingen")
recommendations = []
if not company_sales.empty:
    top_company = company_sales.idxmax()
    recommendations.append(f"Focus op {top_company} voor hogere verkoop.")
if not region_sales.empty:
    top_region = region_sales.idxmax()
    recommendations.append(f"Vergroot aanwezigheid in {top_region}.")
recommendations.append("Overweeg promoties voor top modellen.")
recommendations.append("Monitor voorraad voor populaire modellen.")
recommendations.append("Target hogere inkomensgroepen voor premium modellen.")
for rec in recommendations[:5]:
    st.info(rec)

# Customer Segmentation
st.header("Klant Segmentatie")
# K-means op Annual Income
income = df['Annual Income'].dropna().values.reshape(-1, 1)
if len(income) > 0:
    scaler = StandardScaler()
    income_scaled = scaler.fit_transform(income)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(income_scaled)
    df_seg = df.dropna(subset=['Annual Income']).copy()
    df_seg['Cluster'] = clusters
    # Scatter plot
    fig = px.scatter(df_seg, x='Annual Income', y='Price (€)', color='Cluster', title="Klant Segmenten")
    st.plotly_chart(fig)
    # Segment kenmerken
    for i in range(3):
        seg = df_seg[df_seg['Cluster'] == i]
        if not seg.empty:
            avg_income = seg['Annual Income'].mean()
            avg_price = seg['Price (€)'].mean()
            st.write(f"Segment {i}: Gemiddeld Inkomen €{avg_income:,.2f}, Gemiddelde Prijs €{avg_price:,.2f}")
else:
    st.write("Niet genoeg data voor segmentatie.")

# Detail Tabel
st.header("Detail Tabel")
st.dataframe(filtered_df.head(100))

# Footer
st.write(f"Data range: {date_range[0]} tot {date_range[1]}")