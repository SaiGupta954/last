# app.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse

# --- Streamlit Secrets Configuration ---
# Store these in .streamlit/secrets.toml:
"""
[connections.sql]
url = "mssql+pyodbc://azureuser:YourStrongP%40ssw0rd@newretailserver123.database.windows.net:1433/RetailDB?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
"""

# --- Database Connection using Streamlit's Built-in SQL Connection ---
@st.cache_resource
def get_db_connection():
    try:
        conn = st.connection("sql", type="sql")
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# --- Data Loading with Fallback to Sample Data ---
@st.cache_data(ttl=600)
def load_data():
    conn = get_db_connection()
    if conn:
        try:
            df_transactions = conn.query("SELECT TOP 1000 * FROM Transactions", ttl=600)
            df_households = conn.query("SELECT TOP 1000 * FROM Households", ttl=600)
            df_products = conn.query("SELECT TOP 1000 * FROM Products", ttl=600)
            
            # Clean column names
            df_transactions.columns = df_transactions.columns.str.strip()
            df_households.columns = df_households.columns.str.strip()
            df_products.columns = df_products.columns.str.strip()
            
            return df_transactions, df_households, df_products
            
        except Exception as e:
            st.error(f"Query failed: {e}")
    
    # Fallback to sample data
    st.warning("Using sample data as fallback")
    df_transactions = pd.read_csv("400_transactions.csv")
    df_households = pd.read_csv("400_households.csv")
    df_products = pd.read_csv("400_products.csv")
    return df_transactions, df_households, df_products

# --- Main App ---
st.set_page_config(page_title="Retail Analytics", layout="wide")

# Navigation
page = st.sidebar.radio("Go to", ["Dashboard", "Raw Data"])

if page == "Dashboard":
    st.title("📊 Retail Analytics Dashboard")
    
    # Load data
    df_transactions, df_households, df_products = load_data()
    
    # Show key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Households", df_households.shape[0])
    col2.metric("Total Transactions", df_transactions.shape[0])
    col3.metric("Unique Products", df_products['PRODUCT_NUM'].nunique())
    
    # Visualization 1: Spending Trends
    st.subheader("Weekly Spending Pattern")
    weekly_spending = df_transactions.groupby('WEEK_NUM')['SPEND'].sum()
    st.line_chart(weekly_spending)
    
    # Visualization 2: Top Products
    st.subheader("Top 10 Products by Sales")
    product_sales = df_transactions.merge(df_products, on='PRODUCT_NUM')
    top_products = product_sales.groupby('COMMODITY')['SPEND'].sum().nlargest(10)
    st.bar_chart(top_products)

elif page == "Raw Data":
    st.title("📁 Raw Data Explorer")
    
    df_transactions, df_households, df_products = load_data()
    
    st.subheader("Transactions Data")
    st.dataframe(df_transactions.head(100))
    
    st.subheader("Households Data")
    st.dataframe(df_households.head(100))
    
    st.subheader("Products Data")
    st.dataframe(df_products.head(100))
