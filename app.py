# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import hashlib
from sqlalchemy import text  # For raw SQL queries

# --- Authentication (simple, local memory) ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

if 'user_db' not in st.session_state:
    st.session_state.user_db = {"admin": {"email": "gampara@mail.uc.edu", "password": make_hashes("admin123")}}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login_signup():
    if not st.session_state.authenticated:
        auth_option = st.sidebar.selectbox("Login or Signup", ["Login", "Signup"])
        if auth_option == "Signup":
            st.sidebar.subheader("Create Account")
            new_user = st.sidebar.text_input("Username")
            new_email = st.sidebar.text_input("Email")
            new_password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button("Signup"):
                if new_user and new_password:
                    if new_user in st.session_state.user_db:
                        st.sidebar.error("Username already exists.")
                    else:
                        hashed_pw = make_hashes(new_password)
                        st.session_state.user_db[new_user] = {"email": new_email, "password": hashed_pw}
                        st.sidebar.success("Signup successful. Please login.")
                else:
                    st.sidebar.error("Username and password cannot be empty.")
        elif auth_option == "Login":
            st.sidebar.subheader("Login")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button("Login"):
                user = st.session_state.user_db.get(username)
                if user and check_hashes(password, user['password']):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")

# --- Main App Logic ---
st.set_page_config(page_title="Retail Analytics App", layout="wide")

login_signup()

if not st.session_state.authenticated:
    st.warning("Please log in to access the application.")
    st.stop()

# --- Database Connection Function ---
def get_db_connection():
    try:
        conn = st.connection(
            "sql",
            type="sql",
            dialect="mssql",
            username=st.secrets.db_credentials.username,
            password=st.secrets.db_credentials.password,
            host=st.secrets.db_credentials.server,
            port=1433,
            database=st.secrets.db_credentials.database,
            query={
                "driver": "ODBC Driver 17 for SQL Server",
                "Encrypt": "yes",
                "TrustServerCertificate": "yes",
                "ConnectionTimeout": "30",
            }
        )
        conn.query("SELECT 1")
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.error("Please ensure your database credentials (username, password, server, database) are correctly configured in Streamlit Secrets.")
        st.info("Secrets can be set in your local `.streamlit/secrets.toml` file or in the Streamlit Community Cloud app settings.")
        st.code("""
[db_credentials]
username = "azureuser"
password = "YourStrongP%40ssw0rd"
server = "newretailserver123.database.windows.net"
database = "RetailDB"
        """, language="toml")
        return None

# --- Establish Connection ---
conn = get_db_connection()

if conn:
    st.success("Database connection established successfully!")
else:
    st.error("Database connection could not be established.")
    st.stop()

# --- Data Loading Function ---
@st.cache_data(ttl=600)
def load_data(_conn):
    if _conn is None:
        st.error("Cannot load data: Database connection is not available.")
        return None, None, None
    try:
        limit = 10000
        df_transactions = _conn.query(f"SELECT TOP {limit} * FROM Transactions", ttl=600)
        df_households = _conn.query(f"SELECT TOP {limit} * FROM Households", ttl=600)
        df_products = _conn.query(f"SELECT TOP {limit} * FROM Products", ttl=600)

        df_transactions.columns = df_transactions.columns.str.strip()
        df_households.columns = df_households.columns.str.strip()
        df_products.columns = df_products.columns.str.strip()

        return df_transactions, df_households, df_products
    except Exception as e:
        st.error(f"Failed to load data from database: {e}")
        return None, None, None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
if conn:
    page = st.sidebar.radio("Go to", ["Dashboard", "Household Search", "CLV Calculation", "Data Loader"])
else:
    st.sidebar.warning("Database connection failed. Cannot navigate.")
    st.stop()

# --- Page Implementations ---

# --- Load Data (if not already loaded) ---
df_transactions, df_households, df_products = load_data(conn)

if df_transactions is None:
    st.error("Failed to load data. Please check database or data availability.")
    st.stop()

# --- Logic for Each Page ---
if page == "Dashboard":
    st.title("📊 Retail Analytics Dashboard")

    st.subheader("Transaction Sample")
    st.dataframe(df_transactions.head(), use_container_width=True)

    st.subheader("Spend Over Time")
    if "DATE" in df_transactions.columns:
        df_transactions["DATE"] = pd.to_datetime(df_transactions["DATE"], errors="coerce")
        spend_over_time = df_transactions.groupby(df_transactions["DATE"].dt.to_period('M')).sum(numeric_only=True)
        spend_over_time.index = spend_over_time.index.to_timestamp()
        st.line_chart(spend_over_time["SPEND"])
    else:
        st.warning("DATE column missing in Transactions data.")

elif page == "Household Search":
    st.title("🏠 Household Search")

    selected_household = st.selectbox("Select Household Number", df_households["HSHD_NUM"].unique())

    household_data = df_households[df_households["HSHD_NUM"] == selected_household]
    st.write("Household Details:")
    st.dataframe(household_data)

elif page == "CLV Calculation":
    st.title("💰 Customer Lifetime Value (CLV) Calculation")

    if "HSHD_NUM" in df_transactions.columns and "SPEND" in df_transactions.columns:
        clv_df = df_transactions.groupby("HSHD_NUM")["SPEND"].sum().reset_index()
        st.dataframe(clv_df.head(), use_container_width=True)
        st.success("Simple CLV calculated as total spend per household.")
    else:
        st.warning("Required columns missing in Transactions data.")

elif page == "Data Loader":
    st.title("🛠️ Data Loader")
    st.write("**Transactions Data:**")
    st.dataframe(df_transactions.head(), use_container_width=True)

    st.write("**Households Data:**")
    st.dataframe(df_households.head(), use_container_width=True)

    st.write("**Products Data:**")
    st.dataframe(df_products.head(), use_container_width=True)
