# Final Combined app.py

import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import pyodbc
import hashlib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# ----------------------------------------------
# Helper functions for password hashing
# ----------------------------------------------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# ----------------------------------------------
# Data Loader (Merged from data_loader.py)
# ----------------------------------------------
@st.cache_data(ttl=600)
def load_data():
    server = 'newretailserver123.database.windows.net'
    database = 'RetailDB'
    username = 'azureuser'
    password = 'YourStrongP@ssw0rd'
    driver = '{ODBC Driver 18 for SQL Server}'
    conn_str = (
        f'DRIVER={driver};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
        'Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;'
    )
    conn = pyodbc.connect(conn_str)
    df_transactions = pd.read_sql("SELECT * FROM Transactions", conn)
    df_households = pd.read_sql("SELECT * FROM Households", conn)
    df_products = pd.read_sql("SELECT * FROM Products", conn)
    conn.close()

    df_transactions.columns = df_transactions.columns.str.strip()
    df_households.columns = df_households.columns.str.strip()
    df_products.columns = df_products.columns.str.strip()

    return df_transactions, df_households, df_products

# ----------------------------------------------
# Search Household Function (Merged from search_household.py)
# ----------------------------------------------
def search_household(df, hshd_num):
    return df[df['hshd_num'] == hshd_num]

# ----------------------------------------------
# Streamlit App
# ----------------------------------------------

st.set_page_config(page_title="🍭 Retail Insights Dashboard", layout="wide")

if 'user_db' not in st.session_state:
    st.session_state.user_db = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# ----------------------------------------------
# Login/Signup System
# ----------------------------------------------
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

login_signup()

# ----------------------------------------------
# Main App After Login
# ----------------------------------------------
if st.session_state.authenticated:

    st.title("📊 Retail Customer Analytics Dashboard")

    df_transactions, df_households, df_products = load_data()

    df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
    df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
    df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)

    full_df = df_transactions.merge(df_households, on='hshd_num', how='left')
    full_df = full_df.merge(df_products, on='product_num', how='left')

    df_transactions['date'] = pd.to_datetime(df_transactions['YEAR'].astype(str) + df_transactions['WEEK_NUM'].astype(str) + '0', format='%Y%U%w')

    tab1, tab2, tab3 = st.tabs(["📈 Insights", "⚡️ ML Predictions", "🔍 Household Search"])

    # ----------------------------------------------
    # Tab 1: Insights
    # ----------------------------------------------
    with tab1:
        st.header("📈 Customer Engagement Over Time")
        weekly_engagement = df_transactions.groupby(df_transactions['date'].dt.to_period('W'))['SPEND'].sum().reset_index()
        weekly_engagement['ds'] = weekly_engagement['date'].dt.start_time
        st.line_chart(weekly_engagement.set_index('ds')['SPEND'])

        st.header("👨‍👩‍👧 Demographics and Engagement")
        selected_demo = st.selectbox("Segment by:", ['INCOME_RANGE', 'AGE_RANGE', 'CHILDREN'])
        demo_spending = full_df.groupby(selected_demo)['SPEND'].sum().reset_index()
        st.bar_chart(demo_spending.rename(columns={selected_demo: 'index'}).set_index('index'))

        st.header("🌟 Loyalty Program Effect")
        if 'LOYALTY_FLAG' in df_households.columns:
            loyalty = full_df.groupby('LOYALTY_FLAG')['SPEND'].agg(['sum', 'mean']).reset_index()
            st.dataframe(loyalty)

        st.header("🛂 Basket Analysis")
        basket = df_transactions.groupby(['BASKET_NUM', 'product_num'])['SPEND'].sum().reset_index()
        top_products = basket.groupby('product_num')['SPEND'].sum().nlargest(10).reset_index()
        top_products = top_products.merge(df_products, on='product_num', how='left')
        if 'COMMODITY' in top_products.columns:
            st.bar_chart(top_products.set_index('COMMODITY')['SPEND'])

        st.header("📆 Seasonal Spending Patterns")
        df_transactions['month'] = df_transactions['date'].dt.month_name()
        seasonal = df_transactions.groupby('month')['SPEND'].sum().reset_index()
        seasonal['month'] = pd.Categorical(seasonal['month'], categories=[
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ], ordered=True)
        seasonal = seasonal.sort_values('month')
        st.bar_chart(seasonal.set_index('month'))

    # ----------------------------------------------
    # Tab 2: Machine Learning
    # ----------------------------------------------
    with tab2:
        st.header("⚡️ Churn Prediction")

        now = df_transactions['date'].max()
        rfm = df_transactions.groupby('hshd_num').agg(
            recency=('date', lambda x: (now - x.max()).days),
            frequency=('BASKET_NUM', 'nunique'),
            monetary=('SPEND', 'sum')
        )
        rfm['churn'] = (rfm['recency'] > 84).astype(int)

        X = rfm[['recency', 'frequency', 'monetary']]
        y = rfm['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        st.write("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                  columns=['Predicted Not Churn', 'Predicted Churn'],
                                  index=['Actual Not Churn', 'Actual Churn']))

        feat_imp = pd.Series(clf.feature_importances_, index=['Recency', 'Frequency', 'Monetary'])
        st.bar_chart(feat_imp)

    # ----------------------------------------------
    # Tab 3: Search Household
    # ----------------------------------------------
    with tab3:
        st.header("🔍 Household Search")
        household_id = st.text_input("Enter Household ID:")
        if st.button("Search"):
            result = search_household(full_df, int(household_id))
            if not result.empty:
                st.dataframe(result)
            else:
                st.warning("No records found for the entered Household ID.")
