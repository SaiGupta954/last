# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import hashlib
from sqlalchemy import text # Import text for raw SQL execution with parameters

# --- Authentication (simple, local memory) ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

if 'user_db' not in st.session_state:
    # In a real app, load/save this from a persistent store
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
                    st.rerun() # Rerun the script to reflect login state
                else:
                    st.sidebar.error("Invalid credentials")

# --- Main App Logic ---
st.set_page_config(page_title="Retail Analytics App", layout="wide")

login_signup() # Display login/signup in sidebar

# Stop execution if not authenticated
if not st.session_state.authenticated:
    st.warning("Please log in to access the application.")
    st.stop()

# --- Database Connection using Streamlit's SQLConnection ---
# This relies on .streamlit/secrets.toml
# Ensure secrets.toml uses 'ODBC Driver 17 for SQL Server'
@st.cache_resource # Cache the connection object itself
def get_db_connection():
    try:
        # Use st.connection which reads from secrets.toml by default
        conn = st.connection("sql", type="sql")
        # Test connection with a simple query
        conn.query("SELECT 1")
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.error("Please ensure your `.streamlit/secrets.toml` file is correctly configured with the connection URL.")
        st.code("""
[connections.sql]
url = "mssql+pyodbc://<user>:<password_encoded>@<server>:1433/<db>?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes&ConnectionTimeout=30"
        """, language="toml")
        return None

conn = get_db_connection()

# --- Data Loading Function (uses cached connection) ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_data(_conn): # Pass connection object
    if _conn is None:
        st.error("Cannot load data: Database connection is not available.")
        return None, None, None
    try:
        # Load only a subset for performance in the dashboard view initially
        limit = 10000 # Adjust as needed
        df_transactions = _conn.query(f"SELECT TOP {limit} * FROM Transactions", ttl=600)
        df_households = _conn.query(f"SELECT TOP {limit} * FROM Households", ttl=600)
        df_products = _conn.query(f"SELECT TOP {limit} * FROM Products", ttl=600)

        # Clean column names immediately after loading
        df_transactions.columns = df_transactions.columns.str.strip()
        df_households.columns = df_households.columns.str.strip()
        df_products.columns = df_products.columns.str.strip()

        return df_transactions, df_households, df_products
    except Exception as e:
        st.error(f"Failed to load data from database: {e}")
        return None, None, None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
# Only show navigation if connection is successful
if conn:
    page = st.sidebar.radio("Go to", ["Dashboard", "Household Search", "CLV Calculation", "Data Loader"])
else:
    st.sidebar.warning("Database connection failed. Cannot navigate.")
    st.stop() # Stop further execution if DB connection failed

# --- Page Implementations ---

if page == "Dashboard":
    st.title("📊 Retail Customer Analytics Dashboard")
    df_transactions, df_households, df_products = load_data(conn)

    if df_transactions is None or df_households is None or df_products is None:
        st.warning("Could not load data for the dashboard.")
    else:
        # --- Data Preparation ---
        try:
            df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
            df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
            df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)
            full_df = df_transactions.merge(df_households, on='hshd_num', how='left')
            full_df = full_df.merge(df_products, on='product_num', how='left')
            # Ensure date conversion happens safely
            full_df['date'] = pd.to_datetime(
                full_df['YEAR'].astype(str) + full_df['WEEK_NUM'].astype(str) + '0',
                format='%Y%U%w', errors='coerce' # Coerce errors to NaT
            )
            full_df.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed
        except Exception as e:
            st.error(f"Error during data preparation: {e}")
            st.stop()


        # --- Dashboard Widgets and Charts ---
        st.header("📈 Customer Engagement Over Time")
        weekly_engagement = full_df.groupby(full_df['date'].dt.to_period('W'))['SPEND'].sum().reset_index()
        weekly_engagement['ds'] = weekly_engagement['date'].dt.start_time
        st.line_chart(weekly_engagement.set_index('ds')['SPEND'])

        st.header("👨‍👩‍👧 Demographics and Engagement")
        demo_cols = [col for col in ['INCOME_RANGE', 'AGE_RANGE', 'CHILDREN'] if col in full_df.columns]
        if demo_cols:
            selected_demo = st.selectbox("Segment by:", demo_cols)
            # Handle potential missing values before grouping
            demo_spending = full_df.dropna(subset=[selected_demo]).groupby(selected_demo)['SPEND'].sum().reset_index()
            st.bar_chart(demo_spending.rename(columns={selected_demo: 'index'}).set_index('index'))
        else:
            st.warning("Demographic columns not found for segmentation.")

        st.header("🔍 Customer Segmentation (Top Spenders)")
        segmentation = full_df.groupby(['hshd_num']).agg(
            CLV=('SPEND', 'sum'),
            Income=('INCOME_RANGE', 'first'), # Get first non-null value
            Age=('AGE_RANGE', 'first')
        ).reset_index()
        st.dataframe(segmentation.sort_values(by='CLV', ascending=False).head(10))

        # --- Add other dashboard sections similarly, ensuring column names exist ---
        # Example: Loyalty Program
        if 'LOYALTY_FLAG' in full_df.columns:
            st.header("🌟 Loyalty Program Effect")
            loyalty = full_df.groupby('LOYALTY_FLAG')['SPEND'].agg(['sum', 'mean']).reset_index()
            st.dataframe(loyalty)

        # Example: Basket Analysis
        if 'COMMODITY' in full_df.columns and 'BASKET_NUM' in full_df.columns and 'product_num' in full_df.columns:
             st.header("🧺 Basket Analysis (Top Commodities)")
             top_commodities = full_df.groupby('COMMODITY')['SPEND'].sum().nlargest(10).reset_index()
             st.bar_chart(top_commodities.set_index('COMMODITY')['SPEND'])
             fig_pie = px.pie(top_commodities, values='SPEND', names='COMMODITY', title='Spending Distribution by Top Commodities')
             st.plotly_chart(fig_pie)
        else:
             st.warning("Required columns for Basket Analysis (COMMODITY, BASKET_NUM, product_num) not found.")


        # --- ML Tabs ---
        st.subheader("🤖 Machine Learning Insights")
        tab1, tab2 = st.tabs(["⚠️ Churn Prediction", "🧺 Basket Analysis"])

        with tab1:
            # Churn Prediction (RFM based)
            st.header("ML Churn Prediction: At-Risk Customers")
            if not full_df.empty and 'date' in full_df.columns:
                try:
                    now = full_df['date'].max()
                    rfm = full_df.groupby('hshd_num').agg(
                        recency=('date', lambda x: (now - x.max()).days if pd.notna(x.max()) else 9999),
                        frequency=('BASKET_NUM', 'nunique'),
                        monetary=('SPEND', 'sum')
                    )
                    # Define churn based on recency (e.g., > 84 days)
                    rfm['churn'] = (rfm['recency'] > 84).astype(int)

                    X = rfm[['recency', 'frequency', 'monetary']]
                    y = rfm['churn']

                    if len(X) > 10 and len(y.unique()) > 1: # Need enough data and both classes for meaningful split/train
                        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

                        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Added balanced weight
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        st.write("**Classification Report:**")
                        st.text(classification_report(y_test, y_pred, zero_division=0)) # Handle zero division

                        st.write("**Confusion Matrix:**")
                        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                                  columns=['Predicted Not Churn', 'Predicted Churn'],
                                                  index=['Actual Not Churn', 'Actual Churn']))

                        feat_imp = pd.Series(clf.feature_importances_, index=['Recency', 'Frequency', 'Monetary']).sort_values(ascending=False)
                        st.write("**Feature Importances:**")
                        st.bar_chart(feat_imp)
                    else:
                         st.warning("Not enough data or classes to train churn model.")
                except Exception as e:
                    st.error(f"Error during churn prediction model training: {e}")
            else:
                st.warning("Insufficient data for churn prediction.")

        with tab2:
            # Basket Spend Prediction
            st.header("Basket Analysis - Predicting Total Spend")
            if not full_df.empty:
                try:
                    # Use relevant categorical features for prediction
                    features_to_encode = ['DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']
                    available_features = [f for f in features_to_encode if f in full_df.columns]

                    if available_features:
                        basket_features_df = full_df[['BASKET_NUM', 'SPEND'] + available_features].copy()
                        basket_features_df.dropna(subset=['BASKET_NUM'] + available_features, inplace=True) # Drop rows with NaNs in features

                        basket_features_encoded = pd.get_dummies(basket_features_df, columns=available_features, dummy_na=False) # Encode features

                        # Aggregate features per basket
                        X_basket = basket_features_encoded.drop(columns=['SPEND']).groupby('BASKET_NUM').sum()
                        y_basket = basket_features_encoded.groupby('BASKET_NUM')['SPEND'].sum() # Sum spend per basket

                        if len(X_basket) > 10: # Need enough data
                            X_train, X_test, y_train, y_test = train_test_split(X_basket, y_basket, test_size=0.2, random_state=42)

                            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use parallel jobs
                            rf.fit(X_train, y_train)
                            y_pred = rf.predict(X_test)

                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            st.write(f"**R² Score:** {r2:.3f}")
                            st.write(f"**MSE:** {mse:.2f}")

                            st.subheader("Predicted vs. Actual Basket Spend (Test Set Sample)")
                            chart_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).sample(min(100, len(y_test))) # Show sample
                            st.line_chart(chart_df.reset_index(drop=True))

                            # Feature importance
                            importances = pd.Series(rf.feature_importances_, index=X_basket.columns).sort_values(ascending=False)
                            top_features = importances.head(15) # Show more features
                            st.write("**Top Drivers of Basket Spend:**")
                            st.bar_chart(top_features)
                        else:
                            st.warning("Not enough basket data to train prediction model.")
                    else:
                        st.warning("No suitable features found for basket spend prediction.")
                except Exception as e:
                    st.error(f"Error during basket analysis model training: {e}")
            else:
                st.warning("Insufficient data for basket analysis.")


elif page == "Household Search":
    st.title("Search Household Transactions")
    hshd_num_input = st.text_input("Enter Household Number (HSHD_NUM):")

    if hshd_num_input:
        try:
            hshd_num = int(hshd_num_input)
            # Use parameterized query to prevent SQL injection
            query = text("""
            SELECT
                H.HSHD_NUM, T.BASKET_NUM, T.PURCHASE_,
                P.PRODUCT_NUM, P.DEPARTMENT, P.COMMODITY
            FROM Households H
            JOIN Transactions T ON H.HSHD_NUM = T.HSHD_NUM
            LEFT JOIN Products P ON T.PRODUCT_NUM = P.PRODUCT_NUM
            WHERE H.HSHD_NUM = :hnum
            ORDER BY T.PURCHASE_ DESC, T.BASKET_NUM;
            """)
            # Execute using conn.query with params
            data = conn.query(str(query), params={"hnum": hshd_num})

            if not data.empty:
                st.dataframe(data)
            else:
                st.write("No data found for the entered Household Number.")
        except ValueError:
            st.error("Please enter a valid numeric Household Number.")
        except Exception as e:
            st.error(f"An error occurred during search: {e}")

elif page == "CLV Calculation":
    st.title("Calculate and Save CLV")
    st.write("This calculates Customer Lifetime Value (sum of SPEND) per household and saves it to the `clv_results` table.")

    if st.button("Run CLV Calculation and Save to SQL"):
        try:
            # 1. Calculate CLV
            clv_query = """
            SELECT HSHD_NUM, SUM(SPEND) AS CLV
            FROM transactions
            GROUP BY HSHD_NUM
            """
            clv_df = conn.query(clv_query) # Use st.connection query

            # 2. Use conn.session for DML (CREATE, DELETE, INSERT)
            with conn.session as s:
                # 3. Create table if not exists (idempotent)
                s.execute(text("""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'clv_results')
                BEGIN
                    CREATE TABLE clv_results (
                        HSHD_NUM INT PRIMARY KEY,
                        CLV FLOAT
                    );
                    PRINT 'Table clv_results created.';
                END
                ELSE
                BEGIN
                    PRINT 'Table clv_results already exists.';
                END;
                """))
                s.commit() # Commit table creation

                # 4. Clear existing data before inserting new data (optional, depends on desired behavior)
                delete_stmt = text("DELETE FROM clv_results")
                s.execute(delete_stmt)
                st.write(f"Deleted existing rows from clv_results.")

                # 5. Insert new CLV data using parameterized query
                insert_stmt = text("INSERT INTO clv_results (HSHD_NUM, CLV) VALUES (:hnum, :clv_val)")
                inserted_count = 0
                for _, row in clv_df.iterrows():
                    try:
                        s.execute(insert_stmt, {"hnum": int(row['HSHD_NUM']), "clv_val": float(row['CLV'])})
                        inserted_count += 1
                    except Exception as insert_error:
                        st.warning(f"Could not insert row for HSHD_NUM {row['HSHD_NUM']}: {insert_error}")
                        s.rollback() # Rollback the failed insert, but continue loop
                        s.begin() # Start a new transaction context for the next iteration

                s.commit() # Commit all successful inserts
            st.success(f"✅ CLV results processed. {inserted_count}/{len(clv_df)} rows saved to `clv_results` table in Azure SQL!")
            st.dataframe(clv_df.head())

        except Exception as e:
            st.error(f"An error occurred during CLV calculation/saving: {e}")
            # Check if the error might be due to transaction log full
            if "transaction log for database" in str(e).lower() and "is full" in str(e).lower():
                st.warning("The database transaction log might be full. Consider increasing log size or running backups.")


elif page == "Data Loader":
    st.title("Data Loader")
    st.write("Load data from Database or upload CSV files.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Load from Database")
        if st.button("Load Latest Data from Database"):
            # This button simply triggers a reload using the existing load_data function
            # The results will be cached or re-fetched based on the TTL
            st.cache_data.clear() # Clear cache to force reload
            st.cache_resource.clear() # Clear connection cache if needed
            st.success("Cleared cache. Data will be reloaded from the database on the next page view or interaction.")
            st.rerun() # Force a rerun to reflect the reload

    with col2:
        st.subheader("Upload CSV Files")
        uploaded_transactions = st.file_uploader("Upload Transactions CSV", type="csv")
        uploaded_households = st.file_uploader("Upload Households CSV", type="csv")
        uploaded_products = st.file_uploader("Upload Products CSV", type="csv")

        # Store uploaded data in session state if needed for processing
        if uploaded_transactions:
            st.session_state['uploaded_transactions'] = pd.read_csv(uploaded_transactions)
            st.success("Transactions CSV uploaded.")
        if uploaded_households:
            st.session_state['uploaded_households'] = pd.read_csv(uploaded_households)
            st.success("Households CSV uploaded.")
        if uploaded_products:
            st.session_state['uploaded_products'] = pd.read_csv(uploaded_products)
            st.success("Products CSV uploaded.")

    # Display previews if data loaded from DB or Upload
    st.subheader("Data Preview")
    df_t, df_h, df_p = load_data(conn) # Load data (cached or fresh)

    if df_t is not None:
        st.write("Transactions Preview (from DB):")
        st.dataframe(df_t.head())
    elif 'uploaded_transactions' in st.session_state:
        st.write("Transactions Preview (Uploaded CSV):")
        st.dataframe(st.session_state['uploaded_transactions'].head())

    if df_h is not None:
        st.write("Households Preview (from DB):")
        st.dataframe(df_h.head())
    elif 'uploaded_households' in st.session_state:
         st.write("Households Preview (Uploaded CSV):")
         st.dataframe(st.session_state['uploaded_households'].head())

    if df_p is not None:
        st.write("Products Preview (from DB):")
        st.dataframe(df_p.head())
    elif 'uploaded_products' in st.session_state:
         st.write("Products Preview (Uploaded CSV):")
         st.dataframe(st.session_state['uploaded_products'].head())
