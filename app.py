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

# --- Database Connection using Streamlit's SQLConnection (UPDATED METHOD) ---
@st.cache_resource # Cache the connection object itself
def get_db_connection():
    """
    Establishes and caches the database connection using st.connection
    with keyword arguments, retrieving credentials securely from Streamlit secrets.
    """
    try:
        # Define connection parameters using keyword arguments,
        # retrieving sensitive info from st.secrets.
        # Ensure your secrets are structured accordingly (e.g., in secrets.toml
        # or Streamlit Cloud secrets).
        # Example structure:
        # [db_credentials]
        # username = "your_azure_username"
        # password = "Your_URL_Encoded_Password" # Ensure password is URL encoded if needed
        # server = "your_server_name.database.windows.net"
        # database = "your_database_name"

        conn = st.connection(
            "sql",          # Connection name (used internally by Streamlit)
            type="sql",     # Specify the connection type [2][4]
            # --- Core Connection Details (obtained from st.secrets) ---
            dialect="mssql", # Specify the dialect for Microsoft SQL Server [4]
            username=st.secrets.db_credentials.username,
            password=st.secrets.db_credentials.password,
            host=st.secrets.db_credentials.server,
            port=1433,      # Default SQL Server port
            database=st.secrets.db_credentials.database,
            # --- Driver Specific Arguments ---
            # Passed via the 'query' parameter as a dictionary [4]
            query={
                "driver": "ODBC Driver 17 for SQL Server", # Recommended driver for Cloud compatibility [5]
                "Encrypt": "yes",
                "TrustServerCertificate": "yes", # Set to 'no' in production unless certificate is properly configured
                "ConnectionTimeout": "30",
            }
        )
        # Test connection with a simple query to ensure it's working
        conn.query("SELECT 1")
        # st.success("Database connection established successfully!") # Optional success message
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.error("Please ensure your database credentials (username, password, server, database) are correctly configured in Streamlit Secrets.")
        st.info("Secrets can be set in your local `.streamlit/secrets.toml` file or in the Streamlit Community Cloud application settings under a section like `[db_credentials]`.")
        # Example structure for secrets.toml:
        st.code("""
[db_credentials]
username = "your_azure_username"
password = "Your_URL_Encoded_Password"
server = "your_server_name.database.windows.net"
database = "your_database_name"
        """, language="toml")
        return None # Return None to indicate failure

conn = get_db_connection()

# --- Data Loading Function (uses cached connection) ---
# No changes needed here, it receives the 'conn' object
@st.cache_data(ttl=600) # Cache data for 10 minutes [5]
def load_data(_conn): # Pass connection object
    if _conn is None:
        st.error("Cannot load data: Database connection is not available.")
        return None, None, None
    try:
        # Load only a subset for performance in the dashboard view initially
        limit = 10000 # Adjust as needed
        # Use the connection object's query method [2]
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
# (No changes needed in the page implementations below, they depend on 'conn'
#  and 'load_data' which are handled above)

if page == "Dashboard":
    st.title("📊 Retail Customer Analytics Dashboard")
    df_transactions, df_households, df_products = load_data(conn)

    if df_transactions is None or df_households is None or df_products is None:
        st.warning("Could not load data for the dashboard.")
    else:
        # --- Data Preparation ---
        try:
            # Use lowercase and consistent naming convention
            df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
            df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
            df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)

            # Merge dataframes
            full_df = pd.merge(df_transactions, df_households, on='hshd_num', how='left')
            full_df = pd.merge(full_df, df_products, on='product_num', how='left')

            # Safely create date column
            if 'YEAR' in full_df.columns and 'WEEK_NUM' in full_df.columns:
                 # Ensure YEAR and WEEK_NUM are string type before concatenation
                 full_df['year_str'] = full_df['YEAR'].astype(str)
                 full_df['week_str'] = full_df['WEEK_NUM'].astype(str).str.zfill(2) # Pad week number if needed

                 # Construct date string assuming week starts on Sunday ('%Y%U%w' where w=0 for Sunday)
                 date_str = full_df['year_str'] + full_df['week_str'] + '0'
                 full_df['date'] = pd.to_datetime(date_str, format='%Y%U%w', errors='coerce')
                 full_df.drop(columns=['year_str', 'week_str'], inplace=True) # Remove temporary columns
                 full_df.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed
            else:
                 st.warning("Required columns 'YEAR' or 'WEEK_NUM' not found for date creation.")
                 full_df['date'] = pd.NaT # Assign NaT if columns missing


        except Exception as e:
            st.error(f"Error during data preparation: {e}")
            st.stop()


        # --- Dashboard Widgets and Charts ---
        if not full_df.empty and 'date' in full_df.columns and not full_df['date'].isnull().all():
            st.header("📈 Customer Engagement Over Time (Weekly Spend)")
            # Ensure date column is datetime before using dt accessor
            if pd.api.types.is_datetime64_any_dtype(full_df['date']):
                weekly_engagement = full_df.groupby(full_df['date'].dt.to_period('W'))['SPEND'].sum().reset_index()
                # Convert period to timestamp for plotting
                weekly_engagement['ds'] = weekly_engagement['date'].dt.start_time
                st.line_chart(weekly_engagement.set_index('ds')['SPEND'])
            else:
                st.warning("Date column is not in the expected format for time series analysis.")
        else:
            st.info("Insufficient data to display customer engagement over time.")


        st.header("👨‍👩‍👧 Demographics and Engagement")
        demo_cols = [col for col in ['INCOME_RANGE', 'AGE_RANGE', 'CHILDREN'] if col in full_df.columns]
        if demo_cols and not full_df.empty:
            selected_demo = st.selectbox("Segment by:", demo_cols)
            # Handle potential missing values before grouping
            demo_spending = full_df.dropna(subset=[selected_demo]).groupby(selected_demo)['SPEND'].sum().reset_index()
            if not demo_spending.empty:
                st.bar_chart(demo_spending.rename(columns={selected_demo: 'index'}).set_index('index'))
            else:
                st.warning(f"No spending data available for the selected demographic: {selected_demo}")
        elif not demo_cols:
            st.warning("Demographic columns (INCOME_RANGE, AGE_RANGE, CHILDREN) not found for segmentation.")
        else:
             st.info("Insufficient data for demographic segmentation.")

        st.header("🔍 Customer Segmentation (Top Spenders)")
        if not full_df.empty and 'hshd_num' in full_df.columns and 'SPEND' in full_df.columns:
            # Aggregate safely, using 'first' for potentially varying demographic info per household
            agg_dict = {'CLV': ('SPEND', 'sum')}
            if 'INCOME_RANGE' in full_df.columns: agg_dict['Income'] = ('INCOME_RANGE', 'first')
            if 'AGE_RANGE' in full_df.columns: agg_dict['Age'] = ('AGE_RANGE', 'first')

            segmentation = full_df.groupby('hshd_num').agg(**agg_dict).reset_index()
            st.dataframe(segmentation.sort_values(by='CLV', ascending=False).head(10))
        else:
             st.info("Insufficient data for customer segmentation.")


        # Example: Loyalty Program
        if 'LOYALTY_FLAG' in full_df.columns and not full_df.empty:
            st.header("🌟 Loyalty Program Effect")
            loyalty = full_df.groupby('LOYALTY_FLAG')['SPEND'].agg(['sum', 'mean']).reset_index()
            st.dataframe(loyalty)
        else:
            st.info("LOYALTY_FLAG column not found or insufficient data for loyalty analysis.")


        # Example: Basket Analysis
        basket_cols_present = all(col in full_df.columns for col in ['COMMODITY', 'BASKET_NUM', 'product_num', 'SPEND'])
        if basket_cols_present and not full_df.empty:
             st.header("🧺 Basket Analysis (Top Commodities by Spend)")
             top_commodities = full_df.groupby('COMMODITY')['SPEND'].sum().nlargest(10).reset_index()
             if not top_commodities.empty:
                st.bar_chart(top_commodities.set_index('COMMODITY')['SPEND'])
                fig_pie = px.pie(top_commodities, values='SPEND', names='COMMODITY', title='Spending Distribution by Top Commodities')
                st.plotly_chart(fig_pie)
             else:
                 st.info("No commodity spending data found.")
        elif not basket_cols_present:
             st.warning("Required columns for Basket Analysis (COMMODITY, BASKET_NUM, product_num, SPEND) not found.")
        else:
             st.info("Insufficient data for basket analysis.")


        # --- ML Tabs ---
        st.subheader("🤖 Machine Learning Insights")
        tab1, tab2 = st.tabs(["⚠️ Churn Prediction", "🧺 Basket Spend Prediction"])

        with tab1:
            # Churn Prediction (RFM based)
            st.header("ML Churn Prediction: At-Risk Customers")
            rfm_cols_present = all(col in full_df.columns for col in ['hshd_num', 'date', 'BASKET_NUM', 'SPEND'])
            if rfm_cols_present and not full_df.empty and pd.api.types.is_datetime64_any_dtype(full_df['date']) and not full_df['date'].isnull().all():
                try:
                    now = full_df['date'].max()
                    if pd.isna(now): # Handle case where max date is NaT
                         st.warning("Cannot determine current date for RFM calculation.")
                    else:
                        rfm = full_df.groupby('hshd_num').agg(
                            # Handle potential NaT in date groups
                            recency=('date', lambda x: (now - x.max()).days if pd.notna(x.max()) else 9999),
                            frequency=('BASKET_NUM', 'nunique'),
                            monetary=('SPEND', 'sum')
                        ).reset_index()

                        # Define churn based on recency (e.g., > 84 days = 12 weeks)
                        rfm['churn'] = (rfm['recency'] > 84).astype(int)

                        X = rfm[['recency', 'frequency', 'monetary']]
                        y = rfm['churn']

                        # Need sufficient data and both churn/non-churn examples
                        if len(X) >= 50 and len(y.unique()) > 1 and y.value_counts().min() >= 5:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

                            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                            clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)

                            st.write("**Classification Report:**")
                            st.text(classification_report(y_test, y_pred, zero_division=0))

                            st.write("**Confusion Matrix:**")
                            st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                                      columns=['Predicted Not Churn', 'Predicted Churn'],
                                                      index=['Actual Not Churn', 'Actual Churn']))

                            feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
                            st.write("**Feature Importances:**")
                            st.bar_chart(feat_imp)
                        else:
                             st.warning("Not enough data or class diversity (churn/non-churn) to train a reliable churn prediction model.")
                except Exception as e:
                    st.error(f"Error during churn prediction model training: {e}")
            elif not rfm_cols_present:
                st.warning("Required columns for RFM (hshd_num, date, BASKET_NUM, SPEND) not found.")
            else:
                st.warning("Insufficient or invalid data for churn prediction.")

        with tab2:
            # Basket Spend Prediction
            st.header("Basket Spend Prediction")
            spend_cols_present = all(col in full_df.columns for col in ['BASKET_NUM', 'SPEND'])
            if spend_cols_present and not full_df.empty:
                try:
                    # Use relevant categorical features for prediction, check if they exist
                    features_to_encode = ['DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']
                    available_features = [f for f in features_to_encode if f in full_df.columns]

                    if available_features:
                        # Select necessary columns and drop rows with NaNs in key fields
                        basket_features_df = full_df[['BASKET_NUM', 'SPEND'] + available_features].copy()
                        basket_features_df.dropna(subset=['BASKET_NUM'] + available_features, inplace=True)

                        if not basket_features_df.empty:
                            # One-hot encode available categorical features
                            basket_features_encoded = pd.get_dummies(basket_features_df, columns=available_features, dummy_na=False)

                            # Aggregate features per basket (summing encoded flags)
                            X_basket = basket_features_encoded.drop(columns=['SPEND']).groupby('BASKET_NUM').sum()
                            y_basket = basket_features_encoded.groupby('BASKET_NUM')['SPEND'].sum() # Target: total spend per basket

                            # Need enough unique baskets to train
                            if len(X_basket) >= 50:
                                X_train, X_test, y_train, y_test = train_test_split(X_basket, y_basket, test_size=0.2, random_state=42)

                                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                                rf.fit(X_train, y_train)
                                y_pred = rf.predict(X_test)

                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                st.write(f"**Model Performance (R² Score):** {r2:.3f}")
                                st.write(f"**Model Performance (MSE):** {mse:.2f}")

                                st.subheader("Predicted vs. Actual Basket Spend (Test Set Sample)")
                                # Show a sample for comparison
                                chart_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).sample(min(100, len(y_test)), random_state=42)
                                st.line_chart(chart_df.reset_index(drop=True))

                                # Feature importance
                                importances = pd.Series(rf.feature_importances_, index=X_basket.columns).sort_values(ascending=False)
                                top_features = importances.head(15)
                                st.write("**Top Drivers of Basket Spend (Features):**")
                                st.bar_chart(top_features)
                            else:
                                st.warning("Not enough unique basket data to train prediction model.")
                        else:
                            st.warning("No data remaining after cleaning for basket spend prediction.")
                    else:
                        st.warning("No suitable categorical features (DEPARTMENT, COMMODITY, etc.) found for basket spend prediction.")
                except Exception as e:
                    st.error(f"Error during basket analysis model training: {e}")
            elif not spend_cols_present:
                 st.warning("Required columns (BASKET_NUM, SPEND) not found.")
            else:
                st.warning("Insufficient data for basket spend prediction.")


elif page == "Household Search":
    st.title("Search Household Transactions")
    hshd_num_input = st.text_input("Enter Household Number (HSHD_NUM):")

    if hshd_num_input:
        try:
            hshd_num = int(hshd_num_input) # Validate input is integer
            # Use parameterized query to prevent SQL injection [2][4]
            # Ensure column names match exactly those in your database tables
            query = text("""
            SELECT
                H.HSHD_NUM,
                T.BASKET_NUM,
                T.PURCHASE_   -- Assuming PURCHASE_ is the intended column name
                , T.PRODUCT_NUM
                , T.SPEND
                , T.UNITS
                , T.STORE_R
                , T.WEEK_NUM
                , T.YEAR
                , P.DEPARTMENT
                , P.COMMODITY
                , P.BRAND_TY
                , P.NATURAL_ORGANIC_FLAG
            FROM Households H
            INNER JOIN Transactions T ON H.HSHD_NUM = T.HSHD_NUM -- Use INNER JOIN if household must exist
            LEFT JOIN Products P ON T.PRODUCT_NUM = P.PRODUCT_NUM -- LEFT JOIN if product might not exist
            WHERE H.HSHD_NUM = :hnum -- Parameterized placeholder
            ORDER BY T.YEAR DESC, T.WEEK_NUM DESC, T.BASKET_NUM;
            """)
            # Execute using conn.query with params argument
            data = conn.query(str(query), params={"hnum": hshd_num})

            if not data.empty:
                st.dataframe(data)
            else:
                st.info(f"No data found for Household Number: {hshd_num}")
        except ValueError:
            st.error("Please enter a valid numeric Household Number.")
        except Exception as e:
            # Provide more specific error feedback if possible
            st.error(f"An error occurred during search: {e}")
            if "Invalid object name" in str(e):
                 st.warning("Check if table names (Households, Transactions, Products) and column names in the query are correct.")


elif page == "CLV Calculation":
    st.title("Calculate and Save CLV")
    st.write("This calculates Customer Lifetime Value (total SPEND per household) from the `Transactions` table and saves it to the `clv_results` table in your database.")
    st.warning("This will DELETE all existing data in `clv_results` before inserting new calculations.")

    if st.button("Run CLV Calculation and Save to SQL"):
        if conn: # Ensure connection exists
            try:
                # 1. Calculate CLV using conn.query
                clv_query = """
                SELECT HSHD_NUM, SUM(SPEND) AS CLV
                FROM transactions -- Ensure table name is correct
                WHERE SPEND IS NOT NULL AND HSHD_NUM IS NOT NULL -- Add basic data quality checks
                GROUP BY HSHD_NUM;
                """
                clv_df = conn.query(clv_query)

                if clv_df.empty:
                    st.warning("No data found in Transactions to calculate CLV.")
                else:
                    st.write(f"Calculated CLV for {len(clv_df)} households.")

                    # 2. Use conn.session for DML operations (CREATE, DELETE, INSERT) [2]
                    with conn.session as s:
                        # 3. Create table if it doesn't exist (idempotent)
                        # Use standard SQL for broader compatibility if possible, or keep T-SQL for SQL Server
                        create_table_sql = text("""
                        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'clv_results')
                        BEGIN
                            CREATE TABLE clv_results (
                                HSHD_NUM INT PRIMARY KEY,
                                CLV FLOAT NOT NULL,
                                LastUpdated DATETIME DEFAULT GETDATE() -- Optional: track update time
                            );
                            -- Optional: PRINT 'Table clv_results created.'; -- PRINT might not work via sqlalchemy directly
                        END;
                        """)
                        s.execute(create_table_sql)
                        s.commit() # Commit table creation separately

                        # 4. Clear existing data from the target table
                        st.write("Clearing existing data from `clv_results`...")
                        delete_stmt = text("DELETE FROM clv_results;")
                        s.execute(delete_stmt)
                        # s.commit() # Commit deletion separately or together with inserts

                        # 5. Insert new CLV data using parameterized query within the session
                        st.write(f"Inserting {len(clv_df)} calculated CLV rows...")
                        insert_stmt = text("INSERT INTO clv_results (HSHD_NUM, CLV) VALUES (:hnum, :clv_val);")
                        inserted_count = 0
                        skipped_count = 0

                        # Prepare data for insertion (convert types explicitly)
                        data_to_insert = [
                            {"hnum": int(row['HSHD_NUM']), "clv_val": float(row['CLV'])}
                            for _, row in clv_df.iterrows() if pd.notna(row['HSHD_NUM']) and pd.notna(row['CLV'])
                        ]

                        if len(data_to_insert) < len(clv_df):
                             skipped_count = len(clv_df) - len(data_to_insert)
                             st.warning(f"Skipped {skipped_count} rows due to missing HSHD_NUM or CLV values during preparation.")

                        # Execute inserts (consider bulk insert for large data, though executemany is often efficient enough)
                        if data_to_insert:
                            s.execute(insert_stmt, data_to_insert) # Pass list of dicts to executemany implicitly
                            inserted_count = len(data_to_insert)

                        s.commit() # Commit all inserts together
                        st.success(f"✅ CLV results processed. {inserted_count} rows saved to `clv_results` table.")
                        if skipped_count > 0:
                             st.warning(f"{skipped_count} rows were skipped due to missing values.")
                        st.dataframe(clv_df.head())

            except Exception as e:
                # Attempt to rollback session on error
                try:
                    s.rollback()
                except:
                    pass # Ignore rollback error if session is already closed or invalid
                st.error(f"An error occurred during CLV calculation/saving: {e}")
                # Check for common SQL Server errors
                if "transaction log for database" in str(e).lower() and "is full" in str(e).lower():
                    st.warning("Error suggests the database transaction log might be full. Consider increasing log size or frequency of log backups.")
                elif "Cannot insert duplicate key" in str(e):
                     st.warning("Error suggests duplicate HSHD_NUM values were being inserted. The DELETE step might have failed, or source data has duplicates after aggregation (unlikely with GROUP BY).")
                elif "String or binary data would be truncated" in str(e):
                     st.warning("Check if data being inserted exceeds the column size defined in the `clv_results` table.")

        else:
            st.error("Database connection is not available. Cannot run CLV calculation.")


elif page == "Data Loader":
    st.title("🔄 Data Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reload from Database")
        st.write("Clear cache and reload data from the connected database on the next page visit.")
        if st.button("🔄 Reload Data from Database"):
            # Clear both data and resource caches
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared! Data will be reloaded from the database.")
            # Rerun might be needed if you want immediate effect without navigation
            st.rerun()

    with col2:
        st.subheader("Upload CSV Files (Preview Only)")
        st.write("Upload CSVs to preview their structure. (Note: Uploaded data is not saved to the database here).")
        uploaded_transactions = st.file_uploader("Upload Transactions CSV", type="csv")
        uploaded_households = st.file_uploader("Upload Households CSV", type="csv")
        uploaded_products = st.file_uploader("Upload Products CSV", type="csv")

        # Store uploaded data in session state for previewing below
        if uploaded_transactions:
            try:
                st.session_state['uploaded_transactions'] = pd.read_csv(uploaded_transactions)
                st.success("Transactions CSV uploaded for preview.")
            except Exception as e:
                st.error(f"Error reading Transactions CSV: {e}")
        if uploaded_households:
             try:
                st.session_state['uploaded_households'] = pd.read_csv(uploaded_households)
                st.success("Households CSV uploaded for preview.")
             except Exception as e:
                st.error(f"Error reading Households CSV: {e}")
        if uploaded_products:
             try:
                st.session_state['uploaded_products'] = pd.read_csv(uploaded_products)
                st.success("Products CSV uploaded for preview.")
             except Exception as e:
                st.error(f"Error reading Products CSV: {e}")

    # --- Data Preview Section ---
    st.divider() # Visual separator
    st.subheader("📊 Data Preview")

    # Attempt to load data from DB (will use cache unless cleared)
    if conn:
        df_t, df_h, df_p = load_data(conn)
    else:
        df_t, df_h, df_p = None, None, None # Ensure variables exist even if conn failed

    preview_tabs = st.tabs(["Transactions", "Households", "Products"])

    with preview_tabs[0]:
        st.write("**Transactions Data**")
        if df_t is not None:
            st.write("Preview from Database (first 5 rows):")
            st.dataframe(df_t.head())
        elif 'uploaded_transactions' in st.session_state and st.session_state['uploaded_transactions'] is not None:
            st.write("Preview from Uploaded CSV (first 5 rows):")
            st.dataframe(st.session_state['uploaded_transactions'].head())
        else:
            st.info("No Transactions data loaded or uploaded to preview.")

    with preview_tabs[1]:
        st.write("**Households Data**")
        if df_h is not None:
            st.write("Preview from Database (first 5 rows):")
            st.dataframe(df_h.head())
        elif 'uploaded_households' in st.session_state and st.session_state['uploaded_households'] is not None:
             st.write("Preview from Uploaded CSV (first 5 rows):")
             st.dataframe(st.session_state['uploaded_households'].head())
        else:
            st.info("No Households data loaded or uploaded to preview.")

    with preview_tabs[2]:
        st.write("**Products Data**")
        if df_p is not None:
            st.write("Preview from Database (first 5 rows):")
            st.dataframe(df_p.head())
        elif 'uploaded_products' in st.session_state and st.session_state['uploaded_products'] is not None:
             st.write("Preview from Uploaded CSV (first 5 rows):")
             st.dataframe(st.session_state['uploaded_products'].head())
        else:
             st.info("No Products data loaded or uploaded to preview.")

