import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import hashlib
from sqlalchemy import text

st.set_page_config(page_title="Retail Analytics App", layout="wide")

# --- Authentication (simple, local memory) ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

if 'user_db' not in st.session_state:
    st.session_state.user_db = {}
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

login_signup()
if not st.session_state.authenticated:
    st.stop()

# --- Database connection using Streamlit's SQLConnection ---
# Updated connection code to handle missing configuration gracefully
try:
    # First try to use the connection from secrets.toml
    conn = st.connection("sql")
except Exception as e:
    st.warning("Database connection from secrets.toml failed. Please provide connection details.")
    # Show connection configuration form
    with st.expander("Configure Database Connection"):
        db_type = st.selectbox("Database Type", ["mssql", "mysql", "postgresql", "sqlite"])
        if db_type != "sqlite":
            server = st.text_input("Server", "localhost")
            database = st.text_input("Database Name", "retail")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            port = st.text_input("Port", "1433" if db_type == "mssql" else "3306" if db_type == "mysql" else "5432")
            
            if db_type == "mssql":
                conn_string = f"mssql+pymssql://{username}:{password}@{server}:{port}/{database}"
            elif db_type == "mysql":
                conn_string = f"mysql+pymysql://{username}:{password}@{server}:{port}/{database}"
            elif db_type == "postgresql":
                conn_string = f"postgresql://{username}:{password}@{server}:{port}/{database}"
        else:
            db_path = st.text_input("SQLite Database Path", "retail.db")
            conn_string = f"sqlite:///{db_path}"
        
        if st.button("Connect to Database"):
            try:
                conn = st.connection("sql", type="sql", url=conn_string)
                st.success("Connected successfully!")
            except Exception as e:
                st.error(f"Connection failed: {e}")
                st.info("Make sure you have the appropriate database driver installed.")
                st.stop()

@st.cache_data(ttl=600)
def load_data():
    try:
        df_transactions = conn.query("SELECT * FROM Transactions", ttl=600)
        df_households = conn.query("SELECT * FROM Households", ttl=600)
        df_products = conn.query("SELECT * FROM Products", ttl=600)
        df_transactions.columns = df_transactions.columns.str.strip()
        df_households.columns = df_households.columns.str.strip()
        df_products.columns = df_products.columns.str.strip()
        return df_transactions, df_households, df_products
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Provide sample data if database connection fails
        if st.button("Load Sample Data Instead"):
            return load_sample_data()
        st.stop()

def load_sample_data():
    """Load sample data for demonstration purposes when DB connection fails"""
    # Create sample transaction data
    transactions = pd.DataFrame({
        'HSHD_NUM': range(1000, 1010) * 10,
        'BASKET_NUM': range(100, 200),
        'PURCHASE_': pd.date_range(start='2023-01-01', periods=100),
        'PRODUCT_NUM': range(5000, 5100),
        'SPEND': [round(float(i) * 1.5, 2) for i in range(100)],
        'UNITS': [1] * 100,
        'STORE_R': ['WEST'] * 50 + ['EAST'] * 50,
        'WEEK_NUM': [i % 52 + 1 for i in range(100)],
        'YEAR': [2023] * 100
    })
    transactions['date'] = pd.to_datetime(transactions['YEAR'].astype(str) + transactions['WEEK_NUM'].astype(str) + '0', format='%Y%U%w')
    
    # Create sample household data
    households = pd.DataFrame({
        'HSHD_NUM': range(1000, 1010),
        'L': ['A', 'B', 'C', 'D'] * 2 + ['A', 'B'],
        'AGE_RANGE': ['35-44', '45-54', '25-34', '55-64', '35-44', '65+', '25-34', '35-44', '45-54', '55-64'],
        'MARITAL': ['MARRIED'] * 5 + ['SINGLE'] * 5,
        'INCOME_RANGE': ['50-74K', '75-99K', '35-49K', '100-150K', '50-74K', '25-34K', '35-49K', '50-74K', '75-99K', '100-150K'],
        'HOMEOWNER': ['YES'] * 6 + ['NO'] * 4,
        'HSHD_COMPOSITION': ['2 Adults Kids', '2 Adults No Kids', '1 Adult Kids', '2 Adults Kids', '1 Adult No Kids'] * 2,
        'HH_SIZE': [3, 2, 3, 4, 1, 2, 3, 2, 2, 4],
        'CHILDREN': [1, 0, 1, 2, 0, 0, 1, 0, 0, 2]
    })
    
    # Create sample product data
    products = pd.DataFrame({
        'PRODUCT_NUM': range(5000, 5100),
        'DEPARTMENT': ['GROCERY'] * 30 + ['DAIRY'] * 30 + ['MEAT'] * 20 + ['PRODUCE'] * 20,
        'COMMODITY': ['CEREAL'] * 15 + ['PASTA'] * 15 + ['MILK'] * 15 + ['CHEESE'] * 15 + 
                     ['BEEF'] * 10 + ['CHICKEN'] * 10 + ['FRUITS'] * 10 + ['VEGETABLES'] * 10,
        'BRAND_TY': ['NATIONAL'] * 50 + ['PRIVATE'] * 50,
        'NATURAL_ORGANIC_FLAG': ['N'] * 75 + ['Y'] * 25
    })
    
    st.success("Loaded sample data for demonstration!")
    return transactions, households, products

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Household Search", "CLV Calculation", "Data Loader"])

# --- Dashboard ---
if page == "Dashboard":
    st.title("📊 Retail Customer Analytics Dashboard")
    
    try:
        df_transactions, df_households, df_products = load_data()
        df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
        df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
        df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)
        full_df = df_transactions.merge(df_households, on='hshd_num', how='left')
        full_df = full_df.merge(df_products, on='product_num', how='left')

        st.header("📈 Customer Engagement Over Time")
        if 'date' not in df_transactions.columns:
            df_transactions['date'] = pd.to_datetime(df_transactions['YEAR'].astype(str) + df_transactions['WEEK_NUM'].astype(str) + '0', format='%Y%U%w')
        weekly_engagement = df_transactions.groupby(df_transactions['date'].dt.to_period('W'))['SPEND'].sum().reset_index()
        weekly_engagement['ds'] = weekly_engagement['date'].dt.start_time
        st.line_chart(weekly_engagement.set_index('ds')['SPEND'])

        st.header("👨👩👧 Demographics and Engagement")
        demo_options = [col for col in df_households.columns if col in ['INCOME_RANGE', 'AGE_RANGE', 'CHILDREN']]
        if demo_options:
            selected_demo = st.selectbox("Segment by:", demo_options)
            demo_spending = full_df.groupby(selected_demo)['SPEND'].sum().reset_index()
            st.bar_chart(demo_spending.rename(columns={selected_demo: 'index'}).set_index('index'))

        st.header("🔍 Customer Segmentation")
        segmentation = full_df.groupby(['hshd_num']).agg({'SPEND': 'sum', 'INCOME_RANGE': 'first', 'AGE_RANGE': 'first'})
        st.dataframe(segmentation.sort_values(by='SPEND', ascending=False).head(10))

        st.header("🌟 Loyalty Program Effect")
        if 'LOYALTY_FLAG' in df_households.columns:
            loyalty = full_df.groupby('LOYALTY_FLAG')['SPEND'].agg(['sum', 'mean']).reset_index()
            st.dataframe(loyalty)

        st.header("🧺 Basket Analysis")
        basket = df_transactions.groupby(['BASKET_NUM', 'product_num'])['SPEND'].sum().reset_index()
        top_products = basket.groupby('product_num')['SPEND'].sum().nlargest(10).reset_index()
        top_products = top_products.merge(df_products, on='product_num', how='left')
        if 'COMMODITY' in top_products.columns:
            st.bar_chart(top_products.set_index('COMMODITY')['SPEND'])
            product_spending = top_products.groupby('COMMODITY')['SPEND'].sum().reset_index()
            fig = px.pie(product_spending, values='SPEND', names='COMMODITY', title='Spending Distribution by Product Category')
            st.plotly_chart(fig)
        else:
            st.dataframe(top_products)

        st.header("📆 Seasonal Spending Patterns")
        df_transactions['month'] = df_transactions['date'].dt.month_name()
        seasonal = df_transactions.groupby('month')['SPEND'].sum().reset_index()
        seasonal['month'] = pd.Categorical(seasonal['month'], categories=[
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ], ordered=True)
        seasonal = seasonal.sort_values('month')
        st.bar_chart(seasonal.set_index('month'))

        st.header("💰 Customer Lifetime Value")
        clv = df_transactions.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
        st.dataframe(clv.head(10))

        st.header("📊 Customer Spending by Product Category")
        category_spending = full_df.groupby('COMMODITY')['SPEND'].sum().reset_index()
        st.bar_chart(category_spending.set_index('COMMODITY')['SPEND'])

        st.header("🏆 Top 10 Customers by Spending")
        top_customers = full_df.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
        st.dataframe(top_customers.head(10))

        st.header("📈 Trends in Age Group Spending")
        age_group_spending = full_df.groupby('AGE_RANGE')['SPEND'].sum().reset_index()
        st.bar_chart(age_group_spending.set_index('AGE_RANGE')['SPEND'])
        fig = px.pie(age_group_spending, values='SPEND', names='AGE_RANGE', title='Spending Distribution by Age Group')
        st.plotly_chart(fig)

        tab1, tab2 = st.tabs(["⚠️ Churn Prediction", "🧺 Basket Analysis"])
        with tab1:
            st.header("Churn Prediction: Customer Engagement Over Time")
            departments = ["All"] + sorted(df_products['DEPARTMENT'].dropna().unique())
            commodities = ["All"] + sorted(df_products['COMMODITY'].dropna().unique())
            brand_types = ["All"] + sorted(df_products['BRAND_TY'].dropna().unique())
            organic_flags = ["All"] + sorted(df_products['NATURAL_ORGANIC_FLAG'].dropna().unique())
            col1, col2, col3, col4 = st.columns(4)
            department = col1.selectbox("Select Department:", departments)
            commodity = col2.selectbox("Select Commodity:", commodities)
            brand_type = col3.selectbox("Select Brand Type:", brand_types)
            organic_flag = col4.selectbox("Select Organic:", organic_flags)
            if st.button("Apply Filters", key="churn_apply"):
                filtered = df_transactions.merge(df_products, on='product_num', how='left')
                if department != "All":
                    filtered = filtered[filtered['DEPARTMENT'] == department]
                if commodity != "All":
                    filtered = filtered[filtered['COMMODITY'] == commodity]
                if brand_type != "All":
                    filtered = filtered[filtered['BRAND_TY'] == brand_type]
                if organic_flag != "All":
                    filtered = filtered[filtered['NATURAL_ORGANIC_FLAG'] == organic_flag]
                if not filtered.empty:
                    churn_df = filtered.groupby('date')['SPEND'].sum().reset_index().sort_values('date')
                    st.line_chart(churn_df.set_index("date")["SPEND"])
                    with st.expander("📄 Raw Data"):
                        st.dataframe(churn_df)
                else:
                    st.warning("No data found for selected filters.")

            st.subheader("ML Churn Prediction: At-Risk Customers")
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
            st.write("**Feature Importances:**")
            st.bar_chart(feat_imp)

        with tab2:
            st.header("Basket Analysis - Predicting Total Spend")
            basket_merged = df_transactions.merge(df_products, on='product_num', how='left')
            basket_features = pd.get_dummies(
                basket_merged[['BASKET_NUM', 'DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']]
            )
            X_basket = basket_features.groupby('BASKET_NUM').sum()
            y_basket = basket_merged.groupby('BASKET_NUM')['SPEND'].sum()
            X_train, X_test, y_train, y_test = train_test_split(X_basket, y_basket, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**MSE:** {mse:.2f}")
            st.subheader("Predicted vs. Actual Basket Spend (Test Set)")
            chart_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            st.line_chart(chart_df.reset_index(drop=True))
            st.subheader("📄 Actual vs. Predicted Spend Table")
            st.dataframe(chart_df)
            csv = chart_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name='predicted_vs_actual_basket_spend.csv',
                mime='text/csv',
            )
            importances = pd.Series(rf.feature_importances_, index=X_basket.columns)
            top_features = importances.sort_values(ascending=False).head(10)
            st.write("**Top Drivers of Basket Spend:**")
            st.bar_chart(top_features)
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        st.info("Please check your database connection settings in .streamlit/secrets.toml")

# --- Household Search ---
elif page == "Household Search":
    st.title("Search Household Transactions")
    hshd_num = st.text_input("Enter Household Number (HSHD_NUM):")
    if hshd_num:
        try:
            hshd_num = int(hshd_num)
            try:
                query = f"""
                SELECT H.HSHD_NUM, T.BASKET_NUM, T.PURCHASE_ AS Date,
                       P.PRODUCT_NUM, P.DEPARTMENT, P.COMMODITY
                FROM dbo.households H
                JOIN dbo.transactions T ON H.HSHD_NUM = T.HSHD_NUM
                LEFT JOIN dbo.products P ON T.PRODUCT_NUM = P.PRODUCT_NUM
                WHERE H.HSHD_NUM = {hshd_num}
                ORDER BY H.HSHD_NUM, T.BASKET_NUM, T.PURCHASE_, P.PRODUCT_NUM, P.DEPARTMENT, P.COMMODITY;
                """
                data = conn.query(query)
                if not data.empty:
                    st.dataframe(data)
                else:
                    st.write("No data found for the entered Household Number.")
            except Exception as e:
                st.error(f"Query error: {e}")
                # Fallback to simpler query without schema qualification
                try:
                    simplified_query = f"""
                    SELECT * FROM Transactions 
                    WHERE HSHD_NUM = {hshd_num}
                    """
                    data = conn.query(simplified_query)
                    if not data.empty:
                        st.dataframe(data)
                        st.info("Limited data displayed due to database schema differences.")
                    else:
                        st.write("No data found for the entered Household Number.")
                except Exception as e2:
                    st.error(f"Simplified query also failed: {e2}")
        except ValueError:
            st.error("Please enter a valid number for the Household Number.")

# --- CLV Calculation ---
elif page == "CLV Calculation":
    st.title("Calculate and Save CLV")
    if st.button("Run CLV Calculation and Save to SQL"):
        try:
            clv_query = """
            SELECT HSHD_NUM, SUM(SPEND) AS CLV
            FROM transactions
            GROUP BY HSHD_NUM
            """
            clv_df = conn.query(clv_query)
            
            try:
                with conn.session as s:
                    # First check if we're using SQL Server (which uses different syntax)
                    try:
                        s.execute("""
                        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'clv_results')
                        BEGIN
                            CREATE TABLE clv_results (
                                HSHD_NUM INT PRIMARY KEY,
                                CLV FLOAT
                            );
                        END;
                        """)
                    except Exception:
                        # For other database types like MySQL, PostgreSQL
                        s.execute("""
                        CREATE TABLE IF NOT EXISTS clv_results (
                            HSHD_NUM INT PRIMARY KEY,
                            CLV FLOAT
                        )
                        """)
                    
                    # Clear existing data
                    s.execute("DELETE FROM clv_results")
                    
                    # Insert new data
                    for _, row in clv_df.iterrows():
                        s.execute(
                            "INSERT INTO clv_results (HSHD_NUM, CLV) VALUES (:HSHD_NUM, :CLV)",
                            params={"HSHD_NUM": int(row['HSHD_NUM']), "CLV": float(row['CLV'])}
                        )
                    s.commit()
                st.success("✅ CLV results saved to clv_results table in database!")
            except Exception as session_err:
                st.error(f"Error saving to database: {session_err}")
                st.info("Displaying results without saving to database")
                st.dataframe(clv_df)
                csv = clv_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CLV Results as CSV",
                    data=csv,
                    file_name='clv_results.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Error calculating CLV: {e}")
            st.info("Please check your database connection settings")

# --- Data Loader (CSV Upload) ---
elif page == "Data Loader":
    st.title("Data Loader: Upload Datasets")
    uploaded_transactions = st.file_uploader("Upload Transactions Dataset", type="csv")
    uploaded_households = st.file_uploader("Upload Households Dataset", type="csv")
    uploaded_products = st.file_uploader("Upload Products Dataset", type="csv")
    
    if uploaded_transactions is not None:
        st.session_state['transactions_df'] = pd.read_csv(uploaded_transactions)
    if uploaded_households is not None:
        st.session_state['households_df'] = pd.read_csv(uploaded_households)
    if uploaded_products is not None:
        st.session_state['products_df'] = pd.read_csv(uploaded_products)
    
    if 'transactions_df' in st.session_state:
        st.write("Transactions Data", st.session_state['transactions_df'].head())
    if 'households_df' in st.session_state:
        st.write("Households Data", st.session_state['households_df'].head())
    if 'products_df' in st.session_state:
        st.write("Products Data", st.session_state['products_df'].head())
    
    if (('transactions_df' not in st.session_state) or
        ('households_df' not in st.session_state) or
        ('products_df' not in st.session_state)):
        if st.button("Load Latest Data from Database"):
            try:
                tdf, hdf, pdf = load_data()
                st.session_state['transactions_df'] = tdf
                st.session_state['households_df'] = hdf
                st.session_state['products_df'] = pdf
                st.success("Loaded latest data from database.")
                st.write("Transactions Data", tdf.head())
                st.write("Households Data", hdf.head())
                st.write("Products Data", pdf.head())
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.info("Please check your database connection settings")
                
    # Add functionality to save uploaded data to database
    if ('transactions_df' in st.session_state and 
        'households_df' in st.session_state and 
        'products_df' in st.session_state):
        if st.button("Save Uploaded Data to Database"):
            try:
                with conn.session as s:
                    # Create tables if they don't exist
                    try:
                        # SQL Server syntax
                        for table, df in [
                            ('Transactions', st.session_state['transactions_df']),
                            ('Households', st.session_state['households_df']),
                            ('Products', st.session_state['products_df'])
                        ]:
                            cols = ", ".join([f"{col} VARCHAR(255)" for col in df.columns])
                            s.execute(f"""
                            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table}')
                            BEGIN
                                CREATE TABLE {table} ({cols});
                            END;
                            """)
                    except Exception:
                        # Generic SQL syntax for other databases
                        for table, df in [
                            ('Transactions', st.session_state['transactions_df']),
                            ('Households', st.session_state['households_df']),
                            ('Products', st.session_state['products_df'])
                        ]:
                            cols = ", ".join([f"{col} VARCHAR(255)" for col in df.columns])
                            s.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
                    
                    # Insert data
                    for table, df in [
                        ('Transactions', st.session_state['transactions_df']),
                        ('Households', st.session_state['households_df']),
                        ('Products', st.session_state['products_df'])
                    ]:
                        # Clear existing data
                        s.execute(f"DELETE FROM {table}")
                        
                        # Insert new data
                        for _, row in df.iterrows():
                            placeholders = ", ".join([f":{col}" for col in df.columns])
                            cols = ", ".join(df.columns)
                            params = {col: str(val) for col, val in row.items()}
                            s.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", params=params)
                        
                    s.commit()
                st.success("✅ All data saved to database successfully!")
            except Exception as e:
                st.error(f"Error saving to database: {e}")
                st.info("Unable to save to database. You can still use the uploaded data for analysis.")
