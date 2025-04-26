import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import hashlib
from sqlalchemy import create_engine, text
import urllib.parse

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

# --- Load Sample Data Function ---
@st.cache_data
def load_sample_data():
    try:
        # Try to load from CSV files if available
        df_transactions = pd.read_csv('400_transactions.csv')
        df_households = pd.read_csv('400_households.csv')
        df_products = pd.read_csv('400_products.csv')
    except:
        # Create sample data if files aren't available
        st.warning("Using generated sample data for demonstration")
        # Sample transactions data
        df_transactions = pd.DataFrame({
            'HSHD_NUM': [1, 1, 2, 2, 3, 3, 4, 5],
            'BASKET_NUM': [1001, 1002, 2001, 2002, 3001, 3002, 4001, 5001],
            'PURCHASE_': ['2023-01-01', '2023-01-15', '2023-01-02', '2023-01-20', '2023-01-05', '2023-01-25', '2023-01-10', '2023-01-30'],
            'PRODUCT_NUM': [1, 2, 1, 3, 2, 4, 3, 5],
            'SPEND': [10.5, 20.3, 15.0, 25.5, 12.0, 30.0, 18.5, 22.0],
            'UNITS': [1, 2, 1, 3, 1, 2, 2, 1],
            'STORE_R': [1, 1, 2, 2, 1, 3, 2, 1],
            'WEEK_NUM': [1, 3, 1, 4, 2, 5, 3, 6],
            'YEAR': [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        })
        
        # Sample households data
        df_households = pd.DataFrame({
            'HSHD_NUM': [1, 2, 3, 4, 5],
            'L': [1, 2, 3, 1, 2],
            'AGE_RANGE': ['35-44', '45-54', '25-34', '55-64', '35-44'],
            'MARITAL': ['Married', 'Single', 'Married', 'Divorced', 'Married'],
            'INCOME_RANGE': ['35-49K', '50-74K', '25-34K', '75-99K', '35-49K'],
            'HOMEOWNER': ['Homeowner', 'Renter', 'Homeowner', 'Renter', 'Homeowner'],
            'HSHD_COMPOSITION': ['2 Adults Kids', 'Single', '2 Adults Kids', 'Single', '2 Adults No Kids'],
            'HH_SIZE': [3, 1, 4, 1, 2],
            'CHILDREN': [1, 0, 2, 0, 0]
        })
        
        # Sample products data
        df_products = pd.DataFrame({
            'PRODUCT_NUM': [1, 2, 3, 4, 5],
            'DEPARTMENT': ['Grocery', 'Dairy', 'Produce', 'Meat', 'Bakery'],
            'COMMODITY': ['Vegetables', 'Milk', 'Fruits', 'Beef', 'Bread'],
            'BRAND_TY': ['National', 'Private', 'National', 'Private', 'National'],
            'NATURAL_ORGANIC_FLAG': ['N', 'N', 'Y', 'N', 'Y']
        })
    
    # Clean column names
    df_transactions.columns = df_transactions.columns.str.strip()
    df_households.columns = df_households.columns.str.strip()
    df_products.columns = df_products.columns.str.strip()
    
    return df_transactions, df_households, df_products

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Household Search", "CLV Calculation", "Data Loader"])

# --- Dashboard ---
if page == "Dashboard":
    st.title("📊 Retail Customer Analytics Dashboard")
    
    # Load sample data
    df_transactions, df_households, df_products = load_sample_data()
    
    # Rename columns for consistency in joins
    df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
    df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
    df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)
    
    # Merge dataframes
    full_df = df_transactions.merge(df_households, on='hshd_num', how='left')
    full_df = full_df.merge(df_products, on='product_num', how='left')
    
    # Convert date and create date field
    df_transactions['date'] = pd.to_datetime(df_transactions['YEAR'].astype(str) + df_transactions['WEEK_NUM'].astype(str) + '0', format='%Y%U%w')
    
    # --- Customer Engagement Over Time ---
    st.header("📈 Customer Engagement Over Time")
    weekly_engagement = df_transactions.groupby(df_transactions['date'].dt.to_period('W'))['SPEND'].sum().reset_index()
    weekly_engagement['ds'] = weekly_engagement['date'].dt.start_time
    st.line_chart(weekly_engagement.set_index('ds')['SPEND'])
    
    # --- Demographics and Engagement ---
    st.header("👨👩👧 Demographics and Engagement")
    selected_demo = st.selectbox("Segment by:", ['INCOME_RANGE', 'AGE_RANGE', 'CHILDREN'])
    demo_spending = full_df.groupby(selected_demo)['SPEND'].sum().reset_index()
    st.bar_chart(demo_spending.rename(columns={selected_demo: 'index'}).set_index('index'))
    
    # --- Customer Segmentation ---
    st.header("🔍 Customer Segmentation")
    segmentation = full_df.groupby(['hshd_num']).agg({'SPEND': 'sum', 'INCOME_RANGE': 'first', 'AGE_RANGE': 'first'})
    st.dataframe(segmentation.sort_values(by='SPEND', ascending=False).head(10))
    
    # --- Loyalty Program Effect ---
    st.header("🌟 Loyalty Program Effect")
    if 'LOYALTY_FLAG' in df_households.columns:
        loyalty = full_df.groupby('LOYALTY_FLAG')['SPEND'].agg(['sum', 'mean']).reset_index()
        st.dataframe(loyalty)
    
    # --- Basket Analysis ---
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
    
    # --- Seasonal Spending Patterns ---
    st.header("📆 Seasonal Spending Patterns")
    df_transactions['month'] = df_transactions['date'].dt.month_name()
    seasonal = df_transactions.groupby('month')['SPEND'].sum().reset_index()
    seasonal['month'] = pd.Categorical(seasonal['month'], categories=[
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ], ordered=True)
    seasonal = seasonal.sort_values('month')
    st.bar_chart(seasonal.set_index('month'))
    
    # --- Customer Lifetime Value ---
    st.header("💰 Customer Lifetime Value")
    clv = df_transactions.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
    st.dataframe(clv.head(10))
    
    # --- Customer Spending by Product Category ---
    st.header("📊 Customer Spending by Product Category")
    category_spending = full_df.groupby('COMMODITY')['SPEND'].sum().reset_index()
    st.bar_chart(category_spending.set_index('COMMODITY')['SPEND'])
    
    # --- Top 10 Customers by Spending ---
    st.header("🏆 Top 10 Customers by Spending")
    top_customers = full_df.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
    st.dataframe(top_customers.head(10))
    
    # --- Trends in Age Group Spending ---
    st.header("📈 Trends in Age Group Spending")
    age_group_spending = full_df.groupby('AGE_RANGE')['SPEND'].sum().reset_index()
    st.bar_chart(age_group_spending.set_index('AGE_RANGE')['SPEND'])
    fig = px.pie(age_group_spending, values='SPEND', names='AGE_RANGE', title='Spending Distribution by Age Group')
    st.plotly_chart(fig)
    
    # --- ML Analysis Tabs ---
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
        
        # ML Churn Prediction
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
        
        # Feature importance
        importances = pd.Series(rf.feature_importances_, index=X_basket.columns)
        top_features = importances.sort_values(ascending=False).head(10)
        st.write("**Top Drivers of Basket Spend:**")
        st.bar_chart(top_features)

# --- Household Search ---
elif page == "Household Search":
    st.title("Search Household Transactions")
    hshd_num = st.text_input("Enter Household Number (HSHD_NUM):")
    
    if hshd_num:
        try:
            hshd_num = int(hshd_num)
            
            # Load sample data
            df_transactions, df_households, df_products = load_sample_data()
            
            # Filter data based on household number
            household_data = df_households[df_households['HSHD_NUM'] == hshd_num]
            transactions = df_transactions[df_transactions['HSHD_NUM'] == hshd_num]
            
            if not transactions.empty:
                # Join with products
                result = transactions.merge(df_products, left_on='PRODUCT_NUM', right_on='PRODUCT_NUM', how='left')
                result = result[['HSHD_NUM', 'BASKET_NUM', 'PURCHASE_', 'PRODUCT_NUM', 'DEPARTMENT', 'COMMODITY']]
                st.dataframe(result)
            else:
                st.write("No data found for the entered Household Number.")
                
        except ValueError:
            st.error("Please enter a valid numeric Household Number.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- CLV Calculation ---
elif page == "CLV Calculation":
    st.title("Calculate and Save CLV")
    
    if st.button("Calculate CLV"):
        # Load sample data
        df_transactions, _, _ = load_sample_data()
        
        # Calculate CLV
        clv_df = df_transactions.groupby('HSHD_NUM')['SPEND'].sum().reset_index()
        clv_df.columns = ['HSHD_NUM', 'CLV']
        
        st.success("✅ CLV calculated successfully!")
        st.dataframe(clv_df)
        
        # Download option
        csv = clv_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CLV Results",
            data=csv,
            file_name='clv_results.csv',
            mime='text/csv',
        )

# --- Data Loader ---
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
        if st.button("Load Sample Data"):
            tdf, hdf, pdf = load_sample_data()
            st.session_state['transactions_df'] = tdf
            st.session_state['households_df'] = hdf
            st.session_state['products_df'] = pdf
            st.success("Loaded sample data.")
            st.write("Transactions Data", tdf.head())
            st.write("Households Data", hdf.head())
            st.write("Products Data", pdf.head())
