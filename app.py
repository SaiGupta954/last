# --- app.py (Full Corrected Version) ---

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
from sqlalchemy import text
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- App Config ---
st.set_page_config(page_title="Retail Analytics on Azure", layout="wide")

# --- Utility Functions ---
def find_col(df, *candidates):
    cols = {c.strip().lower(): c.strip() for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    raise KeyError(f"None of {candidates} found in columns")

# --- Azure SQL Connection using st.connection() ---
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
        st.sidebar.error(f"\u274C Database connection failed: {e}")
        return None

# --- Get Connection ---
conn = get_db_connection()

if conn is None:
    st.error("Database connection could not be established.")
    st.stop()

# --- Load Data ---
@st.cache_data(ttl=600)
def load_data(_conn):
    try:
        limit = 10000
        trans = _conn.query(f"SELECT TOP {limit} * FROM Transactions", ttl=600)
        hh    = _conn.query(f"SELECT TOP {limit} * FROM Households", ttl=600)
        prod  = _conn.query(f"SELECT TOP {limit} * FROM Products", ttl=600)

        hh.columns, trans.columns, prod.columns = map(lambda c: c.str.strip(), [hh.columns, trans.columns, prod.columns])

        date_col = find_col(trans, "date", "purchase", "purchase_")
        trans[date_col] = pd.to_datetime(trans[date_col], errors="coerce")
        trans = trans.rename(columns={date_col: "DATE"})

        merged = (trans
                  .merge(prod, on=find_col(trans,"product_num"), how="left")
                  .merge(hh,   on=find_col(trans,"hshd_num"),    how="left")
        )

        return hh, prod, trans, merged
    except Exception as e:
        st.error(f"Failed loading data: {e}")
        return None, None, None, None

hh, prod, trans, merged = load_data(conn)

if merged is None:
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("\u2699\ufe0f Analytics Controls")
hcol     = find_col(merged,"hshd_num")
bcol     = find_col(merged,"basket_num")
icol     = find_col(merged,"income_range")
min_sup  = st.sidebar.slider("Min Support", 0.0,0.10,0.01,step=0.005)
min_conf = st.sidebar.slider("Min Confidence", 0.10,1.00,0.30,step=0.05)
churn_w  = st.sidebar.slider("Churn window (days)",30,180,90,step=10)
sel      = st.sidebar.selectbox("Household #", sorted(merged[hcol].unique()), index=9, format_func=lambda x: f"{int(x):04d}")

# --- Header & KPIs ---
st.title("\U0001F6D2 Retail Analytics on Azure")
total_spend = merged["SPEND"].sum()
avg_basket  = merged.groupby([hcol,bcol])["SPEND"].sum().mean()
c1,c2,c3    = st.columns([1,1,2])
with c1: st.metric("\U0001F4B0 Total Spend", f"${total_spend:,.0f}")
with c2: st.metric("\U0001F6CD\ufe0f Avg Spend/Basket", f"${avg_basket:,.2f}")
with c3: st.markdown(f"**As of:** {merged['DATE'].max().strftime('%b %d, %Y')}")
st.markdown("---")

# --- Household Data Pull ---
st.subheader(f"\U0001F4CB Data Pull for Household #{int(sel):04d}")
dfp = (merged[merged[hcol]==sel]
       .sort_values([hcol,bcol,"DATE",find_col(merged,"product_num"),"DEPARTMENT","COMMODITY"]))
st.dataframe(dfp, height=200, use_container_width=True)
st.markdown("---")

# --- Time Series Analysis ---
colA,colB = st.columns(2, gap="large")
with colA:
    st.subheader("\U0001F4C8 Spend Over Time")
    ts = merged.groupby("DATE")["SPEND"].sum().reset_index()
    st.line_chart(ts.rename(columns={"DATE":"index"}).set_index("index")["SPEND"])
with colB:
    st.subheader("\U0001F3F7\ufe0f Avg Spend/Visit by Income")
    grp = merged.groupby([hcol,bcol,icol])["SPEND"].sum().reset_index()
    avg_inc = grp.groupby(icol)["SPEND"].mean().sort_values(ascending=False)
    st.bar_chart(avg_inc)
st.markdown("---")

# --- Basket Analysis (Apriori Rules) ---
st.subheader("\U0001F6D2 Basket Analysis")
baskets = (merged[merged[hcol]==sel]
           .groupby(bcol)[find_col(merged,"product_num")]
           .apply(lambda x: list(map(str,x))).tolist())
te    = TransactionEncoder()
df_tf = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
freq  = apriori(df_tf, min_support=min_sup, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=min_conf).sort_values("lift",ascending=False)

if rules.empty:
    st.info("No rules at these thresholds.")
else:
    rules["antecedents"] = rules["antecedents"].apply(lambda s:", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s:", ".join(sorted(s)))
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].head(10), height=200)

# --- Basket Analysis (Random Forest) ---
st.subheader("\U0001F527 ML Basket Analysis")
if len(baskets)>10:
    target = merged[merged[hcol]==sel]["PRODUCT_NUM"].astype(str).value_counts().idxmax()
    y = df_tf[target]
    X = df_tf.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    st.markdown(f"**Predicting presence of product {target} → accuracy: {acc:.2%}**")
    imp = pd.Series(clf.feature_importances_, index=X.columns).nlargest(10)
    fig, ax = plt.subplots(figsize=(6,3))
    imp.plot(kind="bar", ax=ax)
    ax.set_xlabel("Product Num"); ax.set_ylabel("Importance")
    fig.tight_layout(); st.pyplot(fig)
else:
    st.info("Not enough baskets for ML analysis.")

# --- Churn Prediction ---
st.subheader("\U0001F6A8 Churn Prediction")
maxd = merged["DATE"].max()
rfm  = merged.groupby(hcol).agg(
    recency=("DATE", lambda x: (maxd-x.max()).days),
    frequency=(bcol,"nunique"),
    monetary=("SPEND","sum")
).reset_index()
rfm["churn"] = (rfm["recency"]>churn_w).astype(int)
rate = rfm["churn"].mean()

if rate>0:
    Xc = rfm[["recency","frequency","monetary"]]
    yc = rfm["churn"]
    model = LogisticRegression(max_iter=500).fit(Xc,yc)
    row = rfm.loc[rfm[hcol]==sel,["recency","frequency","monetary"]]
    p   = model.predict_proba(row)[0,1]
    st.metric("Predicted Churn Risk",f"{p*100:.1f}%",delta=f"{rate*100:.1f}%")
else:
    st.warning("No churn cases; try changing the window.")

st.markdown("""
**How it works**
- Recency = days since last purchase
- Frequency = # of baskets
- Monetary = total spend
- Logistic regression on RFM
""")

# --- Done ---
