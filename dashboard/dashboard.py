import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from babel.numbers import format_currency

# Set aesthetic style for plots
sns.set(style='dark')

# --- Helper Functions (Data Aggregation) ---

def create_daily_orders_df(df):
    """
    Resample data to daily frequency to calculate total orders and revenue.
    """
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)
    return daily_orders_df

def create_sum_order_items_df(df):
    """
    Calculate total revenue contribution per product category.
    """
    sum_order_items_df = df.groupby("product_category_name_english").price.sum().sort_values(ascending=False).reset_index()
    return sum_order_items_df

def create_by_city_df(df):
    """
    Aggregate customer distribution by city.
    """
    by_city_df = df.groupby(by="customer_city").customer_id.nunique().reset_index()
    by_city_df.rename(columns={"customer_id": "customer_count"}, inplace=True)
    return by_city_df

def create_rfm_df(df):
    """
    Compute RFM metrics (Recency, Frequency, Monetary) for customer segmentation.
    """
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max", 
        "order_id": "nunique", 
        "price": "sum"
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    
    return rfm_df

# --- Data Loading & Preprocessing ---

# Define dynamic file path to ensure compatibility across different environments
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'main_data.csv')
all_df = pd.read_csv(file_path)

# Standardize column names for product categories (prefer English)
if "product_category_name_english" in all_df.columns:
    pass 
elif "category" in all_df.columns:
    all_df.rename(columns={"category": "product_category_name_english"}, inplace=True)
elif "product_category_name" in all_df.columns:
    all_df.rename(columns={"product_category_name": "product_category_name_english"}, inplace=True)
else:
    all_df["product_category_name_english"] = "Unknown"

# Convert timestamp columns to datetime objects
datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Sort data chronologically
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

# --- Sidebar Configuration (Filter) ---

min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    # Date Range Filter
    try:
        start_date, end_date = st.date_input(
            label='Rentang Waktu',
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
    except ValueError:
        st.error("Invalid date range selected.")
        start_date, end_date = min_date, max_date

# Apply filter to main dataframe
main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_df["order_purchase_timestamp"] <= str(end_date))]

# --- Data Visualization Preparation ---
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
by_city_df = create_by_city_df(main_df)
rfm_df = create_rfm_df(main_df)

# --- Dashboard Layout ---

st.header('Dicoding E-Commerce Dashboard :sparkles:')

# 1. Sales Trend Section
st.subheader('Daily Orders & Revenue')
col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total Orders", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "AUD", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

# 2. Product Performance Section
st.subheader("Best & Worst Performing Product (By Revenue)")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

# Top 5 Products
sns.barplot(x="price", y="product_category_name_english", data=sum_order_items_df.head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Total Revenue", fontsize=20)
ax[0].set_title("Best Performing Product", loc="center", fontsize=18)
ax[0].tick_params(axis='y', labelsize=15)

# Bottom 5 Products
sns.barplot(x="price", y="product_category_name_english", data=sum_order_items_df.sort_values(by="price", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Total Revenue", fontsize=20)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=18)
ax[1].tick_params(axis='y', labelsize=15)

st.pyplot(fig)

# 3. Customer Demographics Section
st.subheader("Customer Demographics")
fig, ax = plt.subplots(figsize=(20, 10))
colors_demog = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x="customer_count", 
    y="customer_city",
    data=by_city_df.sort_values(by="customer_count", ascending=False).head(5),
    palette=colors_demog,
    ax=ax
)
ax.set_title("Number of Customer by City", loc="center", fontsize=30)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
st.pyplot(fig)

# 4. RFM Analysis Section
st.subheader("Best Customer Based on RFM Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_monetary = format_currency(rfm_df.monetary.mean(), "AUD", locale='es_CO') 
    st.metric("Average Monetary", value=avg_monetary)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors_rfm = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

# Recency Plot
sns.barplot(x="customer_id", y="recency", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors_rfm, hue="customer_id", legend=False, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Customer ID", fontsize=30)
ax[0].set_title("By Recency (Days)", loc="center", fontsize=40)
ax[0].tick_params(axis='x', labelsize=20, rotation=90)
ax[0].tick_params(axis='y', labelsize=30)

# Frequency Plot
sns.barplot(x="customer_id", y="frequency", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors_rfm, hue="customer_id", legend=False, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Customer ID", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=40)
ax[1].tick_params(axis='x', labelsize=20, rotation=90)
ax[1].tick_params(axis='y', labelsize=30)

# Monetary Plot
sns.barplot(x="customer_id", y="monetary", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors_rfm, hue="customer_id", legend=False, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("Customer ID", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=40)
ax[2].tick_params(axis='x', labelsize=20, rotation=90)
ax[2].tick_params(axis='y', labelsize=30)

# Clean up chart aesthetics
for a in ax:
    for spine in ['top', 'right']:
        a.spines[spine].set_visible(False)

st.pyplot(fig)

# --- Strategic Insights Section ---
st.markdown("---")
st.subheader("📋 Conclusion & Recommendations")

tab1, tab2, tab3 = st.tabs(["💡 Product Strategy", "📈 Sales Trend", "🤝 Customer Retention"])

with tab1:
    st.markdown("""
    **Kesimpulan Produk:**
    * **Bed Bath Table** & **Health Beauty** adalah kategori produk dengan performa pendapatan tertinggi.
    * **Security and Services** memiliki performa terendah, memerlukan evaluasi strategis.
    
    **Rekomendasi:**
    * Jaga ketersediaan stok (*Safety Stock*) pada kategori *Best Seller*.
    * Pertimbangkan strategi *bundling* atau promosi untuk kategori dengan performa rendah.
    """)

with tab2:
    st.markdown("""
    **Analisis Tren:**
    * Tren pendapatan perusahaan menunjukkan pola **Positif (Meningkat)**.
    * Terdapat lonjakan signifikan di bulan **November 2017** (indikasi efek musiman/akhir tahun).
    
    **Rekomendasi:**
    * Fokuskan anggaran pemasaran terbesar pada Q4 (Oktober-November) untuk memanfaatkan momentum puncak.
    * Siapkan promosi tematik di awal tahun untuk menjaga stabilitas penjualan.
    """)

with tab3:
    st.markdown("""
    **Analisis RFM:**
    * **Recency:** Mayoritas pelanggan sudah lama tidak bertransaksi.
    * **Frequency:** Sebagian besar pelanggan hanya berbelanja satu kali (*Low Retention*).
    * **Monetary:** Terdapat segmen VIP yang berkontribusi signifikan terhadap revenue.
    
    **Rekomendasi:**
    * Luncurkan **Loyalty Program** untuk meningkatkan retensi pelanggan.
    * Berikan penawaran eksklusif bagi pelanggan VIP untuk mempertahankan loyalitas mereka.
    """)
