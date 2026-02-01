import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Konfigurasi visualisasi seaborn
sns.set(style='dark')

# --- Helper Functions (Data Aggregation & Transformation) ---

def create_daily_orders_df(df):
    """
    Melakukan resampling data time-series ke frekuensi harian ('D').
    Mengagregasi jumlah order unik (nunique) dan total revenue (sum) per hari.
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
    Mengelompokkan data (Group By) berdasarkan kategori produk.
    Menghitung total nilai moneter (Revenue) untuk setiap kategori dan mengurutkannya secara descending.
    """
    sum_order_items_df = df.groupby("product_category_name_english").price.sum().sort_values(ascending=False).reset_index()
    return sum_order_items_df

def create_by_city_df(df):
    """
    Mengagregasi data pelanggan berdasarkan kota (Customer Demographics).
    Menghitung distribusi jumlah pelanggan unik (unique counts) di setiap kota.
    """
    by_city_df = df.groupby(by="customer_city").customer_id.nunique().reset_index()
    by_city_df.rename(columns={"customer_id": "customer_count"}, inplace=True)
    return by_city_df

def create_rfm_df(df):
    """
    Menghitung metrik RFM (Recency, Frequency, Monetary) untuk segmentasi pelanggan.
    - Recency: Menghitung selisih hari (timedelta) antara tanggal analisis dan pembelian terakhir.
    - Frequency: Menghitung jumlah transaksi unik per pelanggan.
    - Monetary: Menjumlahkan total nilai transaksi (Revenue) per pelanggan.
    """
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max", # Mengambil tanggal transaksi terakhir
        "order_id": "nunique",             # Menghitung frekuensi transaksi
        "price": "sum"                     # Menghitung total monetary value
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    # Kalkulasi skor Recency
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    
    return rfm_df

# --- Load Data & Preprocessing ---

# Membaca dataset utama dari file CSV
all_df = pd.read_csv("dashboard/main_data.csv")

# Handling Missing Values / Schema Mismatch pada kolom kategori
if "product_category_name_english" not in all_df.columns:
    if "product_category_name" in all_df.columns:
        # Rename kolom existing jika translasi tidak tersedia
        all_df.rename(columns={"product_category_name": "product_category_name_english"}, inplace=True)
    else:
        # Imputasi nilai default jika data kategori hilang sepenuhnya
        all_df["product_category_name_english"] = "Unknown"

# Type Casting: Konversi kolom tanggal ke format datetime64[ns]
datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Sorting data time-series secara kronologis
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

# --- Sidebar Configuration & Dynamic Filtering ---

min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    # Menampilkan Logo Perusahaan
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    # Widget Date Input untuk filtering data dinamis
    try:
        start_date, end_date = st.date_input(
            label='Rentang Waktu',
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
    except ValueError:
        st.error("Input tanggal tidak valid, menggunakan rentang default.")
        start_date, end_date = min_date, max_date

# Filtering DataFrame utama berdasarkan rentang waktu yang dipilih user
main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                 (all_df["order_purchase_timestamp"] <= str(end_date))]

# --- Data Processing for Visualization ---

# Menjalankan fungsi helper untuk membuat DataFrame spesifik visualisasi
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
by_city_df = create_by_city_df(main_df)
rfm_df = create_rfm_df(main_df)

# --- Dashboard Layout & Visualization ---

st.header('Dicoding E-Commerce Dashboard :sparkles:')

# 1. Metric Visualization: Daily Orders & Revenue
st.subheader('Daily Orders & Revenue')
col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total Orders", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "AUD", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)

# Plotting Line Chart untuk Tren Penjualan
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

# 2. Bar Chart: Product Performance (By Revenue)
st.subheader("Best & Worst Performing Product (By Revenue)")
st.markdown("*Pertanyaan: Produk apa yang menyumbang pendapatan terbesar dan terendah?*")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

# Subplot 1: Best Performing Products
sns.barplot(x="price", y="product_category_name_english", data=sum_order_items_df.head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Total Revenue", fontsize=20)
ax[0].set_title("Best Performing Product", loc="center", fontsize=18)
ax[0].tick_params(axis='y', labelsize=15)

# Subplot 2: Worst Performing Products
sns.barplot(x="price", y="product_category_name_english", data=sum_order_items_df.sort_values(by="price", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Total Revenue", fontsize=20)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=18)
ax[1].tick_params(axis='y', labelsize=15)

st.pyplot(fig)

# 3. Bar Chart: Customer Demographics
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

# 4. Bar Chart: RFM Analysis Visualization
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

# Visualisasi Recency
temp_df_r = rfm_df.sort_values(by="recency", ascending=True).head(5)
sns.barplot(x="customer_id", y="recency", data=temp_df_r, palette=colors_rfm, hue="customer_id", legend=False, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Customer ID", fontsize=30)
ax[0].set_title("By Recency (Days)\n(Lower is Better)", loc="center", fontsize=40)
ax[0].tick_params(axis='x', labelsize=20, rotation=90)
ax[0].tick_params(axis='y', labelsize=30)

# Visualisasi Frequency
temp_df_f = rfm_df.sort_values(by="frequency", ascending=False).head(5)
sns.barplot(x="customer_id", y="frequency", data=temp_df_f, palette=colors_rfm, hue="customer_id", legend=False, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Customer ID", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=40)
ax[1].tick_params(axis='x', labelsize=20, rotation=90)
ax[1].tick_params(axis='y', labelsize=30)

# Visualisasi Monetary
temp_df_m = rfm_df.sort_values(by="monetary", ascending=False).head(5)
sns.barplot(x="customer_id", y="monetary", data=temp_df_m, palette=colors_rfm, hue="customer_id", legend=False, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("Customer ID", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=40)
ax[2].tick_params(axis='x', labelsize=20, rotation=90)
ax[2].tick_params(axis='y', labelsize=30)

# Menghilangkan spines (batas chart) untuk estetika
for a in ax:
    for spine in ['top', 'right']:
        a.spines[spine].set_visible(False)

st.pyplot(fig)

st.caption('Copyright (c) Dicoding 2023')