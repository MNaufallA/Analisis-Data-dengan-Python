import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Data Wrangling

### Gathering Data

#### Load Tabel Stasiun Aotizhongxin
Aotizhongxin_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Aotizhongxin_20130301-20170228.csv')
Aotizhongxin_df.head()

#### Load Tabel Stasiun Changping
Changping_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Changping_20130301-20170228.csv')
Changping_df.head()

#### Load Tabel Stasiun Dingling
Dingling_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Dingling_20130301-20170228.csv')
Dingling_df.head()

#### Load Tabel Stasiun Dongsi
Dongsi_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Dongsi_20130301-20170228.csv')
Dongsi_df.head()

#### Load Tabel Stasiun Guanyuan
Guanyuan_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Guanyuan_20130301-20170228.csv')
Guanyuan_df.head()

#### Load Tabel Stasiun Gucheng
Gucheng_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Gucheng_20130301-20170228.csv')
Gucheng_df.head()

#### Load Tabel Stasiun Huairou
Huairou_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Huairou_20130301-20170228.csv')
Huairou_df.head()

#### Load Tabel Stasiun Nongzhanguan
Nongzhanguan_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Nongzhanguan_20130301-20170228.csv')
Nongzhanguan_df.head()

#### Load Tabel Stasiun Shunyi
Shunyi_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Shunyi_20130301-20170228.csv')
Shunyi_df.head()

#### Load Tabel Stasiun Tiantan
Tiantan_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Tiantan_20130301-20170228.csv')
Tiantan_df.head()

#### Load Tabel Stasiun Wanliu
Wanliu_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Wanliu_20130301-20170228.csv')
Wanliu_df.head()

#### Load Tabel Stasiun Wanshouxigong

Wanshouxigong_df = pd.read_csv('https://raw.githubusercontent.com/MNaufallA/Analisis-Data-dengan-Python/main/submission/data/PRSA_Data_Wanshouxigong_20130301-20170228.csv')
Wanshouxigong_df.head()


### Cleaning Data

#### Membersihkan Data Aotizhongxin_df
##### Menangani missing value
# Metode interpolasi
Aotizhongxin_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Aotizhongxin_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Aotizhongxin_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Aotizhongxin_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Aotizhongxin_df.isna().sum()

#### Membersihkan Data Changping_df
##### Menangani missing value
# Metode interpolasi
Changping_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Changping_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Changping_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Changping_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Aotizhongxin_df.isna().sum()

#### Membersihkan Data Dingling_df
##### Menangani missing value
# Metode interpolasi
Dingling_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Dingling_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Dingling_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Dingling_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Aotizhongxin_df.isna().sum()

#### Membersihkan Data Dongsi_df
##### Menangani missing value
# Metode interpolasi
Dongsi_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Dongsi_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Dongsi_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Dongsi_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Dongsi_df.isna().sum()

#### Membersihkan Data Guanyuan_df
##### Menangani missing value
# Metode interpolasi
Guanyuan_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Guanyuan_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Guanyuan_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Guanyuan_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Guanyuan_df.isna().sum()

#### Membersihkan Data Gucheng_df
##### Menangani missing value
# Metode interpolasi
Gucheng_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Gucheng_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Gucheng_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Gucheng_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')

# Metode backward fill
Gucheng_df[['NO2']] = Gucheng_df[['NO2']].fillna(method='bfill')
Gucheng_df.isna().sum()

#### Membersihkan Data Huairou_df
##### Menangani missing value
# Metode interpolasi
Huairou_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Huairou_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Huairou_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Huairou_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Guanyuan_df.isna().sum()

#### Membersihkan Data Nongzhanguan_df
##### Menangani missing value
# Metode interpolasi
Nongzhanguan_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Nongzhanguan_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Nongzhanguan_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Nongzhanguan_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Nongzhanguan_df.isna().sum()

#### Membersihkan Data Shunyi_df
##### Menangani missing value
# Metode interpolasi
Shunyi_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Shunyi_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Shunyi_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Shunyi_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Shunyi_df.isna().sum()

#### Membersihkan Data Tiantan_df
##### Menangani missing value
# Metode interpolasi
Tiantan_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Tiantan_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Tiantan_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Tiantan_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Tiantan_df.isna().sum()

#### Membersihkan Data Wanliu_df
##### Menangani missing value
# Metode interpolasi
Wanliu_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Wanliu_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Wanliu_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Wanliu_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Wanliu_df.isna().sum()

#### Membersihkan Data Wanshouxigong_df
##### Menangani missing value
# Metode interpolasi
Wanshouxigong_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] = Wanshouxigong_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].interpolate(method='linear')

# Metode forward fill
Wanshouxigong_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']] = Wanshouxigong_df[['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']].fillna(method='ffill')
Wanshouxigong_df.isna().sum()


## Exploratory Data Analysis (EDA)
### Explore Data all_df
#### Penggabungan seluruh DataFrame
# List semua dataframe
list_df = [Aotizhongxin_df, Changping_df, Dingling_df, Dongsi_df, Guanyuan_df, Gucheng_df, Huairou_df, Nongzhanguan_df, Shunyi_df, Tiantan_df, Wanliu_df, Wanshouxigong_df]

# Gabungkan semua dataframe
all_df = pd.concat(list_df, ignore_index=True)
all_df.head()
all_df.isna().sum()
print("Jumlah duplikasi: ",all_df.duplicated().sum())
all_df.describe()

# Metode interpolasi
all_df[['NO2']] = all_df[['NO2']].interpolate(method='linear')
all_df.isna().sum()


#### Explore variabel polutan dan stasiun

# Mendefinisikan variabel
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
stations = ['Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']

# Memuat pivot table dengan 'station' sebagai indeks dan rata-rata untuk setiap polutan
pivot_table = all_df.pivot_table(index='station', values=pollutants, aggfunc='mean')
print(pivot_table)


#### Explore variabel lingkungan dan polutan
# Mendefinisikan variabel
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
environmental_conditions = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']

# Buat pivot table dengan 'environmental_conditions' sebagai indeks dan rata-rata untuk setiap polutan
pivot_table = all_df.pivot_table(index=environmental_conditions, values=pollutants, aggfunc='mean')
print(pivot_table)


#### Explore variabel polutan dan waktu (tahun)
# Mendefinisikan variabel
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# Buat pivot table dengan 'tahun' sebagai indeks dan rata-rata untuk setiap polutan
pivot_table = all_df.pivot_table(index='year', values=pollutants, aggfunc='mean')
print(pivot_table)


#### Explore variabel waktu dan stasiun
# Mendifinisikan variabel
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
stations = ['Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']

# Buat pivot table dengan 'tahun' dan 'stasiun' sebagai indeks dan rata-rata untuk setiap polutan
pivot_table = all_df.pivot_table(index=['year', 'station'], values=pollutants, aggfunc='mean')
print(pivot_table)



## Visualization & Explanatory Analysis

### Pertanyaan 1: Bagaimana perbandingan tingkat polutan total antar stasiun?
# Mendifinikan variabel
pollutants = ['SO2', 'NO2', 'O3', 'PM2.5', 'PM10', 'CO']
stations = ['Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']

# Groupby 'station' dan hitung rata-rata untuk setiap polutan
grouped = all_df.groupby('station')[pollutants].mean()

# Mengurutkan data berdasarkan total polutan
grouped['total'] = grouped.sum(axis=1)
grouped = grouped.sort_values('total', ascending=False)

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))

# Plot gabungan dengan urutan station dari yang terbesar
grouped[pollutants].plot(kind='barh', stacked=True, ax=ax, title='Perbandingan Tingkat Polutan Total Antar Station')
ax.set_xlabel('Konsentrasi')
ax.invert_yaxis()  # Membalik urutan station

plt.tight_layout()
plt.show()



### Pertanyaan 2: Bagaimana perbandingan tiap polutan pada tiap stasiun?
#### Polutan SO2 (Sulfur Dioksida)
# Plot untuk SO2
fig, ax = plt.subplots(figsize=(10, 7))
grouped['SO2'].sort_values(ascending=False).plot(kind='bar', color='b', ax=ax, title='Perbandingan Tingkat SO2 Antar Station')
ax.set_ylim([0, grouped['SO2'].max() * 1.1])  # Meningkatkan tinggi sumbu y
ax.set_ylabel('Konsentrasi')
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.show()

#### Polutan NO2 (Nitrogen Dioksida)
# Plot untuk NO2
fig, ax = plt.subplots(figsize=(10, 7))
grouped['NO2'].sort_values(ascending=False).plot(kind='bar', color='g', ax=ax, title='Perbandingan Tingkat NO2 Antar Station')
ax.set_ylim([0, grouped['NO2'].max() * 1.1])  # Meningkatkan tinggi sumbu y
ax.set_ylabel('Konsentrasi')
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.show()

#### Polutan O3 (Ozon)
# Plot untuk O3
fig, ax = plt.subplots(figsize=(10, 7))
grouped['O3'].sort_values(ascending=False).plot(kind='bar', color='r', ax=ax, title='Perbandingan Tingkat O3 Antar Station')
ax.set_ylim([0, grouped['O3'].max() * 1.1])  # Meningkatkan tinggi sumbu y
ax.set_ylabel('Konsentrasi')
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.show()

#### Polutan PM2.5 (Partikulat 2.5μm)
# Plot untuk PM2.5
fig, ax = plt.subplots(figsize=(10, 7))
grouped['PM2.5'].sort_values(ascending=False).plot(kind='bar', color='c', ax=ax, title='Perbandingan Tingkat PM2.5 Antar Station')
ax.set_ylim([0, grouped['PM2.5'].max() * 1.1])  # Meningkatkan tinggi sumbu y
ax.set_ylabel('Konsentrasi')
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.show()

#### Polutan PM10 (Partikulat 10μm)
# Plot untuk PM10
fig, ax = plt.subplots(figsize=(10, 7))
grouped['PM10'].sort_values(ascending=False).plot(kind='bar', color='m', ax=ax, title='Perbandingan Tingkat PM10 Antar Station')
ax.set_ylim([0, grouped['PM10'].max() * 1.1])  # Meningkatkan tinggi sumbu y
ax.set_ylabel('Konsentrasi')
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.show()

#### Polutan CO (Karbon Monoksida)
# Plot untuk CO
fig, ax = plt.subplots(figsize=(10, 7))
grouped['CO'].sort_values(ascending=False).plot(kind='bar', color='y', ax=ax, title='Perbandingan Tingkat CO Antar Station')
ax.set_ylim([0, grouped['CO'].max() * 1.1])  # Meningkatkan tinggi sumbu y
ax.set_ylabel('Konsentrasi')
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.show()



### Pertanyaan 3: Bagaimana pengaruh kondisi lingkungan terhadap tingkat polutan?
#### Pengaruh curah hujan terhadap kadar CO (Karbon Monoksida)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_df, x='RAIN', y='CO', hue='station')
plt.title('Scatter plot of CO vs RAIN for each station')
plt.show()

#### Pengaruh temperatur terhadap kadar O3 (Ozon)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_df, x='TEMP', y='O3', hue='station')
plt.title('Scatter plot of O3 vs TEMP for each station')
plt.show()



### Pertanyaan 4: Bagaimana fluktuasi tingkat polutan sepanjang waktu?
#### Fluktuasi Karbon Monoksida sepanjang waktu
plt.figure(figsize=(10, 6))

# Buat line chart untuk CO
all_df.groupby('year')['CO'].mean().plot(label='CO')

plt.title('Fluctuation of CO over years')
plt.xlabel('Year')
plt.ylabel('CO concentration')
plt.legend()
plt.show()

#### Fluktuasi polutan lain sepanjang waktu
# Daftar variabel polutan kecuali CO
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']

plt.figure(figsize=(10, 6))

# Buat line chart untuk setiap polutan
for pollutant in pollutants:
    all_df.groupby('year')[pollutant].mean().plot(label=pollutant)

plt.title('Fluctuation of pollutants over years')
plt.xlabel('Year')
plt.ylabel('Pollutants concentration')
plt.legend()
plt.show()




### Pertanyaan 5: Bagaimana distribusi stasiun berdasarkan tingkat polutan dan kondisi lingkungan? (Cluster)
all_df = pd.get_dummies(all_df, columns=['wd'])

# Pilih fitur yang akan digunakan untuk clustering
features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
            'wd_E', 'wd_ENE', 'wd_ESE', 'wd_N', 'wd_NE', 'wd_NNE', 'wd_NNW', 'wd_NW', 'wd_S',
            'wd_SE', 'wd_SSE', 'wd_SSW', 'wd_SW', 'wd_W', 'wd_WNW', 'wd_WSW']
X = all_df[features]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tentukan jumlah cluster
n_clusters = 3

# Lakukan clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
all_df['cluster'] = kmeans.fit_predict(X_scaled)

# Cetak hasil clustering
for i in range(n_clusters):
    print(f"Cluster {i}:")
    print(all_df[all_df['cluster'] == i])

# Visualisasi hasil clustering
plt.figure(figsize=(10, 7))
sns.scatterplot(data=all_df, x='PM2.5', y='PM10', hue='cluster', palette='viridis')
plt.title('Hasil Clustering')
plt.xlabel('PM2.5')
plt.ylabel('PM10')
plt.show()
