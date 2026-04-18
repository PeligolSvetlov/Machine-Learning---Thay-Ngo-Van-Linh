# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

base_path = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
file_path = os.path.join(base_path, 'Bengaluru_House_Data.csv')

try:
    df = pd.read_csv(file_path)
    print(">>> Nạp dữ liệu thành công!")
except:
    print(f"LỖI: Không tìm thấy file tại {file_path}")
    exit()

# %%
print("--- 5 dòng đầu tiên ---")
print(df.head())


print("\n--- 5 dòng cuối cùng ---")
print(df.tail())

print(f"\nKích thước dữ liệu (hàng, cột): {df.shape}")
    
# %%

print("--- Thông tin tổng quan ---")
df.info()

print("\n--- Thống kê mô tả ---")
print(df.describe())

def quick_count(df):
    for col in df.columns:
        print(df[col].value_counts())
        print("-" * 30)
        
quick_count(df)


# %% 
sns.pairplot(df)
plt.show()
num_vars = ["bath", "balcony", "price"]
sns.heatmap(df[num_vars].corr(),cmap="coolwarm", annot=True)
plt.show()
# %%
print(df.isnull().sum())
print()
print(df.isnull().mean()*100)

#%%
df2 = df.drop('society', axis = 'columns')
print(df2.shape)
print()
df2['balcony'] = df2['balcony'].fillna(df2['balcony'].mean())
print(df2.isnull().sum())

#%%
df3 = df2.dropna()
print(df3.shape)
print()
print(df3.isnull().sum())

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

df3['total_sqft'].value_counts()

#%%
total_sqft_float = []
for str_val in df3['total_sqft']:
    try:
        total_sqft_float.append(float(str_val))
    except:
        try:
            temp = []
            temp = str_val.split('-')
            total_sqft_float.append((float(temp[0])+float(temp[-1]))/2)
        except:
            total_sqft_float.append(np.nan)
            
df4 = df3.reset_index(drop=True)

df5 = df4.join(pd.DataFrame({'total_sqft_float':total_sqft_float}))
print(df5.head())
print()
print(df5.isnull().sum())
print()

# %%
df6 = df5.dropna()
print(df6.shape)
print()
print(df6.info())
print()
print(df6['size'].value_counts())

# %%
size_int = []
for str_val in df6['size']:
    temp=[]
    temp = str_val.split(" ")
    try:
        size_int.append(int(temp[0]))
    except:
        size_int.append(np.nan)
        print("Noice = ",str_val)
        
df6 = df6.reset_index(drop=True)

# %%

df7 = df6.join(pd.DataFrame({'bhk':size_int}))
print(df7.shape)
print()
print(df7.tail())

sns.boxplot(x = df7['total_sqft_float'])

df7[df7['total_sqft_float']/df7['bhk'] < 350].head()

# %%
df8 = df7[~(df7['total_sqft_float']/df7['bhk'] < 350)]
print(df8.shape)
print()
df8['price_per_sqft'] = df8['price']*100000 / df8['total_sqft_float']
print(df8.head())
print()
print(df8.price_per_sqft.describe())


# %%
#Bai tap 0
vars = ['price', 'total_sqft_float', 'price_per_sqft', 'balcony', 'bath', 'bhk']
plt.figure(figsize=(16,12))

for i, var in enumerate(vars):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x = df8[var], color = 'skyblue')
    
    plt.title(f'Phân bổ Outlier của {var}')
    plt.grid(True, axis = 'x', linestyle = '--', alpha = 0.7)
    
plt.tight_layout()
plt.show()

# %%
#Bai tap 1
def remove_pps_outliers(df):
    out = pd.DataFrame()
    
    for i, j in df.groupby('location'):
        m = np.mean(j.price_per_sqft)
        n = np.std(j.price_per_sqft)
        
        reduce = j[(j.price_per_sqft > (m - n)) & (j.price_per_sqft <= (m + n))]
        out = pd.concat([out, reduce], ignore_index=True)
    
    return out

df9 = remove_pps_outliers(df8)
print(f"Kich thuoc truoc khi loc: {df8.shape}")
print(f"Kich thuoc sau khi loc: {df9.shape}")

# %%
#Bai tap 2
def reomove_bhk_ouliers(df):
    k = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {'mean': np.mean(bhk_df.price_per_sqft), 'std': np.std(bhk_df.price_per_sqft), 'count': bhk_df.shape[0]}
            
        for bhk, bhk_df in location_df.groupby('bhk'):
            prev = bhk_stats.get(bhk - 1)
            
            if prev and prev['count'] > 5:
                k = np.append(k, bhk_df[bhk_df.price_per_sqft < prev['mean']].index.values)
    
    return df.drop(k, axis = 'index')

df10 = reomove_bhk_ouliers(df9)
print(df10.shape)

# %%
#Bai tap 3
print(df10.bath.unique())
print()
print(df10[df10.bath > df10.bhk+2])
print()
df11 = df10[df10.bath < df10.bhk+2]
print(df11.shape)
print(df11.head())

# %%
#Bai tap 4
df12 = df11.drop(['area_type',  'availability','location', 'size', 'total_sqft'], axis = 'columns')
print(df12.head())
df12.to_csv("clean_data.csv", index=False)

# %%
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('1. Bản đồ dữ liệu thiếu (Màu vàng là ô trống)')
plt.show()

# %%
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if pd.notnull(x) else 0)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['price'], color='salmon')
plt.title('2a. Phân bổ Giá nhà (Nhiều Outlier)')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['bhk'], color='skyblue')
plt.title('2b. Phân bổ Số phòng ngủ (BHK)')
plt.show()

# %%
# Làm sạch diện tích để vẽ biểu đồ
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2: return (float(tokens[0]) + float(tokens[1])) / 2
    try: return float(x)
    except: return None

df_clean = df.copy()
df_clean['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df_clean = df_clean.dropna(subset=['total_sqft', 'price'])

plt.figure(figsize=(10, 5))
sns.histplot(df_clean['total_sqft'], bins=50, kde=True, color='purple')
plt.xlim(0, 5000) # Giới hạn để nhìn rõ hơn
plt.title('3. Phân bổ diện tích nhà (Dưới 5000 sqft)')
plt.show()

# %%
#Bai tap 5
def plot_scatter_chart(data, location):
    bhk2 = data[(data.location == location) & (data.bhk == 2)]
    bhk3 = data[(data.location == location) & (data.bhk == 3)]
    plt.figure(figsize=(10, 6))
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Diện tích (sqft)")
    plt.ylabel("Giá (Lakhs)")
    plt.title(f'4. So sánh giá nhà tại: {location}')
    plt.legend()
    plt.show()

plot_scatter_chart(df9, "Rajaji Nagar")

# %%
plt.figure(figsize=(10, 6))
df.groupby('area_type')['price'].mean().sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title('5. Giá nhà trung bình theo Loại khu vực')
plt.ylabel('Giá trung bình (Lakhs)')
plt.xticks(rotation=30)
plt.show()
# %%
area_stats = df9.groupby('area_type')['price_per_sqft'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))

area_stats.plot(kind='bar', color='teal', width=0.6)

plt.title('Giá nhà trung bình (per Sqft) theo Loại khu vực', fontsize=14, fontweight='bold')
plt.ylabel('Giá trung bình (Rupee/sqft)', fontsize=12)
plt.xlabel('Loại khu vực', fontsize=12)

plt.xticks(rotation=30, ha='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(area_stats):
    plt.text(i, v + 100, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
# %%
