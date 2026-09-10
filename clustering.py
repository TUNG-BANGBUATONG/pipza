import requests
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. ดึงข้อมูลจาก API ตัวเอง (พอร์ต 3040)
API_URL = "http://119.59.102.161:3040/api/products"
res = requests.get(API_URL)
data = res.json()

# แปลงเป็น DataFrame
df = pd.DataFrame(data)

# แปลงข้อมูลตัวเลขให้พร้อมคำนวณ
df['price'] = pd.to_numeric(df['price'])
df['stock'] = pd.to_numeric(df['stock'])

# 2. ทำ Data Scaling (สไลด์หน้า 8)
features = df[['price', 'stock']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 3. รันโมเดล K-Means (แบ่ง 3 กลุ่ม ตามสไลด์หน้า 8)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_features)

# 4. แสดงผลตารางใน Terminal
print("\n========== K-MEANS CLUSTERING RESULT ==========")
print(df[['id', 'name', 'price', 'stock', 'cluster']])
print("================================================\n")

# 5. พล็อตกราฟ Scatter Plot (ตามสไลด์หน้า 10)
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green']
for cluster_id in range(3):
    cluster_data = df[df['cluster'] == cluster_id]
    plt.scatter(
        cluster_data['price'], 
        cluster_data['stock'], 
        label=f'Cluster {cluster_id}',
        s=120
    )

plt.title('Gunpla K-Means Clustering (Price vs Stock)')
plt.xlabel('Product Price (THB)')
plt.ylabel('Units in Stock')
plt.legend()
plt.grid(True)
plt.show()