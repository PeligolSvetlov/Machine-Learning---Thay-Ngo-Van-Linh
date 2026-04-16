# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


base_path = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
file_path = os.path.join(base_path, 'Loan Modelling Thera Bank.csv')

try:
    loan_dataset = pd.read_csv(file_path)
    print(">>> Nạp dữ liệu thành công!")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file tại {file_path}")
    exit()



# %%
# Biểu đồ 1: Kiểm tra độ lệch dữ liệu
plt.figure(figsize=(6, 4))
sns.countplot(x='Personal Loan', data=loan_dataset, palette='viridis')
plt.title('Phân bổ khách hàng Chấp nhận vay (1) và Từ chối (0)')
plt.show()

# Biểu đồ 2: Ma trận tương quan
plt.figure(figsize=(12, 8))
correlation = loan_dataset.drop(['ID', 'ZIP Code'], axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan giữa các đặc trưng')
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.histplot(data=loan_dataset, x='Income', hue='Personal Loan', kde=True, palette='magma')
plt.title('Mối liên hệ giữa Thu nhập và Khả năng vay vốn')
plt.show()
# %%
X = loan_dataset.drop(['Personal Loan', 'ID', 'ZIP Code'], axis=1)
y = loan_dataset['Personal Loan']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=5)
print("Dữ liệu đã được chuẩn hóa và chia tập thành công.")


# %%
k_values = range(1, 21)
k_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, knn.predict(X_test)))

plt.figure(figsize=(8, 4))
plt.plot(k_values, k_scores, marker='o', color='green')
plt.title('Độ chính xác theo giá trị K (Elbow Method)')
plt.xlabel('K neighbors')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# %%
knn_model = KNeighborsClassifier(n_neighbors=2, weights='distance').fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=20, random_state=51).fit(X_train, y_train)

print("--- SO SÁNH ĐỘ CHÍNH XÁC ---")
print(f"KNN Accuracy: {accuracy_score(y_test, knn_model.predict(X_test)):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")


# %%
new_customer = pd.DataFrame({
    'Age': [25], 'Experience': [1], 'Income': [49], 'Family': [4], 
    'CCAvg': [1.6], 'Education': [1], 'Mortgage': [0], 
    'Securities Account': [1], 'CD Account': [0], 'Online': [0], 'CreditCard': [0]
})

new_customer_scaled = scaler.transform(new_customer)

prediction = rf_model.predict(new_customer_scaled)
print(f"Kết quả dự đoán (Random Forest): {'Được duyệt vay' if prediction[0] == 1 else 'Không được duyệt vay'}")
# %%
