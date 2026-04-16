import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

base_path = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
file_path = os.path.join(base_path, 'Loan Modelling Thera Bank.csv')

try:
    loan_dataset = pd.read_csv(file_path)
    print(">>> Nạp dữ liệu thành công!")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file tại {file_path}. Hãy kiểm tra lại thư mục!")
    exit()

X = loan_dataset.drop(['Personal Loan', 'ID', 'ZIP Code'], axis=1)
y = loan_dataset['Personal Loan']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=5)

k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
k_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, knn.predict(X_test)))

plt.figure(figsize=(8, 4))
plt.plot(k_values, k_scores, marker='o', linestyle='--', color='blue')
plt.title('Tìm giá trị K tối ưu cho KNN')
plt.xlabel('Số lượng láng giềng (K)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


knn_model = KNeighborsClassifier(n_neighbors=2, weights='distance')
knn_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=51)
rf_model.fit(X_train, y_train)

print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"KNN Accuracy: {accuracy_score(y_test, knn_model.predict(X_test)):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")

new_customer_data = pd.DataFrame({
    'Age': [25], 'Experience': [1], 'Income': [49], 'Family': [4], 
    'CCAvg': [1.6], 'Education': [1], 'Mortgage': [0], 
    'Securities Account': [1], 'CD Account': [0], 'Online': [0], 'CreditCard': [0]
})


new_customer_scaled = scaler.transform(new_customer_data)


prediction = rf_model.predict(new_customer_scaled)
result = "Được duyệt vay" if prediction[0] == 1 else "Không được duyệt vay"

print(f"\nDự đoán cho khách hàng mới: {result}")