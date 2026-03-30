# ==========================================
# PHẦN 1: NẠP THƯ VIỆN & CHUẨN BỊ DỮ LIỆU
# ==========================================

# Nạp thư viện pandas để đọc và xử lý dữ liệu dạng bảng (DataFrame) [cite: 2]
import pandas as pd
# Nạp thư viện numpy để tính toán các phép toán mảng và ma trận [cite: 3]
import numpy as np
# Nạp thư viện matplotlib.pyplot để vẽ các biểu đồ cơ bản [cite: 5]
import matplotlib.pyplot as plt
# Nạp thư viện seaborn để vẽ biểu đồ thống kê trực quan và đẹp mắt hơn [cite: 6]
import seaborn as sns

# Nạp hàm chia tập dữ liệu thành phần huấn luyện (train) và kiểm thử (test) [cite: 558]
from sklearn.model_selection import train_test_split
# Nạp thuật toán phân loại K-Nearest Neighbors (KNN) [cite: 567]
from sklearn.neighbors import KNeighborsClassifier
# Nạp các hàm tính toán điểm đánh giá mô hình: Độ chính xác, Độ chuẩn xác, Độ phủ [cite: 567]
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Nạp thuật toán phân loại Naive Bayes (phiên bản phân phối Gaussian) [cite: 893]
from sklearn.naive_bayes import GaussianNB
# Nạp thư viện Support Vector Machine (SVM) [cite: 904]
from sklearn import svm
# Nạp thuật toán Hồi quy Logistic (Logistic Regression) [cite: 917]
from sklearn.linear_model import LogisticRegression
# Nạp thuật toán Cây quyết định (Decision Tree) [cite: 926]
from sklearn.tree import DecisionTreeClassifier
# Nạp thuật toán Rừng ngẫu nhiên (Random Forest) [cite: 938]
from sklearn.ensemble import RandomForestClassifier

# 1. ĐỌC DỮ LIỆU
# Đọc file dữ liệu dạng CSV và lưu vào biến loan_dataset [cite: 10]
loan_dataset = pd.read_csv('Loan Modelling Thera Bank.csv') 

# 2. TIỀN XỬ LÝ DỮ LIỆU (BASELINE)
# Tạo tập đặc trưng X: Cắt bỏ cột mục tiêu 'Personal Loan' và cột 'ID' (do ID không mang ý nghĩa dự đoán) [cite: 416-420]
X = loan_dataset.drop(['Personal Loan', 'ID'], axis=1)

# Chuẩn hóa Min-Max (Min-Max Scaling): Ép toàn bộ các biến số về cùng một thang đo từ 0 đến 1 [cite: 418-419]
# Bằng cách lấy giá trị trừ đi min, sau đó chia cho khoảng giá trị (max - min)
X = (X - X.min()) / (X.max() - X.min()) 

# Tạo tập nhãn y: Trích xuất riêng cột 'Personal Loan' để làm mục tiêu dự đoán (1 là vay, 0 là không vay) [cite: 501-502]
y = loan_dataset['Personal Loan']

# 3. CHIA TẬP DỮ LIỆU TRAIN / TEST
# Phân chia ngẫu nhiên dữ liệu: 80% đưa vào tập Train để huấn luyện, 20% đưa vào tập Test để kiểm tra [cite: 558-561]
# random_state=5 giúp khóa hạt giống ngẫu nhiên, đảm bảo các lần chạy code khác nhau vẫn ra cùng một kết quả chia
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# ==========================================
# PHẦN 2: THỬ NGHIỆM & TINH CHỈNH MÔ HÌNH KNN
# ==========================================

# 1. MÔ HÌNH KNN CƠ BẢN (BASELINE)
# Khởi tạo mô hình KNN: Dùng 10 láng giềng gần nhất, hàm khoảng cách Minkowski bậc 2 (tương đương Euclidean), trọng số dồn về các điểm gần hơn [cite: 589-594]
knn_base = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2, weights='distance')
# Đưa dữ liệu Train vào để mô hình học các mẫu (patterns) [cite: 593]
knn_base.fit(X_train, y_train)
# Dùng tập X_test đưa cho mô hình dự đoán, sau đó so sánh với y_test thực tế để lấy điểm Accuracy [cite: 599-601]
print("1. KNN Cơ bản (k=10) Accuracy:", accuracy_score(y_test, knn_base.predict(X_test)))

# 2. TÌM SỐ LÁNG GIỀNG (K) TỐT NHẤT
print("\nĐang chạy thử nghiệm các giá trị K...")
# Tạo danh sách các giá trị K muốn thử nghiệm [cite: 627]
n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
# Vòng lặp chạy qua từng giá trị K [cite: 628]
for k in n_neighbors_list:
    # Khởi tạo mô hình với biến k tương ứng ở vòng lặp hiện tại [cite: 629-630]
    knn_temp = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2, weights='distance')
    # Huấn luyện mô hình [cite: 631]
    knn_temp.fit(X_train, y_train)
    # (Tại đây tài liệu gốc lưu lại điểm số vào mảng để vẽ đồ thị so sánh) [cite: 636-641]

# 3. TÌM HÀM KHOẢNG CÁCH (P) TỐT NHẤT VỚI K=2
print("Đang chạy thử nghiệm các hàm khoảng cách Minkowski (p)...")
# Tạo danh sách các bậc p của khoảng cách Minkowski [cite: 676]
ps_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 'inf']
# Vòng lặp chạy qua từng giá trị p [cite: 677]
for p in ps_list:
    # Nếu p là vô cực ('inf'), ta dùng chuẩn khoảng cách Chebyshev [cite: 678-679]
    metric_name = 'chebyshev' if p == 'inf' else 'minkowski'
    # Gán giá trị p_val hợp lệ (do hàm Chebyshev trong sklearn không cần p, ta gán cờ bằng 2 cho khỏi lỗi) [cite: 680-682]
    p_val = 2 if p == 'inf' else p
    # Khởi tạo mô hình KNN với hàm khoảng cách tương ứng [cite: 692-693]
    knn_temp = KNeighborsClassifier(n_neighbors=2, metric=metric_name, p=p_val, weights='distance')
    # Huấn luyện mô hình [cite: 694]
    knn_temp.fit(X_train, y_train)
    # (Tài liệu gốc tiếp tục lưu điểm số vào mảng để trực quan hóa) [cite: 696-702]

# ==========================================
# PHẦN 3: TỐI ƯU HÓA ĐẶC TRƯNG & SO SÁNH CHÉO
# ==========================================

# 1. MÔ HÌNH KNN TỐI ƯU (SAU KHI LOẠI BỎ THUỘC TÍNH NHIỄU)
# Liệt kê các cột dữ liệu không có tác dụng tốt, thậm chí làm giảm độ chính xác của mô hình [cite: 867-874]
cols_to_drop = ['Personal Loan', 'ID', 'Age', 'Experience', 'ZIP Code', 'Mortgage', 'Securities Account', 'Online', 'CreditCard']
# Xóa các cột đó để tạo tập đặc trưng X mới (X_final) [cite: 867-874]
X_final = loan_dataset.drop(cols_to_drop, axis=1)
# Chuẩn hóa Min-Max lại một lần nữa cho tập dữ liệu đã lược bớt [cite: 875]
X_final = (X_final - X_final.min()) / (X_final.max() - X_final.min()) 

# Chia tập Train/Test mới với bộ X_final [cite: 877]
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_final, y, test_size=0.2, random_state=5)

# Khởi tạo mô hình KNN có thông số tốt nhất từ các bước thử nghiệm phía trên: k=2, metric=minkowski p=2 [cite: 878-879]
knn_final = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2, weights='distance')
# Huấn luyện mô hình tối ưu [cite: 880]
knn_final.fit(X_train_f, y_train_f)
# In kết quả. Lần này điểm Accuracy sẽ đạt mức rất cao (khoảng 0.986) [cite: 884-888]
print("\n2. KNN Tối ưu (Bỏ biến nhiễu, k=2) Accuracy:", accuracy_score(y_test_f, knn_final.predict(X_test_f)))

# 2. SO SÁNH VỚI CÁC THUẬT TOÁN HỌC MÁY KHÁC
# (Quay lại sử dụng tập X_train và X_test ban đầu để việc so sánh được công bằng)
print("\n--- KẾT QUẢ CÁC MÔ HÌNH KHÁC ---")

# Thuật toán Naive Bayes: Dựa trên xác suất thống kê [cite: 893-896]
nb = GaussianNB().fit(X_train, y_train)
print('Naive Bayes Accuracy:', accuracy_score(y_test, nb.predict(X_test)))

# Thuật toán Support Vector Machine (SVM): Dùng kernel dạng tuyến tính (linear) để vẽ đường phân định [cite: 904-907]
svm_classifier = svm.SVC(kernel='linear').fit(X_train, y_train)
print('SVM Accuracy:', accuracy_score(y_test, svm_classifier.predict(X_test)))

# Thuật toán Hồi quy Logistic: Dùng hàm Sigmoid để phân loại nhị phân [cite: 915-920]
lr = LogisticRegression(random_state=0).fit(X_train, y_train)
print('Logistic Regression Accuracy:', accuracy_score(y_test, lr.predict(X_test)))

# Thuật toán Cây Quyết Định (Decision Tree): Chia nhánh dựa trên độ hỗn loạn thông tin (entropy) [cite: 926-929]
dt = DecisionTreeClassifier(criterion='entropy', random_state=51).fit(X_train, y_train)
print('Decision Tree Accuracy:', accuracy_score(y_test, dt.predict(X_test)))

# Thuật toán Rừng Ngẫu Nhiên (Random Forest): Gộp 20 cây quyết định lại với nhau để đưa ra kết quả vững chắc nhất [cite: 938-944]
rf = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=51).fit(X_train, y_train)
print('Random Forest Accuracy:', accuracy_score(y_test, rf.predict(X_test)))



# 1. Tạo một DataFrame chứa thông tin của khách hàng mới (Khách hàng ID 1)
new_customer_data = {
    'Age': [25], 'Experience': [1], 'Income': [49], 'ZIP Code': [91107], 
    'Family': [4], 'CCAvg': [1.6], 'Education': [1], 'Mortgage': [0], 
    'Securities Account': [1], 'CD Account': [0], 'Online': [0], 'CreditCard': [0]
}
test_df = pd.DataFrame(new_customer_data)

# 2. Tiền xử lý dữ liệu cho khách hàng này (bắt buộc phải làm giống tập Train)
# Lấy giá trị min/max từ tập dataset gốc để chuẩn hóa
X_raw = loan_dataset.drop(['Personal Loan', 'ID'], axis=1)
test_scaled = (test_df - X_raw.min()) / (X_raw.max() - X_raw.min())

# 3. Chạy thử với mô hình Random Forest (mô hình xịn nhất đạt 99% accuracy)
rf_prediction = rf.predict(test_scaled)
print(f"Random Forest dự đoán: {'Có vay' if rf_prediction[0] == 1 else 'Không vay'}")

# 4. Chạy thử với mô hình KNN Tối ưu (đã bỏ đi các biến nhiễu)
# Lọc lại đúng 5 cột mà mô hình KNN tối ưu (knn_final) sử dụng
cols_to_keep = ['Income', 'Family', 'CCAvg', 'Education', 'CD Account']
test_scaled_final = test_scaled[cols_to_keep]

knn_prediction = knn_final.predict(test_scaled_final)
print(f"KNN Tối ưu dự đoán: {'Có vay' if knn_prediction[0] == 1 else 'Không vay'}")