from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import pandas as pd 
df = pd.read_csv('D:/code hoc may/Breast_cancer_data.csv') # đọc tập csv và lưu trữ dạng dataframe
data = np.array(df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']].values) 
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42, shuffle = True)

# Chia tập dữ liệu thành 90% huấn luyện và 10% kiểm tra
np.set_printoptions(suppress=True)
# Khởi tạo mô hình KMeans và huấn luyện trên tập dữ liệu huấn luyện
b = []

# Số lượng cụm bạn muốn thử
num_clusters = 9

for i in range( 2,num_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(train_data)
    test_labels = kmeans.predict(test_data)
    
    silhouette = silhouette_score(test_data, test_labels)
    davies_bouldin = davies_bouldin_score(test_data, test_labels)
    new_list = [silhouette, davies_bouldin,i]
    b.append(new_list)
max_b = max(b)
print(max_b)

kmeans = KMeans(n_clusters=2, init='k-means++')
#cluster=2 số lượng cụm,k-means phương thức khởi tạo
kmeans.fit(train_data)
labels = kmeans.labels_

def predict_cluster():
    # Lấy giá trị đầu vào từ các ô nhập liệu
    new_sample = []
    for entry in entry_fields:
        value = entry.get()
        if value == "":
            messagebox.showerror("Lỗi", "Vui lòng điền đầy đủ thông tin vào các trường.")
            return
        try:
            new_sample.append(float(value))
        except ValueError:
            messagebox.showerror("Lỗi", f"Giá trị không hợp lệ: {value}")
            return

    # Dự đoán nhãn cụm cho mẫu mới
    cluster_label = kmeans.predict([new_sample])[0]
    result_label.configure(text=f"Nhãn dự đoán: Cụm {cluster_label + 1}")
def evaluate_model():
    # Đánh giá mô hình sử dụng các chỉ số Silhouette và Davies-Bouldin
    test_labels = kmeans.predict(test_data)
    silhouette = silhouette_score(test_data, test_labels)
    davies_bouldin = davies_bouldin_score(test_data, test_labels)
    evaluation_label.configure(text=f"Độ đo Silhouette: {silhouette:.10f}\n Độ đo Davies-Bouldin: {davies_bouldin:.10f}")

# Tạo giao diện người dùng
form = Tk()
form.title("Dự đoán nhãn phân cụm Ung thư vú")
form.geometry("300x400")

# Tạo các ô nhập liệu cho thông tin của mẫu mới
entry_fields = []
labels = ['bán kính trung bình', 'kết cấu trung bình', 'chu vi trung bình', 'diện tích trung bình', 'độ mịn trung bình']
for i, label in enumerate(labels):
    label = Label(form, text=label)
    label.grid(row=i, column=0, sticky='e')
    entry = Entry(form)
    entry.grid(row=i, column=1)
    entry_fields.append(entry)

# Tạo các nút dự đoán và đánh giá
predict_button = Button(form, text="Dự đoán Cụm", command=predict_cluster)
predict_button.grid(row=len(labels), columnspan=2, pady=10)
evaluate_button = Button(form, text="Đánh giá Mô hình", command=evaluate_model)
evaluate_button.grid(row=len(labels)+1, columnspan=2, pady=10)

# Label hiển thị nhãn dự đoán
result_label = Label(form, text="")
result_label.grid(row=len(labels)+2, columnspan=2)

# Label hiển thị điểm đánh giá
evaluation_label = Label(form, text="")
evaluation_label.grid(row=len(labels)+3, columnspan=2)

form.mainloop()
