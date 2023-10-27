from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import pandas as pd 

df = pd.read_csv('D:/code hoc may/Breast_cancer_data.csv') # đọc tập csv và lưu trữ dạng dataframe
data = np.array(df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','diagnosis']].values) 
# Chia tập dữ liệu thành 90% huấn luyện và 10% kiểm tra
np.set_printoptions(suppress=True)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42, shuffle = True)
kmeans = KMeans(n_clusters=2, init='k-means++',random_state=42)
kmeans.fit(train_data)
centers = kmeans.cluster_centers_
train_labels = kmeans.labels_
# Dự đoán nhãn cho tập kiểm tra 
test_labels = kmeans.predict(test_data) 
# In kết quả dự đoán trên tập kiểm tra
for i in range(len(test_data)):
    print(f"Mẫu {test_data[i]} thuộc cụm {test_labels[i] + 1}")

# In trung tâm của các cụm
print("\nTrung tâm của các cụm:")
for center in centers:
    print(center)

# Đánh giá mô hình sử dụng độ đo Silhouette và Davies-Bouldin
test_silhouette_score = silhouette_score(test_data, test_labels)
test_davies_bouldin_score = davies_bouldin_score(test_data, test_labels)
print(f"Độ đo Silhouette trên tập kiểm tra: {test_silhouette_score}")
print(f"Độ đo Davies-Bouldin trên tập kiểm tra: {test_davies_bouldin_score}")
