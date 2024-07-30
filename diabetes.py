import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Memuat dataset
data = pd.read_csv('diabetes.csv')

# Melihat informasi dataset
print(data.head())
print(data.info())

# Melihat statistik deskriptif
print(data.describe())

# Memeriksa missing values
print(data.isnull().sum())

# Misalkan kolom 'Outcome' adalah target, dan fitur lainnya adalah input
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Melihat distribusi kelas
print(y.value_counts())

# Visualisasi distribusi fitur
sns.pairplot(data, hue='Outcome')
plt.show()

# Korelasi antar fitur
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Memisahkan data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test)

# Membuat model Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Memprediksi kelas pada data uji
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Untuk melihat pentingnya fitur
feature_importances = model.feature_importances_
print(f'Feature Importances: {feature_importances}')

# Menyimpan model dan scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
