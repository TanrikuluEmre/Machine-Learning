import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

data = {
    'Weight': [70, np.nan, 90, np.nan, 65, 45, np.nan, 55, 95, 60],
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Height': [np.nan, 175, 180, 175, np.nan, 190, 185, 175, 200, np.nan],
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']
}

df = pd.DataFrame(data)

# Gender sütununu düşürme
df_numeric = df.drop('Gender', axis=1)

# Gender sütununu sayısal formata dönüştürme
gender_mapping = {'M': 1, 'F': 0}
gender_numeric = df['Gender'].map(gender_mapping)

# KNNImputer kullanarak eksik değerleri doldurma
imputer = KNNImputer(n_neighbors=2)
df_imputed_numeric = imputer.fit_transform(df_numeric)

# Dönüştürülen veriyi DataFrame olarak oluşturma
df_imputed_numeric = pd.DataFrame(df_imputed_numeric, columns=df_numeric.columns)

# Dönüştürülen Gender sütununu DataFrame'e ekleyerek sonuç oluşturma
df_imputed = pd.concat([df_imputed_numeric, gender_numeric], axis=1)

print(df_imputed)
