import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

data = {
    'Weight': [70, np.nan, 90, np.nan, 65, 45, np.nan, 55, 95, 60],
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Height': [np.nan, 175, 180, 175,  np.nan, 190,185, 175, 200, np.nan],
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']
}

df = pd.DataFrame(data) 
#DataFrame fonksiyonu verileri satır ve sütunlara bölerek tablo haline getirir
print(df)

imputer = KNNImputer(n_neighbors=2)
#KNNImputer, eksik değerleri K-En Yakın Komşular algoritması kullanarak dolduran bir sınıftır. K değeri belirlenir

df_numeric = df.drop('Gender', axis=1) 
gender = df['Gender']
#Veri setindeki sayısal sütunlar 'Gender' sütunu hariç bırakılarak bir df_numeric DataFrame'i oluşturulur ve 'Gender' sütunu ayrı bir değişkene atanır.

df_imputed_numeric = imputer.fit_transform(df_numeric)
#KNNImputer nesnesi kullanılarak eksik değerler K-En Yakın Komşular yöntemiyle doldurulur.


df_imputed_numeric = pd.DataFrame(df_imputed_numeric, columns=df_numeric.columns)
df_imputed = pd.concat([df_imputed_numeric, gender], axis=1) # pd.concat bir veya daha fazla DataFrame'i veya seriyi birleştirmek için kullanılır.
#Son olarak, doldurulmuş sayısal sütunlar bir DataFrame'e dönüştürülür ve 'Gender' sütunu ile birleştirilerek orijinal veri seti elde edilir.

print(df_imputed)
