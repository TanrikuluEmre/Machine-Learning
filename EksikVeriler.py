import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

# Verileri oku
veriler = pd.read_csv("EksikVeriler.csv")

# Eksik değerleri -1 ile kodla
veriler = veriler.fillna(-1)

# SimpleImputer'ı kullanarak eksik değerleri doldur
imputer = SimpleImputer(missing_values=-1, strategy="mean")

# Eğitim verisi
yas = veriler.iloc[:, 1:4].values

# SimpleImputer'ı eğit
imputer = imputer.fit(yas[:, 1:4])

# Eksik değerleri doldur
yas[:, 1:4] = imputer.transform(yas[:, 1:4])

print(yas)