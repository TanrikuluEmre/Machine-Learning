import numpy as np

# Input1, input2 ve bias
train_in = np.array([
    [1., 1., 1],
    [1., 0, 1],
    [0, 1., 1],
    [0, 0, 1]
])
# Output
train_out = np.array([
    [1.],
    [0],
    [0],
    [0]
])

# Random_normal() kullanarak rasgele değerler ile başlatılan ağırlık değişkeni
np.random.seed(12)
w = np.random.normal(size=(3, 1))

# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Çıktıyı hesapla
output = sigmoid(np.dot(train_in, w))

# Hata hesapla
loss = np.sum((output - train_out) ** 2)

# 0.01'lik bir öğrenme oranıyla Gradient Descent kullanarak kaybı en aza indirin.
learning_rate = 0.01

# Gradyan inişi ile ağırlıkları güncelle
for i in range(1000):
    output = sigmoid(np.dot(train_in, w))
    error = train_out - output
    w += learning_rate * np.dot(train_in.T, error * output * (1 - output))

# Giriş vektörüne göre çıktı ve maliyeti hesaplayın
output = sigmoid(np.dot(train_in, w))
cost = np.sum((output - train_out) ** 2)
print('Loss:', cost)
