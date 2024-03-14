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

# Random_normal() kullanarak rasgele değerler ile başlatılan 3 satır 1 sütundan oluşan ağırlık değişkeni
np.random.seed(12)
w = np.random.normal(size=(3, 1))

# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 1/1+e^(-x)

# Girdileri ağırlık değerleri ile çarparak sigmoid fonksiyonuna gönderilir ve çıktı alınır 
output = sigmoid(np.dot(train_in, w))

# Sigmoid fonksiyonundan alınan çıktıdan gerçek çıktı çıkarılarak karesi alınır ve kayıp oranı hesaplanır.
loss = np.sum((output - train_out) ** 2)

# Öğrenme oranı, modelin her adımda ağırlıklarını ne kadar güncelleyeceğini belirler. Daha büyük bir öğrenme oranı, her adımda daha büyük ağırlık güncellemeleri yapılmasını sağlar, 
#dolayısıyla model daha hızlı bir şekilde eğitilebilir. Ancak, çok büyük bir öğrenme oranı, modelin istenmeyen davranışlarına yol açabilir ve ağırlıkların sıçramasına neden olabilir. 
#Öte yandan, çok küçük bir öğrenme oranı daha istikrarlı ve güvenilir bir eğitim sağlayabilir, ancak eğitim sürecinin daha uzun sürmesine neden olabilir ve yerel minimumlara takılı kalma riski artabilir.
learning_rate = 0.1

# Gradyan inişi ile ağırlıkların güncellenmesi
# range(100000) Döngü iterasyon sayısını ifade eder, modelin eğitim sürecinde kaç adım (iterasyon) alınacağını belirler. Daha fazla iterasyon, modelin daha uzun süre eğitilmesini sağlar, 
#bu da daha iyi bir model performansına ve daha düşük bir kayıp değerine ulaşılabilir. Ancak, çok fazla iterasyon, aşırı uyuma (overfitting) ve eğitim setine özgü hataların model 
#tarafından öğrenilmesine yol açabilir. Bu nedenle, iterasyon sayısı dikkatlice seçilmelidir.
for i in range(100000):
    output = sigmoid(np.dot(train_in, w))
    error = train_out - output
    w += learning_rate * np.dot(train_in.T, error * output * (1 - output))

# Giriş vektörüne göre çıktı ve maliyeti hesapla.
output = sigmoid(np.dot(train_in, w))
cost = np.sum((output - train_out) ** 2)
print('Loss:', cost)
