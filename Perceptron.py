import tensorflow as tf

# Input1, input2 ve bias
train_in = [
    [1., 1., 1],
    [1., 0, 1],
    [0, 1., 1],
    [0, 0, 1]
]
# Output
train_out = [
    [1.],
    [0],
    [0],
    [0]
]

# Random_normal() kullanarak rasgele değerler ile başlatılan ağırlık değişkeni
w = tf.Variable(tf.random.normal([3, 1], seed=12))

# Girdi ve çıktı için tensor tanımla
x = tf.Variable(train_in, dtype=tf.float32)
y = tf.Variable(train_out, dtype=tf.float32)

# Output(çıktıyı) hesapla.
output = tf.nn.relu(tf.matmul(x, w))

# Mean Squared Loss or Error(Ortalama Kareli Hata)
loss = tf.reduce_sum(tf.square(output - y))

# 0.01'lik bir öğrenme oranıyla GradientDescentOptimizer kullanarak kaybı en aza indirin.
optimizer = tf.keras.optimizers.SGD(0.01)

# Gradyanları hesapla ve optimize et
with tf.GradientTape() as tape:
    output = tf.nn.relu(tf.matmul(x, w))
    loss = tf.reduce_sum(tf.square(output - y))
grads = tape.gradient(loss, [w])
optimizer.apply_gradients(zip(grads, [w]))

# Giriş vektörüne göre çıktı ve maliyeti hesaplayın
cost = tf.reduce_sum(tf.square(output - y))
print('Loss:', cost)
