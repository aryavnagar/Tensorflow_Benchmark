import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

tf.config.experimental.list_physical_devices()
tf.test.is_built_with_cuda()

(X_train, y_train), (X_test,y_test) = tf.keras.datasets.cifar10.load_data()


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

y_train_categorical = keras.utils.to_categorical(
    y_train, num_classes=10, dtype='float32'
)
y_test_categorical = keras.utils.to_categorical(
    y_test, num_classes=10, dtype='float32'
)



def get_model():
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(32,32,3)),
            keras.layers.Dense(4500, activation='relu'),
            keras.layers.Dense(1200, activation='relu'),
            keras.layers.Dense(10, activation='sigmoid')    
        ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


%timeit -n1 -r1 
with tf.device('/GPU:0'):
    cpu_model = get_model()
    cpu_model.fit(X_train_scaled, y_train_categorical, epochs=10)
    

%timeit -n1 -r1 
with tf.device('/CPU:0'):
    cpu_model = get_model()
    cpu_model.fit(X_train_scaled, y_train_categorical, epochs=10)

