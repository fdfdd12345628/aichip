import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, MaxPool2D, Conv1D
from tensorflow.keras.datasets import mnist

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf_config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=tf_config))

LEARNINT_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255
# x_train = x_train.reshape(x_train.shape[0], 28 * 28)
# x_test = x_test.reshape(x_test.shape[0], 28 * 28)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

input_layer = Input((28, 28, 1))
# d1 = Dense(units=128, activation='sigmoid')(input_layer)
c1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
p1 = MaxPool2D(pool_size=(2, 2))(c1)
c2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(p1)
f1 = Flatten()(c2)
d2 = Dense(units=10, activation='softmax')(f1)
model = Model(inputs=input_layer, outputs=d2)
model.compile(
    # optimizer=keras.optimizers.SGD(learning_rate=LEARNINT_RATE,),
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['acc'],
)

train_jistory = model.fit(
    x=x_train, y=y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_split=0.2,
)

test_result = model.evaluate(x=x_test, y=y_test, batch_size=x_test.shape[0])
print(test_result)
