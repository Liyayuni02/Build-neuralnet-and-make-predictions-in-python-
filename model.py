from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras import optimizers
import time
from sklearn import model_selection
from tensorflow import keras
import numpy
import pandas
seed = 7
numpy.random.seed(seed)

#Meninialisasi class
names = ['Jumlah_Tetes', 'Sudut', 'Error', 'Output']

#Mengimport file dataset
dataset = pandas.read_csv('datatraining.data', delimiter=',', names=names)
arrays = dataset.values
X=arrays[:,0:3]
Y=arrays[:,3]

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, random_state=seed)

#Pembuatan model dan input hidden layer
model = Sequential()
model.add(Dense(48, input_dim=3, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(20, kernel_initializer='random_uniform', activation='relu',  bias_initializer='zeros'))
model.add(Dense(40, kernel_initializer='random_uniform', activation='relu',  bias_initializer='zeros'))
model.add(Dense(1, kernel_initializer='random_uniform', activation='sigmoid', bias_initializer='zeros'))

#Mengcompile model
opt = keras.optimizers.Adam(learning_rate=0.05)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#Memasukan log direktori pada tensorboard
tensorboard = TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True, write_images=True)

#Mengatur berapa kali training pada model fit
model.fit(X_train, Y_train, epochs=550, batch_size=10, callbacks=[tensorboard])

#Menserialisasikan model ke json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#Menserelisasikan weights ke HDF5
model.save_weights("model.dataepoch10")
print("Saved model")

#Print nilai weight
for lay in model.layers:
    print(lay.name)
    print(lay.get_weights())

# print (dataset.shape)
# print (dataset.head(10))
# print (dataset.describe())
# print (dataset.groupby('Jumlah_Tetes').size())

