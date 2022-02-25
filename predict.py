from keras.models import model_from_json
from sklearn import model_selection
import pandas
#Menload JSON dan membuat model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#Menload weight dari model
loaded_model.load_weights("model.dataepoch10")
print("Loaded model")

#Mengimport dataset validasi
dataset = pandas.read_csv("datavalidasi.data", delimiter=",")
array = dataset.values
X = array[:, 0:3]
Y = array[:, 3]
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, random_state=seed)

#Mengcompile data load model
loaded_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
scores = loaded_model.evaluate(X_validation, Y_validation)
print("%s: %.5f%%" % (loaded_model.metrics_names[1], scores[1]*100))
