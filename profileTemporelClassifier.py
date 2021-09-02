from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from pyimagesearch.conv import shallownet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = simplepreprocessor.SimplePreprocessor(4,5)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = shallownet.ShallowNet.build(width=4, height=5, depth=1, classes=1)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32,
              epochs=300, verbose=1)
print("[INFO] serializing network...")
model.save(args["model"])
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=100000)


predictions = np.where(predictions > 0.5, 1, 0)
print(classification_report(testY,
    predictions,
    target_names = ["ProfileTemporelleStable","ProfileTemporelleUnStable"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 300), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 300), H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('Training_Loss.png')




#python main.py --dataset ../Profile_temporelle  --model model.hdf5