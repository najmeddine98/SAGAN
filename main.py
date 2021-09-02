import matplotlib.pyplot as plt
import numpy as np
import cv2
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from numpy import array
from keras.layers import (
Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
#from preprocessing import SimplePreprocessor
#from datasets import simpledatasetloader
from imutils import paths
import argparse
import numpy as np
#from skimage.exposure import rescale_intensityp
from numba import jit, cuda


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
 help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
sp = simplepreprocessor.SimplePreprocessor(4,5)
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
#data = data.reshape((data.shape[0], 20))
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1000.0)))
#le = LabelEncoder()
#print(labels)
#labels = le.fit_transform(labels)
print("//////////////////////////////////")
#print(labels)
#print(data)
print(np.shape(data))
img_rows = 5
img_cols = 4
channels = 1
data = array(data)
#data = data.reshape((data.shape[0], data.shape[1], 1))
print(data.shape)
img_shape = (img_rows, img_cols, channels)
z_dim = 100
#data = np.expand_dims(data, axis=3)
print(data.shape)


def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(256 * 5 * 2, input_dim=z_dim))
    model.add(Reshape((5, 2, 256)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=(1,1), padding='same')) #(5,2,128)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=(1,2), padding='same')) #(5,4,64)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same')) #(5,4,1)
    model.add(Activation('tanh'))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(
    Conv2D(32,
    kernel_size=3,
    strides=(1,1),
    input_shape=img_shape,
    padding='same'))        #(5,4,,32)
    model.add(LeakyReLU(alpha=0.01))
    model.add(
    Conv2D(64,
    kernel_size=3,
    strides=1,
    input_shape=img_shape,
    padding='same'))    #(5,4,64)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(
    Conv2D(128,
    kernel_size=3,
    strides=2,
    input_shape=img_shape,
    padding='same'))    #(2,2,128)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])
generator = build_generator(z_dim)
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []

def sample_images(generator,iteration,image_grid_rows=4, image_grid_columns=4):


    z = np.random.normal(0, 1, (40, z_dim))
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5

    Nbimg=iteration*40

    for k in np.arange(40):
        i=Nbimg+k
        fileName = "Generated_Profile_temporelle_stable/%d.npy" % (i)

        with open(fileName, 'wb') as f:
            np.save(f, gen_imgs[k, :, :, 0])

    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("Generated_Profile_temporelle_stable_Img/%d.png" % iteration)
    plt.close()


def train(iterations, batch_size, sample_interval,data):
    X_train= data
    X_train = X_train / 127.5 - 1.0
    #print(np.shape(X_train))
    X_train = np.expand_dims(X_train, axis=3)
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    for iteration in range(iterations):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)
        g_loss = gan.train_on_batch(z, real)

        losses.append((d_loss, g_loss))
        accuracies.append(100.0 * accuracy)
        iteration_checkpoints.append(iteration + 1)
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(iteration + 1, d_loss, 100.0 * accuracy, g_loss))
        sample_images(generator,iteration)

iterations = 20000
batch_size = 512
sample_interval = 1000
#train(iterations, batch_size, sample_interval,data)

a=np.zeros((1,4))
i=0
for k in np.arange(550000,683720):

    fileName = "Generated_Profile_temporelle_stable/%d.npy" % (k)
    i=k-393193
    image_nb= "Generate_img/%d.jpg" % (i)

    with open(fileName, 'rb') as f:
        inter=np.load(f)
        output = (inter * 255).astype("uint8")
        np.array(output)
        cv2.imwrite(image_nb,output)
    if (k % 500 == 0):
        print(k)

#(data1, labels1) = sdl.load("../Profile_temporelle/Generate_img", verbose=500)

#print(labels)
