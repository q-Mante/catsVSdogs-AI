import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, image_utils 
load_img = image_utils.load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
import random
import os

# Image constants
img_width = 128
img_height = 128
img_size = (128, 128)
img_channels = 3

# Constants
batch_size = 15

# Path to training data (images)
Directory = os.listdir("../../train-1")

# Ask to train new model
text = input("Train? y/n: ")
epochs = int(input("Epochs: "))
if text == 'y':
    train = True
else:
    train = False

if train:
    # Preparing training data
    labels = []
    for name in Directory:
        label = name.split('.')[0]
        if label == 'dog':
            labels.append(1)
        else:
            labels.append(0)

    df = pd.DataFrame({
        'filename': Directory,
        'label': labels
    })

    # View data frame
    print("DATA FRAME:")
    print(df.head(-1))

    # Build a model

    # Input layer which represents input image data, which will reshape the image into a single dimensional array. In this we also have,
    # convolutional layer, which will extract features from image,
    # pooling layer, which will reduce spatial volume of input image after convolution,
    # fully connected layer, which connects the network from a layer to another layer and,
    # output layer, which will predict values

    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

    model = Sequential(name = "catsVSdogs")

    model.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = (128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

    # View model summary
    print()
    print("MODEL SUMMARY:")
    print(model.summary(line_length = 100))

    # Callbacks
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    earlystop = EarlyStopping(patience = 10)
    learning_rate_reduction = ReduceLROnPlateau(
        monitor = 'val_acc', 
        patience = 2, 
        verbose = 1, 
        factor = 0.5, 
        min_lr = 0.00001
    )
    callbacks = [earlystop, learning_rate_reduction]

    # Prepare data
    df["label"] = df["label"].replace({0: 'cat', 1: 'dog'}) 
    train_data, validation_data = train_test_split(df, test_size = 0.20, random_state = 42)
    train_data = train_data.reset_index(drop = True)
    validation_data = validation_data.reset_index(drop = True)
    final_train_data = train_data.shape[0]
    final_validation_data = validation_data.shape[0]

    # Create training generator
    generate_train_data = ImageDataGenerator(
        rotation_range = 15,
        rescale = 1./255,
        shear_range = 0.1,
        zoom_range = 0.2,
        horizontal_flip = True,
        width_shift_range = 0.1,
        height_shift_range = 0.1
    )

    train_gen = generate_train_data.flow_from_dataframe(
        train_data,
        "../../train-1",
        x_col = 'filename',
        y_col = 'label',
        target_size = img_size,
        class_mode = 'categorical',
        batch_size = batch_size
    )

    # Create validation data generator
    generate_validation_data = ImageDataGenerator(rescale = 1./255)
    validation_gen = generate_validation_data.flow_from_dataframe(
        validation_data, 
        "../../train-1", 
        x_col = 'filename',
        y_col = 'label',
        target_size = img_size,
        class_mode = 'categorical',
        batch_size = batch_size
    )

    # Train the model
    history = model.fit(
        train_gen, 
        epochs = epochs,
        validation_data = validation_gen,
        validation_steps = final_validation_data//batch_size,
        steps_per_epoch = final_train_data//batch_size,
        callbacks = callbacks
    )

    # Save the model
    model.save("../../model_catsVSdogs_" + str(epochs) + "epoch.h5")
else:
    model = load_model("../../model_catsVSdogs_" + str(epochs) + "epoch.h5")

# Prepare test dataset
test_filenames = os.listdir("../../test-1")
test_data = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_data.shape[0]

# Create test data generator
generate_test_data = ImageDataGenerator(rescale = 1./255)
test_gen = generate_test_data.flow_from_dataframe(
    test_data, 
    "../../test-1", 
    x_col = 'filename',
    y_col = None,
    class_mode = None,
    target_size = img_size,
    batch_size = batch_size,
    shuffle = False
)

# Perform prediction
prediction = model.predict(test_gen, steps = np.ceil(nb_samples/batch_size))

# Convert predicted category
test_data['label'] = np.argmax(prediction, axis = -1)

label_map = {
    "cat": 0,
    "dog": 1
}
test_data['label'] = test_data['label'].replace(label_map)

test_data['label'] = test_data['label'].replace({ 'dog': 1, 'cat': 0 })

# Visualize
testing = test_data.head(18)
testing.head()
plt.figure(figsize = (12, 24))
for index, row in testing.iterrows():
    filename = row['filename']
    label = row['label']
    image = load_img("../../test-1/" + filename, target_size = img_size)
    plt.subplot(6, 3, index+1)
    plt.imshow(image)
    plt.xlabel(filename + '(' + "{}".format(label) + ')' )
plt.tight_layout()
plt.show()

# Test custom data
results = {
    0:'cat',
    1:'dog'
}
prediction_results = []
from PIL import Image
test = os.listdir("../../images-1")
for name in test:
    im = Image.open("../../images-1/" + name)
    im = im.resize(img_size)
    im = np.expand_dims(im, axis=0)
    im = np.array(im)
    im = im/255
    pred = model.predict([im])[0]
    num = np.argmax(pred)
    prediction_results.append(name + " " + str(pred) + " " + results[num])

for text in prediction_results:
    print(text)
