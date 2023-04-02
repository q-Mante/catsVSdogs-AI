import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

img_width=128
img_height=128
img_size=(128,128)
img_channels=3

Directory = os.listdir("./train/train")

# Directory = r'C:\Users\KRISHNA\CatsDogData\train\train'

labels=[]
for name in Directory:
    label =name.split('.')[0]
    if label=='dog':
        labels.append(1)
    else:
        labels.append(0)

df=pd.DataFrame({
    'filename':Directory,
    'label':labels
})


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization


model=Sequential()

model.add(Conv2D(64,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])

model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
df["label"] = df["label"].replace({0: 'cat', 1: 'dog'}) 

train_data, validation_data = train_test_split(df, test_size=0.20, random_state=42)
train_data = train_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

final_train_data = train_data.shape[0]
final_validation_data = validation_data.shape[0]
batch_size=15

generate_train_data = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )

train_gen = generate_train_data.flow_from_dataframe(train_data,
                                             "./train/train/",x_col='filename',y_col='label',
                                                 target_size=img_size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)


generate_validation_data = ImageDataGenerator(rescale=1./255)
validation_gen = generate_validation_data.flow_from_dataframe(
    validation_data, 
    "./train/train/", 
    x_col='filename',
    y_col='label',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size
)


epochs=10
history = model.fit_generator(
    train_gen, 
    epochs=epochs,
    validation_data=validation_gen,
    validation_steps=final_validation_data//batch_size,
    steps_per_epoch=final_train_data//batch_size,
    callbacks=callbacks
)
model.save("model1_catsVSdogs_10epoch.h5")

test_filenames = os.listdir("./test1/test1")
test_data = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_data.shape[0]

generate_test_data= ImageDataGenerator(rescale=1./255)
test_gen = generate_test_data.flow_from_dataframe(
    test_data, 
    "./test1/test1/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

prediction = model.predict_generator(test_gen, steps=np.ceil(nb_samples/batch_size))

test_data['label'] = np.argmax(prediction, axis=-1)

label_map = dict((v,k) for k,v in train_gen.class_indices.items())
test_data['label'] = test_data['label'].replace(label_map)

test_data['label'] = test_data['label'].replace({ 'dog': 1, 'cat': 0 })


testing = test_data.head(10)
testing.head()
plt.figure(figsize=(12, 24))
for index, row in testing.iterrows():
    filename = row['filename']
    label = row['label']
    image = load_img("./test1/test1/"+filename, target_size=img_size)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(label) + ')' )
plt.tight_layout()
plt.show()