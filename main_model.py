import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.optimizers import Adam
import torch
from sklearn.metrics import confusion_matrix as cm
from keras.utils.vis_utils import plot_model
import random
from sklearn.metrics import classification_report

"""
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.compat.v1.Session(config=config))
"""

# Belirli bir isabet yüzdesine ulaşıldığında eğitimi durdurma fonksiyonu
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('accuracy') > .90) & (logs.get('val_accuracy') > .8):
            print("%90 Doğruluk oranına ulaşıldı!")
            self.model.stop_training = True
callbacks = myCallback()

# Görüntülerin piksel değerleri 0-1 arasına normalize edilmiştir.
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

test_datagen = test_datagen.flow_from_directory(
        'test/',  # Test resimleri için kaynak dosya
        target_size=(200, 150),  # Resimlerin boyutu veya resimlerin yeniden boyutlandırılması
        batch_size=8,
        # 7 Adet sınıf mevcut olduğu için, sınıf modu "kategori" şeklinde ayarlanmıştır
        class_mode='categorical')

train_generator = train_datagen.flow_from_directory(
        'train/',  # Eğitim resimleri için kaynak dosya
        target_size=(200, 150),  # Resimlerin boyutu veya resimlerin yeniden boyutlandırılması
        batch_size=64,
        # 7 Adet sınıf mevcut olduğu için, sınıf modu "kategori" şeklinde ayarlanmıştır
        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
        'val/',  # Doğrulama resimleri için kaynak dosya
        target_size=(200, 150),  
        batch_size=8,
        class_mode='categorical')

print(train_generator.class_indices)


model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(7, activation='softmax')
])

#Genelde multi-class classification da categorical_crossentropy ve adam optimizer kullanılır.
opt = Adam(learning_rate=0.01)
model.compile(loss ='categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

# print(model.summary())

# model.summary()

# history = model.fit(
#       train_generator,
#       steps_per_epoch=1,  
#       epochs=2,
#       validation_data = validation_generator,
#       callbacks=[callbacks])

history = model.fit(train_generator,
                     epochs=8,
                      validation_data=validation_generator,
                      callbacks=[callbacks])

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_confusion(confusion_mat):
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    tick_marks = np.arange(len(classes))
    plt.imshow(confusion_mat, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(tick_marks,classes, rotation=45)
    plt.yticks(tick_marks,classes)
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


plot_acc(history)
plot_loss(history)

predict_score = model.predict_generator(train_generator,
         steps = np.ceil(train_generator.samples / train_generator.batch_size), verbose=1, workers=0)
predict_score = np.argmax(predict_score, axis=1)
confusion_matrix = cm(train_generator.classes, predict_score)

plot_confusion(confusion_matrix)

score = model.evaluate_generator(test_datagen,1506)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

def pred_one_img(model):   
    classes = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6} 
    print("*****************************************")
    path = "test/"

    files=os.listdir(path)
    class_of_img=random.choice(files)
    path = path + class_of_img + "/"

    files=os.listdir(path)
    d=random.choice(files)
    

    image_to_test_1 = image.load_img(path + "/" + d, target_size=(200, 150,3))

    image_to_test = np.asarray(image_to_test_1)
    image_to_test = np.expand_dims(image_to_test, axis=0)
    prediction = model.predict_classes(image_to_test)  
    print(prediction)

    prediction = next( k for (k, v) in classes.items() if v == prediction[0] )

    plt.imshow(image_to_test_1)
    plt.ylabel("Sınıf: "+ class_of_img)
    plt.xlabel(prediction)
    plt.show()

pred_one_img(model)
pred_one_img(model)
pred_one_img(model)

# image_to_test = image.load_img(img_path, target_size=(224, 224))
