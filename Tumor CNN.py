try:
    import tensorflow as tf
except:
    print("Install tensorflow")
try:
    from keras.preprocessing.image import ImageDataGenerator
except:
    print("Install keras")

# Data Preprocessing
# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


from PIL import Image
test = Image.open("dataset/pred/test1.jpg")



class Tumor_CNN:
    def Classification(self,training_set,test_set,ep=1):
        # Building the CNN
        # Initialising the CNN
        self.cnn = tf.keras.models.Sequential()

        # Step 1 - Convolution
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

        # Step 2 - Pooling
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Adding a second convolutional layer
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
        # Step 3 - Flattening
        self.cnn.add(tf.keras.layers.Flatten())

        # Step 4 - Full Connection
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Step 5 - Output Layer
        self.cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # Part 3 - Training the CNN

        # Compiling the CNN
        self.cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Training the CNN on the Training set and evaluating it on the Test set
        self.cnn.fit(x = training_set, validation_data = training_set, epochs = ep)

        # Part 4 - Making a single prediction
        import numpy as np
        from keras.preprocessing import image
        
        # Load Image without tumor
        test_image = image.load_img('dataset/pred/test1.jpg', target_size = (64, 64))
        test_image = np.expand_dims(test_image, axis = 0)
        
        # Load Image with tumor
        test_image1 = image.load_img('dataset/pred/test2.jpg', target_size = (64, 64))
        test_image1 = np.expand_dims(test_image1, axis = 0)
        
        result = self.cnn.predict(test_image)
        result1 = self.cnn.predict(test_image1)
        training_set.class_indices
        
        if result[0][0] == 1:
            prediction = 'Found'
        else:
            prediction = 'Not Tumor Found'
        print(prediction)
        
        if result1[0][0] == 1:
            prediction = 'Found'
        else:
            prediction = 'Not Tumor Found'
        print(prediction)

Tumor_CNN().Classification(training_set, test_set,10)
