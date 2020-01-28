def model_comp():
  model=Sequential()
  
  model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D(32, (3,3),  padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))


  model.add(Conv2D(64, (3,3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3,3),  padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
 
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))


  model.add(Conv2D(128, (3,3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3,3),   padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  #model.add(Conv2D(128, (3,3),   padding='same'))
  #model.add(Activation('elu'))
  #model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.4))

  model.add(Conv2D(64, (3,3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3,3),  padding='same' ))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(32, (3,3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(BatchNormalization())
  model.add(Conv2D(16, (3,3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D(16, (3,3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  #model.add(Dropout(0.5))
  model.add(Flatten())
 

  model.add(Dense(10))
  model.add(Activation('softmax'))
  
  return model


def aug():
  datagen = ImageDataGenerator(
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      )

  rts=RMSprop(learning_rate=0.001, decay=1e-6)
  model.compile(optimizer=rts,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  steps = int(len(train_images)/64)

  history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=64), steps_per_epoch=steps, epochs=100, 
                      validation_data=(test_images, test_labels))


def starting(num):
  opt_RMS = RMSprop(learning_rate=0.001, decay=1e-6)
  model.compile(optimizer=opt_RMS,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])   
  history = model.fit(train_images, train_labels, batch_size=64, epochs=num, 
                        validation_data=(test_images, test_labels))


aug()
starting(15)