def model_comp():
  model=Sequential()
  w=1e-4
  model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3), kernel_regularizer=l2(w)))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D(32, (3,3),  kernel_regularizer=l2(w), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))


  model.add(Conv2D(64, (3,3)))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3,3),  kernel_regularizer=l2(w), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))


  model.add(Conv2D(128, (3,3), kernel_regularizer=l2(w)))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3,3),  kernel_regularizer=l2(w),  padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.4))
  


  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(32, activation='elu'))
 

  model.add(Dense(10))
  model.add(Activation('softmax'))
  
  return model


model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=64, epochs=100, 
                      validation_data=(test_images, test_labels))