import kerastuner

from kerastuner.tuners import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

def build_model(hp):
  model = Sequential()

  d1 = hp.Float('Dropout_1', 0, 0.5, 0.1)
  d2 = hp.Float('Dropout_2', 0, 0.5, 0.1)
  d3 = hp.Float('Dropout_3', 0, 0.5, 0.1)
  d4 = hp.Float('Dropout_4', 0, 0.5, 0.1)
  
  model.add(Conv2D((hp.Int('input_units',
                                min_value=32,
                                max_value=256,
                                step=32)), (3,3), input_shape=(32, 32, 3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D((hp.Int('input_units_2',
                                min_value=32,
                                max_value=256,
                                step=32)), (3,3),  padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(d1))


  model.add(Conv2D((hp.Int('deep_layer_1',
                                min_value=32,
                                max_value=256,
                                step=32)), (3,3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D((hp.Int('deep_layer_1',
                                min_value=32,
                                max_value=256,
                                step=32)), (3,3),  padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
 
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(d2))


  model.add(Conv2D((hp.Int('deep_layer_2',
                                min_value=32,
                                max_value=256,
                                step=32)), (3,3), padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  model.add(Conv2D((hp.Int('deep_layer_3',
                                min_value=32,
                                max_value=256,
                                step=32)), (3,3),   padding='same'))
  model.add(Activation('elu'))
  model.add(BatchNormalization())
  #model.add(Conv2D(128, (3,3),   padding='same'))
  #model.add(Activation('elu'))
  #model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(d3))

  for i in range(hp.Int('n_layers_1', 0, 3)):  # adding variation of layers.
        model.add(Conv2D(hp.Int(f'conv_{i}_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(d4))

  for j in range(hp.Int('n_layers_2', 0, 3)):  # adding variation of layers.
        model.add(Conv2D(hp.Int(f'conv_{j}_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
  
  #model.add(Dropout(0.5))
  model.add(Flatten())
  #model.add(Dropout(0.5))
  #model.add(Dense(32, activation='elu'))
 

  model.add(Dense(10))
  model.add(Activation('softmax'))



  model.compile(optimizer=Adam(
            hp.Float('learning_rate', 0.001, 0.031, 0.002)),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
  return model
  return model

tuner=BayesianOptimizationh(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='test_dir'
)

tuner.search(x=train_images, 
             y=train_labels,
             verbose=2,
             epochs=25,
             batch_size=64,
             validation_data=(test_images, test_labels))