import kerastuner

from kerastuner.tuners import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

def modificate_tuner(hp):
 
  model = Sequential()
  act=hp.Choice('act', ['elu', 'relu', 'selu', 'tanh'])
  model.add(Conv2D(128, (3,3), input_shape=(32, 32, 3)))
  model.add(Activation(act))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))

  model.add(Conv2D(480, (3,3)))
  model.add(Activation(act))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Conv2D(128, (3,3)))
  model.add(Activation(act))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))
  #model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(416, (3,3)))
  model.add(Activation(act))
  model.add(BatchNormalization())

  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(32, activation=act))

  model.add(Dense(10))
  model.add(Activation('softmax'))

  model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
  
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