import kerastuner

from kerastuner.tuners import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

def build_model(hp):
  model = Sequential()
  model.add(Conv2D(hp.Int('first_layer', 32, 512, 32), (3,3), input_shape=(32,32,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  for i in range(hp.Int('deep_layers', 1, 4)):
      model.add(Conv2D(hp.Int(f'conv_{i}_unit', 
                              min_value =32,
                              max_value = 512,
                              step = 32), (3,3)))
      model.add(Activation('relu'))
     # model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(hp.Int('last_layer', 
                          min_value=32,
                          max_value=512,
                          step=32), (3,3)))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dropout(hp.Float('drop_layer', 0, 0.5, 0.2)))


  model.add(Dense(10))
  model.add(Activation('softmax'))

  model.compile(optimizer="adam",
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