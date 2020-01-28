x=Input(shape=(32,32,3))

branch_e = Conv2D(32, (3,3), activation='elu', padding='same')(x)
branch_e = BatchNormalization()(branch_e)
branch_e = Conv2D(32, (3,3), activation='elu', padding='same')(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e = MaxPooling2D(2,2)(branch_e)
branch_e = Dropout(0.2)(branch_e)

branch_e = Conv2D(64, (3,3), activation='elu', padding='same' )(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e = Conv2D(64, (3,3), activation='elu', padding='same')(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e = MaxPooling2D(2,2)(branch_e)
branch_e = Dropout(0.3)(branch_e)

branch_e=Conv2D(128, (3,3), activation='elu', padding='same')(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e =Conv2D(128, (3,3), activation='elu', padding='same')(branch_e)
branch_e = BatchNormalization()(branch_e)
resediual = Conv2D(128, 1, strides=2, padding='same')(x)
#resediual = Flatten()(resediual)
layers.add([branch_e, resediual])
branch_e = Dropout(0.4)(branch_e)


branch_e = Conv2D(64, (3,3), activation='elu', padding='same' )(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e = Conv2D(64, (3,3), activation='elu', padding='same')(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e = MaxPooling2D(2,2)(branch_e)
branch_e = Dropout(0.5)(branch_e)


branch_e = Conv2D(32, (3,3), activation='elu', padding='same' )(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e = Conv2D(32, (3,3), activation='elu', padding='same')(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e = Conv2D(16, (3,3), activation='elu', padding='same' )(branch_e)
branch_e = BatchNormalization()(branch_e)
branch_e = Conv2D(16, (3,3), activation='elu', padding='same')(branch_e)
branch_e = BatchNormalization()(branch_e)


branch_e = Flatten()(branch_e)

model = Model(x, branch_e)