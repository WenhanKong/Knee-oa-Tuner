from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import Xception, ResNet50, InceptionV3, EfficientNetV2M

def build_base_model(name):
    if name == 'Xception':
        return Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            name=name
        )
    elif name == 'renet50':
         return ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            name=name
        )
    elif name == 'efficient_net_v2':
        return EfficientNetV2M(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
        )
    else:
        raise ValueError()
    
def build_xception(hp):
    
    # Base model
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        name='xception'
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(224, 224, 3)) # layer 0
    x = base_model(inputs, training=False) # layer 1
    
    # Tunable top layers
    x = layers.GlobalAveragePooling2D()(x)
    
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        x = layers.Dense(
            units=hp.Int(f'dense_units_{i}', 128, 512, step=128),
            activation='relu',
            kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 1e-5, 1e-2))
        )(x)
        x = layers.Dropout(hp.Float('dropout', 0.3, 0.6))(x)
    
    outputs = layers.Dense(3, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    
    # Tune optimizer and learning rate
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Float('lr', 1e-5, 1e-3, sampling='log')
    
    model.compile(
        optimizer=optimizers.get({'class_name': optimizer, 'config': {'learning_rate': lr}}),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_resnet50(hp):
        
    # Base model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        name='resnet50'
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(224, 224, 3)) # layer 0
    x = base_model(inputs, training=False) # layer 1

    # Tunable top layers
    x = layers.GlobalAveragePooling2D()(x)
    
    for i in range(hp.Int('num_dense_layers', 1, 2, 3)):
        x = layers.Dense(
            units=hp.Int(f'dense_units_{i}', 256, 1024, step=256),
            activation='relu',
            kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 1e-5, 1e-2))
        )(x)
        x = layers.Dropout(hp.Float('dropout', 0.3, 0.6))(x)
        
    # x = layers.BatchNormalization()(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    
    # Tune optimizer and learning rate
    optimizer = hp.Choice('optimizer', ['adam'])
    # optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])

    lr = hp.Float('lr', 1e-5, 1e-3, sampling='log')
    
    model.compile(
        optimizer=optimizers.get({'class_name': optimizer, 'config': {'learning_rate': lr}}),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

    
def build_efficient_net_v2(hp):
    base_model = build_base_model('efficient_net_v2')
    base_model.trainable = False
        
    inputs = layers.Input(shape=(224, 224, 3)) # layer 0
    x = base_model(inputs, training=False) # layer 1
    
    # Tunable top layers
    x = layers.GlobalAveragePooling2D()(x)
    
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        x = layers.Dense(
            units=hp.Int(f'dense_units_{i}', 256, 1024, step=256),
            activation='relu',
            kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 1e-5, 1e-2))
        )(x)
        x = layers.Dropout(hp.Float('dropout', 0.3, 0.6))(x)
    
    outputs = layers.Dense(3, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    
    # Tune optimizer and learning rate
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Float('lr', 1e-5, 1e-3, sampling='log')
    
    model.compile(
        optimizer=optimizers.get({'class_name': optimizer, 'config': {'learning_rate': lr}}),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model