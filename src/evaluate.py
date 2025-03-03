# %%
from tensorflow import keras
from .data_loader import get_test_data_generator, get_data_generators
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def fine_tune(model):
    # Unfreeze top layers of Xception
    base_model = model.layers[1]  #  Xception is the 2nd layer    

    # Unfreeze top 20% of Xception layers
    num_unfreeze = int(len(base_model.layers) * 0.2)
    for layer in base_model.layers[-num_unfreeze:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Freeze bottom 80%
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False
    
    # model.compile()
    model.compile(
    optimizer=Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Train (fine-tune)
    
    train_gen, valid_gen = get_data_generators()
    
    history = model.fit(
        train_gen,
        epochs=50,  # Adjust as needed
        validation_data=valid_gen,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            # keras.callbacks.TensorBoard(log_dir='logs/finetune_logs')
        ]
    )
    
    return model
    
    
def evalueate(model_path):
    
    # tf.config.run_functions_eagerly(True)
    
    test_gen, _ = get_data_generators()
    
    # Load the saved model
    best_model = keras.models.load_model(model_path)
    
    # print(best_model.summary())
    # print(best_model.layers[1].summary())
    # best_model = fine_tune(best_model)
    
    # Evaluate on test data
    test_loss, test_acc = best_model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Get true labels and predictions
    y_true = test_gen.labels
    y_pred_probs = best_model.predict(test_gen, steps=len(test_gen))
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="local")

    args = parser.parse_args()
    project_name = args.project_name
    
    model_path = f'/WAVE/users2/unix/wkong/ml240/models/best_model_{project_name}.keras'
    evalueate(model_path)
    
    