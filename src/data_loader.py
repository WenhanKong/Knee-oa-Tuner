# %%
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_test_data_generator(batch_size=16, image_size=(224, 224)):
    
    with open('/WAVE/users2/unix/wkong/ml240/data/test_df_new.pkl', 'rb') as f:
        test_df_new = pickle.load(f)
    
    ts_gen = ImageDataGenerator(rescale=1./255)
    
    test_gen_new = ts_gen.flow_from_dataframe(
        test_df_new,x_col='image_path',
        y_col='category_encoded',
        target_size=image_size,
        class_mode='sparse',
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )
    
    return test_gen_new
    
    return test_gen_new
def get_data_generators(batch_size=16, image_size=(224, 224)):
    
    # Load the serialized DataFrames
    with open('/WAVE/users2/unix/wkong/ml240/data/train_df_new.pkl', 'rb') as f:
        train_df_new = pickle.load(f)

    with open('/WAVE/users2/unix/wkong/ml240/data/valid_df_new.pkl', 'rb') as f:
        valid_df_new = pickle.load(f)

    tr_gen = ImageDataGenerator(rescale=1./255)

    train_gen_new = tr_gen.flow_from_dataframe(
        train_df_new,
        x_col='image_path',
        y_col='category_encoded',
        target_size=image_size,
        class_mode='sparse',
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    valid_gen_new = tr_gen.flow_from_dataframe(
        valid_df_new,
        x_col='image_path',
        y_col='category_encoded',
        target_size=image_size,
        class_mode='sparse',
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    return train_gen_new, valid_gen_new