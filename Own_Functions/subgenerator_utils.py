# subgenerator_utils.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def create_subgenerator(generator, indices):
    subset_data = [generator.filepaths[idx] for idx in indices if idx < len(generator.filepaths)]
    subset_labels = [generator.classes[idx] for idx in indices if idx < len(generator.filepaths)]
    idx_to_class = {v: k for k, v in generator.class_indices.items()}
    subset_labels = [idx_to_class[idx] for idx in subset_labels]

    sub_datagen = ImageDataGenerator(rescale=1./255)
    sub_generator = sub_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': subset_data, 'class': subset_labels}),
        x_col='filename',
        y_col='class',
        target_size=generator.target_size,
        batch_size=generator.batch_size,
        class_mode=generator.class_mode,
        shuffle=False
    )
    return sub_generator
