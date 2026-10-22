"""Module untuk proses Hyperparameter Tuning menggunakan Keras Tuner di TFX."""

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.components.tuner.component import TunerFnResult

LABEL_KEY = 'Experience_Level'


def transformed_name(key):
    """Mengubah nama fitur asli menjadi nama fitur setelah transformasi (_xf).

    Args:
        key (str): Nama asli dari fitur.

    Returns:
        str: Nama fitur baru yang bersih dari spasi dan karakter khusus.
    """
    cleaned_key = (
        key.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return f"{cleaned_key}_xf"


NUMERIC_FEATURE_KEYS = [
    'Age',
    'Weight (kg)',
    'Height (m)',
    'Max_BPM',
    'Avg_BPM',
    'Resting_BPM',
    'Session_Duration (hours)',
    'Fat_Percentage',
    'Water_Intake (liters)',
    'Workout_Frequency (days/week)',
    'BMI'
]

CATEGORICAL_FEATURE_KEYS = ['Gender', 'Workout_Type']


def gzip_reader_fn(filenames):
    """Membaca file dataset TFRecord yang dikompresi dengan format GZIP.

    Args:
        filenames: Pattern atau daftar nama file data TFRecord.

    Returns:
        tf.data.TFRecordDataset objek.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Membuat batch dataset dari file TFRecord yang sudah ditransformasi.

    Args:
        file_pattern: Pola jalur file data input.
        tf_transform_output: Output grafik transformasi TFX.
        batch_size (int): Jumlah data per batch.

    Returns:
        tf.data.Dataset objek yang siap dikonsumsi model.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY))
    return dataset


def build_model(hp):
    """Membangun arsitektur model Keras DNN dengan Hyperparameter Search Space.

    Args:
        hp: Objek HyperParameters dari Keras Tuner.

    Returns:
        tf.keras.Model objek yang sudah dikompilasi.
    """
    inputs = []
    encoded_features = []

    for feature in NUMERIC_FEATURE_KEYS:
        input_layer = tf.keras.Input(
            shape=(1,), name=transformed_name(feature))
        inputs.append(input_layer)
        encoded_features.append(input_layer)

    for feature in CATEGORICAL_FEATURE_KEYS:
        input_layer = tf.keras.Input(
            shape=(1,),
            name=transformed_name(feature),
            dtype=tf.int64)
        embedding = tf.keras.layers.Embedding(
            input_dim=10, output_dim=4)(input_layer)
        flatten = tf.keras.layers.Flatten()(embedding)
        inputs.append(input_layer)
        encoded_features.append(flatten)

    concatenate = tf.keras.layers.concatenate(encoded_features)

    units_1 = hp.Choice('units_1', values=[32, 64])
    units_2 = hp.Choice('units_2', values=[16, 32])
    dropout_rate = hp.Choice('dropout', values=[0.3, 0.4])
    learning_rate = hp.Choice('learning_rate', values=[1e-3])

    deep = tf.keras.layers.Dense(units_1, activation="relu")(concatenate)
    deep = tf.keras.layers.Dropout(dropout_rate)(deep)
    deep = tf.keras.layers.Dense(units_2, activation="relu")(deep)

    outputs = tf.keras.layers.Dense(4, activation="softmax")(deep)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def tuner_fn(fn_args):
    """Fungsi callback TFX Tuner untuk mengeksekusi pencarian hyperparameter.

    Args:
        fn_args: Argumen pelari TFX yang berisi jalur data dan parameter langkah.

    Returns:
        TunerFnResult objek yang berisi tuner dan argumen fitting model.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective='val_accuracy',
        max_trials=8,
        directory=fn_args.working_dir,
        project_name='gym_project'
    )

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
        }
    )
    