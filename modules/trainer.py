"""Module untuk proses pelatihan (training) model Keras di dalam pipeline TFX."""

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from keras.utils.vis_utils import plot_model

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
    """Membangun arsitektur model Keras DNN menggunakan hyperparameter terbaik.

    Args:
        hp (dict): Kamus berisi nilai hyperparameter hasil tuning.

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

    units_1 = hp.get('units_1', 64)
    units_2 = hp.get('units_2', 16)
    dropout_rate = hp.get('dropout', 0.4)
    learning_rate = hp.get('learning_rate', 1e-3)

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


def run_fn(fn_args: FnArgs):
    """Fungsi callback TFX Trainer untuk mengeksekusi pelatihan model.

    Args:
        fn_args: Argumen pelari TFX yang berisi jalur data dan parameter langkah.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    hp = fn_args.hyperparameters['values']
    model = build_model(hp)

    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=30,
        callbacks=[earlystop_cb]
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model,
            tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples'))}

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures)

    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True)


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membuat fungsi handler untuk memproses raw data saat model di-serve."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn
    