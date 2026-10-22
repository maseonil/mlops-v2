"""Module untuk mendefinisikan preprocessing transformasi fitur menggunakan TFX."""

import tensorflow_transform as tft

LABEL_KEY = 'Experience_Level'

NUMERIC_FEATURE_KEYS = [
    'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
    'Session_Duration (hours)', 'Fat_Percentage', 'Water_Intake (liters)',
    'Workout_Frequency (days/week)', 'BMI'
]

CATEGORICAL_FEATURE_KEYS = {'Gender': 2, 'Workout_Type': 4}


def transformed_name(key):
    """Mengubah nama fitur asli menjadi nama fitur setelah transformasi (_xf).

    Args:
        key (str): Nama asli dari fitur.

    Returns:
        str: Nama fitur baru yang sudah bersih dari spasi dan karakter khusus.
    """
    cleaned_key = (
        key.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return f"{cleaned_key}_xf"


def preprocessing_fn(inputs):
    """Fungsi callback dari TensorFlow Transform untuk preprocessing data.

    Args:
        inputs: Kamus (dict) berisi tensor fitur mentah sebelum transformasi.

    Returns:
        dict: Kamus berisi tensor fitur yang sudah ditransformasi.
    """
    outputs = {}

    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    for key, dim in CATEGORICAL_FEATURE_KEYS.items():
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )

    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        inputs[LABEL_KEY]
    )

    return outputs
    