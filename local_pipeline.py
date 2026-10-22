"""Module untuk menginisialisasi dan menjalankan TFX pipeline secara lokal."""

import os
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "maseonil-pipeline"

DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TUNER_MODULE_FILE = "modules/tuner.py"
TRAINER_MODULE_FILE = "modules/trainer.py"

OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_local_pipeline(
    pipeline_components, custom_pipeline_root: Text
) -> pipeline.Pipeline:
    """Menginisialisasi pipeline TFX lokal dengan komponen yang diberikan.

    Args:
        components: Daftar komponen TFX yang akan dijalankan.
        pipeline_root: Jalur direktori root untuk output pipeline.

    Returns:
        Objek pipeline TFX.
    """
    logging.info(f"Pipeline root set to: {custom_pipeline_root}")
    beam_args = []
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=custom_pipeline_root,
        components=pipeline_components,
        enable_cache=False,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    # pylint: disable=import-outside-toplevel
    from modules.components import init_components

    # pipeline component
    components = init_components(
        data_dir=DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=serving_model_dir,
    )

    local_pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=local_pipeline)
    