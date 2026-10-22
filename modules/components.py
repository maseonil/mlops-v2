"""Memulai komponen pipeline TFX."""

import os

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)

# pylint: disable=too-many-arguments
def init_components(
    *,
    data_dir,
    transform_module,
    tuner_module,
    training_module,
    training_steps,
    eval_steps,
    serving_model_dir,
):
    """Initiate tfx pipeline components.

    Args:
        data_dir (str): path ke data
        transform_module (str): path ke transform_module
        tuner_module (str): path ke tuner_module
        training_module (str): path ke transform_module
        training_steps (int): jumlah training steps
        eval_steps (int): jumlah of eval steps
        serving_model_dir (str): path ke direktori serving model

    Returns:
        TFX components
    """
    # pylint: disable=too-many-locals
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    components = {}

    components['example_gen'] = CsvExampleGen(
        input_base=data_dir,
        output_config=output
    )

    components['statistics_gen'] = StatisticsGen(
        examples=components['example_gen'].outputs["examples"]
    )

    components['schema_gen'] = SchemaGen(
        statistics=components['statistics_gen'].outputs["statistics"]
    )

    components['example_validator'] = ExampleValidator(
        statistics=components['statistics_gen'].outputs['statistics'],
        schema=components['schema_gen'].outputs['schema']
    )

    components['transform'] = Transform(
        examples=components['example_gen'].outputs['examples'],
        schema=components['schema_gen'].outputs['schema'],
        module_file=os.path.abspath(transform_module)
    )

    components['tuner'] = Tuner(
        module_file=os.path.abspath(tuner_module),
        examples=components['transform'].outputs['transformed_examples'],
        transform_graph=components['transform'].outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'], num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps)
    )

    components['trainer'] = Trainer(
        module_file=os.path.abspath(training_module),
        examples=components['transform'].outputs['transformed_examples'],
        transform_graph=components['transform'].outputs['transform_graph'],
        schema=components['schema_gen'].outputs['schema'],
        hyperparameters=components['tuner'].outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps)
    )

    components['model_resolver'] = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=[
            "Gender",
        ])
    ]

    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(
                class_name='SparseCategoricalAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.20}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': -1e-10})
                )
            )
        ])
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Experience_Level')],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )

    components['evaluator'] = Evaluator(
        examples=components['example_gen'].outputs['examples'],
        model=components['trainer'].outputs['model'],
        baseline_model=components['model_resolver'].outputs['model'],
        eval_config=eval_config)

    components['pusher'] = Pusher(
        model=components['trainer'].outputs["model"],
        model_blessing=components['evaluator'].outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=os.path.join(
                    serving_model_dir,
                    'gym-experience-model'))),
    )

    return tuple(components.values())
    