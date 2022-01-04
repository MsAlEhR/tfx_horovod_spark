import tensorflow as tf
from tfx import v1 as tfx
import os
import multiprocessing
import socket
from typing import List, Optional

from absl import logging
logging.set_verbosity(logging.INFO)  # Set default logging level.

PIPELINE_NAME = "penguin-sample"

_trainer_module_file = 'trainer.py'

# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = os.path.join('/YOUR_SHARED_STORAGE_PATH', PIPELINE_NAME)
# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join('/YOUR_SHARED_STORAGE_PATH', PIPELINE_NAME, 'metadata.db')
# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('/YOUR_SHARED_STORAGE_PATH', PIPELINE_NAME)


DATA_ROOT="./data"

_data_filepath = os.path.join(DATA_ROOT, "data.csv")


try:
  _parallelism = multiprocessing.cpu_count()
except NotImplementedError:
  _parallelism = 1

_beam_portable_pipeline_args = [

	'--runner=SparkRunner',
	 "--job_endpoint=YOUR_MASTER_IP:8099",
	 "--environment_type=EXTERNAL",
	 "--environment_config=localhost:50000",
	'--spark_version=3',
	'--worker_harness_container_image=None',
	'--experiments=use_loopback_process_worker=True',
	'--sdk_worker_parallelism=%d' % _parallelism,
	'--environment_cache_millis=1000000',
	'--experiments=pre_optimize=all',
]

def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     beam_pipeline_args: List[str],
                     module_file: str, serving_model_dir: str,
                     metadata_path: str) -> tfx.dsl.Pipeline:
  """Creates a three component penguin pipeline with TFX."""
  # Brings data into the pipeline.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5))


  # Following three components will be included in the pipeline.
  components = [
     example_gen,
      trainer,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      enable_cache=True,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_pipeline_args,
      components=components)



tfx.orchestration.LocalDagRunner().run(
  _create_pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      data_root=DATA_ROOT,
      module_file=_trainer_module_file,
      beam_pipeline_args=_beam_portable_pipeline_args, 
      serving_model_dir=SERVING_MODEL_DIR,
      metadata_path=METADATA_PATH))

