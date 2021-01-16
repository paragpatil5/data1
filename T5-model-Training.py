
# coding: utf-8

# In[1]:


import gin
import t5
import os
import json
import random
import functools
import tensorflow as tf
import tensorflow_datasets as tfds


# =============================================================================
#  only change finetune steps 
# =============================================================================
FINETUNE_STEPS = 2000 


pwd=os.getcwd()

# training data path 
nq_tsv_path = {
    "train": pwd+"/data/train.csv",
    "validation":pwd+"/data/validation.csv"
}

# model save directory 
MODEL_DIR = pwd+"/model"

input_sec=3000
output_sec=8

batch_sz=4

def nq_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path[split])
  # Split each "<question>\t<answer>" example into (question, answer) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["",""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"question": ... "answer": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["text", "summary"], ex)))
    
  return ds

# print("A few raw validation examples...")
for ex in tfds.as_numpy(nq_dataset_fn("validation").take(2)):
  print(ex)


# In[4]:


def trivia_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(
                 ["text: ", normalize_text(ex["text"])]),
        "targets": normalize_text(ex["summary"])
        
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


# In[5]:


t5.data.TaskRegistry.remove("nq_context_free")

t5.data.TaskRegistry.add(
    "nq_context_free",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=nq_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[trivia_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    #num_input_examples=num_nq_examples
)


# In[6]:


nq_task = t5.data.TaskRegistry.get("nq_context_free")
ds = nq_task.get_dataset(split="validation", sequence_length={"inputs": input_sec, "targets": output_sec})
print("A few preprocessed validation examples...")
for ex in tfds.as_numpy(ds.take(2)):
  print(ex)




t5.data.MixtureRegistry.remove("trivia_all")
t5.data.MixtureRegistry.add(
    "trivia_all",
    ["nq_context_free"],
     default_rate=1.0
)


# In[8]:


sequence_length={"inputs": input_sec, "targets": output_sec}

# export model directory 

PRETRAINED_DIR = f'gs://t5-data/pretrained_models/small'


# In[9]:


model = t5.models.MtfModel(
        model_dir=MODEL_DIR,
    tpu=None,
    tpu_topology=None,
 
    batch_size=batch_sz,
    sequence_length=sequence_length,
    learning_rate_schedule=0.003,
    save_checkpoints_steps=500,   # change this value for checkpoins 
    keep_checkpoint_max=40,
    iterations_per_loop=100,
)






model.finetune(
    mixture_or_task_name="trivia_all",
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=FINETUNE_STEPS
)



export_dir = os.path.join(MODEL_DIR, "export")

model.batch_size = 1 # make one prediction per call
saved_model_path = model.export(
    export_dir,
    checkpoint_step=-1,  # use most recent
    beam_size=1,  # no beam search
    temperature=1.0,  # sample according to predicted distribution
)
print("Model saved to:", saved_model_path)

