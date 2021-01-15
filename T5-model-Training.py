
# coding: utf-8

# In[1]:

import os
#On CPU only # on GPU comment below lines 1 line
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gin
import t5

import json
import random
import functools
import tensorflow as tf
import tensorflow_datasets as tfds


# In[220]:



# import csv
# import os
# import sys

# file="/home/parag.patil/Parag/greenField/t5_data/train_new9k.csv"
# newfile = "/home/parag.patil/Parag/greenField/t5_data/train_new9k.txt"
# csv.field_size_limit(sys.maxsize)

# with open(file,'r') as csv_file:
#     csv_reader = csv.reader(csv_file)

#     #csv_reader.next()  ## skip one line (the first one)

#     #newfile = file + '.txt'

#     for line in csv_reader:
#         with open(newfile, 'a') as new_txt:    #new file has .txt extn
#             txt_writer = csv.writer(new_txt, delimiter = '\t') #writefile
#             txt_writer.writerow(line)  


# In[2]:


nq_tsv_path = {
    "train": "/home/parag/model_training/test/data/train.csv",
    "validation":"/home/parag/model_training/test/data/validation.csv"
}


# In[208]:


# def dataset_fn(split, shuffle_files=False):
#     # We only have one file for each split.
#     del shuffle_files

#     # Load lines from the text file as examples.
#     ds = tf.data.TextLineDataset(nq_tsv_path[split])
#     # Split each "<question>\t<answer>" example into (question, answer) tuple.
#     print(" >>>> about to read csv . . . ")
#     ds = ds.map(
#         functools.partial(tf.io.decode_csv, record_defaults=["", ""],
#                           field_delim="\t", use_quote_delim=False),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     print(" >>>> after reading csv . . . ")
#     # Map each tuple to a {"question": ... "answer": ...} dict.
#     ds = ds.map(lambda *ex: dict(zip(["text", "summary"], ex)))
#     print(" >>>> after mapping . . . ")
#     return ds


# In[222]:


# for ex in tfds.as_numpy(dataset_fn("train").take(2)):
#   print(ex)


# In[3]:


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
   # text = tf.strings.lower(text)
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
ds = nq_task.get_dataset(split="validation", sequence_length={"inputs": 2000, "targets": 8})
print("A few preprocessed validation examples...")
for ex in tfds.as_numpy(ds.take(2)):
  print(ex)


# t5.data.TaskRegistry.add(
#     "nq_context_free",
#     # Supply a function which returns a tf.data.Dataset.
#     dataset_fn=nq_dataset_fn,
#     splits=["train", "validation"],
#     # Supply a function which preprocesses text from the tf.data.Dataset.
#     text_preprocessor=[trivia_preprocessor],
#     # Lowercase targets before computing metrics.
#     postprocess_fn=t5.data.postprocessors.lower_text, 
#     # We'll use accuracy as our evaluation metric.
#     metric_fns=[t5.evaluation.metrics.accuracy],
#     # Not required, but helps for mixing and auto-caching.
#     #num_input_examples=num_nq_examples
# ) 

# In[7]:


t5.data.MixtureRegistry.remove("trivia_all")
t5.data.MixtureRegistry.add(
    "trivia_all",
    ["nq_context_free"],
     default_rate=1.0
)


# In[8]:


sequence_length={"inputs": 2000, "targets": 8}

# export model directory 
MODEL_DIR = "/home/parag/model_training/test/model"
PRETRAINED_DIR = f'gs://t5-data/pretrained_models/small'


# In[9]:


model = t5.models.MtfModel(
        model_dir=MODEL_DIR,
    tpu=None,
    tpu_topology=None,
 
    batch_size=8,
    sequence_length=sequence_length,
    learning_rate_schedule=0.003,
    save_checkpoints_steps=500,
    keep_checkpoint_max=40,
    iterations_per_loop=100,
)



# In[217]:


# import tensorboard as tb
# tb.notebook.start("--logdir " + MODEL_DIR)


# In[ ]:


FINETUNE_STEPS = 1500 #@param {type: "integer"}

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


# In[206]:


# import tensorflow as tf
# import tensorflow_text  # Required to run exported model.

# def load_predict_fn(model_path):
#   if tf.executing_eagerly():
#     print("Loading SavedModel in eager mode.")
#     imported = tf.saved_model.load(model_path, ["serve"])
#     return lambda x: imported.signatures['serving_default'](tf.constant(x))['outputs'].numpy()
#   else:
#     print("Loading SavedModel in tf 1.x graph mode.")
#     tf.compat.v1.reset_default_graph()
#     sess = tf.compat.v1.Session()
#     meta_graph_def = tf.compat.v1.saved_model.load(sess, ["serve"], model_path)
#     signature_def = meta_graph_def.signature_def["serving_default"]
#     return lambda x: sess.run(
#         fetches=signature_def.outputs["outputs"].name, 
#         feed_dict={signature_def.inputs["input"].name: x}
#     )

# predict_fn = load_predict_fn(saved_model_path)


# In[207]:


# def answer(question):
#   return predict_fn([question])[0].decode('utf-8')

# for question in ["trivia question: where is the google headquarters?",
#                  "trivia question: what is the most populous country in the world?",
#                  "trivia question: who are the 4 members of the beatles?",
#                  "trivia question: how many teeth do humans have?"]:
#     print(answer(question))

