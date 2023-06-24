import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Read file to string
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# setup encode and decode functions
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# split into train/test sets
data = tf.convert_to_tensor(np.array(encode(text), dtype=np.int64), dtype=tf.int64)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(data) - block_size, dtype=tf.int32)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# TF Dataset generators
def train_dataset():
  xb, yb = get_batch('train')
  yield xb, yb

def val_dataset():
  xb, yb = get_batch('validation')
  yield xb, yb


class BigramModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size, vocab_size)

  def call(self, inputs, training=False):
    logits = self.embedding(inputs)
    return logits
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits  = self.call(idx, training=False)
        logits = logits[:, -1, :]
        vocab = np.array([i for i in range(0, vocab_size)], dtype=np.float32)
        idx_next = [tf.cast(tf.tensordot(tfp.distributions.Multinomial(1, logits=logits).sample(), vocab, 1), dtype=tf.int64)]
        idx = tf.concat((idx, idx_next), axis=1)
    return idx

model = BigramModel()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

history = model.fit(
  x=tf.data.Dataset.from_generator(train_dataset, (tf.float32, tf.float32),
                                  (tf.TensorShape([batch_size, block_size]),
                                  tf.TensorShape([batch_size, block_size]))),
  validation_data=tf.data.Dataset.from_generator(val_dataset, (tf.float32, tf.float32),
                                  (tf.TensorShape([batch_size, block_size]),
                                  tf.TensorShape([batch_size, block_size]))),
  epochs=10
)

gen = model.generate(np.array([[0,0]]), 100)
print(decode(gen.numpy()[0]))



