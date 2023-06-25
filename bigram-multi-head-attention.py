import numpy as np
import math
import tensorflow as tf
import tensorflow_probability as tfp

# Read file to string
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
n_embed = 32

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
batch_size = 16

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



class Head(tf.keras.Model):
  def __init__(self, head_size):
    super().__init__()
    self.head_size = head_size
    self.key = tf.keras.layers.Dense(head_size, activation=None, use_bias=False)
    self.query = tf.keras.layers.Dense(head_size, activation=None, use_bias=False)
    self.value = tf.keras.layers.Dense(head_size, activation=None, use_bias=False)

  def call(self, inputs, training=False):
    B,T,C = inputs.shape
    k = self.key(inputs)
    q = self.query(inputs)

    raw_affinity = tf.matmul(q, tf.transpose(k,perm=[0,2,1]))
    num = int((T*(T+1))/2)
    mask_matrix = tfp.math.fill_triangular(tf.ones(num))
    final_affinity = raw_affinity * tf.where(mask_matrix!=1, -99999999999999., tf.cast(mask_matrix, tf.float32))
    final_affinity = tf.nn.softmax(final_affinity * self.head_size**-0.5)
    out = tf.matmul(final_affinity, self.value(inputs))
    return out
  
class MultiHead(tf.keras.Model):
    def __init__(self, n, total_head_size):
        super().__init__()
        self.heads = [Head(total_head_size // n) for i in range(0, n)]

    def call(self, inputs, training=False):
        results = []
        for head in self.heads:
            curr = head.call(inputs, training)
            results.append(curr)
        return tf.concat(tuple(results), axis=2)

class FeedForward(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(n_embed, activation=None)
        self.layer2 = tf.keras.layers.Dense(n_embed, activation='relu')

    def call(self, inputs, training=False):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x

class BigramModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size, n_embed)
    self.pos_embedding = tf.keras.layers.Embedding(block_size, n_embed)
    self.head = MultiHead(4, n_embed)
    self.ffwd = FeedForward()
    self.lm_head = tf.keras.layers.Dense(vocab_size, activation=None)

  def call(self, inputs, training=False):
    B,T = inputs.shape
    embedding = self.embedding(inputs)
    pos_embedding = self.pos_embedding(tf.range(T))
    x = embedding + pos_embedding
    x = self.head(x)
    x = self.ffwd(x)
    logits = self.lm_head(x)
    return logits
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cropped = idx[:, -block_size:]
        logits  = self.call(idx_cropped, training=False)
        logits = logits[:, -1, :]
        probs= tf.nn.softmax(logits, axis=-1)
        print("probs", probs.numpy()[0])
        idx_next = encode(np.random.choice(np.array(chars), p=probs.numpy()[0]))
        idx = tf.concat((idx, np.array([idx_next])), axis=1)
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
  epochs=1000
)

gen = model.generate(np.array([[0,0]]), 200)
print(decode(gen.numpy()[0]))



