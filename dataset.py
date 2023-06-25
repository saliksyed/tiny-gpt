import tiktoken
import torch

class PKD:
    def __init__(self):
        with open('pkd.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.encoder = tiktoken.get_encoding("r50k_base")
            self.tokens = torch.tensor(self.encoder.encode(self.text))
            self.vocab = sorted(list(set(self.tokens)))
            self.vocab_size = len(self.vocab)
            train = int(0.9 * len(self.tokens))
            self.training_data = self.tokens[:train]
            self.test_data = self.tokens[train:]

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, encoding):
        return self.encoder.decode([self.vocab[e] for e in encoding])
    
    def get_batch(self, batch_size, block_size, validation=False):
        data = self.test_data if validation else self.training_data
        rand_idx = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in rand_idx])
        y = torch.stack([data[i + 1: i + 1 + block_size] for i in rand_idx])
        return x,y
    

def test_PKD_encoding():
    dataset = PKD()
    text = "hello brave new world"
    assert dataset.decode(dataset.encode(text)) == text

def test_PKD_get_batch():
    dataset = PKD()
    x , y = dataset.get_batch(32, 8)
    assert x.shape == (32, 8)
    assert y.shape == (32, 8)