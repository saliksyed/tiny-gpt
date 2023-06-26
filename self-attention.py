import torch
from dataset import Dataset


class SelfAttention(torch.nn.Module):
    def __init__(self, head_input_size, head_size=32):
        super().__init__()
        self.query = torch.nn.Linear(head_input_size, head_size)
        self.key = torch.nn.Linear(head_input_size, head_size)
        self.value = torch.nn.Linear(head_input_size, head_size)

    def forward(self, examples):
        batch_size, block_size, vocab_size = examples.shape

        # for each example compute it's key and query
        q = self.query(examples)
        k = self.key(examples)
        v = self.value(examples)

        affinity = k @ torch.transpose(q, 2, 1)
        mask = torch.tril(torch.ones(block_size, block_size))
        affinity = affinity.masked_fill(mask == 0, float('-inf'))
        affinity = torch.nn.functional.softmax(affinity, dim=-1)
        return affinity @ v

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, head_input_size, n_heads=4, total_head_size=32):
        super().__init__()
        self.heads = [SelfAttention(head_input_size, total_head_size // n_heads) for _ in range(n_heads)]
    
    def forward(self, batch, targets=None):
        return torch.cat([h(batch) for h in self.heads], dim=-1)

class BaseModel(torch.nn.Module):
    def __init__(self, dataset, block_size=16, n_embed=64):
        super().__init__()
        self.dataset = dataset
        self.block_size = block_size
        self.vocab_embedding = torch.nn.Embedding(dataset.vocab_size, n_embed)
        self.pos_embedding = torch.nn.Embedding(self.block_size, n_embed)
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_embed, n_embed),
            MultiHeadSelfAttention(n_embed, 8, total_head_size=n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(n_embed, dataset.vocab_size),
        )

    def forward(self, batch, targets=None):
        B, T = batch.shape
        encoded = self.vocab_embedding(batch)
        logits = self.layers(encoded)
        batch_size, block_size, vocab_size = logits.shape
        loss = None
        if (targets != None):
            # targets shape is batch_size X block_size
            loss = torch.nn.functional.cross_entropy(logits.view(batch_size*block_size, vocab_size), targets.view(batch_size*block_size))
        return logits, loss
    
    def generate(self, max_tokens=100, init=torch.zeros((1,1), dtype=torch.long)):
        look_behind = self.block_size
        so_far = init
        
        for _ in range(max_tokens):
            logits, loss = self.forward(so_far[:,-look_behind:]) # logits is of size 1 x min(len(so_far), look_behind) x vocab_size
            # use the prediction being informed by the most recent logit:
            logits = logits[:, -1, :]
            # create probability dist from lgogits
            dist = torch.nn.functional.softmax(logits, dim=-1)
            next = torch.multinomial(dist, num_samples=1)
            so_far = torch.cat((so_far, next), dim=1)
        return self.dataset.decode(so_far.tolist()[0])
    
    def train(self, batch_size=16, num_iterations=1000, learning_rate=1e-2):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        print(sum(p.numel() for p in self.parameters())/1e6, 'M parameters')
        for iter in range(num_iterations):
            xb, yb = self.dataset.get_batch(batch_size, self.block_size, validation=False)
            logits, loss = self(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if (iter % 100 == 0):
                print(f'Iteration {iter} loss={loss}')

dataset = Dataset()
model = BaseModel(dataset, 256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.train(num_iterations=1000)
print(model.generate(1000))