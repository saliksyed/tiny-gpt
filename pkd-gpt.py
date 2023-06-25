import torch
from dataset import PKD


class BaseModel(torch.nn.Module):
    def __init__(self, dataset, n_embed=32):
        super().__init__()
        self.dataset = dataset
        self.token_embedding = torch.nn.Embedding(dataset.vocab_size, n_embed)
        self.relu = torch.nn.ReLU()
        self.reconstruction = torch.nn.Linear(n_embed, dataset.vocab_size)
        

    def forward(self, batch, targets=None):
        logits = self.reconstruction(self.relu(self.token_embedding(batch)))
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
    
    def train(self, block_size=8, batch_size=16, num_iterations=1000, learning_rate=1e-2):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.block_size = block_size
        for iter in range(num_iterations):
            xb, yb = self.dataset.get_batch(batch_size, block_size, validation=False)
            logits, loss = self(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if (iter % 100 == 0):
                print(f'Iteration {iter}')

dataset = PKD()
model = BaseModel(dataset)
model.train(num_iterations=1)
print(model.generate(1000))