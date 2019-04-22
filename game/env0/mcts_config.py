## Hyperparameters
epsilon = 0
cpuct   = 1
alpha   = 0.8
sims    = 30

## Training parameters
momentum    = 0.9
lr          = 0.1
reg         = 10e-4
batch_size  = 256
epochs      = 1
loops       = 10
memory_size = 30000

hidden_layers = [
    {'filters': 75, 'kernel_size': (3,5)} for _ in range(10)
]

## Evaluation
episodes          = 30
scoring_threshold = 1.3
