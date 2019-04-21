## Hyperparameters
epsilon = 0
cpuct   = 1
alpha   = 0.8
sims    = 100

## Training parameters
momentum    = 0.9
lr          = 0.1
reg         = 10e-4
batch_size  = 256
epochs      = 10
loops       = 10
memory_size = 300

hidden_layers = [
    {'filters': 15, 'kernel_size': (2,2)} for _ in range(10)
]

## Evaluation
episodes          = 30
scoring_threshold = 1.3
