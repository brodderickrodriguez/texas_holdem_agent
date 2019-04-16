## Hyperparameters
epsilon = 0
cpuct   = 1
alpha   = 0.8
sims    = 100

## Training parameters
momentum   = 0.9
lr         = 0.1
reg_const  = 10e-4
batch_size = 256
epochs     = 10
loops      = 10

CNN_LAYERS = [
    {'filters': 75, 'kernel_size': (8,8)} for _ in range(50)
]

## Evaluation
episodes          = 30
scoring_threshold = 1.3