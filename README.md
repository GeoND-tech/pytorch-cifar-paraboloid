# Using paraboloid neurons to train CIFAR10 with PyTorch

Paraboloid neuron demonstration for [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Requirements
- Python 3.9+
- Install the rest of the requirements by running:
```
pip install -r requirements.txt
```

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Training Loss and Accuracy
|   Model           | Training Loss        | Accuracy |
| ----------------- | -------------        | -------- |
| [DLA] (400 epochs)                  | 0.001210       | 96.03% |
| [DLA_paraboloid] (400 epochs)       | 0.000218       | 96.08% |
| [DLA_paraconv] (400 epochs)         | 0.001263       | 96.11% |
| [DLA_paraconv_half] (400 epochs)    | 0.001279       | 96.15% |
| [DLA_paraconv_quarter] (200 epochs) | 0.001007       | 96.04% |
