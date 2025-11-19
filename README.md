# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.9+
- PyTorch 2.5.1+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy
| Model             | Training Loss        | Accuracy |
| ----------------- | ----------- | |
| [DLA]                   | 95.47%      | |
| [DLA_paraboloid]        | 92.64%      | |
| [DLA_paraconv]          | 93.02%      | |
| [DLA_paraconv_half]     | 93.62%      | |
| [DLA_paraconv_quarter]  | 93.75%      | |
