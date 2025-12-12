# Using paraboloid neurons to train DLA models on CIFAR10 with PyTorch

Paraboloid neuron demonstration for [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Requirements
- Linux only
- Python 3.9+, use of a virtual environment recommended
- Install the rest of the requirements by running:
```
pip install -r requirements.txt
```
- (Optional) Download the pre-trained models by running:
```
wget -i models.txt -P checkpoint
```

## Models
### DLA
Our baseline Deep Layer Aggregation model.
### DLA_paraboloid
A DLA model with an additional layer of paraboloid neurons before the output layer. In terms of code, first we import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```

Then we insert the layer between these two lines:
```
self.layer6 = Tree(block, 256, 512, level=1, stride=2)
self.linear = nn.Linear(512, num_classes)
```
as so:
```
self.layer6 = Tree(block, 256, 512, level=1, stride=2)
self.paraboloid = gpt.Paraboloid(512, 1024, h_factor = 0.01, lr_factor = 1., wd_factor = 10., grad_factor = 1., input_factor = 0.1, output_factor = 0.1)
self.linear = nn.Linear(1024, num_classes)
```
Note that ```Paraboloid(512, 1024, h_factor = 0.01, lr_factor = 1., wd_factor = 10., grad_factor = 1., input_factor = 0.1, output_factor = 0.1)``` uses the default arguments and is equivalent to ```Paraboloid(512, 1024)```. We include the assignments here to show which parameters can be changed to fine tune the Paraboloid layer.

Remember to update the forward function:
```
out = self.layer6(out)
out = F.avg_pool2d(out, 4)
out = out.view(out.size(0), -1)
out = self.paraboloid(out)
out = self.linear(out)
```

### DLA_paraconv
A DLA model with the first convolutional layer replaced with a paraboloid convolutional layer. In terms of code, again, we first import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```
Then we find the line with the first convolutional layer:
```
nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
```
and replace it with:
```
gpt.ParaConv2d(3, 16, kernel_size=3, stride=1, padding=1, wd_factor = 2., lr_factor = 1., output_factor = 0.1, h_factor = 0.01),
```
Again, ```ParaConv2d(3, 16, kernel_size=3, stride=1, padding=1, wd_factor = 2., lr_factor = 1., output_factor = 0.1, h_factor = 0.01)``` is equivalent to ```ParaConv2d(3, 16)```. We include the assignments here to show which parameters can be changed to fine tune the ParaConv2d layer.

In this case, we do not need to update the forward function, as we replaced an existing layer.

### DLA_paraconv_half and DLA_paraconv_quarter
The same as DLA_paraconv, but only using 8 and 4 neurons (respectively) instead of 16. Note the input parameter of the following layer must also be adjusted accordingly.

## IMPORTANT
Including a layer with paraboloid neurons requires a specialized optimizer:
```
#optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                      momentum=0.9, weight_decay=5e-4, nesterov = True)
optimizer = gpt.GeoNDSGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4, nesterov = True)
```
It is recommended that ```nesterov = True``` be included in the arguments when using paraboloid neurons.

## Evaluation and Training

### Evaluate the models by running: 
```
python main.py --model dla --eval dla.pth
python main.py --model dla_paraboloid --eval dla_paraboloid.pth
python main.py --model dla_paraconv --eval dla_paraconv.pth
python main.py --model dla_paraconv_half --eval dla_paraconv_half.pth
python main.py --model dla_paraconv_quarter --eval dla_paraconv_quarter.pth
```
### Train a model from scratch by ommitting the --eval argument, e.g.:
```
python main.py --model dla_paraconv_half
```
### You can resume the training with:
```
python main.py --model dla_paraconv_half --resume
```

## Training Loss and Accuracy
|   Model           | Training Loss        | Accuracy |
| ----------------- | -------------        | -------- |
| [DLA] (400 epochs) - BASELINE       | 0.001210       | 96.03% |
| [DLA_paraboloid] (400 epochs)       | 0.000218       | 96.08% |
| [DLA_paraconv] (400 epochs)         | 0.001263       | 96.11% |
| [DLA_paraconv_half] (400 epochs)    | 0.001279       | 96.15% |
| [DLA_paraconv_quarter] (200 epochs) | 0.001007       | 96.04% |
