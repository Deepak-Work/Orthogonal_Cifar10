# Orthogonal_Cifar10
Implements Orthogonal Low Rank Embedding for CIFAR10 dataset using DenseNet Architecture. Also returns the calibrated confidences.

<H3>To run:</H3>

```python
python softmax_calibration.py --layers 30 --growth 9 --epochs 50 
--no-bottleneck --reduce 1.0 --name DenseNet-30-9
```
<H3>Description of each file:</H3>
 AngularLoss.py: Implements additive softmax and returns parameterized values of deep features as a function of theta.
<br> Densenet.py: Contains the architecture of the model. It is the standard Densenet architecture which is publically available.
<br> OLE.py: Implements orthogonal Low Rank Embedding loss function.
<br> Softmax_calibration.py: Main file.
<br> Temperature scaling: Implements softmax calibration.
  
  
  

