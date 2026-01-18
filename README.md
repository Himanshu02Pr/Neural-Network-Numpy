## Neural-Network-Numpy
**Two Layer Simple Neural Network Implementation Using Numpy.**

Basic Neural Network architecture to train upon MNIST dataset, to identify numbers from images.
Total 70,000 images, each of 28 X 28 pixels, divided as 60,000 training and 10,000 testing image sets.

Key Points :
- Reshaping and normalization of image data, to operate upon with numpy.
- Weight initialization using 'He(Kaiming)' scaling technique, to avoid gradient explosion or vanishing during ReLU activation.
- ReLU and Softmax implementation, with numerical overflow control in softmax.
- Simple yet efficient architecture, achieving accuracy of ~93 % in 10 epochs.

The underlying matrix operations -
```math
\begin{aligned}
Forward \ Propagation \\\\
Z[1]=W[1]X+b[1] \\\\
A[1]=gReLU(Z[1])) \\\\
Z[2]=W[2]A[1]+b[2] \\\\
A[2]=gsoftmax(Z[2]) \\\\\\
\\
Backward \ Propagation \\\\
dZ[2]=A[2]−Y \\\\ 
dW[2]=(1/m) \ dZ[2]A[1]T \\\\
dB[2]=(1/m) \ ΣdZ[2]\\\\
dZ[1]=W[2]TdZ[2].∗g[1]′(z[1])\\\\
dW[1]=(1/m) \ dZ[1]A[0]T\\\\
dB[1]=(1/m) \ ΣdZ[1]\\\\
\\
Parameter \ Updates\\\\
W[2]:=W[2]−αdW[2] \\\\ 
b[2]:=b[2]−αdb[2] \\\\ 
W[1]:=W[1]−αdW[1] \\\\ 
b[1]:=b[1]−αdb[1] \\\\
\end{aligned}


```
<br>
  <img width="673" height="496" alt="image" src="https://github.com/user-attachments/assets/1fc856d3-fbae-414f-8dc8-da50d12dc3c1" /> 
<br><br>
  <img width="673" height="682" alt="image" src="https://github.com/user-attachments/assets/f68f83e5-66c6-47ca-8dde-57a58ddde31e" />

 
