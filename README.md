# Lane Detection
Prototype for rear signal detection system.    
This model detects   
- no activity  
- left turn  
- right turn  
- hazard lights  

## Results <a name="results"></a>

https://github.com/user-attachments/assets/147eabe2-7884-447d-afca-3849f4a53990  

https://github.com/user-attachments/assets/3919735e-b428-4152-b44c-c9e08e740c51  

https://github.com/user-attachments/assets/b3c941e0-24fa-48be-9d0d-3fccb723f67d  

https://github.com/user-attachments/assets/82cd37bd-abb5-4a89-b755-6e083eeadfac  


## Discussion <a name="discussion">
I attempted to recreate the LSTM/CNN work described in the paper "Learning to Tell Brake and Turn Signals in Videos Using CNN-LSTM Structure" (https://ieeexplore.ieee.org/document/8317782). The dataset used is available: http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal.

Using classical computer visions such as Lucas-Kanade and ORB and image subtraction to mask the lights as that paper discussed, I had unsatisfactory results. Instead my method uses just a CNN wrapped in an LSTM. While I struggled to find a model that would converge, I eventually used a simple VGG16 like CNN, untrained, with its last layer as a simple Flatten" (instead of GlobalMaxPooling).

## 4D Augmentation
Part of this project was to use existing 3D torchvision image augmentation on a 4D image sequence tensor.
This repo uses the idea that an RGB image is of the shape (3, rows, cols) and therefore a 2x2 image would look like:
[[[100, 200, 150], [101, 202, 153]],
 [[101, 204, 151], [101, 202, 154]]]
With the RGB channel values augmented indepentently.

A 4D sequence of 2 2x2 images would like:
[[[[100, 200, 150], [101, 202, 153]],
  [[101, 204, 151], [101, 202, 154]]]],
 [[[105, 205, 155], [106, 207, 158]],
  [[106, 209, 156], [106, 207, 159]]]]]
This 4D sequence of 2x3x2x2 gets squashed to 6x2x2 or [R0 G0 B0 R1 G1 B0] x 2x2 and the augmentation works as expected.
For the example above:
[[[100, 200, 150, 105, 205, 155], [101, 202, 153, 106, 207, 158]],
 [[101, 204, 151, 106, 209, 156], [101, 202, 154, 106, 207, 159]]]
We can then augment and unsquash back to 2x3x2x2.






