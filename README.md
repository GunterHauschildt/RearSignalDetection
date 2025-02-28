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

