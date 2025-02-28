# Lane Detection
Prototype for rear signal detection.
Detects 
- no activity
- left turn
- right turn
- hazard lights.

## Results <a name="results"></a>

## Discussion <a name="discussion">
I attempted to recreate the LSTM/CNN work described in the paper
"Learning to Tell Brake and Turn Signals in Videos Using CNN-LSTM Structure"
(https://ieeexplore.ieee.org/document/8317782)

The dataset used is available: http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal

Using classical computer visions such as Lucas-Kanade and ORB and image subtraction to mask the lights as that paper discussed, I had unsatisfactory results.

Instead my method uses just a CNN wrapped in an LSTM. 
While I struggled to find a model that would converge, I eventually used a simple VGG16 like CNN, with its last layer as a simple Flatten" instead of GlobalMaxPooling.

