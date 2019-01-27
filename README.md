AHAB
=======

Trying to get a good score using Deep Learninf at this challenge : 
https://codegolf.stackexchange.com/questions/152856/write-moby-dick-approximately

# Model

Using a shallow LSTM network : 
- 3 LSTM Cells of 16 units, with a timestep of 20
- 1 fully connected layer with softmax for the classification (82 units, the number of classes)

# Colab

The ipynb file is supposed to run on Google Colab
