


Anthony Dye - awd240000

Image Captioning Model

This Project is an image captioning model that uses a Convolutional Nueral Network and 
LSTM or long-short-term-memory architecture to predict next word tokens for 
generating captions for images. We use VGG as our convolutional neural network by 
taking of its classification layers to produce image vectors that provide us with a 
numbered summary of each image. We then clean, and process captions and save them to 
our unique vocabulary. Our LSTM is built from scratch using numPY.

Dataset - Flickr8k from Kaggle. https://www.kaggle.com/datasets/adityajn105/flickr8k

To use this program it would be easiest to downlaod the the dataset and put the contents inside data/archive -> images and dump all the files into that

python3 and pip need to be installed

run in terminal:

python3 -m venv venv
source venv/bin/activate
pip install numpy tensorflow pillow nltk matplotlib


1st time run: uncomment our model.extractFeatures() in main.py and run it in terminal: python3 main.py
***after running it comment model.extractFeatures() out again so its not done everytime.


To change parameters: find def train() at the bottom of main.py and change the epochs(amount of time ran through dataset),
learning_rate(weight change rate), embed size, and hidden size. 

