# Image Captioning with a CNN and a From-Scratch LSTM

Takes an image and writes a caption for it, one word at a time. The LSTM is written by hand in NumPy. No framework is doing the sequence modeling.

## What it does

You give it a picture and it gives you a sentence. VGG16 turns the image into a vector of numbers, and an LSTM I wrote from scratch uses that vector as its starting memory and predicts the caption one word at a time until it decides to stop.

## How it works

Image features. VGG16 pretrained on ImageNet with the three fully connected layers cut off and global average pooling on what's left, which gives a 512 number vector per image. The weights are frozen. Features get cached to features.npy so the CNN only has to run once instead of once per epoch.

Captions. Lowercase, strip punctuation and digits, then wrap each caption in startseq and endseq tokens so the model knows where a sentence begins and ends. The vocabulary is built from the words that actually appear in the dataset, and two dictionaries map words to indices and back.

The LSTM. Written from scratch in NumPy, forward pass and backprop both by hand. Four gates per step: forget, input, cell candidate, output. Three use sigmoid because they are dials from 0 to 1, and the candidate uses tanh because it is content and needs to go negative. The image vector is projected down to hidden size and used as the initial hidden state, so the model starts every caption already holding the picture.

Generating. Start with startseq, take the highest scoring word, feed it back in, and stop when the model predicts endseq or hits max length.

## Results

BLEU-1 of 0.5442 and BLEU-4 of 0.1391 on Flickr8k after 10 epochs.

Worth being upfront about: the evaluation scores images that were also used in training, so these are not generalization numbers. A real held-out test split would give lower and more honest ones.

## What I ran into

Writing backprop by hand was the hard part. Getting the gradient math right for four gates, and keeping the matrix shapes straight through all of it, took a lot longer than the forward pass did.

## What I'd do differently

- The backprop only corrects the word it just got wrong. The error never travels back through earlier words in the caption, so the model cannot learn longer phrases. I think that is why BLEU-1 is decent and BLEU-4 falls off. The fix is to run the whole caption forward while saving each step, then walk it backward carrying the gradient, and update once at the end.
- The matrix that projects the image into the hidden state never gets a gradient update, so it stays at its random initialization the whole time. The model can tell images apart but never learns what to focus on.
- No train and test split. Evaluation runs on training images.
- The trained weights are never saved, so the model retrains from scratch every run.
- Generation takes the single highest scoring word at each step. Beam search would keep a few options open and usually scores better.

## Setup

Dataset is Flickr8k from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k

Download it and put the images in data/archive/images.

You need python3 and pip. Then:

    python3 -m venv venv
    source venv/bin/activate
    pip install numpy tensorflow pillow nltk matplotlib

First run: uncomment model.extractFeatures() in main.py, then run python3 main.py. Comment it back out afterward so it does not redo the feature extraction every time.

To change parameters, find the train() call at the bottom of main.py and adjust epochs, learning_rate, embed size, and hidden size.
