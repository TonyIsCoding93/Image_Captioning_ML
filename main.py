import numpy as np
import os, re
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class LSTM:
    

    def __init__(self, vocab_size, embed_size, hidden_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        #forget gate
        self.weight_forget_gate = np.random.randn(hidden_size, hidden_size + embed_size) *.01
        self.bias_forget_gate = np.zeros((hidden_size, 1))
        #input gate
        self.weight_input = np.random.randn(hidden_size, hidden_size + embed_size) * .01
        self.bias_input = np.zeros((hidden_size, 1))
        #cell gate
        self.weight_cell = np.random.randn(hidden_size, hidden_size + embed_size) * .01
        self.bias_cell = np.zeros((hidden_size, 1))
        #output gate
        self.weight_output = np.random.randn(hidden_size, hidden_size + embed_size) * .01
        self.bias_output = np.zeros((hidden_size, 1))
        #embed matrix 
        self.embedding = np.random.randn(vocab_size, embed_size) * .01
        #output layer
        self.weight_vocab = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bias_vocab = np.zeros((vocab_size, 1))
    #squish each number between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    #create probabilites of each array
    def to_probabilities(self, x):
        exp_x = np.exp(x-np.max(x))
        return exp_x / exp_x.sum(axis=0)
    
    def forward_pass(self, x, hidden_state, cell_state):
        #grab row from embed matrix and make it a column
        word_embed = self.embedding[x].reshape(-1, 1)
        #combing hidden state and word embedding into 1 long vertical vector
        combined = np.vstack((hidden_state, word_embed))
        #decide on what to forget from long term memory
        forget = self.sigmoid(self.weight_forget_gate @ combined + self.bias_forget_gate)
        #decide how much we should save into long term memory
        input_gate = self.sigmoid(self.weight_input @ combined + self.bias_input)
        #create new information that could be stored into long term memory
        cell_update = np.tanh(self.weight_cell @ combined + self.bias_cell)
        #determine what long term mem is relevent 
        output_gate = self.sigmoid(self.weight_output @ combined + self.bias_output)
        #erase and add 
        cell_state = forget * cell_state + input_gate * cell_update
        #put the cell state through tanh to get -1 to 1 output, filter exposure
        hidden_state = output_gate * np.tanh(cell_state)
        #prediciton, get raw scores for each word then turn them into percentages. highest prob wins!
        output = self.weight_vocab @ hidden_state + self.bias_vocab
        probabilities = self.to_probabilities(output)

        return probabilities, hidden_state, cell_state

class ImageCaptioningModel:

    def __init__(self, captions_path, images_path):
        self.captions_path = captions_path
        self.images_path = images_path
        self.captions = {}
        self.features = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
    # function that loads our image files and captions into a dictionary
    def loadCaptions(self):

        hashmap = {}
        #"data/archive/captions.txt"
        with open(self.captions_path, "r") as file:
            lines = file.readlines()
            
            for line in lines[1:]:
                seperatedLine = line.strip().split(",", 1)
                if seperatedLine[0] not in hashmap:
                    hashmap[seperatedLine[0]] = [seperatedLine[1]]
                else:
                    hashmap[seperatedLine[0]].append(seperatedLine[1])
        self.captions = hashmap

    def extractFeatures(self):
        #using pre-trained weights from imagenet and taking off fully connected layers
        #to allow our to make classifications 
        baseModel = VGG16(weights='imagenet', include_top=False, pooling='avg')
        #we dont want to train this model. we just want to extract the features and numbers
        #assocciated with them
        baseModel.trainable = False
        for filename in os.listdir(self.images_path):
            #create full path so when we load the image it knows exactly where to get it from
            full_path = os.path.join(self.images_path, filename)
            #resize image to 224x224 which is what KGG16 works with
            image = load_img(full_path, target_size=(224, 224))
            #using keras, convert to numPy array
            image = img_to_array(image)
            #VGG16 is meant to process images by batches so here we add a batch number for processing
            image = np.expand_dims(image, axis=0)
            #normalize pixels so we have proper feature vectors
            image = preprocess_input(image)
            self.features[filename] = baseModel.predict(image, verbose=0) ##verbose to cancel progress bar

        np.save('features.npy', self.features) #saving our vector numbers to the disk so we dont 
        #have to run VGG16 everytime our program runs. 
    def preProcessCaptions(self):
        for file, captions in self.captions.items():
            cleaned= []
            for caption in captions:
                caption = caption.lower()
                caption = re.sub("[^a-z ]", "", caption)
                caption = caption.strip()
                caption = "startseq " + caption + " endseq"
                cleaned.append(caption)
            self.captions[file] = cleaned

    def buildVocabulary(self):
        vocab = set()
        for filename, captions in self.captions.items():
            for caption in captions:
                for word in caption.split():
                    vocab.add(word)
        for index, word in enumerate(vocab, start=1):
            self.word_to_index[word] = index    #going into the LSTM
            self.index_to_word[index] = word    #coming out of the LSTM
        self.vocab_size = len(vocab) + 1

    











model = ImageCaptioningModel("data/archive/captions.txt", "data/archive/Images/")
model.loadCaptions()
model.preProcessCaptions()
model.buildVocabulary()

print(f"Vocabulary size: {model.vocab_size}")
print(f"Sample mappings:")
print(f"  'dog' → {model.word_to_index.get('dog')}")
print(f"  'startseq' → {model.word_to_index.get('startseq')}")
print(f"  'endseq' → {model.word_to_index.get('endseq')}")
print(f"  Index 1 → {model.index_to_word.get(1)}")


lstm = LSTM(vocab_size=model.vocab_size, embed_size=256, hidden_size=256)

hidden_state = np.zeros((256, 1))
cell_state = np.zeros((256, 1))

word_index = model.word_to_index["startseq"]
probs, hidden_state, cell_state = lstm.forward_pass(word_index, hidden_state, cell_state)

predicted_index = np.argmax(probs)
predicted_word = model.index_to_word.get(predicted_index, "unknown")
print(f"Input: 'startseq'")
print(f"Predicted next word: '{predicted_word}'")
print(f"Probability: {probs[predicted_index][0]:.4f}")
print(f"Output shape: {probs.shape}")

