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



        #creating a matrix of random numbers that will be multiplied by the input vector 
        # of the vocab word in the embed matrix + whatever is in the hidden state matrix,
        #this gate in particular is designed to forget words in our long term memory that isnt useful anymore
        self.weight_forget_gate = np.random.randn(hidden_size, hidden_size + embed_size) *.01 #we are multiplying by .01 here for the reason that we need numbers that will feed into our 
        self.bias_forget_gate = np.zeros((hidden_size, 1))                                    #sigmoid and tanh functions properly for training. 
        #same as above but this is for the input. this is deciding if 
        #what new information to put into memory
        self.weight_input = np.random.randn(hidden_size, hidden_size + embed_size) * .01 
        self.bias_input = np.zeros((hidden_size, 1))
        #also same as above but instead this is the actual cell of information that will be stored in memory
        self.weight_cell = np.random.randn(hidden_size, hidden_size + embed_size) * .01
        self.bias_cell = np.zeros((hidden_size, 1))#we create a bias vector as a baseline for when there is no input
        #the output gate isnt our output layer per se. its the part of memory thats relevent to our next predicted word
        self.weight_output = np.random.randn(hidden_size, hidden_size + embed_size) * .01
        self.bias_output = np.zeros((hidden_size, 1))
        #our embed matrix is a matrix that holds the values of each word. each word will have 256 values associated with it. 
        self.embedding = np.random.randn(vocab_size, embed_size) * .01
        #our output layer matrix is going to consist of every word in our vocabulary as rows and
        # our hidden layer as columns. this multiplied by a the current hidden vector will 
        # output a score per word giving us the probability we can choose from
        self.weight_vocab = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bias_vocab = np.zeros((vocab_size, 1))


    #squish each number between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    #takes our output layer and converts our numbers to probabilities which allows us to choose the best fit
    def to_probabilities(self, x):
        exp_x = np.exp(x-np.max(x))
        return exp_x / exp_x.sum(axis=0)
    

    
    def forward_pass(self, x, hidden_state, cell_state):


        #these are created to store the states information pre-pass to reference while doing our back propgation
        self.last_prev_hidden = hidden_state.copy()
        self.last_prev_cell = cell_state.copy()
        self.last_word_index = x


        #make our embed row a column to make it a vector to multiply against weights
        word_embed = self.embedding[x].reshape(-1, 1)
        #combing hidden state and word embedding into 1 long vertical vector to multiply against our gates
        combined = np.vstack((hidden_state, word_embed))
        # here we are turning doing the math of multiplying our gates with our combined vectors. Then making them values of 0-1.
        # Except our cell_update as we could be removing or adding information so we use a -1 to 1 range
        forget = self.sigmoid(self.weight_forget_gate @ combined + self.bias_forget_gate)
        input_gate = self.sigmoid(self.weight_input @ combined + self.bias_input)
        cell_update = np.tanh(self.weight_cell @ combined + self.bias_cell)
        output_gate = self.sigmoid(self.weight_output @ combined + self.bias_output)
        #cell_state is an update that includes information of what should be forgotten and what should be added to the relevant information
        cell_state = forget * cell_state + input_gate * cell_update
        #put the cell state through tanh to get -1 to 1 and mutiply it by our output gate which will give us the hidden state which is the most relevant words in our cell state
        hidden_state = output_gate * np.tanh(cell_state)
        #prediciton, get raw scores for each word then turn them into percentages. highest prob wins!
        output = self.weight_vocab @ hidden_state + self.bias_vocab
        probabilities = self.to_probabilities(output)



        #save what we just did for our backprop.
        self.last_combined = combined
        self.last_forget = forget
        self.last_input = input_gate
        self.last_cell_update = cell_update
        self.last_output = output_gate
        self.last_cell_state = cell_state
        self.last_hidden_state = hidden_state
        self.last_probabilities = probabilities
        

        return probabilities, hidden_state, cell_state
    
    def back_prop(self, target_index, learning_rate = 0.001):
        #copy the probabilities we just got and subtract our chosen word by 1. we can then see how far off we were for every word.
        #the math here makes it very simple by seeing that the chosen word is too low while everything else is too high
        error = self.last_probabilities.copy()
        error[target_index] -= 1

        #this is the main chunk for the back prop. we are finding why each gate gave us a bad result with our probability
        #to do this we are finding the gradient adjustment at each gate throughout our process and we are finding out which
        #one cuased the most issues
        grad_weight_vocab = error @ self.last_hidden_state.T
        grad_bias_vocab = error
        grad_hidden = self.weight_vocab.T @ error
        tanh_cell = np.tanh(self.last_cell_state)
        grad_output_gate = grad_hidden * tanh_cell * self.last_output * (1 - self.last_output)
        grad_cell = grad_hidden * self.last_output * (1 - tanh_cell ** 2)
        grad_forget = grad_cell * self.last_prev_cell * self.last_forget * (1 - self.last_forget)
        grad_input = grad_cell * self.last_cell_update * self.last_input * (1 - self.last_input)
        grad_cell_update = grad_cell * self.last_input * (1 - self.last_cell_update ** 2) #

        #now we are correcting the weights since we now know how wrong each gate was. we use .T here for easy matrix multiplication!!
        grad_weight_forget = grad_forget @ self.last_combined.T
        grad_bias_forget = grad_forget

        grad_weight_input = grad_input @ self.last_combined.T
        grad_bias_input = grad_input

        grad_weight_cell = grad_cell_update @ self.last_combined.T
        grad_bias_cell = grad_cell_update

        grad_weight_output = grad_output_gate @ self.last_combined.T
        grad_bias_output = grad_output_gate

        # we are fixing our weights here in our long combined vector and using a splice to fix our embedded vector only.
        grad_combined = (self.weight_forget_gate.T @ grad_forget +
                        self.weight_input.T @ grad_input +
                        self.weight_cell.T @ grad_cell_update +
                        self.weight_output.T @ grad_output_gate)
        grad_embed = grad_combined[self.hidden_size:]

        #update every single weight. nudge each one based on its gradient
        self.weight_vocab -= learning_rate * grad_weight_vocab
        self.bias_vocab -= learning_rate * grad_bias_vocab
        self.weight_forget_gate -= learning_rate * grad_weight_forget
        self.bias_forget_gate -= learning_rate * grad_bias_forget
        self.weight_input -= learning_rate * grad_weight_input
        self.bias_input -= learning_rate * grad_bias_input
        self.weight_cell -= learning_rate * grad_weight_cell
        self.bias_cell -= learning_rate * grad_bias_cell
        self.weight_output -= learning_rate * grad_weight_output
        self.bias_output -= learning_rate * grad_bias_output
        self.embedding[self.last_word_index] -= learning_rate * grad_embed.flatten()

        #calculate the loss so we can track if training is getting better
        loss = -np.log(self.last_probabilities[target_index] + 1e-8)
        return loss[0]

class ImageCaptioningModel:

    def __init__(self, captions_path, images_path):
        self.captions_path = captions_path
        self.images_path = images_path
        self.captions = {}
        self.features = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
    # preproccessing. this function is to strip and save each caption to an image by way of hashmap 
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
        # to get the feature vectors 
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

        #this is where we are cleaning the captions. we are making them all lowercase and removing anything other then alphabetical letters. 
        # we are also adding start and end to our captions so our model knows when to start and stop
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
     # this is where we simply build a vocabulary set of all our words. more importantly we are creating two different hashmaps
     # that are made for the purpose of calling upon a word using a number that is decided by the lstm or is used during training. 
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





    # for each image we are using our foward pass and back prop to adjust our 
    # weights of our original captions and images to get our weights correct for our model
    def train(self, epochs=10, learning_rate=0.001):
        #load our saved features from disk
        features = np.load('features.npy', allow_pickle=True).item()

        # LSTM creation. Our baby
        lstm = LSTM(self.vocab_size, embed_size=256, hidden_size=256)

        # The amount of epochs we are running through. an epoch is a single pass through our entire dataset. this can be adjusted. 
        for epoch in range(epochs):
            total_loss = 0
            num_words = 0

            for filename, captions in self.captions.items():
                


                if filename not in features:
                    continue

                # feature vector
                image_feature = features[filename].reshape(-1, 1)



                for caption in captions:
                    words = caption.split()

                    #. start with blank memory for each caption
                    hidden_state = np.zeros((256, 1))
                    cell_state = np.zeros((256, 1))



                    #go through each word in the caption
                    for i in range(len(words) - 1):
                        #current word is the input
                        input_word = self.word_to_index[words[i]]
                        #next word is what we want to predict
                        target_word = self.word_to_index[words[i + 1]]

                        # foward pass follow by our backprop. what this does is stated in our LSTM class
                        probs, hidden_state, cell_state = lstm.forward_pass(input_word, hidden_state, cell_state)
                        loss = lstm.back_prop(target_word, learning_rate)
                        total_loss += loss
                        num_words += 1
                        #aded this because i spent way too much time figuring out why nothing wasnt working... just needed a print statement
                        #every so often words, 10000 worked great
                        if num_words % 10000 == 0:
                            print(num_words, "words done -", "loss:", round(total_loss / num_words, 4))

            # this was perfect for an update on how training is progressing
            avg_loss = total_loss / num_words
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        #done
        self.lstm = lstm
        print("Training complete!")


    #this is the end result. this is what we spent weeks on, this is what we worked so hard for, this is it. an image caption generator ill 
    #probably never use in the future but ill take everything that i learned to try and become an AI/ML engineer.
    def generateCaption(self, filename, max_length=20):
        #load features
        features = np.load('features.npy', allow_pickle=True).item()
        
        #start with blank memory
        hidden_state = np.zeros((256, 1))
        cell_state = np.zeros((256, 1))
        current_word = "startseq"
        caption = []
        
        for i in range(max_length):
       
            word_index = self.word_to_index.get(current_word)
            if word_index is None:
                break
            #forward pass
            probs, hidden_state, cell_state = self.lstm.forward_pass(word_index, hidden_state, cell_state)
            
            #quick note: no back prop here as we are only predicting next word. back prop is used for the training


            #pick the word with highest probability
            predicted_index = np.argmax(probs)
            predicted_word = self.index_to_word.get(predicted_index, "unknown")
            
            if predicted_word == "endseq":
                break
            
            caption.append(predicted_word)
            current_word = predicted_word
        
        return " ".join(caption)

    def evaluate(self, num_images=100):


        from nltk.translate.bleu_score import corpus_bleu
        


        features = np.load('features.npy', allow_pickle=True).item()
        

        references = []
        predictions = []
        count = 0
        
        for filename, captions in self.captions.items():
            count += 1
            if count > num_images:
                break
                
            if filename not in features:
                continue
            
            #generate a caption for this image
            generated = self.generateCaption(filename)
            
            #the reference captions need to be lists of words
            #each image has 5 reference captions to compare against
            ref = [caption.split() for caption in captions]
            
            #remove startseq and endseq from references
            ref = [[w for w in r if w not in ["startseq", "endseq"]] for r in ref]
            
            references.append(ref)
            predictions.append(generated.split())
        
        #calculate BLEU scores
        bleu1 = corpus_bleu(references, predictions, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references, predictions, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))
        
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-2: {bleu2:.4f}")
        print(f"BLEU-3: {bleu3:.4f}")
        print(f"BLEU-4: {bleu4:.4f}")
        
        return bleu1, bleu2, bleu3, bleu4


model = ImageCaptioningModel("data/archive/captions.txt", "data/archive/Images/")
model.loadCaptions()
model.preProcessCaptions()
model.buildVocabulary()
print(f"Vocabulary size: {model.vocab_size}")
model.train(epochs=10, learning_rate=0.001)
caption = model.generateCaption("1000268201_693b08cb0e.jpg")
print(f"Generated caption: {caption}")
model.evaluate(num_images=100)