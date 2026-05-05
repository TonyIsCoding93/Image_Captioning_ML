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



        #forget gate. random matrix that gets multiplied by our input vector
        #basically decides what info isnt useful anymore in memory
        self.weight_forget_gate = np.random.randn(hidden_size, hidden_size + embed_size) *.01 #we are multiplying by .01 here for the reason that we need numbers that will feed into our 
        self.bias_forget_gate = np.zeros((hidden_size, 1))                                    #sigmoid and tanh functions properly for training. 
        #same as above but this is for the input. this is deciding
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

        #converts our image vector into a hidden state size so we can use it as our starting memory
        self.image_to_hidden = np.random.randn(hidden_size, 512) * 0.01




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
        wordVector = self.embedding[x].reshape(-1, 1)

        #combing hidden state and word embedding into 1 long vertical vector to multiply against our gates
        mergedInput = np.vstack((hidden_state, wordVector))
        # here we are turning doing the math of multiplying our gates with our combined vectors. Then making them values of 0-1.
        # Except our cell_update as we could be removing or adding information so we use a -1 to 1 range
        forget_result = self.sigmoid(self.weight_forget_gate @ mergedInput + self.bias_forget_gate)
        inputResult = self.sigmoid(self.weight_input @ mergedInput + self.bias_input)
        cell_newInfo = np.tanh(self.weight_cell @ mergedInput + self.bias_cell)
        outputResult = self.sigmoid(self.weight_output @ mergedInput + self.bias_output)

        #cell_state is an update that includes information of what should be forgotten and what should be added to the relevant information
        cell_state = forget_result * cell_state + inputResult * cell_newInfo
        #put the cell state through tanh to get -1 to 1 and mutiply it by our output gate which will give us the hidden state which is the most relevant words in our cell state
        hidden_state = outputResult * np.tanh(cell_state)

        #prediciton, get raw scores for each word then turn them into percentages. highest prob wins!
        rawScores = self.weight_vocab @ hidden_state + self.bias_vocab
        probabilities = self.to_probabilities(rawScores)



        #save what we just did for our backprop.
        self.last_mergedInput = mergedInput
        self.last_forget_result = forget_result
        self.last_inputResult = inputResult
        self.last_cell_newInfo = cell_newInfo
        self.last_outputResult = outputResult
        self.last_cell_state = cell_state
        self.last_hidden_state = hidden_state
        self.last_probabilities = probabilities
        # make sure done properly ----delete 

        return probabilities, hidden_state, cell_state
    






    def back_prop(self, target_index, learning_rate = 0.001):
        #subtract 1 from the right answer so we can see how off we were
        error = self.last_probabilities.copy()
        error[target_index] -= 1


        #finding how wrong each gate was so we can fix the weights
        errorVocabWeights = error @ self.last_hidden_state.T
        errorVocab_bias = error

        error_hidden = self.weight_vocab.T @ error
        tanhOfCell = np.tanh(self.last_cell_state)

        errorOutputGate = error_hidden * tanhOfCell * self.last_outputResult * (1 - self.last_outputResult)
        error_cell = error_hidden * self.last_outputResult * (1 - tanhOfCell ** 2)
        errorForgetGate = error_cell * self.last_prev_cell * self.last_forget_result * (1 - self.last_forget_result)
        error_inputGate = error_cell * self.last_cell_newInfo * self.last_inputResult * (1 - self.last_inputResult)
        errorCellInfo = error_cell * self.last_inputResult * (1 - self.last_cell_newInfo ** 2)

        #now we are correcting the weights since we now know how wrong each gate was. we use .T here for easy matrix multiplication!!
        fixForget_weights = errorForgetGate @ self.last_mergedInput.T
        fixForget_bias = errorForgetGate



        fix_inputWeights = error_inputGate @ self.last_mergedInput.T
        fix_inputBias = error_inputGate
        fixCellWeights = errorCellInfo @ self.last_mergedInput.T
        fixCell_bias = errorCellInfo

        fix_outputWeights = errorOutputGate @ self.last_mergedInput.T 
        fixOutputBias = errorOutputGate


        # fixing our weights in our long merged vector and using a splice to fix our embedded vector only
        error_merged = (self.weight_forget_gate.T @ errorForgetGate +
                        self.weight_input.T @ error_inputGate +
                        self.weight_cell.T @ errorCellInfo +
                        self.weight_output.T @ errorOutputGate)
        errorEmbedding = error_merged[self.hidden_size:]

        #update every single weight. nudge each one based on its gradient
        self.weight_vocab -= learning_rate * errorVocabWeights
        self.bias_vocab -= learning_rate * errorVocab_bias
        self.weight_forget_gate -= learning_rate * fixForget_weights
        self.bias_forget_gate -= learning_rate * fixForget_bias
        self.weight_input -= learning_rate * fix_inputWeights
        self.bias_input -= learning_rate * fix_inputBias
        self.weight_cell -= learning_rate * fixCellWeights
        self.bias_cell -= learning_rate * fixCell_bias
        self.weight_output -= learning_rate * fix_outputWeights
        self.bias_output -= learning_rate * fixOutputBias
        self.embedding[self.last_word_index] -= learning_rate * errorEmbedding.flatten()


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
            runningLoss = 0
            wordCount = 0

            for filename, captions in self.captions.items():

                if filename not in features:
                    continue


                # feature vector
                imageVector = features[filename].reshape(-1, 1)

                for caption in captions:
                    words = caption.split()

                    #use image features as starting point for the hidden state so the lstm knows whats in the picture
                    hidden = lstm.image_to_hidden @ imageVector
                    memory = np.zeros((256, 1))


                    #go through each word in the caption
                    for i in range(len(words) - 1):
                        #current word is the input
                        currentWord = self.word_to_index[words[i]]
                        #next word is what we want to predict
                        answer = self.word_to_index[words[i + 1]]

                        # foward pass follow by our backprop. what this does is stated in our LSTM class
                        probs, hidden, memory = lstm.forward_pass(currentWord, hidden, memory)
                        loss = lstm.back_prop(answer, learning_rate)

                        runningLoss += loss
                        wordCount += 1

                        #aded this because i spent way too much time figuring out why nothing wasnt working... just needed a print statement
                        #every so often words, 10000 worked great
                        if wordCount % 10000 == 0:
                            print(wordCount, "words done -", "loss:", round(runningLoss / wordCount, 4))

            # this was perfect for an update on how training is progressing
            epochLoss = runningLoss / wordCount
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epochLoss:.4f}")

        #done
        self.lstm = lstm
        print("Training complete!")








    #this is the end result. this is what we spent weeks on, this is what we worked so hard for, this is it. an image caption generator ill 
    #probably never use in the future but ill take everything that i learned to try and become an AI/ML engineer.
    def generateCaption(self, filename, max_length=20):
        #load features
        features = np.load('features.npy', allow_pickle=True).item()
        
        #use image features as starting hidden state
        imageVector = features[filename].reshape(-1, 1)
        hidden = self.lstm.image_to_hidden @ imageVector
        memory = np.zeros((256, 1))

        currentWord = "startseq"
        result = []
        
        for i in range(max_length):

            wordIdx = self.word_to_index.get(currentWord)
            if wordIdx is None:
                break

            #forward pass
            probs, hidden, memory = self.lstm.forward_pass(wordIdx, hidden, memory)
            
            #quick note: no back prop here as we are only predicting next word. back prop is used for the training

            #pick the word with highest probability
            bestPick = np.argmax(probs)
            predictedWord = self.index_to_word.get(bestPick, "unknown")
            
            if predictedWord == "endseq":
                break

            result.append(predictedWord)
            currentWord = predictedWord
        
        return " ".join(result)








    #bleu scores. very boilerplate style code but its standard for image captioning. we use it to compare our outputed captions to
    #the correct captions and score



    def evaluate(self, num_images=100):

        from nltk.translate.bleu_score import corpus_bleu

        features = np.load('features.npy', allow_pickle=True).item()
        
        realCaptions = []
        ourCaptions = []
        imgCount = 0
        



        for filename, captions in self.captions.items():
            imgCount += 1


            if imgCount > num_images:
                break
                
            if filename not in features:
                continue

            generatedCaption = self.generateCaption(filename)


            #   need to split the real captions into words and get rid of our start and stop tokens
            refWords = []
            for cap in captions:
                words = cap.split()
                cleaned = []
                for w in words:
                    if w != "startseq" and w != "endseq":
                        cleaned.append(w)
                refWords.append(cleaned)
            
            realCaptions.append(refWords)
            ourCaptions.append(generatedCaption.split())
        

        #these weights are standard for bleu scoring
        score1 = corpus_bleu(realCaptions, ourCaptions, weights=(1, 0, 0, 0))
        score2 = corpus_bleu(realCaptions, ourCaptions, weights=(0.5, 0.5, 0, 0))
        score3 = corpus_bleu(realCaptions, ourCaptions, weights=(0.33, 0.33, 0.33, 0))
        score4 = corpus_bleu(realCaptions, ourCaptions, weights=(0.25, 0.25, 0.25, 0.25))
        


        print(f"BLEU-1: {score1:.4f}")

        print(f"BLEU-2: {score2:.4f}")
        print(f"BLEU-3: {score3:.4f}")
        print(f"BLEU-4: {score4:.4f}")

        return score1, score2, score3, score4





















model = ImageCaptioningModel("data/archive/captions.txt", "data/archive/Images/")
model.loadCaptions()
model.preProcessCaptions()
model.buildVocabulary()
print(f"Vocabulary size: {model.vocab_size}")
model.train(epochs=10, learning_rate=0.001)
caption1 = model.generateCaption("1000268201_693b08cb0e.jpg")
caption2 = model.generateCaption("1001773457_577c3a7d70.jpg")
caption3 = model.generateCaption("1002674143_1b742ab4b8.jpg")
print(f"Caption 1: {caption1}")
print(f"Caption 2: {caption2}")
print(f"Caption 3: {caption3}")
model.evaluate(num_images=100)