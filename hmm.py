# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier

class HMM():
    """A Hidden Markov Model classifier."""
    def __init__(self):
        self.initProM = []                                  
        self.tranProM = []                               
        self.emiProM = {}                                   
        self.states = []                                    
        self.vocabulary = []                            
        self.vocabularyIndex = {}
        self.statesIndex={}

    def train(self, test, initial_probabilities=[], transition_probabilities=[], emission_probabilities=[], states=[], vocabulary=[]):
        '''maximum likelihood
        '''
        if len(test)==0:
            self.initProM = initial_probabilities
            self.tranProM = transition_probabilities
            self.vocabulary = vocabulary
            for i in range(len(self.vocabulary)):
                self.vocabularyIndex[self.vocabulary[i]] = i
            self.states = states
            for i in range(len(self.states)):
                self.statesIndex[self.states[i]] = i
            self.emiProM = emission_probabilities
        else:
            vocabulary_size=0
            state_size=0
            for sentence in test:
                words=sentence.data #list of words in one sentence
                labels=sentence.label#list of label in one sentence
                for word in words:
                    if word not in self.vocabularyIndex.keys():
                        self.vocabularyIndex[word]=vocabulary_size
                        self.vocabulary.append(word)
                        vocabulary_size+=1
                for label in labels:
                    if label not in self.statesIndex.keys():
                        self.statesIndex[label]=state_size
                        self.states.append(label)
                        state_size+=1

            #in this loop decide the size of the word and the state    
            self.initProM=[0 for i in range(state_size)]
            self.emiProM =[[0 for i in range(vocabulary_size)] for j in range(state_size)]
            self.tranProM =[[0 for i in range(state_size)] for j in range(state_size)]
            count_emission=[0 for i in range(state_size)]
            #initialize the j jtProM
            for sentence in test:
                #Build the a and b matrix
                words=sentence.data
                labels=sentence.label
                last_label_id=None
                
                for word,label in zip(words,labels):
                    label_id=self.statesIndex[label]
                    word_id=self.vocabularyIndex[word]
                    if last_label_id==None: 
                        self.initProM[label_id]+=1# from state start to state label_id
                    else:
                        self.tranProM[last_label_id][label_id]+=1# transition count
                    self.emiProM[label_id][word_id]+=1
                    count_emission[label_id]+=1
                    last_label_id=label_id
            #Get the probability for transition, initial,emission
            d=float(sum(self.initProM[:]))
            for i in range (state_size):
                self.initProM[i]/=d
            # Transfer the emission count into probability
            for i in range(state_size):
                d=float(count_emission[i])
                for j in range(vocabulary_size):
                    self.emiProM[i][j]/=d
            for i in range(state_size):
                d=float(sum(self.tranProM[i][:]))
                for j in range (state_size):
                    self.tranProM[i][j]/=d
                        
                
            


    def likelihood(self, instances):
        '''forward algorthm
        '''
        observe = instances.data
        alpha = [self.initProM[i] * self.emiProM[i][self.vocabularyIndex[observe[0]]] for i in range(len(self.states))]

        for i in range(1, len(observe)):
            tempAlapha = alpha[:]
            for j in range(len(self.states)):
                sum = 0
                for k in range(len(self.states)):
                    sum += tempAlapha[k] * self.tranProM[k][j]
                alpha[j] = sum*self.emiProM[j][self.vocabularyIndex[observe[i]]]
        result = alpha[0]
        for i in range(1, len(self.states)):
            result += alpha[i]
        return result

   
    def classify(self, instances):
        '''viterbi algorthm
        '''
        Path={}                                           
        observe = instances.data
        delta = [[None] * len(self.states) for x in range(len(observe))]
        for i in range(len(self.states)):
            if observe[0] not in self.vocabulary:
                delta[0][i] =self.initProM[i]
            else:  
                delta[0][i] = self.emiProM[i][self.vocabularyIndex[observe[0]]] * self.initProM[i]
            Path['0'+self.states[i]] = [self.states[i]]
      
        index = 0
        for i in range(1, len(observe)):                                                      
            for j in range(len(self.states)):                                                 
                maxPro = 0
                for k in range(len(self.states)):
                    if delta[i-1][k] * self.tranProM[k][j] > maxPro:
                        maxPro = delta[i-1][k] * self.tranProM[k][j]
                        index = k
                        Path[str(i)+self.states[j]] = Path[str(i-1)+self.states[index]]+[self.states[j]]
                if observe[i] not in self.vocabulary:
                    delta[i][j] = maxPro
                else:
                    delta[i][j] = maxPro * self.emiProM[j][self.vocabularyIndex[observe[i]]]
        index = 0
        maxPro = delta[len(observe)-1][0]
        for i in range(len(self.states)):
            if delta[len(observe)-1][i] > maxPro:
                index = i
                maxPro = delta[len(observe)-1][i]
        return Path[str(len(observe)-1)+self.states[index]]


