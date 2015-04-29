from corpus import Document
from hmm import HMM
class Gene(Document):
    def features(self):
        return self.data
         
if __name__=="__main__":
    hmm=HMM()
    hmm.train([],
            initial_probabilities=[.5, .5], # P(non-coding, Coding)
            transition_probabilities=[[.95, .05],  # P(Hot, Cold|Hot)
                                        [.15, .85]], # P(Hot, Cold|Cold)
            emission_probabilities=[[.4, .1, .1, .4],  # P(1, 2, 3|Hot)
                                    [.2, .3, .3, .2]], # P(1, 2, 3|Cold)
            states=("1", "2"),#noncoding,coding
            vocabulary=('A', 'C', 'G','T'))
    print hmm.likelihood(Gene(['T','G','C','A']))
    print hmm.classify(Gene(['G','C','G','C','A','T','T','A','A','T','C','G','T','C','G','T','C','G','T','A','G','T','T','C','C','T','T']))
    print hmm.classify(Gene(['G','C','G','C','A','T','T','A','A','T','C','G','T','C','G','G','T','C','G','T','A','G','T','T','C','C','T','T']))
