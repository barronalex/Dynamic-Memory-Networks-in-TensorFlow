""" from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/ """


import os as os
import numpy as np

# can be sentence or word
input_mask_mode = "sentence"

def init_babi(fname):
    print "==> Loading test from %s" % fname
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "S": ""} 
            counter = 0
            id_map = {}
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            task["C"] += line
            id_map[id] = counter
            counter += 1
            
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = []
            for num in tmp[2].split():
                task["S"].append(id_map[int(num.strip())])
            tasks.append(task.copy())

    return tasks


def get_babi_raw(id, test_id):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    if (test_id == ""):
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_train.txt' % babi_name))
    babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_test.txt' % babi_test_name))
    return babi_train_raw, babi_test_raw

            
def load_glove(dim):
    word2vec = {}
    
    print "==> loading glove"
    with open(("./data/glove/glove.6B/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print "==> glove is loaded"
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector

def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        if next_index == 97:
            print word
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")

# from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def process_input(data_raw, floatX, self):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    relevant_labels = []
    for x in data_raw:
        inp = x["C"].lower().split(' ') 
        inp = [w for w in inp if len(w) > 0]
        q = x["Q"].lower().split(' ')
        q = [w for w in q if len(w) > 0]

        
        inp_vector = [process_word(word = w, 
                                    word2vec = self.word2vec, 
                                    vocab = self.vocab, 
                                    ivocab = self.ivocab, 
                                    word_vector_size = self.config.embed_size, 
                                    to_return = "index") for w in inp]
                                    
        q_vector = [process_word(word = w, 
                                    word2vec = self.word2vec, 
                                    vocab = self.vocab, 
                                    ivocab = self.ivocab, 
                                    word_vector_size = self.config.embed_size, 
                                    to_return = "index") for w in q]
        
        inputs.append(np.vstack(inp_vector).astype(floatX))
        questions.append(np.vstack(q_vector).astype(floatX))
        answers.append(process_word(word = x["A"], 
                                        word2vec = self.word2vec, 
                                        vocab = self.vocab, 
                                        ivocab = self.ivocab, 
                                        word_vector_size = self.config.embed_size, 
                                        to_return = "index"))
        # NOTE: here we assume the answer is one word! 
        if input_mask_mode == 'word':
            input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
        elif input_mask_mode == 'sentence': 
            input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
        else:
            raise Exception("invalid input_mask_mode")

        relevant_labels.append(x["S"])
    
    return inputs, questions, answers, input_masks, relevant_labels 

def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding
        
