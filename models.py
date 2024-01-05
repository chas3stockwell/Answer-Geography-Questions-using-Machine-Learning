import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List
import torch.optim as optim

def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap with Jaccard similarity
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # Note that this is a list of a single Derivation
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs

class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size, output_size, embedding_dropout=0.2, bidirect=True):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        
        self.input_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = RNNEncoder(emb_dim, hidden_size, bidirect)
        self.decoder = RNNDecoder(emb_dim, hidden_size, output_size)
        self.output_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        

    def forward(self, x_tensor, inp_lens_tensor, y_tensor):
        
        criterion = torch.nn.NLLLoss(reduction = 'mean', ignore_index = 0) #define your sum
        
        
        enc_output_each_word, enc_context_mask, enc_final_states_reshaped = self.encode_input( x_tensor, inp_lens_tensor )
        #enc_final_states_reshaped: tuple, with [1,1,hid_size] each 
        
        inputs = torch.tensor([1]).long() #[1], value is 1. 
        total_loss = 0
        
        for word_pos in range( len(y_tensor[0]) ): #use an EOS token to end. Add padd tokens. 
            
            emb_inputs = self.output_emb(inputs) #[1, embed_size]
            outputs, hidden = self.decoder.forward(emb_inputs, enc_final_states_reshaped, False, enc_output_each_word)
            #outputs = [1, output_size] [1,153]
            #hidden = tuple, [1,1,hid_size] each
            
            target = torch.unsqueeze(y_tensor[0][word_pos], 0).long() #[1], value is actual target. 
            
            
            loss = criterion(outputs, target) #Output [1, 153], Target: [1], loss: [], loss is value of the loss. Ranges from 0.002 to 4
            
            inputs = target #input: [1], target: [1]
            enc_final_states_reshaped = hidden #tuple, with [1,1,hid_size] replaced by [1,1,hid_size]
            
            total_loss += loss #Total Loss: [], with value from all loss
        
        return total_loss #loses about 20-90 per word. 
        
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len] vector of indices or a batched input/output
        [batch size x sent len]. y_tensor contains the gold sequence(s) used for training
        :param inp_lens_tensor/out_lens_tensor: either a vector of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        
        #Encode each example
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, reverse_input=False) 
        #massaged data. 
            
            
        deriv_list = []
        for pos, val in enumerate(all_test_input_data): #iterate through example len 120
            
            val = [x for x in val if x != 0] #this gets rid of all padded 0's
            #val = np.insert(val, 0, 1) #add a <SOS> token to the val
            
            x_tensor = torch.unsqueeze( torch.FloatTensor(val).long(), 0) #[1, actual sent length] OR [1,18] on dev [1,24] on blind. 
            inp_lens_tensor = torch.FloatTensor( [len(val)] ).long() #[1], value is actual sent length  
            enc_output_each_word, enc_context_mask, enc_final_states_reshaped = self.encode_input( x_tensor, inp_lens_tensor )
            #enc_final_states_reshaped: [1,1,hid_size]
            
            
            arr = [] #sent arr of tokens.Returns one less than the sentence length back (since we do not code the <EOS> token. 
            word_pos = 0
            curr_tok = 0
            max_itrs = 100
            
            counter = 0
            
            #it's one of these two, I don't really know
            #inputs = x_tensor[0][0] #introductory guess. 
            
            inputs = torch.ones(1).long()
            inputs = inputs[0]
            
            while (inputs != 2) and (counter <= max_itrs): 
                
            #while (word_pos < len(x_tensor[0]) ): #just iterates through the sentence without 0's
            #while (x_tensor[0][word_pos] != 8): #8 is the ) token, just through an output_indexer. 
                
                inputs = torch.unsqueeze( inputs, 0) #[1], value is the value of the word token. 
                emb_inputs = torch.unsqueeze( self.output_emb( inputs  ), 0) #[1, 1, emb_dim]
                
                outputs, hidden = self.decoder.rnn(emb_inputs, enc_final_states_reshaped) #outputs: [1,1,hid_size] hidden: [1,1,hid_size]
                classification = self.decoder.classification(outputs) #[1,1,output_size]
                log_probs = self.decoder.log_softmax(classification)[0][0] #[output_size]. Without the indices [1,1,output_size]
                curr_tok = torch.argmax(log_probs)
                
                arr.append( curr_tok.item() ) #returns int guess from the log_probs. 
                
                inputs = curr_tok
                
                enc_final_states_reshaped = hidden #enc_final_states_reshaped and hidden are tuples is [1,1,hid_size], [1,1,hid_size]
                
                counter += 1
            
            arr = arr[:len(arr)-1] #remove the <EOS> token
            
            word = []
            for i in arr: 
                word.append( str(self.output_indexer.get_object( i )) ) #convert to actual string tokens. 
            
            deriv_list.append( [Derivation( val, 1.0, word)] ) #append derivation object with answer. 
      
        print(deriv_list)
        return deriv_list #this is correct. 
        
        
        
    def encode_input(self, x_tensor, inp_lens_tensor):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple). ONLY THE ENCODER FINAL
        STATES are needed for the basic seq2seq model. enc_output_each_word is needed for attention, and
        enc_context_mask is needed to batch attention.

        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0],
        [1, 2, 3],
        [2, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 3 x dim, enc_context_mask = [[1, 1, 0], [1, 1, 1], [1, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor) #here's where you call forward. 
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)
        
    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, input_emb_dim: int, hidden_size: int, bidirect: bool):
        """
        :param input_emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_emb_dim, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        Note that output is only needed for attention, and context_mask is only used for batched attention.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = max(input_lens.data).item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)
    
    
class RNNDecoder(nn.Module):
    
    def __init__(self,  input_emb_dim, hidden_size, output_size):
        """
        :param input_emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_emb_dim, hidden_size, num_layers=1, batch_first=True, dropout=0.)
        self.classification = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        
        
        self.linear = nn.Linear(2*hidden_size, output_size)
        self.softmax = nn.Softmax(dim = 2)
        self.tanx = nn.Tanh()
        
        
    def forward( self, emb_inputs, enc_hidden, isHidden, context ): #two questions, one where is the weight vector. 
        
        emb_inputs = torch.unsqueeze( emb_inputs, 0) #[1, 1, embed_size]
        
        outputs, hidden = self.rnn(emb_inputs, enc_hidden) #outputs: [1,1,hid_size] hidden: tuple [1,1,hid_size] for each
        
        classification = self.classification(outputs) #[1,1,153] [1,1,output_size]
        
        log_probs = self.log_softmax(classification)[0] #[1, 153], logsoftmax on dim 2
        
        
        #print(context.shape)
        
        if isHidden: 
            print(outputs.shape)
            linear = self.linear(context)
            print(linear.shape)
            
            #matrix dimensions [1,1,400], [1,1,153]
            #product = torch.bmm(context, log_probs)
            product = torch.bmm(context, outputs) #[1,1,400] [1,1,153]
            
            post_tan = self.tanx(product) #[20, 1, 153]
            
            new_log_probs = self.softmax(weighted_sum) 
            
            #sum up over timestep to get c
            #concatenate with outputs. 
            emb_inputs = torch.cat(new_log_probs, log_probs)
            print(emb_inputs)
            
        return log_probs, hidden

def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])


def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Create indexed input & padded
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False) #these are the train 
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False) 

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len) #these are the labels. 
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    num_epochs = 10
    output_size = len(output_indexer) #153
    network = Seq2SeqSemanticParser(input_indexer, output_indexer, emb_dim = 256, hidden_size = 400, output_size = output_size)
    optimizer = optim.Adam(network.parameters(), lr=0.001) #create ADAM optimizer
    
    n = []
    for i in range(len(all_train_input_data)): 
        n.append(i)
    
    for epoch in range(num_epochs): 
        
        
        total_loss = 0.0
        random.shuffle(n) #random shuffle indices.
        
        for pos in n: 
            #val = np.insert(all_train_input_data[pos], 0, 1) #add the <SOS> token into sequence
            val = [x for x in all_train_input_data[pos] if x != 0]
            y_val = [x for x in all_train_output_data[pos] if x != 0]
            
            x_tensor = torch.unsqueeze( torch.FloatTensor(val).long(), 0) #[1, 20]
            inp_lens_tensor = torch.FloatTensor( [len(val)] ).long() #[1], and the value in the 1 is 20. 
            
            y_tensor = torch.unsqueeze( torch.FloatTensor( y_val ).long(), 0) #[1, 65]
           
            
            network.zero_grad() #zero out the gradient
            
            loss = network.forward(x_tensor, inp_lens_tensor, y_tensor)
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print("Total Loss for Epoch ", epoch, " is ", total_loss)
    return network
    