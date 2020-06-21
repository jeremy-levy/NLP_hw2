from collections import defaultdict

import torch
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
from utils import split

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
# ROOT_TOKEN = PAD_TOKEN # this can be used if you are not padding your batches
ROOT_TOKEN = "<root>"  # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN, ROOT_TOKEN]


class PosDataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            curr_sentence = [(ROOT_TOKEN, ROOT_TOKEN, -1)]
            # add the ROOT token to the start of the sentence
            for line in f:
                word_splitted_fields = split(line, (' ', '\n', '\t'))
                # verify this is not a end of a sentence
                if len(word_splitted_fields) > 8:
                    word = word_splitted_fields[1]
                    pos_tag = word_splitted_fields[3]
                    tree_head = int(word_splitted_fields[6])
                    curr_sentence.append((word, pos_tag, tree_head))
                else:  # this is an end of a sentence
                    self.sentences.append(curr_sentence)
                    curr_sentence = [(ROOT_TOKEN, ROOT_TOKEN, -1)]
                    # add the ROOT token to each start of a sentence

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class PosDataset(Dataset):
    def __init__(self, file_path: str, word_dict, pos_dict, padding=False, word_embeddings=None):
        super().__init__()
        self.file = file_path
        self.datareader = PosDataReader(self.file, word_dict, pos_dict)
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.datareader.word_dict)
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)
        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len, tree_heads = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len, tree_heads

    @staticmethod
    def init_word_embeddings(word_dict):
        # TODO - for Model1 do not use Glove !
        glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_tree_head_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            tree_head_list = []
            for word, pos, tree_head in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
                tree_head_list.append(tree_head)
            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)
            sentence_tree_head_list.append(torch.tensor(tree_head_list, dtype=torch.long, requires_grad=False))

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_len_list,
                                                                     sentence_tree_head_list))}
