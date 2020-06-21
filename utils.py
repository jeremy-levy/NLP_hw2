from collections import defaultdict
import torch.nn as nn
import torch


def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters string: characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


def get_vocabs(list_of_paths):
    """
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param file_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - tag2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n', '\t'))
                if len(splited_words) > 4:
                    word = splited_words[1]
                    pos_tag = splited_words[3]
                    word_dict[word] += 1
                    pos_dict[pos_tag] += 1

    return word_dict, pos_dict


def word_idx_to_onehot(word_idx, num_of_words, sentence_len):
    words_onehot = torch.FloatTensor(sentence_len, num_of_words)
    words_onehot.zero_()
    words_onehot.scatter_(1, word_idx.reshape(sentence_len, 1), 1)
    return words_onehot


cross_entropy_loss = nn.CrossEntropyLoss()


def nll_loss(score_mat, true_heads, sentence_len):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # first, remove the ROOT column from mat_score and from true_heads
    edge_scores = score_mat[:, 1:]
    true_heads = true_heads[1:]

    '''
    print("score")
    print(score_mat)
    print("edge")
    print(edge_scores)
    print("true_head")
    print(true_heads)
    '''

    loss = torch.zeros(1, device=device)
    sentence_len = sentence_len - 1  # minus 1 because we do not count "ROOT"

    for modifyer_word in range(sentence_len):
        edge = edge_scores[:, modifyer_word].unsqueeze(dim=0)
        true_score = true_heads[modifyer_word:modifyer_word + 1]
        cross = cross_entropy_loss(edge, true_score)
        loss += cross
    return (1.0 / sentence_len) * loss


def UAS(predicted_tree, true_tree):
    equal = torch.eq(torch.from_numpy(predicted_tree[1:]).cpu(), true_tree[1:].cpu())
    equal = torch.sum(equal)
    equal = (equal.item() / (len(predicted_tree) - 1))
    return equal * 100
