import torch.nn as nn
import torch.nn.functional as F
import torch
from data_reader import SPECIAL_TOKENS


def word_idx_to_onehot(word_idx, num_of_words, sentence_len):
    words_onehot = torch.FloatTensor(sentence_len, num_of_words)
    words_onehot.zero_()
    words_onehot.scatter_(1, word_idx.reshape(sentence_len, 1), 1)
    return words_onehot


class model_1(nn.Module):
    def __init__(self, words_dict, poses_dict, word_emb_dim, pos_emb_dim, dropout=0.25, lstm_h_dim=256,
                 mlp_h_dim=128, *args):
        super(model_1, self).__init__()

        self.words_dict = words_dict
        self.poses_dict = poses_dict
        self.num_of_words = len(words_dict) + len(SPECIAL_TOKENS)
        self.num_of_poses = len(poses_dict) + len(SPECIAL_TOKENS)
        self.dropout = dropout
        self.lstm_h_dim = lstm_h_dim
        self.mlp_h_dim = mlp_h_dim
        self.lstm_layers = 2

        self.word_embedding = nn.Embedding(self.num_of_words, word_emb_dim)
        self.pos_embedding = nn.Embedding(self.num_of_poses, pos_emb_dim)
        self.emb_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder = nn.LSTM(self.emb_dim, lstm_h_dim, num_layers=self.lstm_layers, bidirectional=True,
                               dropout=self.dropout, batch_first=True)

        # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * self.lstm_layers * lstm_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, 1)
        )

    def forward(self, sentence):
        # word_idx_tensor, pos_idx_tensor, true_tree_heads, sentence_len = sentence
        word_idx_tensor, pos_idx_tensor, sentence_len = sentence

        # Pass word_idx and pos_idx through their embedding layers
        '''
        words_onehot = word_idx_to_onehot(word_idx_tensor, self.num_of_words, sentence_len)
        poses_onehot = word_idx_to_onehot(pos_idx_tensor, self.num_of_poses, sentence_len)
        '''
        words_embedding = self.word_embedding(word_idx_tensor)
        pos_embedding = self.pos_embedding(pos_idx_tensor)

        # Concat both embedding outputs        
        embeded_words_poses = torch.cat((words_embedding, pos_embedding), 1)

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        embeded_words_poses = embeded_words_poses.unsqueeze(0)
        bi_lstm_output, _ = self.encoder(embeded_words_poses)
        bi_lstm_output = bi_lstm_output.squeeze()

        # Get score for each possible edge in the parsing graph, construct score matrix
        score_matrix = torch.FloatTensor(sentence_len, sentence_len)
        # loop over heads (the pointers)
        for head_idx in range(sentence_len):
            # loop over modifiers
            for modifyer_idx in range(sentence_len):
                if head_idx == modifyer_idx:
                    score_matrix[head_idx][modifyer_idx] = 0
                    continue
                tmp_concat_tensor = torch.cat((bi_lstm_output[head_idx], bi_lstm_output[modifyer_idx]), 0)
                curr_word_head_score = self.edge_scorer(tmp_concat_tensor)
                score_matrix[head_idx][modifyer_idx] = curr_word_head_score

        return score_matrix


class model_2(nn.Module):
    def __init__(self, words_dict, poses_dict, word_emb_dim, pos_emb_dim, dropout=0.25, lstm_h_dim=256,
                 mlp_h_dim=128, *args):
        super(model_2, self).__init__()

        self.words_dict = words_dict
        self.poses_dict = poses_dict
        self.num_of_words = len(words_dict) + len(SPECIAL_TOKENS)
        self.num_of_poses = len(poses_dict) + len(SPECIAL_TOKENS)
        self.dropout = dropout
        self.lstm_h_dim = lstm_h_dim
        self.mlp_h_dim = mlp_h_dim
        self.lstm_layers = 2

        self.word_embedding = nn.Embedding(self.num_of_words, word_emb_dim)
        self.pos_embedding = nn.Embedding(self.num_of_poses, pos_emb_dim)
        self.emb_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.2),
        )
        self.encoder1 = nn.LSTM(int(self.emb_dim/3), lstm_h_dim, num_layers=self.lstm_layers, bidirectional=True,
                                dropout=self.dropout, batch_first=True)
        self.encoder2 = nn.LSTM(lstm_h_dim*2, lstm_h_dim, num_layers=self.lstm_layers, bidirectional=True,
                                dropout=self.dropout, batch_first=True)

        # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * self.lstm_layers * lstm_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, 1)
        )

    def forward(self, sentence):
        # word_idx_tensor, pos_idx_tensor, true_tree_heads, sentence_len = sentence
        word_idx_tensor, pos_idx_tensor, sentence_len = sentence

        # Pass word_idx and pos_idx through their embedding layers
        words_embedding = self.word_embedding(word_idx_tensor)
        pos_embedding = self.pos_embedding(pos_idx_tensor)

        # Concat both embedding outputs
        embeded_words_poses = torch.cat((words_embedding, pos_embedding), 1)

        # Feed the CNN with the embedding outputs
        embeded_words_poses = embeded_words_poses.unsqueeze(1)
        cnn_out = self.cnn(embeded_words_poses)
        cnn_out = cnn_out.squeeze()

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        cnn_out = cnn_out.unsqueeze(0)
        bi_lstm_output, _ = self.encoder1(cnn_out)
        bi_lstm_output, _ = self.encoder2(bi_lstm_output)
        bi_lstm_output = bi_lstm_output.squeeze()

        # Get score for each possible edge in the parsing graph, construct score matrix
        score_matrix = torch.FloatTensor(sentence_len, sentence_len)
        # loop over heads (the pointers)
        for head_idx in range(sentence_len):
            # loop over modifiers
            for modifyer_idx in range(sentence_len):
                if head_idx == modifyer_idx:
                    score_matrix[head_idx][modifyer_idx] = 0
                    continue
                tmp_concat_tensor = torch.cat((bi_lstm_output[head_idx], bi_lstm_output[modifyer_idx]), 0)
                curr_word_head_score = self.edge_scorer(tmp_concat_tensor)
                score_matrix[head_idx][modifyer_idx] = curr_word_head_score

        return score_matrix
