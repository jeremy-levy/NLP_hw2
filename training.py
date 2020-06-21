from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from chu_liu_edmonds import decode_mst
from utils import get_vocabs, nll_loss, UAS
from models import model_1, model_2
from hp import hp_dict

from data_reader import PosDataset

data_dir = "C:\\Users\\jeremy.levy\\OneDrive - Technion\\MSc\\Courses\\courses_gal\\NLP\\HW\\HW2 - wet\\HW2-files\\"
# data_dir = "C:\\Users\\galye\\Dropbox\\studies\\MSc\\NLP\\HW2 - wet\\HW2-files\\"

path_train = data_dir + "train.labeled"
path_test = data_dir + "test.labeled"

word_dict, pos_dict = get_vocabs([path_train, path_test])

dataset_saved = False

if dataset_saved is True:
    print("Loading dataset")
    training_sentences = torch.load('training_sentences.pt')
    test_sentences = torch.load('test_sentences.pt')
else:
    print("Extracting dataset")
    training_sentences = PosDataset(path_train, word_dict, pos_dict, padding=False)
    test_sentences = PosDataset(path_test, word_dict, pos_dict, padding=False)

    torch.save(training_sentences, 'training_sentences.pt')
    torch.save(test_sentences, 'test_sentences.pt')

# in order to use real batches - need to make all sentences in the same length within a batch.
train_dataloader = DataLoader(training_sentences, shuffle=False, batch_size=1)
test_dataloader = DataLoader(test_sentences, shuffle=False)

print("Number of Train Tagged Sentences ", len(training_sentences))
print("Number of Test Tagged Sentences ", len(test_sentences))

# first sentence 3 tuple is accessed through: "training_sentences.sentences_dataset[0]"
model_choosed = 2
if model_choosed == 1:
    curr_model = model_1(word_dict, pos_dict, hp_dict["word_emb_dim"], hp_dict["pos_emb_dim"],
                          hp_dict["lstm_dropout"], hp_dict["lstm_h_dim"], hp_dict["mlp_h_dim"])
else:
    curr_model = model_2(word_dict, pos_dict, hp_dict["word_emb_dim"], hp_dict["pos_emb_dim"],
                          hp_dict["lstm_dropout"], hp_dict["lstm_h_dim"], hp_dict["mlp_h_dim"])
    curr_model.load_state_dict(torch.load("model1_epoch34.pt"))

if torch.cuda.is_available():
    curr_model.cuda()

# optimizer = torch.optim.Adam(curr_model.parameters(), lr=hp_dict["learning_rate"],
#                             weight_decay=hp_dict["weight_decay"])
optimizer = torch.optim.Adam(curr_model.parameters(), lr=hp_dict["learning_rate"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Working on GPU")
else:
    print("Working on cpu")


max_uas = 0


def train_epoch(train, dl):
    global max_uas
    losses_test = []
    losses_train = []
    UAS_train = []
    UAS_test = []
    loss = None
    acumulate_grad_steps = 256

    for i, data_batch in enumerate(dl):
        curr_sentence = data_batch

        curr_sentence[0] = curr_sentence[0].squeeze().to(device)
        curr_sentence[1] = curr_sentence[1].squeeze().to(device)
        curr_sentence[2] = curr_sentence[2].squeeze().to(device)
        curr_sentence[3] = curr_sentence[3].squeeze().to(device)

        sentence_inputs = curr_sentence[0:3]
        sentence_len = curr_sentence[2].item()
        sentence_labels = curr_sentence[3].to(device)
        score_mat = curr_model.forward(sentence_inputs)  # do not forward the labels.
        score_mat = score_mat.to(device)

        '''
        for head_idx in range(sentence_len):
            for modifyer_idx in range(sentence_len):
                if head_idx == modifyer_idx:
                    score_mat[head_idx][modifyer_idx] = 0
                    continue
                if modifyer_idx == sentence_labels[head_idx]:
                    score_mat[modifyer_idx][head_idx] = 100
                else:
                    score_mat[modifyer_idx][head_idx] = 0
        '''

        # Calculate the negative log likelihood loss described above
        loss = nll_loss(score_mat, sentence_labels, sentence_len)
        loss = loss / acumulate_grad_steps

        if train is True:
            # optimizer.zero_grad()
            loss.backward()
            if i % acumulate_grad_steps == 0:
                optimizer.step()
                curr_model.zero_grad()

            losses_train.append(loss.item())
            if i % acumulate_grad_steps == 0:
                predicted_tree, _ = decode_mst(energy=score_mat.cpu().detach(), length=score_mat.shape[0],
                                               has_labels=False)
                uas_score = UAS(predicted_tree, sentence_labels)
                UAS_train.append(uas_score)
        else:
            # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix
            predicted_tree, _ = decode_mst(energy=score_mat.cpu().detach(), length=score_mat.shape[0], has_labels=False)
            uas_score = UAS(predicted_tree, sentence_labels)

            losses_test.append(loss.item())
            UAS_test.append(uas_score)

    if train is True:
        print("\nTrain: epoch number", epoch, ":  loss = ", np.mean(losses_train),
              ": UAS = ", np.mean(UAS_train), "%")
    else:
        print("Test: epoch number", epoch, ":  loss = ", np.mean(losses_test),
              ": UAS = ", np.mean(UAS_test), "%")

        if np.mean(UAS_test) > max_uas:
            print("saving the model")
            max_uas = np.mean(UAS_test)
            torch.save(curr_model.state_dict(), "model" + str(model_choosed) + "_epoch" + str(epoch) + ".pt")

    return np.mean(losses_train), np.mean(UAS_train), np.mean(losses_test), np.mean(UAS_test)


def plot(variable, ylabel):
    plt.plot(variable)
    plt.ylabel(ylabel)
    plt.xlabel("epochs")
    plt.show()


epochs = 40
losses_test_epochs = []
losses_train_epochs = []
UAS_train_epochs = []
UAS_test_epochs = []

for epoch in range(epochs):
    loss_train, UAS_train, _, _ = train_epoch(train=True, dl=train_dataloader)
    _, _, loss_test, UAS_test = train_epoch(train=False, dl=test_dataloader)

    losses_test_epochs.append(loss_test)
    losses_train_epochs.append(loss_train)
    UAS_train_epochs.append(UAS_train)
    UAS_test_epochs.append(UAS_test)

plot(losses_test_epochs, " test loss")
plot(losses_train_epochs, "train loss")
plot(UAS_test_epochs, "test UAS")
plot(UAS_train_epochs, "train UAS")
