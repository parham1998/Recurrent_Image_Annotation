# =============================================================================
# Install necessary packages
# =============================================================================
# pip install inplace-abn
# pip install timm


# =============================================================================
# Import required libraries
# =============================================================================
import torch
from torch import nn
import timm


# =============================================================================
# CNN (Encoder)
# =============================================================================
class TResNet(nn.Module):
    def __init__(self,
                 args,
                 pretrained,
                 num_classes):
        super(TResNet, self).__init__()
        self.path = args.save_dir + 'TResNet_Corel-5k.pth'

        tresnet = timm.create_model('tresnet_m', pretrained=pretrained)
        if args.method == 'RIA':
            self.features = nn.Sequential(
                tresnet.body,
                tresnet.head.global_pool,
            )
        elif args.method == 'SR-CNN-RNN':
            tresnet.head.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features=2048, out_features=num_classes)
                # nn.Sigmoid()
            )
            self.features = tresnet

    def forward(self, x):
        return self.features(x)


# =============================================================================
# LSTM (Decoder)
# =============================================================================
class Anotator(nn.Module):
    '''
        input-size: (number of classes) + 2
        output-size: (number of classes) + 1
    
        y-hats dim: (batch-size, max-seq-len, (number of classes + 1))
    '''

    def __init__(self,
                 args,
                 input_size,
                 hidden_size,
                 output_size,
                 num_classes,
                 emb_size,
                 is_glove,
                 glove_weights=None):
        super(Anotator, self).__init__()
        self.args = args
        self.path = self.args.save_dir + 'LSTM_Corel-5k.pth'
        self.hidden_size = hidden_size
        if self.args.method == 'RIA':
            self.encoder_size = 2048
        elif self.args.method == 'SR-CNN-RNN':
            self.encoder_size = num_classes

        self.word_emb = nn.Embedding(input_size, emb_size)
        # utilizing GLOVE pre-trained weights in the embedding matrix
        if is_glove:
            self.word_emb.weight.data.copy_(glove_weights)
            self.word_emb.weight.requires_grad_(False)
        #
        self.features_embedding = nn.Linear(self.encoder_size, hidden_size)
        #
        self.lstm_cell = nn.LSTMCell(input_size=emb_size,
                                     hidden_size=hidden_size)

        self.emb_dropout = nn.Dropout(0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def init_state(self, image_features):
        '''
            hidden-state & cell-state dims: (batch-size, hidden-size)
        '''

        if self.args.method == 'SR-CNN-RNN':
            image_features = torch.sigmoid(image_features)
        hidden_state = self.features_embedding(image_features)
        #
        cell_state = torch.zeros_like(hidden_state)
        return hidden_state, cell_state

    def annotator_output(self, annotation_X, prev_state):
        '''
            y-hat dim: (batch-size, (number of classes + 1))
        '''

        embeddings = self.word_emb(annotation_X)
        embeddings = self.emb_dropout(embeddings)
        # state = hidden state & cell state
        hidden_state, cell_state = self.lstm_cell(embeddings, prev_state)
        out = self.dropout(hidden_state)
        yhat = self.fc(out)
        return yhat, hidden_state, cell_state

    def forward(self, image_features, annotations_X, is_train):
        '''
            is_train: True
            annotations_X dim = (batch-size, (max-seq-len + 1))
            
            is_train: False
            annotations_X dim = (batch-size)
        '''

        hidden_state, cell_state = self.init_state(image_features)
        #
        yhats = []
        #
        for t in range(self.args.max_seq_len + 2):
            if is_train == True:
                y_hat, hidden_state, cell_state = self.annotator_output(annotations_X[:, t],
                                                                        (hidden_state, cell_state))
                yhats.append(y_hat.unsqueeze(1))
            else:
                y_hat, hidden_state, cell_state = self.annotator_output(annotations_X,
                                                                        (hidden_state, cell_state))
                y_hat = y_hat.unsqueeze(1)
                yhats.append(y_hat)
                #
                _, annotations_X = torch.max(y_hat, 2)
                annotations_X = annotations_X.squeeze(1)
        #
        yhats = torch.cat(yhats, 1)
        return yhats
