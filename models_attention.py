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
class TResNet_att(nn.Module):
    '''        
        features-out dim: (batch-size, encoded-image-size, encoded-image-size, 2048)
        pool_out dim: (batch-size, 2048)
    '''

    def __init__(self, args, pretrained):
        super(TResNet_att, self).__init__()
        self.path = args.save_dir + 'TResNet_att_Corel-5k.pth'

        tresnet = timm.create_model('tresnet_m', pretrained=pretrained)
        self.features = nn.Sequential(tresnet.body)
        self.avgpool = tresnet.head.global_pool

    def forward(self, x):
        features = self.features(x)
        features_out = features.permute(0, 2, 3, 1)
        #
        pool_out = self.avgpool(features)
        return (features_out, pool_out)


# =============================================================================
# Attention
# =============================================================================
class Attention(nn.Module):
    '''        
        encoder-feature_out dim: (batch-size, num-pixels, 2048)
        decoder-hidden dim: (batch-size, 2048)
    '''

    def __init__(self, encoder_size, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_size, attention_size)
        self.decoder_att = nn.Linear(hidden_size, attention_size)
        self.full_att = nn.Linear(attention_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_feature_out, decoder_hidden):
        # (batch-size, num-pixels, attention-size)
        att1 = self.encoder_att(encoder_feature_out)
        # (batch-size, attention-size)
        att2 = self.decoder_att(decoder_hidden)
        # (batch-size, num-pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        # (batch-size, num-pixels)
        alpha = self.softmax(att)
        # (batch-size, encoder-size)
        attention_weighted_encoding = (
            encoder_feature_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


# =============================================================================
# LSTM (Decoder)
# =============================================================================
class Anotator_att(nn.Module):
    '''
    input-size: (number of classes) + 2,
    output-size: (number of classes) + 1
    
    y-hats dim: (batch-size, max-seq-len, (number of classes + 1))
    '''

    def __init__(self,
                 args,
                 input_size,
                 hidden_size,
                 output_size,
                 attention_size,
                 emb_size,
                 is_glove,
                 glove_weights=None):
        super(Anotator_att, self).__init__()
        self.args = args
        self.path = self.args.save_dir + 'LSTM_att_Corel-5k.pth'
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder_size = 2048

        self.word_emb = nn.Embedding(input_size, emb_size)
        # utilizing GLOVE pre-trained weights in the embedding matrix
        if is_glove:
            self.word_emb.weight.data.copy_(glove_weights)
            self.word_emb.weight.requires_grad_(False)
        #
        self.features_embedding = nn.Linear(self.encoder_size, hidden_size)
        #
        self.lstm_cell = nn.LSTMCell(input_size=emb_size + self.encoder_size,
                                     hidden_size=self.hidden_size)
        #
        self.attention = Attention(encoder_size=self.encoder_size,
                                   hidden_size=self.hidden_size,
                                   attention_size=attention_size)
        #
        self.f_beta = nn.Linear(self.hidden_size, self.encoder_size)
        #
        self.sigmoid = nn.Sigmoid()
        self.emb_dropout = nn.Dropout(0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def init_state(self, image_features):
        '''
            hidden-state & cell-state dims: (batch-size, hidden-size)
        '''
        
        hidden_state = self.features_embedding(image_features)
        #
        cell_state = torch.zeros_like(hidden_state)
        return hidden_state, cell_state

    def annotator_output(self,
                         annotation_X,
                         attention_weighted_encoding,
                         prev_state):
        '''
            y-hat dim: (batch-size, (number of classes + 1))
        '''
        
        embeddings = self.word_emb(annotation_X)
        embeddings = self.emb_dropout(embeddings)
        hidden_state, cell_state = self.lstm_cell(torch.cat([embeddings, attention_weighted_encoding],
                                                            dim=1), prev_state)
        out = self.dropout(hidden_state)
        yhat = self.fc(out)
        return yhat, hidden_state, cell_state

    def forward(self,
                encoder_feature_out,
                encoder_out,
                annotations_X,
                is_train):
        '''
            is_train: True
            annotations_X dim = (batch-size, (max-seq-len + 1))
            
            is_train: False
            annotations_X dim = (batch-size)
        '''
        
        # flatten image
        # (batch-size, num-pixels, encoder-size)
        encoder_feature_out = encoder_feature_out.view(
            encoder_out.size(0), -1, self.encoder_size)
        #
        hidden_state, cell_state = self.init_state(encoder_out)
        #
        yhats = []
        alphas = []
        #
        for t in range(self.args.max_seq_len + 2):
            attention_weighted_encoding, alpha = self.attention(
                encoder_feature_out, hidden_state)
            alphas.append(alpha.unsqueeze(1))
            gate = self.sigmoid(self.f_beta(hidden_state))
            attention_weighted_encoding = gate * attention_weighted_encoding
            if is_train == True:
                y_hat, hidden_state, cell_state = self.annotator_output(annotations_X[:, t],
                                                                        attention_weighted_encoding,
                                                                        (hidden_state, cell_state))
                yhats.append(y_hat.unsqueeze(1))
            else:
                y_hat, hidden_state, cell_state = self.annotator_output(annotations_X,
                                                                        attention_weighted_encoding,
                                                                        (hidden_state, cell_state))
                y_hat = y_hat.unsqueeze(1)
                yhats.append(y_hat)
                #
                _, annotations_X = torch.max(y_hat, 2)
                annotations_X = annotations_X.squeeze(1)
        #
        yhats = torch.cat(yhats, 1)
        alphas = torch.cat(alphas, 1)
        return yhats, alphas
