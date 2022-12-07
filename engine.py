# =============================================================================
# Import required libraries
# =============================================================================
import timeit
from tqdm import tqdm
import numpy as np

import torch
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils.rnn import pack_padded_sequence
# hungarian algorithm
from munkres import Munkres
m = Munkres()

from evaluation_metrics import EvaluationMetrics
from beam_search import annotate_batch_beam_search
from utils import init_input, convert_to_one_hot


# checking the availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine():
    def __init__(self,
                 args,
                 cnn_model,
                 lstm_model,
                 criterion,
                 train_loader,
                 validation_loader,
                 classes,
                 word_map):
        self.args = args
        self.cnn = cnn_model
        self.lstm = lstm_model
        self.criterion = criterion
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.classes = classes
        self.word_map = word_map

    def learnabel_parameters(self, model):
        return [p for p in model.parameters() if p.requires_grad == True]

    def count_learnabel_parameters(self, parameters):
        return sum(p.numel() for p in parameters)

    def scheduler(self, optimizer, lr):
        steps_per_epoch = len(self.train_loader)
        return OneCycleLR(optimizer,
                          max_lr=lr,
                          steps_per_epoch=steps_per_epoch,
                          epochs=self.args.epochs,
                          pct_start=0.2)

    def initialize_optimizer(self):
        # CNN
        lr_cnn = self.args.encoder_lr
        self.optimizer_cnn = optim.Adam(self.learnabel_parameters(self.cnn),
                                        lr_cnn)
        self.scheduler_cnn = self.scheduler(self.optimizer_cnn, lr_cnn)
        # LSTM
        lr_lstm = self.args.decoder_lr
        self.optimizer_lstm = optim.Adam(self.learnabel_parameters(self.lstm),
                                         lr_lstm)
        self.scheduler_lstm = self.scheduler(self.optimizer_lstm, lr_lstm)

    def initialization(self):
        if not self.args.evaluate:
            self.initialize_optimizer()
            self.best_f1_score = 0

            print('Number of CNN\'s learnable parameters: ' +
                  str(self.count_learnabel_parameters(self.learnabel_parameters(self.cnn))))
            if self.args.method == 'Attention':
                print('Number of Attention\'s learnable parameters: ' +
                      str(self.count_learnabel_parameters(self.learnabel_parameters(self.lstm.attention))))
                lstm_param = self.count_learnabel_parameters(
                    self.learnabel_parameters(self.lstm)) - self.count_learnabel_parameters(
                        self.learnabel_parameters(self.lstm.attention))
                print('Number of LSTM\'s learnable parameters: ' + str(lstm_param))
            else:
                print('Number of LSTM\'s learnable parameters: ' +
                      str(self.count_learnabel_parameters(self.learnabel_parameters(self.lstm))))
            #
            print('CNN Optimizer: {}'.format(self.optimizer_cnn))
            print('LSTM Optimizer: {}'.format(self.optimizer_lstm))

        self.metrics = EvaluationMetrics()

        if not torch.cuda.is_available():
            print('CUDA is not available. Training on CPU ...')
        else:
            print('CUDA is available! Training on GPU ...')
            print(torch.cuda.get_device_properties('cuda'))
        #
        self.cnn = self.cnn.to(device)
        self.lstm = self.lstm.to(device)

    def PR_RC_F1_Nplus(self, results):
        N_plus = 'N+: {:.0f}'.format(results['N+'])
        per_class_metrics = 'per-class precision: {:.4f} \t per-class recall: {:.4f} \t per-class f1: {:.4f}'.format(
            results['per_class/precision'], results['per_class/recall'], results['per_class/f1'])
        per_image_metrics = 'per-image precision: {:.4f} \t per-image recall: {:.4f} \t per-image f1: {:.4f}'.format(
            results['per_image/precision'], results['per_image/recall'], results['per_image/f1'])
        return N_plus, per_class_metrics, per_image_metrics

    def load_model(self):
        self.cnn.load_state_dict(torch.load(self.cnn.path))
        self.lstm.load_state_dict(torch.load(self.lstm.path))

    def save_model(self):
        torch.save(self.cnn.state_dict(), self.cnn.path)
        torch.save(self.lstm.state_dict(), self.lstm.path)

    def cal_probability(self, x):
        '''
            calculate P(x1,x2,...,xk | image)
        '''
        x = x.transpose(1, 2)
        p = F.softmax(x, 1)
        val, idxs = torch.max(p, 1)
        batch_size = idxs.size(0)
        val_new = torch.zeros((batch_size))
        for i in range(batch_size):
            hold = 1
            for t in range(idxs.size(1)):
                if idxs[i][t].item() == self.word_map['stop']:
                    break
                hold = hold * val[i][t].item()
            val_new[i] = hold
        return torch.mean(val_new)

    def order_the_targets_mla(self, outputs, targets, labels_lengths):
        batch_size = targets.shape[0]
        #
        outputs_tensor = outputs.clone()
        targets_tensor = targets.clone()
        #
        targets = targets.data.cpu().numpy()
        targets_new = targets.copy()
        #
        for i in range(batch_size):
            n_labels = labels_lengths[i] - 1
            if n_labels != 0:
                current_labels = targets_tensor[i][0:n_labels]
                cost_matrix = np.zeros((n_labels, n_labels), dtype=np.float32)
                for j in range(n_labels):
                    losses = -F.log_softmax(outputs_tensor[i][j], dim=0)
                    temp = losses[current_labels]
                    cost_matrix[j, :] = temp.data.cpu().numpy()
                indexes = m.compute(cost_matrix)
                new_labels = [x[1] for x in indexes]
                current_labels = current_labels.tolist()
                new_labels = [current_labels[x] for x in new_labels]
                targets_new[i][0:n_labels] = new_labels
        return torch.LongTensor(targets_new).to(device)

    def order_the_targets_pla(self, outputs, targets, labels_lengths):
        batch_size = targets.shape[0]
        #
        outputs_tensor = outputs.clone()
        #
        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        targets_new = targets.copy()
        targets_newest = targets.copy()
        indexes = np.argmax(outputs, axis=2)
        #
        for i in range(batch_size):
            n_labels = labels_lengths[i] - 1
            common_indexes = set(
                targets[i][0:n_labels]).intersection(set(indexes[i]))
            diff_indexes = set(
                targets[i][0:n_labels]).difference(set(indexes[i]))
            if common_indexes != set():
                for j in range(n_labels):
                    if indexes[i][j] in common_indexes:
                        if indexes[i][j] != targets_new[i][j].item():
                            old_value = targets_new[i][j]
                            new_value = indexes[i][j]
                            new_value_index = np.where(
                                targets_new[i] == new_value)[0][0]
                            targets_new[i][j] = new_value
                            targets_new[i][new_value_index] = old_value
                        common_indexes.remove(indexes[i][j].item())

            targets_newest[i] = targets_new[i]
            n_different = len(diff_indexes)
            if n_different > 1:
                diff_indexes_tuples = [[count, elem]
                                       for count, elem in enumerate(
                    targets_new[i][0:n_labels])
                    if elem in diff_indexes]
                diff_indexes_locations, diff_indexes_ordered = zip(
                    *diff_indexes_tuples)
                cost_matrix = np.zeros((n_different, n_different),
                                       dtype=np.float32)
                for diff_count, diff_index_location in enumerate(diff_indexes_locations):
                    losses = -F.log_softmax(
                        outputs_tensor[i][diff_index_location], dim=0)
                    temp = losses[torch.LongTensor(diff_indexes_ordered)]
                    cost_matrix[diff_count, :] = temp.data.cpu().numpy()
                indexes2 = m.compute(cost_matrix)
                new_labels = [x[1] for x in indexes2]
                for new_label_count, new_label in enumerate(new_labels):
                    targets_newest[i][diff_indexes_locations[new_label_count]
                                      ] = diff_indexes_ordered[new_label]
        return torch.LongTensor(targets_newest).to(device)

    def train(self, dataloader, epoch=None):
        train_loss = 0
        if self.args.method == 'SR-CNN-RNN':
            train_loss_cnn = 0
            train_loss_lstm = 0
        total_outputs = []
        total_targets = []
        self.cnn.train()
        self.lstm.train()

        for batch_idx, (images, binary_annotations, annotations_X, label_lengths) in enumerate(tqdm(dataloader)):

            images = images.to(device)
            binary_annotations = binary_annotations.to(device)
            annotations_X = annotations_X.to(device)
            label_lengths = label_lengths.to(device)

            # sort input data by decreasing lengths
            label_lengths_sorted, sort_ind = label_lengths.sort(
                dim=0, descending=True)
            images_sorted = images[sort_ind]
            binary_annotations_sorted = binary_annotations[sort_ind]
            annotations_X_sorted = annotations_X[sort_ind]

            # adding STOP to label lengths
            label_lengths_sorted = (label_lengths_sorted + 1).tolist()

            # since we decoded starting with START,
            # the targets are all words after START, up to STOP
            targets = annotations_X_sorted[:, 1:]

            # zero the gradients parameter
            self.optimizer_cnn.zero_grad()
            self.optimizer_lstm.zero_grad()

            if self.args.method == 'RIA':
                fc_features = self.cnn(images_sorted)
                # forward pass: compute predicted outputs by passing inputs to
                # the model
                outputs = self.lstm(fc_features, annotations_X_sorted, True)
                # packing a Tensor containing padded (STOP) sequences of variable length
                outputs_pack, _, _, _ = pack_padded_sequence(
                    outputs, label_lengths_sorted, batch_first=True)
                targets_pack, _, _, _ = pack_padded_sequence(
                    targets, label_lengths_sorted, batch_first=True)
                # calculate the batch loss
                loss = self.criterion[0](outputs_pack, targets_pack)
                # backward pass: compute gradient of the loss with respect to
                # the model parameters
                loss.backward()
                #
                train_loss += loss.item()
            elif self.args.method == 'SR-CNN-RNN':
                fc_features = self.cnn(images_sorted)
                #
                loss_1 = self.criterion[1](
                    fc_features, binary_annotations_sorted)
                outputs = self.lstm(fc_features.detach(),
                                    annotations_X_sorted,
                                    True)
                #
                outputs_pack, _, _, _ = pack_padded_sequence(
                    outputs, label_lengths_sorted, batch_first=True)
                targets_pack, _, _, _ = pack_padded_sequence(
                    targets, label_lengths_sorted, batch_first=True)
                loss_2 = self.criterion[0](outputs_pack, targets_pack)
                #
                loss_1.backward()
                loss_2.backward()
                #
                train_loss_cnn += loss_1.item()
                train_loss_lstm += loss_2.item()
                train_loss = train_loss_cnn + train_loss_lstm
            elif self.args.method == 'Attention':
                image_features, fc_features = self.cnn(images_sorted)
                #
                if self.args.order_free == 'None':
                    outputs, _ = self.lstm(image_features,
                                           fc_features,
                                           annotations_X_sorted,
                                           True)
                #
                else:
                    # initial input
                    annotations_X_1 = init_input(
                        images.shape[0], self.word_map)
                    #
                    outputs, _ = self.lstm(image_features,
                                           fc_features,
                                           annotations_X_1,
                                           False)

                    if self.args.order_free == 'MLA':
                        targets = self.order_the_targets_mla(
                            outputs, targets, label_lengths_sorted)
                    elif self.args.order_free == 'PLA':
                        targets = self.order_the_targets_pla(
                            outputs, targets, label_lengths_sorted)
                #
                outputs_pack, _, _, _ = pack_padded_sequence(
                    outputs, label_lengths_sorted, batch_first=True)
                targets_pack, _, _, _ = pack_padded_sequence(
                    targets, label_lengths_sorted, batch_first=True)
                #
                loss = self.criterion[0](outputs_pack, targets_pack)
                #
                loss.backward()
                #
                train_loss += loss.item()

            # parameters update
            self.optimizer_lstm.step()
            self.optimizer_cnn.step()
            # learning rate update
            self.scheduler_lstm.step()
            self.scheduler_cnn.step()

            outputs = convert_to_one_hot(outputs,
                                         len(self.classes),
                                         self.word_map)
            #
            total_targets.append(binary_annotations_sorted)
            total_outputs.append(outputs)

        results = self.metrics.calculate_metrics(
            torch.cat(total_targets),
            torch.cat(total_outputs))

        print('Epoch: {}'.format(epoch+1))
        if self.args.method == 'SR-CNN-RNN':
            print('Train Loss: {:.5f} \t Train Loss CNN: {:.5f} \t Train Loss LSTM: {:.5f}'.format(
                train_loss/(batch_idx+1), train_loss_cnn/(batch_idx+1), train_loss_lstm/(batch_idx+1)))
        else:
            print('Train Loss: {:.5f}'.format(train_loss/(batch_idx+1)))
        #
        N_plus, per_class_metrics, per_image_metrics = self.PR_RC_F1_Nplus(
            results)
        print(N_plus)
        print(per_class_metrics)
        print(per_image_metrics)

    def validation(self, dataloader, epoch=None):
        valid_loss = 0
        if self.args.method == 'SR-CNN-RNN':
            valid_loss_cnn = 0
            valid_loss_lstm = 0
        total_outputs = []
        total_targets = []
        self.cnn.eval()
        self.lstm.eval()
        #
        # predicted_prob = 0
        # actual_prob = 0

        with torch.no_grad():
            for batch_idx, (images, binary_annotations, annotations_X, label_lengths) in enumerate(tqdm(dataloader)):

                images = images.to(device)
                binary_annotations = binary_annotations.to(device)
                annotations_X = annotations_X.to(device)
                label_lengths = label_lengths.to(device)

                # sort input data by decreasing lengths
                label_lengths_sorted, sort_ind = label_lengths.sort(
                    dim=0, descending=True)
                images_sorted = images[sort_ind]
                binary_annotations_sorted = binary_annotations[sort_ind]
                annotations_X_sorted = annotations_X[sort_ind]

                # adding STOP to label lengths
                label_lengths_sorted = (label_lengths_sorted + 1).tolist()

                # since we decoded starting with START,
                # the targets are all words after START, up to STOP
                targets = annotations_X_sorted[:, 1:]

                # initial input
                annotations_X_1 = init_input(
                    images.shape[0], self.word_map)

                if self.args.method == 'RIA':
                    fc_features = self.cnn(images_sorted)
                    outputs = self.lstm(fc_features, annotations_X_1, False)
                    #
                    outputs_pack, _, _, _ = pack_padded_sequence(
                        outputs, label_lengths_sorted, batch_first=True)
                    targets_pack, _, _, _ = pack_padded_sequence(
                        targets, label_lengths_sorted, batch_first=True)
                    #
                    loss = self.criterion[0](outputs_pack, targets_pack)
                    valid_loss += loss.item()
                elif self.args.method == 'SR-CNN-RNN':
                    fc_features = self.cnn(images_sorted)
                    loss_1 = self.criterion[1](
                        fc_features, binary_annotations_sorted)
                    outputs = self.lstm(fc_features, annotations_X_1, False)
                    #
                    outputs_pack, _, _, _ = pack_padded_sequence(
                        outputs, label_lengths_sorted, batch_first=True)
                    targets_pack, _, _, _ = pack_padded_sequence(
                        targets, label_lengths_sorted, batch_first=True)
                    # calculate the batch loss
                    loss_2 = self.criterion[0](outputs_pack, targets_pack)
                    #
                    valid_loss_cnn += loss_1.item()
                    valid_loss_lstm += loss_2.item()
                    valid_loss = valid_loss_cnn + valid_loss_lstm
                elif self.args.method == 'Attention':
                    image_features, fc_features = self.cnn(images_sorted)
                    #
                    outputs, _ = self.lstm(image_features,
                                           fc_features,
                                           annotations_X_1,
                                           False)
                    #
                    if self.args.order_free == 'None':
                        outputs_pack, _, _, _ = pack_padded_sequence(
                            outputs, label_lengths_sorted, batch_first=True)
                        targets_pack, _, _, _ = pack_padded_sequence(
                            targets, label_lengths_sorted, batch_first=True)
                        # calculate the batch loss
                        loss = self.criterion[0](outputs_pack, targets_pack)
                        #
                        valid_loss += loss.item()

                '''
                comparing P(y_hat | image) with P(y | image)
                
                predicted_prob += self.cal_probability(outputs)
                if (self.args.method == 'RIA' or self.args.method == 'SR-CNN-RNN'):
                    actual_outputs = self.lstm(
                        fc_features, annotations_X_sorted, True)
                elif self.args.method == 'Attention':
                    actual_outputs, _ = self.lstm(image_features,
                                                  fc_features,
                                                  annotations_X_sorted,
                                                  True)
                actual_prob += self.cal_probability(actual_outputs)
                '''

                outputs = convert_to_one_hot(outputs,
                                             len(self.classes),
                                             self.word_map)
                #
                total_targets.append(binary_annotations_sorted)
                total_outputs.append(outputs)

        results = self.metrics.calculate_metrics(
            torch.cat(total_targets),
            torch.cat(total_outputs))

        if self.args.method == 'SR-CNN-RNN':
            print('Validation Loss: {:.5f} \t Validation Loss CNN: {:.5f} \t Validation Loss LSTM: {:.5f}'.format(
                valid_loss/(batch_idx+1), valid_loss_cnn/(batch_idx+1), valid_loss_lstm/(batch_idx+1)))
        else:
            print('Validation Loss: {:.5f}'.format(valid_loss/(batch_idx+1)))
        #
        N_plus, per_class_metrics, per_image_metrics = self.PR_RC_F1_Nplus(
            results)
        print(N_plus)
        print(per_class_metrics)
        print(per_image_metrics)

        # save model when 'per-class f1-score' of the validation set improved
        if not self.args.evaluate:
            '''
            print('predicted prob: {:e} \t actual prob: {:e}'.format(
                predicted_prob/(batch_idx+1), actual_prob/(batch_idx+1)))
            if actual_prob <= predicted_prob:
                print('beam search does not work')
            '''
            #
            if results['per_class/f1'] > self.best_f1_score:
                print('per-class f1 increased ({:.4f} --> {:.4f}). saving model ...'.format(
                    self.best_f1_score, results['per_class/f1']))
                # save the model's best result on the 'checkpoints' folder
                self.save_model()
                #
                lines = ['Epoch: ' + str(epoch+1),
                         N_plus,
                         per_class_metrics,
                         per_image_metrics]
                with open(self.args.save_dir + 'Corel-5k_validation_results.txt', 'w') as f:
                    f.write('\n'.join(lines))
                f.close()
                #
                self.best_f1_score = results['per_class/f1']

    def beam_search_validation(self, dataloader, beam_width):
        total_outputs = []
        total_targets = []

        for batch_idx, (images, binary_annotations, _, _) in enumerate(tqdm(dataloader)):

            images = images.to(device)
            binary_annotations = binary_annotations.to(device)

            outputs = annotate_batch_beam_search(self.args,
                                                 self.cnn,
                                                 self.lstm,
                                                 images,
                                                 self.word_map,
                                                 beam_width)

            outputs_ohv = convert_to_one_hot(outputs,
                                             len(self.classes),
                                             self.word_map,
                                             indexes=True)

            total_outputs.append(outputs_ohv)
            total_targets.append(binary_annotations)

        results = self.metrics.calculate_metrics(
            torch.cat(total_targets),
            torch.cat(total_outputs))
        N_plus, per_class_metrics, per_image_metrics = self.PR_RC_F1_Nplus(
            results)
        print(N_plus)
        print(per_class_metrics)
        print(per_image_metrics)

    def train_iteration(self):
        print('==> Start of Training ...')
        for epoch in range(self.args.epochs):
            start = timeit.default_timer()
            self.train(self.train_loader, epoch)
            self.validation(self.validation_loader, epoch)
            print('CNN LR {:.1e}'.format(
                self.scheduler_cnn.get_last_lr()[0]))
            print('LSTM LR {:.1e}'.format(
                self.scheduler_lstm.get_last_lr()[0]))
            stop = timeit.default_timer()
            print('time: {:.3f}'.format(stop - start))
            # early stop
            if epoch == 39:
                print('Early stop is active')
                break
        print('==> End of training ...')
