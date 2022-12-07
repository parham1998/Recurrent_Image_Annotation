# =============================================================================
# Import required libraries
# =============================================================================
import torch
import torch.nn.functional as F

from utils import init_input

# checking the availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# just for attention model
def annotate_image_beam_search(cnn,
                               lstm,
                               image,
                               word_map,
                               beam_size=1):
    k = beam_size
    #
    cnn.eval()
    lstm.eval()
    # CNN
    image_features, fc_features = cnn(image.unsqueeze(0))
    enc_image_size = image_features.size(1)
    encoder_dim = image_features.size(3)
    # flatten image
    # (1, enc-image-size * enc-image-size, encoder-dim)
    image_features = image_features.view(1, -1, encoder_dim)
    # the problem will be treated as having k batches
    # (k, enc-image-size * enc-image-size, encoder-dim)
    image_features = image_features.expand(
        k, image_features.size(1), encoder_dim)
    # (k, encoder-dim)
    fc_features = fc_features.expand(k, encoder_dim)
    # tensor to store top k previous words at each step; now they're just START
    # (k)
    annotations_X_i = init_input(k, word_map)
    # tensor to store top k sequences; now they're just START
    # (k, 1)
    seqs = annotations_X_i.unsqueeze(1)
    # tensor to store top k sequences' outputs; now they're just 0
    # (k, 1)
    top_k_outputs = torch.zeros(k, 1).to(device)
    # tensor to store top k sequences' alphas; now they're just 1s
    # (k, 1, enc-image-size, enc-image-size)
    seqs_alpha = torch.ones(k, 1, enc_image_size,
                            enc_image_size).to(device)
    # lists to store completed sequences, their alphas and outputs
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_outputs = list()

    # start decoding
    step = 1
    ht, ct = lstm.init_state(fc_features)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit STOP
    while True:
        # (s, encoder-dim), (s, enc-image-size * enc-image-size)
        attention_weighted_encoding, alpha = lstm.attention(
            image_features, ht)
        # (s, enc-image-size, enc-image-size)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        # (s, encoder-dim)
        gate = lstm.sigmoid(lstm.f_beta(ht))
        attention_weighted_encoding = gate * attention_weighted_encoding
        #
        yhat, ht, ct = lstm.annotator_output(
            annotations_X_i, attention_weighted_encoding, (ht, ct))
        outputs = F.log_softmax(yhat, dim=1)
        #
        outputs = top_k_outputs.expand_as(outputs) + outputs
        # for the first step, all k points will have the same outputs (since same (beam-size) previous words, h, c)
        if step == 1:
            top_k_outputs, top_k_words = outputs[0].topk(
                k, 0, True, True)
        else:
            # unroll and find top outputs, and their unrolled indices
            top_k_outputs, top_k_words = outputs.view(
                -1).topk(k, 0, True, True)
        # convert unrolled indices to actual indices of outputs
        prev_word_inds = torch.div(
            top_k_words, outputs.size(1), rounding_mode='floor')
        next_word_inds = top_k_words % outputs.size(1)
        # add new words to sequences, alphas
        # (s, step+1)
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        # (s, step+1, enc-image-size, enc-image-size)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)
        # which sequences are incomplete (didn't reach STOP)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['stop']]
        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))
        # set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_outputs.extend(top_k_outputs[complete_inds])
        # reduce beam length accordingly
        k -= len(complete_inds)
        # proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        ht = ht[prev_word_inds[incomplete_inds]]
        ct = ct[prev_word_inds[incomplete_inds]]
        image_features = image_features[prev_word_inds[incomplete_inds]]
        fc_features = fc_features[prev_word_inds[incomplete_inds]]
        top_k_outputs = top_k_outputs[incomplete_inds].unsqueeze(1)
        annotations_X_i = next_word_inds[incomplete_inds]

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_outputs.index(max(complete_seqs_outputs))
    seq = complete_seqs[i]
    # remove START and STOP
    seq.pop(0)
    seq.pop(-1)
    seq = torch.LongTensor(seq)
    seq = seq.reshape(1, -1)
    #
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def annotate_batch_beam_search(args,
                               cnn,
                               lstm,
                               images,
                               word_map,
                               beam_width=1):

    batch_size = images.size(0)
    #
    hypotheses_tensor = word_map['stop'] * torch.ones(
        (batch_size, args.max_seq_len + 2)).long()
    #
    cnn.eval()
    lstm.eval()
    # for each image
    for i in range(batch_size):
        k = beam_width
        # CNN
        if (args.method == 'RIA' or args.method == 'SR-CNN-RNN'):
            fc_features = cnn(images[i].unsqueeze(0))
            encoder_dim = fc_features.size(1)
        elif args.method == 'Attention':
            image_features, fc_features = cnn(images[i].unsqueeze(0))
            encoder_dim = fc_features.size(1)
            #
            image_features = image_features.view(1, -1, encoder_dim)
            #
            image_features = image_features.expand(
                k, image_features.size(1), encoder_dim)
        #
        fc_features = fc_features.expand(k, encoder_dim)
        #
        annotations_X_i = init_input(k, word_map)
        #
        seqs = annotations_X_i.unsqueeze(1)
        #
        top_k_outputs = torch.zeros(k, 1).to(device)
        #
        complete_seqs = list()
        complete_seqs_outputs = list()
        #
        step = 1
        ht, ct = lstm.init_state(fc_features)
        #
        while True:
            if (args.method == 'RIA' or args.method == 'SR-CNN-RNN'):
                yhat, ht, ct = lstm.annotator_output(
                    annotations_X_i, (ht, ct))
            elif args.method == 'Attention':
                attention_weighted_encoding, _ = lstm.attention(
                    image_features, ht)
                #
                gate = lstm.sigmoid(lstm.f_beta(ht))
                attention_weighted_encoding = gate * attention_weighted_encoding
                #
                yhat, ht, ct = lstm.annotator_output(
                    annotations_X_i, attention_weighted_encoding, (ht, ct))
            #
            outputs = F.log_softmax(yhat, dim=1)
            #
            outputs = top_k_outputs.expand_as(outputs) + outputs
            #
            if step == 1:
                top_k_outputs, top_k_words = outputs[0].topk(
                    k, 0, True, True)
            else:
                top_k_outputs, top_k_words = outputs.view(
                    -1).topk(k, 0, True, True)
            #
            prev_word_inds = torch.div(
                top_k_words, outputs.size(1), rounding_mode='floor')
            next_word_inds = top_k_words % outputs.size(1)
            #
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            #
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['stop']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))
            # set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_outputs.extend(top_k_outputs[complete_inds])
            k -= len(complete_inds)
            #
            if k == 0:
                break
            #
            seqs = seqs[incomplete_inds]
            ht = ht[prev_word_inds[incomplete_inds]]
            ct = ct[prev_word_inds[incomplete_inds]]
            if args.method == 'Attention':
                image_features = image_features[prev_word_inds[incomplete_inds]]
            fc_features = fc_features[prev_word_inds[incomplete_inds]]
            top_k_outputs = top_k_outputs[incomplete_inds].unsqueeze(1)
            annotations_X_i = next_word_inds[incomplete_inds]
            #
            if step > 50:
                break
            step += 1

        idx = complete_seqs_outputs.index(max(complete_seqs_outputs))
        seq = complete_seqs[idx]
        # remove START
        seq.pop(0)
        #
        hypotheses_tensor[i][0:len(seq)] = torch.LongTensor(seq)

    return hypotheses_tensor
