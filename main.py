# =============================================================================
# Import required libraries
# =============================================================================
import argparse
import os
import numpy as np

import torch
from torch import nn

from datasets import *
from utils import *
from models import *
from models_attention import *
from beam_search import *
from engine import Engine

# checking the availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Define hyperparameters
# =============================================================================
parser = argparse.ArgumentParser(
    description='PyTorch Training for Automatic Image Annotation')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--data_root_dir', default='./Corel-5k/', type=str)
parser.add_argument('--image-size', default=448, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--encoder-lr', default=1e-4, type=float)
parser.add_argument('--decoder-lr', default=1e-4, type=float)
parser.add_argument('--num-workers', default=2, type=int,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--beam-width', default=10, type=int)
parser.add_argument('--max-seq-len', metavar='NAME', type=int,
                    help='maximum sequence length (e.g. 5)')
parser.add_argument('--method', metavar='NAME', help='method name (e.g. RIA)')
parser.add_argument('--order-free', metavar='NAME')
parser.add_argument('--sort', dest='sort', action='store_true',
                    help='sorting labels by frequency')
parser.add_argument('--is_glove', dest='is_glove', action='store_true',
                    help='utilizing GLOVE pre-trained weights in the embedding matrix')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluation of the model on the validation set')
parser.add_argument(
    '--save_dir', default='./checkpoints/', type=str, help='save path')


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    is_train = True if not args.evaluate else False

    train_loader, validation_loader, classes, word_map = make_data_loader(args)

    #
    if (args.method == 'RIA' or args.method == 'SR-CNN-RNN'):
        input_dim = 260+2
        hidden_dim = 2048
        output_dim = 260+1
        cnn = TResNet(args, pretrained=is_train, num_classes=len(classes))
    elif args.method == 'Attention':
        input_dim = 260+2
        hidden_dim = 2048
        output_dim = 260+1
        attention_dim = 1024
        cnn = TResNet_att(args, pretrained=is_train)
    #
    if args.is_glove:
        emb_dim = 300
        if os.path.exists('./glove/Corel-5k_glove.pkl'):
            glove_weights = torch.load('./glove/Corel-5k_glove.pkl')
        else:
            # download "glove.6B.300d.txt" and place it in glove folder
            glove_weights = word_embedding(
                './glove/glove.6B.300d.txt', classes)
            torch.save(glove_weights, './glove/Corel-5k_glove.pkl')
        #
        if (args.method == 'RIA' or args.method == 'SR-CNN-RNN'):
            lstm = Anotator(args,
                            input_size=input_dim,
                            hidden_size=hidden_dim,
                            output_size=output_dim,
                            num_classes=len(classes),
                            emb_size=emb_dim,
                            is_glove=args.is_glove,
                            glove_weights=glove_weights)
        elif args.method == 'Attention':
            lstm = Anotator_att(args,
                                input_size=input_dim,
                                hidden_size=hidden_dim,
                                output_size=output_dim,
                                attention_size=attention_dim,
                                emb_size=emb_dim,
                                is_glove=args.is_glove,
                                glove_weights=glove_weights)
    else:
        emb_dim = 1024
        if (args.method == 'RIA' or args.method == 'SR-CNN-RNN'):
            lstm = Anotator(args,
                            input_size=input_dim,
                            hidden_size=hidden_dim,
                            output_size=output_dim,
                            num_classes=len(classes),
                            emb_size=emb_dim,
                            is_glove=args.is_glove)
        elif args.method == 'Attention':
            lstm = Anotator_att(args,
                                input_size=input_dim,
                                hidden_size=hidden_dim,
                                output_size=output_dim,
                                attention_size=attention_dim,
                                emb_size=emb_dim,
                                is_glove=args.is_glove)

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MultiLabelSoftMarginLoss()
    criterion = (criterion_1, criterion_2)

    engine = Engine(args,
                    cnn,
                    lstm,
                    criterion,
                    train_loader,
                    validation_loader,
                    classes,
                    word_map)

    if is_train:
        engine.initialization()
        engine.train_iteration()
    else:
        engine.initialization()
        engine.load_model()
        engine.validation(dataloader=validation_loader)
        #
        print('Applying beam search algorithm')
        engine.beam_search_validation(dataloader=validation_loader,
                                      beam_width=args.beam_width)
        # show images and predicted labels
        images, binary_annotations, _, _ = iter(validation_loader).next()
        images = images.to(device)
        #
        if args.method == 'Attention':
            for i in range(0, 32, 2):
                output, alphas = annotate_image_beam_search(cnn,
                                                            lstm,
                                                            images[i],
                                                            word_map,
                                                            args.beam_width)
                visualize_att(args,
                              images[i],
                              output,
                              alphas,
                              classes,
                              word_map,
                              smooth=False)
        #
        outputs = annotate_batch_beam_search(args,
                                             cnn,
                                             lstm,
                                             images,
                                             word_map,
                                             args.beam_width)
        batch_plot(args,
                   images,
                   outputs,
                   binary_annotations,
                   classes,
                   word_map)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
