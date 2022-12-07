# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
import skimage.transform

from datasets import get_mean_std

# checking the availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(args, tensor):
    mean, std = get_mean_std()
    #
    tensor = tensor.numpy()
    # img shape => (3, h, w), img shape after transpose => (h, w, 3)
    tensor = tensor.transpose(1, 2, 0)
    image = tensor * np.array(std) + np.array(mean)
    image = image.clip(0, 1)
    plt.imshow(image)


def convertBinaryAnnotationsToClasses(annotations, classes):
    labels = []
    annotations = np.array(annotations, dtype='int').tolist()
    for i in range(len(classes)):
        if annotations[i] == 1:
            labels.append(classes[i])
    return labels


def init_input(dim, word_map):
    annotations_X_i = word_map['start'] * torch.ones((dim)).long()
    return annotations_X_i.to(device)


def convert_to_one_hot(x, num_classes, word_map, indexes=False):
    if indexes:
        idxs = x
    else:
        x = x.transpose(1, 2)
        _, idxs = torch.max(x, 1)
    #
    batch_size = idxs.size(0)
    preds = torch.zeros((batch_size, num_classes)).long()
    for i in range(batch_size):
        preds_image = []
        for t in range(idxs.size(1)):
            if idxs[i][t] == word_map['stop']:
                break
            preds_image.append(idxs[i][t])
        preds[i, preds_image] = 1
    return preds.to(device)


# plot one batch of images with grand-truth and predicted annotations
def batch_plot(args,
               images,
               outputs,
               annotations,
               classes,
               word_map):
    #
    outputs = convert_to_one_hot(outputs,
                                 len(classes),
                                 word_map,
                                 indexes=True)

    fig = plt.figure(figsize=(80, 30))
    for i in np.arange(args.batch_size):
        ax = fig.add_subplot(4, 8, i+1)
        imshow(args, images[i].cpu())
        #
        gt_anno = convertBinaryAnnotationsToClasses(annotations[i], classes)
        #
        o = np.array(outputs.cpu(), dtype='int')
        pre_anno = convertBinaryAnnotationsToClasses(o[i], classes)
        #
        string_gt = 'GT: '
        string_pre = 'Pre: '
        if len(gt_anno) != 0:
            for ele in gt_anno:
                string_gt += ele if ele == gt_anno[-1] else ele + ' - '
        #
        if len(pre_anno) != 0:
            for ele in pre_anno:
                string_pre += ele if ele == pre_anno[-1] else ele + ' - '

        ax.set_title(string_gt + '\n' + string_pre)
        plt.savefig(args.data_root_dir + 'batch_plot.jpg')


def visualize_att(args,
                  image,
                  output,
                  alphas,
                  classes,
                  word_map,
                  smooth=True):

    words = ['START']
    for i in range(output.size(1)):
        idx = F.one_hot(output[0, i], len(classes))
        words.extend(convertBinaryAnnotationsToClasses(idx.cpu(), classes))
    words.append('STOP')
    #
    oc_set = set()
    duplicate_idxs = []
    for idx, val in enumerate(words):
        if val not in oc_set:
            oc_set.add(val)
        else:
            duplicate_idxs.append(idx)
    # remove duplicate indexes
    words = [ele for idx, ele in enumerate(words) if idx not in duplicate_idxs]
    alphas = [ele for idx, ele in enumerate(
        alphas) if idx not in duplicate_idxs]
    alphas = torch.FloatTensor(alphas)

    plt.figure(figsize=(40, 20))
    for t in range(len(words)):
        if t > 50:
            break
        #
        plt.subplot(np.int64(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black',
                 backgroundcolor='white', fontsize=20)
        imshow(args, image.cpu())
        #
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(
                current_alpha.numpy(), upscale=32, sigma=8)
        else:
            alpha = skimage.transform.resize(
                current_alpha.numpy(), [14 * 32, 14 * 32])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()
