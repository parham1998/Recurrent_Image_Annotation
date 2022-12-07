# =============================================================================
# Import required libraries
# =============================================================================
import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.utils import shuffle


# =============================================================================
# Create annotation dataset
# =============================================================================
class AnnotationDataset(torch.utils.data.Dataset):
    '''
        image dim: (batch-size, 3, image-size, image-size)
        binary-annotations dim: (batch-size, (number of classes))
        annotations-X dim: (batch-size, max-seq-len + 1)
        label_lengths dim: (batch-size)
    '''

    def __init__(self,
                 root,
                 annotation_path,
                 max_seq_len,
                 sorted_labels=None,
                 transforms=None):
        self.root = root
        # maximum number of annotated labels + START and STOP
        self.max_seq_len = max_seq_len + 2
        self.transforms = transforms
        #
        with open(annotation_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        samples = shuffle(samples, random_state=0)
        self.classes = json_data['labels']
        #
        self.imgs = []
        self.annotations = []
        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annotations.append(sample['image_labels'])
        # converting all labels of each image into a binary array
        # of the class length
        self.binary_annotations = []
        for idx in range(len(self.annotations)):
            item = self.annotations[idx]
            vector = [cls in item for cls in self.classes]
            self.binary_annotations.append(np.array(vector, dtype=float))
        # sorting each label set according to rare-first ordering
        # the rare-first order put the rarer label before
        # the more frequent ones (based on label frequency in the dataset)
        if sorted_labels is not None:
            for idx in range(len(self.annotations)):
                self.annotations[idx] = sorted(
                    self.annotations[idx], key=lambda x: sorted_labels[x])
        #
        classes_new = self.classes.copy()
        classes_new.append('stop')
        classes_new.append('start')
        self.word_map = {cl: i for i, cl in enumerate(classes_new)}

    def __getitem__(self, idx):
        # image
        img_path = os.path.join(self.root, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        binary_annotations = torch.tensor(self.binary_annotations[idx])
        # annotations
        annotations_X = []
        annotations_X.append(self.word_map['start'])
        for c in self.annotations[idx]:
            annotations_X.append(self.word_map[c])
        for _ in range(self.max_seq_len - len(self.annotations[idx])):
            annotations_X.append(self.word_map['stop'])
        annotations_X = torch.tensor(annotations_X)
        label_lengths = len(self.annotations[idx])
        return image, binary_annotations, annotations_X, label_lengths

    def __len__(self):
        return len(self.imgs)


# =============================================================================
# Make data loader
# =============================================================================
def get_mean_std():
    mean = [0.3928, 0.4079, 0.3531]
    std = [0.2559, 0.2436, 0.2544]
    return mean, std


def get_transforms(args):
    mean, std = get_mean_std()
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])
    transform_validation = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])
    return transform_train, transform_validation


def sort_labels(annotation_path):
    ''' 
        sorting all labels in training data based on their frequency 
    '''

    with open(annotation_path) as fp:
        json_data = json.load(fp)
    samples = json_data['samples']
    #
    annotations = []
    for sample in samples:
        annotations.append(sample['image_labels'])
    all_annotations = [
        item for sublist in annotations for item in sublist]
    num_labels = {i: all_annotations.count(i) for i in all_annotations}
    return {k: v for k, v in sorted(
        num_labels.items(), key=lambda item: item[1])}


def make_data_loader(args):
    root_dir = args.data_root_dir
    transform_train, transform_validation = get_transforms(args)
    if args.sort:
        sorted_labels = sort_labels(os.path.join(root_dir, 'train.json'))
    else:
        sorted_labels = None
    #
    train_set = AnnotationDataset(root=os.path.join(root_dir, 'images'),
                                  annotation_path=os.path.join(
                                      root_dir, 'train.json'),
                                  max_seq_len=args.max_seq_len,
                                  sorted_labels=sorted_labels,
                                  transforms=transform_train)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    #
    validation_set = AnnotationDataset(root=os.path.join(root_dir, 'images'),
                                       annotation_path=os.path.join(
                                           root_dir, 'test.json'),
                                       max_seq_len=args.max_seq_len,
                                       sorted_labels=sorted_labels,
                                       transforms=transform_validation)
    validation_loader = DataLoader(validation_set,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=False)
    # all vocabulary except START and STOP 
    classes = train_set.classes
    #
    classes_new = classes.copy()
    classes_new.append('stop')
    classes_new.append('start')
    word_map = {cl: i for i, cl in enumerate(classes_new)}
    
    return train_loader, validation_loader, classes, word_map


# =============================================================================
# Word embedding
# =============================================================================
def word_embedding(glove_path, classes):
    '''
        embedding each class including START and STOP
    '''

    with open(glove_path, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)
    #
    emb = []
    classes_plus = classes.copy()
    classes_plus.append('stop')
    classes_plus.append('start')
    for word in classes_plus:
        emb.append(word_to_vec_map[word])
    return torch.from_numpy(np.array(emb))
