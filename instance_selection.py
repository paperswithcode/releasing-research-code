import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from embeddings import *
from density_estimators import *


def get_embedder(embedding):
    if embedding == 'inceptionv3':
        embedder = Inceptionv3Embedding().eval().cuda()
    elif embedding == 'resnet50':
        embedder = ResNet50Embedding().eval().cuda()
    elif embedding == 'places365':
        embedder = Places365Embedding().eval().cuda()
    elif embedding == 'resnextwsl':
        embedder = ResNextWSL().eval().cuda()
    elif embedding == 'swav':
        embedder = SwAVEmbedding().eval().cuda()

    if torch.cuda.current_device() > 1:
        embedder = nn.DataParallel(embedder)
    return embedder


def get_embeddings_from_loader(dataloader,
                               embedder,
                               return_labels=False,
                               verbose=False):
    embeddings = []
    labels = []

    with torch.no_grad():
        if verbose:
            dataloader = tqdm(dataloader, desc='Extracting embeddings')
        for data in dataloader:
            if len(data) == 3:
                images, label, _ = data
                images = images.cuda()  
            else:
                images = data.cuda()
                labels.append(torch.zeros(len(images)))

            embed = embedder(images)
            embeddings.append(embed.cpu())
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    if return_labels:
        return embeddings, labels
    else:
        return embeddings


def get_keep_indices(embeddings, 
                     labels, 
                     density_measure, 
                     retention_ratio, 
                     verbose=False):
    keep_indices = []
    scores_all = []
    unique_labels = torch.unique(labels)
    if verbose:
        unique_labels = tqdm(unique_labels, desc='Scoring instances')

    for label in unique_labels:
        class_indices = torch.where(labels == label)[0]
        class_embeddings = embeddings[class_indices]

        if density_measure == 'ppca':
            scores = PPCA(class_embeddings)
        elif density_measure == 'gaussian':
            scores = GaussianModel(class_embeddings)
        elif density_measure == 'nn_dist':
            # make negative so that larger values are better
            scores = -compute_nearest_neighbour_distances(class_embeddings,
                                                          nearest_k=5)

        cutoff = np.percentile(scores, (100 - retention_ratio))
        keep_mask = torch.from_numpy(scores > cutoff).bool()
        keep_indices.append(class_indices[keep_mask])
        scores_all.append(torch.from_numpy(scores))
    keep_indices = torch.cat(keep_indices, dim=0)
    scores_all = torch.cat(scores_all, dim=0)
    return keep_indices, scores_all


def select_instances(dataset,
                     retention_ratio,
                     embedding='inceptionv3',
                     density_measure='gaussian',
                     scores_filepath=None,
                     labels_filepath=None,
                     indices_filepath=None,
                     batch_size=128,
                     num_workers=4):
    """
    Args:
        dataset (Dataset): dataset to be subsampled with instance selection.
        retention_ratio (float): percentage of the dataset to keep.
        embedding (str): embedding function for extracting image features. 
            Options are 'inceptionv3', 'resnet50', 'places365', 'resnextwsl', 
            and 'swav'.
        density_measure (str): scoring function to use when determining whether 
            to select instances. Options are 'ppca', 'gaussian', or 'nn_dist'.
        indices_filepath (str): filepath for saving indices so that they don't
            need to be recomputed each time. Should have .pkl file extension.
        batch_size (int): how many samples per batch to load.
        num_workers (int): how many subprocesses to use for data loading.
    Returns:
        instance_selected_dataset (Subset): subset of original dataset 
            containing the best scoring instances.
    """
    
    assert (retention_ratio > 0) and retention_ratio <= 100, \
        'retention_ratio should be betwee 0 and 100'

    if retention_ratio == 100:
        if scores_filepath is not None:
            if os.path.exists(scores_filepath):
                scores_all = torch.load(scores_filepath)
                labels_all = torch.load(labels_filepath)

                print('Retention ratio = 100, skipping dataset reduction')
                return dataset, scores_all, labels_all

    if indices_filepath is not None:
        if os.path.exists(indices_filepath):
            keep_indices = torch.load(indices_filepath)
            scores_all = torch.load(scores_filepath)
            labels_all = torch.load(labels_filepath)
            return Subset(dataset, keep_indices), scores_all, labels_all

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers,
                            pin_memory=True)

    embedder = get_embedder(embedding)
    embeddings, labels = get_embeddings_from_loader(dataloader,
                                                    embedder,
                                                    return_labels=True,
                                                    verbose=True)
    torch.save(labels, labels_filepath)
    labels_all = labels
    keep_indices, scores_all = get_keep_indices(embeddings,
                                    labels,
                                    density_measure,
                                    retention_ratio=retention_ratio,
                                    verbose=True)

    torch.save(scores_all, scores_filepath)
    if indices_filepath is not None:
        torch.save(keep_indices, indices_filepath)

    if retention_ratio == 100:
        print('Retention ratio = 100, skipping dataset reduction')
        return dataset, scores_all, labels_all
    else:
        return Subset(dataset, keep_indices), scores_all, labels_all
