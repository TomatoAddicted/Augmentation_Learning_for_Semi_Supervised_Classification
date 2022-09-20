# Augmentation Learning for Semi Supervised Classification

This is the repository for the paper "Augmentation Learning for Semi Supervised Classification" published at GCPR 2022. The code was further used for a master thesis.

The paper can be found here: https://arxiv.org/abs/2208.01956

Abstract: Recently, a number of new Semi-Supervised Learning methods have emerged. As the accuracy for ImageNet and similar datasets increased over time, the performance on tasks beyond the classification of natural images is yet to be explored. Most Semi-Supervised Learning methods rely on a carefully manually designed data augmentation pipeline that is not transferable for learning on images of other domains. In this work, we propose a Semi-Supervised Learning method that automatically selects the most effective data augmentation policy for a particular dataset. We build upon the Fixmatch method and extend it with meta-learning of augmentations. The augmentation is learned in additional training before the classification training and makes use of bi-level optimization, to optimize the augmentation policy and maximize accuracy. We evaluate our approach on two domain-specific datasets, containing satellite images and hand-drawn sketches, and obtain state-of-the-art results. We further investigate in an ablation the different parameters relevant for learning augmentation policies and show how policy learning can be used to adapt augmentations to datasets beyond ImageNet.

Authors: Tim Frommknecht, Pedro Alves Zipf, Quanfu Fan, Nina Shvetsova, Hilde Kuehne

Institutes: Goethe University Frankfurt, MIT-IBM Watson AI Lab

The code contains fractions of the following repositories:
- https://github.com/kekmodel/FixMatch-pytorch
- https://github.com/VDIGPKU/DADA
