# Data Augmentation — Resources (Project 2421)

Ingested 2026-09-15 from https://intranet-dlh.hbtn.io/projects/2421.
Sections on page: Read-or-watch (6 items). No Definitions-to-skim or
References sections present — nothing to gloss or link-check.

## Data Augmentation: train deep learning models with less data

URL: https://www.labellerr.com/blog/data-augmentation-train-deep-learning-models-with-limited-data/  ·  Date: 2022-11-07 (byline)  ·  Status: summary

Introductory vendor blog arguing augmentation substitutes for costly
collection/labeling when data is scarce, acting as regularization that
reduces overfitting and class-imbalance pain.

- augmentation = small transforms of existing samples (or generative-model samples) to widen the training set
- core image ops: flip (horizontal; vertical via horizontal + 180° rotation), rotation (right angles preserve size; small angles change canvas), scale in/out, random crop, X/Y translation, Gaussian noise against high-frequency overfitting
- motivation: deep models have many parameters; collection costs money, labor, compute, time
- benefits claimed: better generalization, lower collection cost, handles limited/imbalanced data
- caveat echoed: prefer high-quality, high-quantity data when obtainable; augmentation complements it

## A Complete Guide to Data Augmentation

URL: https://www.datacamp.com/tutorial/complete-guide-data-augmentation  ·  Date: 2022-11-23 (byline)  ·  Status: summary

Broad tutorial covering image, text, and audio augmentation plus a
hands-on Keras/TensorFlow cats-vs-dogs walkthrough and a survey of
standalone tooling.

- augmented data (transforms of real samples) vs synthetic data (newly generated, e.g. GANs); both fight overfitting and small/imbalanced sets
- image ops: geometric (flip, crop, rotate, stretch, zoom), color-space (channels, contrast, brightness), kernel filters (blur/sharpen), random erasing, image mixing
- audio ops: noise injection, shifting, speed/pitch changes; text ops: shuffling, synonym replacement, syntax-tree paraphrase, random insert/delete
- Keras patterns: Sequential preprocessing layers (Resizing, Rescaling, RandomFlip, RandomRotation) active only in fit, not evaluate/predict; or Dataset.map with training=True plus prefetch
- tf.image pattern: fine-grained stateless ops (flip_left_right, rgb_to_grayscale, adjust_brightness/saturation, central_crop, rot90, stateless_random_*) mapped over the dataset
- limits/ethics: source bias persists, QA is expensive, GAN hi-res is hard, document transforms, watch fairness/privacy/compliance

## tf.image

URL: https://www.tensorflow.org/api_docs/python/tf/image  ·  Date: 2024-04-26 (page last modified)  ·  Status: summary

API reference for the TensorFlow image-ops module; the direct source
for every Task 0-5 primitive (flip, crop, rotate, contrast, brightness,
hue) and the random/stateless variants used for stochastic augmentation.

- deterministic ops used by the tasks: flip_left_right, flip_up_down, central_crop, crop_to_bounding_box, rot90, adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation, resize
- stochastic ops: random_brightness, random_contrast, random_crop, random_flip_left_right/up_down, random_hue, random_saturation, random_jpeg_quality, plus stateless_random_* taking explicit seeds
- dtype helpers matter: convert_image_dtype, grayscale_to_rgb, rgb_to_grayscale/hsv/yiq/yuv and back
- geometry/quality extras: crop_and_resize, pad_to_bounding_box, resize_with_crop_or_pad, sample_distorted_bounding_box, sobel_edges, psnr/ssim metrics
- checker note: tasks call the plain (non-random) op for 0-2 and the random op for 3-5; keep the exact tf.image.* name per spec

## Data augmentation

URL: https://www.tensorflow.org/tutorials/images/data_augmentation  ·  Date: 2024-07-19 (page last modified)  ·  Status: summary

Official tutorial showing the two supported augmentation styles on
tf_flowers: Keras preprocessing layers and raw tf.image ops, composed
with tf.data (map, shuffle, batch, prefetch).

- style 1: Keras layers (Resizing, Rescaling, RandomFlip, RandomRotation) inside or alongside the model
- style 2: tf.image functions (flip_left_right, rgb_to_grayscale, adjust_brightness, central_crop, stateless_random_*) applied in a Dataset.map augment function
- pipeline shape: download via tensorflow-datasets, map resize/rescale, optional shuffle/batch, augment training split only, prefetch with AUTOTUNE
- conceptual point: transforms must be random yet realistic (e.g. rotation) to raise training diversity
- project parallel: Task 0 mains load stanford_dogs via tfds, set a seed, and imshow the transformed sample

## Image Data Augmentation using TensorFlow

URL: https://medium.com/@speaktoharisudhan/image-data-augmentation-using-tensorflow-46d884f420f6  ·  Date: 2024-02-23 (byline)  ·  Status: summary

Short practitioner piece mapping each project operation to its
tf.image call and its Keras preprocessing-layer counterpart, framed by
the standard limited-data/overfitting/imbalance/domain-shift motives.

- tf.image mapping: rot90, flip_left_right/flip_up_down, resize, adjust_brightness/contrast, adjust_hue, adjust_saturation
- Keras mapping: RandomFlip, RandomRotation, RandomContrast, RandomBrightness, RandomZoom, Rescaling, RandomTranslation inside a Sequential model
- why-augment recap: scarce labels, overfitting as memorization, minority-class balancing, domain adaptation via invariant features, robustness to lighting/orientation/background shifts
- practical tip: rescale pixels (1/255) before/while augmenting so imshow ranges stay valid
- closest single-page cheat sheet to Tasks 0-5 signatures

## Automating Data Augmentation: Practice, Theory and New Direction

URL: http://ai.stanford.edu/blog/data-augmentation/  ·  Date: 2020-04-24 (byline)  ·  Status: summary

Stanford SAIL research survey (Sharon Y. Li) on learning augmentation
policies instead of hand-tuning them, plus theory (augmentation as
kernel/variance regularization) and model patching for subgroup failures.

- hand-tuned transform sequences vary widely; learned search (TANDA adversarial TF-sequence generator, AutoAugment/RandAugment/Adversarial AutoAugment optimizing validation accuracy) beats heuristics
- theory lens: Markov-chain augmentation with kNN behaves like a kernel classifier (averaged transformed features + variance regularization); in over-parametrized linear models, label-invariant transforms add out-of-span information and mixup acts as regularization
- uncertainty sampling: training on highest-loss transforms beat RandAugment on CIFAR-10/100 in their experiments
- model patching (CLAMP): learn inter-subgroup transforms (e.g. add/remove bandages in skin-lesion images) and retrain with a group-robust objective to close subgroup gaps
- directly answers the project's fifth learning objective (ML-automated augmentation) and the advanced Task 6 blog-post topic

## Quiz Hooks

- data augmentation — artificially widening the training set with transformed or generated variants of real samples
- overfitting — memorizing training noise/detail; augmentation regularizes by making memorization harder
- horizontal flip — tf.image.flip_left_right; vertical flip equals horizontal flip plus 180° rotation
- rot90 — rotates an image 90° counter-clockwise (Task 2)
- random crop — sample a sub-window then resize back (Task 1); central_crop takes the middle fraction
- contrast vs brightness vs hue vs saturation — tf.image.adjust_contrast / adjust_brightness / adjust_hue / adjust_saturation (Tasks 3-5)
- Rescaling(1/255) — normalizes pixel range before display/training
- augmentation active in fit only — Keras preprocessing layers are inert in evaluate/predict
- Dataset.map + prefetch(AUTOTUNE) — applies augmentation across the pipeline with overlapped GPU feeding
- stateless_random_* — seeded random ops for reproducible augmentation
- TANDA — adversarially learned transformation-function sequences producing realistic augmentations
- AutoAugment / RandAugment — learned policies optimizing validation accuracy; reduced-cost successors
- mixup as regularization — shrinks training-data weight relative to L2 in the linear analysis
- model patching / CLAMP — learn inter-subgroup transforms and retrain group-robustly to fix deployed subgroup gaps
