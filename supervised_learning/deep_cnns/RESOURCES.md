# Deep Convolutional Architectures — Resources

Project: Deep Convolutional Architectures (intranet 2305) · Dir: `supervised_learning/deep_cnns`
Ingested: 2026-09-16 · Session alive (health check passed, all 24 rltokens resolved in-context: 17 via page navigation, 7 arxiv via in-page fetch).
Sections on project page: Read or watch (24 items: 8 articles, 9 videos, 7 papers) · Definitions to skim (none — section absent) · References (none — section absent).

## Vanishing Gradient Problem

URL: https://en.wikipedia.org/wiki/Vanishing_gradient_problem  ·  Date: 2026-08-29 (last edited)  ·  Status: summary

Training deep nets with backprop multiplies many small derivatives along the backward path, so early-layer gradients shrink exponentially with depth (tanh/sigmoid derivatives live in [0,1]) while the mirror failure — repeated large factors — explodes them. Hochreiter's 1991 thesis named the problem; it bites both deep feedforward nets and RNNs unfolded through time. The page's fix catalog is the motivation for this whole project: ReLU-style activations, careful initialization, batch normalization, and — directly on point — residual and highway skip connections that give gradients a clear path back to early layers.

- Mechanism: n small factors multiplied → ~0 gradient; n large factors → overflow; early layers stall or destabilize.
- Scope: deep feedforward stacks and RNNs via backprop through time, not just one architecture.
- Degradation vs vanishing: related but distinct — plain nets can degrade (higher training error when deeper) even where gradients flow.
- Task link: the reason identity/projection blocks exist; every block here carries a shortcut for the gradient.
- Fix list to remember: ReLU, He/Xavier init, batch norm, skip/highway connections, LSTM gating for sequences.

## What does 1x1 convolution mean in a neural network?

URL: https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network  ·  Date: 2016-02-04 (publication)  ·  Status: summary

A StackExchange Q&A (accepted answer by Indie AI, 2016-02-05) explaining that a 1x1 convolution leaves spatial dims untouched and operates purely across channels: with F1 filters over F input maps it re-mixes each pixel's channel vector, so it raises dimensionality when F1 > F and compresses it when F1 < F. The accepted answer quotes the GoogLeNet paper's own rationale — 1x1 reductions before costly 3x3/5x5 convolutions, with ReLU making each reduction dual-purpose (compression plus nonlinearity). Follow-up answers add the cost intuition (a 256→64→256 sandwich runs several times faster than a direct wide convolution) and the per-pixel fully-connected-network mental model.

- 1x1 conv = channel mixer: spatial size unchanged, depth becomes the filter count.
- Reduction role: shrink channels first, then run the expensive spatial convolution cheaply.
- Dual purpose: dimensionality cut plus one more ReLU nonlinearity per block.
- Task link: the F11 and F12 1x1 layers in identity/projection blocks are exactly these bottlenecks.
- ResNet-50 connection: every stage funnels through 1x1 → 3x3 → 1x1 for the same reason.

## Review: GoogLeNet (Inception v1) — Winner of ILSVRC 2014

URL: https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7  ·  Date: 2018-08-24 (publication)  ·  Status: summary

Sik-Ho Tsang's walkthrough of the 2015 CVPR GoogLeNet paper (ILSVRC 2014 winner): a 22-layer network built from Inception modules that run 1x1, 3x3, 5x5 and pooling side by side and concatenate, with 1x1 bottlenecks (a Network-in-Network borrowing) keeping the compute budget near 1.5B multiply-adds — about 12x fewer parameters than AlexNet at higher accuracy. Two more NIN borrowings shape the design: global average pooling replaces the parameter-heavy fully connected tail (zero weights, ~0.6% top-1 gain, less overfitting), and auxiliary classifiers tapped at intermediate layers (loss weight 0.3, training only) fight vanishing gradients. The 6.67% top-5 result leans on 7-model ensembling plus 144-crop multi-scale testing, a reminder that evaluation protocol carries part of every leaderboard number.

- Inception module: parallel multi-scale filters + pooling, concatenated — no committing to one kernel size.
- Bottleneck math: a 5x5 branch drops from ~113M ops to ~5M ops via a 1x1 reduction first.
- Global average pooling: replaces FC layers, kills tens of millions of weights, regularizes.
- Auxiliary heads: intermediate softmax branches supervise mid-network features during training only.
- Testing protocol matters: ensemble + multi-scale + multi-crop is a large share of the headline error rate.

## Residual Neural Network

URL: https://en.wikipedia.org/wiki/Residual_neural_network  ·  Date: 2026-09-14 (last edited)  ·  Status: summary

The reference article on the residual motif x → F(x) + x: instead of fitting a desired mapping directly, stacked layers fit the residual F(x) = H(x) − x, with an identity skip carrying the input forward untouched. Because the forward signal accumulates additively across blocks, the backward gradient always contains an unattenuated term from deeper layers, which is why hundreds of layers train stably. The page distinguishes our two task blocks precisely: the basic/bottleneck block (two 3x3s, or 1x1 → 3x3 → 1x1 with channel squeeze-and-restore, as in ResNet-50/101/152) for same-shape skips, and the projection variant y = F(x) + Mx with a learned 1x1 projection when dimensions change. It also notes the degradation framing (identity-by-default beats struggling-to-be-identity optimizers), pre-activation reorderings, and the lineage through Highway Networks and LSTM gating.

- Core identity: learn the residual F(x) = H(x) − x; zero weights recover the identity for free.
- Gradient math: ∂E/∂x_l always includes the later ∂E/∂x_L term additively — the anti-vanishing mechanism.
- Bottleneck block: 1x1 reduce, 3x3 process, 1x1 restore — the ResNet-50 unit and our task-0/task-1 shape.
- Projection connection: a 1x1 (optionally strided) convolution Mx on the shortcut when shapes differ — our task-1 shortcut.
- Scope note: the same motif now appears in transformers, AlphaGo/AlphaFold systems, not just vision.

## An Overview of ResNet and its Variants

URL: https://stephanosterburg.gitbook.io/scrapbook/coding/intro-to-tensorflow-for-ai-ml-and-dl/an-overview-of-resnet-and-its-variants  ·  Date: unknown  ·  Status: summary

A long survey (GitBook mirror of a ResNet review article) that starts from why depth alone fails — gradients shrink as they propagate back, so plain nets saturate then degrade — and presents the identity shortcut as the fix: if extra layers default to identity, a deeper net can never be worse than its shallow counterpart. It then tours the family: Highway Networks as gated shortcuts (ResNet is the open-gate special case, and clear gradient paths beat larger solution spaces), pre-activation blocks that let gradients flow unimpeded to 1000+ layers, ResNeXt's parallel equal-topology paths merged by addition with a new cardinality dimension, DenseNet's concatenation of all preceding feature maps for feature reuse and parameter efficiency, stochastic depth (drop whole layers in training, keep all at test — an ensemble view), and Veit et al.'s finding that ResNets behave as ensembles of shallow paths where most gradient flows through 9–18 effective layers.

- Identity-default argument: extra layers that can be identity never hurt; residuals make identity trivial (zero the weights).
- Clear highways beat big solution spaces: ungated ResNet matches or beats gated Highway nets.
- Cardinality (ResNeXt): widening via parallel identical paths beats going deeper or wider per parameter.
- Concatenation (DenseNet): preserve all feature maps instead of adding; reuse features, cut parameters.
- Ensemble view: a ResNet is 2^i implicit paths; deleting layers barely hurts, unlike VGG's single path.
- Task link: identity vs projection blocks, bottleneck 1x1-3x3-1x1, and why stage counts [3,4,6,3] train at all.

## Review: ResNet — Winner of ILSVRC 2015

URL: https://sh-tsang.medium.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8  ·  Date: 2018-09-15 (publication)  ·  Status: summary

Tsang's review of the 2016 CVPR ResNet paper: plain nets degrade (a 56-layer plain net trains worse than a 20-layer one on CIFAR-10), and the residual block H(x) = F(x) + x answers it — even with vanished weight gradients the identity x still carries signal backward. For shape-changing shortcuts the paper tests three options: zero-padding (A, no extra params), projection-only-where-needed (B), and all-projection (C, marginally best). The bottleneck redesign (1x1, 3x3, 1x1) turns ResNet-34 into ResNet-50 and scales to 101/152, with ResNet-152 holding lower FLOPs than VGG-16/19 despite 8x the depth. Numbers to anchor on: 3.57% top-5 on ImageNet (152-layer, 6-model ensemble, multi-scale), CIFAR-10 down to 6.43% at 110 layers, and a 28% relative COCO detection gain when ResNet-101 features replace VGG-16 in Faster R-CNN — evidence the features transfer.

- Degradation demo: deeper plain nets get higher training error, not just test error — an optimization failure.
- Shortcut options A/B/C: zero-pad vs projection; projections add parameters, help marginally.
- Bottleneck: 34-layer basic design becomes the 50/101/152 bottleneck family we implement.
- Batch norm after every conv plus 10-crop/multi-scale testing sit underneath the headline numbers.
- Transfer result: ResNet features lifted detection and segmentation wins, not just classification.

## Review: ResNeXt — 1st Runner Up in ILSVRC 2016

URL: https://medium.com/data-science/review-resnext-1st-runner-up-of-ilsvrc-2016-image-classification-15d7f17b42ac  ·  Date: 2018-12-09 (publication)  ·  Status: summary

Tsang's review of the 2017 CVPR ResNeXt paper (UCSD + FAIR, ILSVRC 2016 runner-up at 3.03% top-5): the ResNet bottleneck is split into C parallel identical paths whose outputs are summed — "Network-in-Neuron" — introducing cardinality C as a third scaling dimension beside depth and width. With C=32 paths of width d=4, ResNeXt-50 hits 22.2% top-1 against ResNet-50's 23.9% at matched complexity, and ablations show raising cardinality beats deepening or widening. Implementation-wise the block has three equivalent forms (split-transform-merge, early concatenation à la Inception-ResNet, grouped convolution), with grouped conv the practical choice; residual connections remain essential, since removing them degrades both ResNet and ResNeXt sharply.

- Cardinality: number of identical parallel paths; a new scaling axis alongside depth and width.
- Merge by addition (not concatenation) keeps the residual form; grouped conv is the efficient twin.
- ResNet-50 is ResNeXt-50 with C=1, d=64 — a clean special case.
- Efficiency claim: cardinality gains beat depth/width gains under fixed complexity.
- Context: advanced-task territory (Inception/ResNeXt/DenseNet blocks), not the three mandatory ResNet tasks.

## Review: DenseNet — Dense Convolutional Network

URL: https://medium.com/data-science/review-densenet-image-classification-b6631a8ef803  ·  Date: 2018-11-25 (publication)  ·  Status: summary

Tsang's review of the CVPR 2017 Best Paper (Cornell/Tsinghua/FAIR): where ResNet adds the past (x + F(x)), DenseNet concatenates it — every layer receives all preceding feature maps, giving L(L+1)/2 connections, implicit deep supervision, and feature reuse so aggressive that thin layers suffice (params scale with layer-index × growth-rate² instead of width²). Two controls keep it tractable: bottleneck 1x1 convs before each 3x3 (DenseNet-B) and transition layers (1x1 conv + 2x2 average pool) with compression factor θ=0.5 between same-size dense blocks (DenseNet-BC). The efficiency numbers are stark: DenseNet-BC with 0.8M params matches a 10.2M-param 1001-layer ResNet on CIFAR-10, and DenseNet-201 rivals ResNet-101 on ImageNet with half the parameters and FLOPs. A feature-reuse heatmap confirms early features feed late layers directly, which is why DenseNet degrades gracefully on small data.

- Concatenate, don't add: preserves all feature maps, diversifies features, smooths decision boundaries.
- Growth rate k: each layer adds only k channels, keeping the network narrow and cheap.
- Transition + compression: 1x1 conv and pooling between blocks, θ=0.5 halves cross-block maps.
- Small-data edge: classifier sees all complexity levels, so it overfits less than deep ResNets.
- Context: advanced-task territory (dense/transition blocks), beyond the mandatory ResNet-50 build.

## 1x1 Convolutions

URL: https://www.youtube.com/watch?v=SIpcirNNGAk  ·  Date: 2017-11-03 (YouTube upload)  ·  Status: summary

A 71-second excerpt (unofficial re-upload) of Ng's lecture making the 1x1 case: over a single-channel image a 1x1 filter is just scalar multiplication, but over a multi-channel volume each 1x1 filter is a learned linear mix of one pixel's full channel vector followed by ReLU — a tiny per-pixel network. Interspersing these is a cheap way to deepen a model and add parameters without restructuring it, and since the operation is really a matrix multiply with few weights, it is inexpensive. The clip frames this as the prerequisite for shrinking channel counts (pooling shrinks space; 1x1 convs shrink depth) before the Inception discussion.

- Single channel: 1x1 is trivial scaling; multi-channel: a per-pixel fully connected layer over channels.
- Adds depth, parameters, and nonlinearity at low cost (matrix multiply, few weights).
- Channel shrinker: the counterpart to pooling for the depth axis.
- Short format: complete as-is (71s excerpt), pairs with the full Network-in-Network lecture below.

## GoogLeNet Tutorial

URL: https://www.youtube.com/watch?v=_XF7N6rp9Jw  ·  Date: 2017-01-04 (YouTube upload)  ·  Status: summary

A ~22-minute tutorial tracing the road from LeNet through AlexNet to GoogLeNet with parameter and connection counting worked by hand: GoogLeNet reaches higher accuracy than AlexNet with roughly 12x fewer parameters inside a 1.5B multiply-add inference budget. Its core message matches the paper — the Inception module concatenates 1x1, 3x3, 5x5 and pooling branches per patch (multi-scale features aggregated), 1x1 reductions tame the compute blowup, stacked modules with occasional stride-2 pooling form the macro-architecture, auxiliary classifiers (0.3-weighted) rescue mid-network gradients, and global average pooling replaces the fully connected tail. Useful as the most explicit of these resources on why pre-GoogLeNet design (fixed kernel per layer, dense FC heads) was wasteful.

- Efficiency framing: smaller, faster, lower-memory models for real (mobile) deployment, not accuracy alone.
- Correlation intuition: local clusters → 1x1, spread clusters → 3x3/5x5, concatenate them all.
- Five distilled insights: multi-scale concat, 1x1 reduction, module stacking, auxiliary losses, GAP tail.
- Overfitting angle: fewer parameters plus GAP/dropout beats the old conv-stack-then-FC habit.
- Meme note: the Inception paper cites the "we need to go deeper" meme — the project's header image.

## Deep Residual Learning for Image Recognition

URL: https://www.youtube.com/watch?v=C6tLw-rPQ2o  ·  Date: 2016-09-02 (YouTube upload)  ·  Status: summary

Kaiming He's own CVPR talk (ComputerVisionFoundation recording, ~12 min): ResNet swept ILSVRC and COCO 2015 — classification, detection, localization, COCO detection and segmentation, most by large margins. The talk's center is the degradation paradox with a construction argument: copy a trained shallow net, set extra layers to identity, and the deeper net should match it — so worse training error means the optimizer, not capacity, is failing. Residual reformulation (fit F(x), output F(x)+x) makes identity a zero-weights away, and VGG-style 3x3 stacks plus shortcuts train cleanly to 152 layers with less complexity than VGG. Closing Q&A is worth remembering: He expects depth is only one design axis and widening can beat deepening under fixed compute — the tradeoff ResNeXt later explored.

- Primary source: the paper's first author presenting, including live detection demos.
- Degradation ≠ overfitting: training error itself rises with depth in plain nets.
- Solution-space argument: deeper nets contain the shallow solution; residuals make it findable.
- Transfer claim: ResNet-101 features gave ~28% relative gain on PASCAL VOC detection.
- Design humility: depth is one axis among many, not an automatic win — match to your compute budget.

## Densely Connected Convolutional Networks

URL: https://www.youtube.com/watch?v=-W6y8xnd--U  ·  Date: 2017-07-25 (YouTube upload)  ·  Status: summary

Gao Huang's CVPR 2017 talk (~10 min, Best Paper): plain nets pass each layer only its predecessor, ResNet adds skips for gradient flow, and DenseNet pushes the idea to its limit by wiring every layer to every other with channel-wise concatenation. Because each layer sees all prior knowledge directly, layers can be thin (small growth rate k), 1x1 bottlenecks (to 4k channels) protect deep layers from wide inputs, and same-size dense blocks joined by transition layers handle downsampling. Reported payoffs: a 100-layer, 0.8M-param DenseNet matches a 1001-layer ResNet on CIFAR-10; without augmentation DenseNet still scores ~5–6% where ResNets overfit past 10%; ImageNet top-1 20.27% at 264 layers. The talk also previews Multi-Scale DenseNets with early-exit classifiers for fast anytime inference.

- Connectivity ladder: plain (1 path) → ResNet (skip-add) → DenseNet (concat-everything).
- Thin layers + 1x1 bottlenecks + transition layers = parameter and compute efficiency.
- Implicit deep supervision: early layers get direct signal from the final loss.
- Small-data robustness: reuses low-complexity features instead of forcing everything through deep transforms.
- Advanced-task link: dense and transition blocks are optional-task blocks, not mandatory ones.

## Network In Network (C4W2L05)

URL: https://www.youtube.com/watch?v=c1RBQzKsDCk  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

Ng's Deep Learning Specialization lecture (CNN course, week 2): a 1x1 convolution over a multi-channel volume is a fully connected network applied independently at every spatial position — each filter dots one pixel's full channel vector, ReLU follows, and multiple filters give a 6x6x(#filters) output from a 6x6x32 input. The worked example shrinks 28x28x192 to 28x28x32 with thirty-two 1x1x192 filters, showing how channel count can be dialed down, held, or raised exactly as pooling dials spatial size. The lecture credits the Lin/Chen/Yan Network-in-Network paper for the idea and positions it as the enabler for the Inception bottleneck in the next videos. Watch at 1.5x–2x per the intranet note.

- Mental model: 1x1 conv = per-pixel MLP over channels, replicated across space.
- Filter depth must equal input channels (1x1x192 over a 192-deep volume).
- Channel count is a free dial: reduce, preserve, or expand — pooling only shrinks space.
- Direct task link: F11/F12 1x1 layers and the "same" padding that preserves spatial dims.

## Inception Network Motivation (C4W2L06)

URL: https://www.youtube.com/watch?v=C86ZXvgpejM  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

Ng's motivation lecture for Inception: instead of choosing one kernel size or pooling per layer, run 1x1, 3x3, 5x5 (same convolutions) and pooling in parallel and concatenate — letting the network learn which scales matter. The naive version is compute-prohibitive (one 5x5 branch costs ~120M multiplies on a 28x28x192 input), so a 1x1 bottleneck first squeezes 192 channels to 16 and the same branch drops to ~12.4M — a ~10x saving that makes the module affordable. The bottleneck reading (shrink, process, restore) is exactly the shape reused by the ResNet bottleneck blocks in this project's tasks. Watch at 1.5x–2x per the intranet note.

- Don't choose, concatenate: parallel branches replace the kernel-size dilemma.
- Bottleneck arithmetic: 120M → 12.4M multiplies worked step by step — the canonical cost demo.
- Shrinking channels aggressively before expensive convs does not hurt accuracy in practice.
- Same-convolutions everywhere keep branch outputs stackable.
- Task link: our 1x1 → 3x3 → 1x1 bottleneck is this idea inside a residual shell.

## Inception Network (C4W2L07)

URL: https://www.youtube.com/watch?v=KfV8CJh7hE0  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

Ng assembles the full module and network: each branch (1x1; 1x1→3x3; 1x1→5x5; pooling→1x1 shrink so the pool path doesn't dominate channels) concatenates to 28x28x256, and GoogLeNet is these modules stacked with occasional max-pool downsampling. Two finishing details: auxiliary softmax side-branches on intermediate layers regularize training by forcing mid-network features to stay predictive, and the "GoogLeNet" spelling honors LeNet while the "Inception" name cites the need-to-go-deeper meme — literally referenced in the paper. Later variants (v2/v3/v4, Inception-ResNet) all build on this module. Watch at 1.5x–2x per the intranet note.

- Full module recipe: reduce → process → concatenate, with the pool path also 1x1-shrunk.
- Macro-architecture: repeat module, downsample occasionally with max-pool.
- Side branches: intermediate classifiers as regularizers, discarded at inference.
- Lineage: Inception v2/v3/v4 and Inception-ResNet extend this exact module.
- Advanced-task link: Inception blocks are optional tasks; the stacking discipline previews ResNet-50's stage stacking.

## Resnets (C4W2L03)

URL: https://www.youtube.com/watch?v=ZILIbUvp5lk  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

Ng's residual-block lecture: a plain two-layer path (linear → ReLU → linear → ReLU from a^l to a^{l+2}) becomes a residual block by adding the a^l shortcut before the final ReLU, so the output is g(z^{l+2} + a^l). Stacking five such blocks converts a "plain net" into a ResNet, and the empirical split is stark — plain nets' training error falls then rises with depth while ResNets' keeps falling past 100 layers. The lecture frames the shortcut (also called skip connection) as the direct remedy for vanishing/exploding gradients in very deep stacks and notes the course's programming exercise implements exactly this. Watch at 1.5x–2x per the intranet note.

- Residual block = two layers plus an identity shortcut added pre-activation.
- Plain vs ResNet training curves: depth hurts plain nets, keeps helping ResNets.
- Skip connection is the anti-vanishing device for 100+ layer training.
- Direct task link: this lecture's block is the task-0 identity block (without bottleneck slimming).
- Vocabulary: shortcut and skip connection are synonyms here.

## Why ResNets Work (C4W2L04)

URL: https://www.youtube.com/watch?v=RYth6EbBUqM  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

Ng's intuition lecture for why depth stops hurting: with ReLU activations (non-negative) plus L2 weight decay pushing weights toward zero, a residual block a^{l+2} = g(W·a^{l+1} + b + a^l) collapses to the identity a^{l+2} = a^l when the block learns nothing — so added layers start from "do no harm" and gradient descent can only improve from there, whereas plain nets cannot even learn identity reliably. The lecture also covers the dimension-mismatch case that motivates our projection block: when shapes differ, insert a learned matrix Ws (or zero-padding) on the shortcut so the addition stays valid, and rely on same-convolutions elsewhere to keep shortcut dimensions aligned. Watch at 1.5x–2x per the intranet note.

- Identity-by-default: zero weights + ReLU + shortcut = free identity mapping.
- Deeper can only help (in training): extra blocks start harmless, then optionally useful.
- Ws matrix / zero-pad: the two fixes for shortcut shape mismatch — our task-1 shortcut conv.
- Same-convolutions preserve dims so most shortcuts need no Ws at all.
- Direct task link: identity block (same dims) vs projection block (Ws 1x1 conv, stride s).

## Network in Network (2014)

URL: https://arxiv.org/pdf/1312.4400  ·  Date: 2013-12-16 (publication)  ·  Status: summary

Lin, Chen and Yan's ICLR-2014 paper (intranet labels it 2014 for the venue; arXiv v1 is Dec 2013): replace the linear conv filter with an mlpconv micro-network — 1x1 convolutions interleaved with ReLUs sliding over the input — so each local patch gets a nonlinear rather than linear classifier, then replace the fully connected classification head with global average pooling over the final feature maps (zero parameters, less overfitting, more interpretable). The paper reached state of the art on CIFAR-10/100 with reasonable SVHN/MNIST results, and its two exports — the 1x1 convolution and global average pooling — became standard parts inside GoogLeNet, ResNet bottlenecks, and DenseNet transitions. Summary grounded in the arXiv record (title, authors, abstract) plus the reviewed secondary sources above; the PDF is the canonical full text.

- mlpconv: micro-MLP (1x1 conv + nonlinearity) as the patch classifier instead of a linear filter.
- Global average pooling: parameter-free classifier head, the anti-overfitting move.
- Dual legacy: both the 1x1 bottleneck and GAP reappear in nearly every later architecture here.
- Empirical anchor: CIFAR-10/100 state of the art at publication.
- Venue note: arXiv Dec 2013, ICLR 2014 — hence the intranet's "(2014)" label.

## Going Deeper with Convolutions (2014)

URL: https://arxiv.org/pdf/1409.4842  ·  Date: 2014-09-17 (publication)  ·  Status: summary

Szegedy et al.'s GoogLeNet paper (ILSVRC 2014 winner, "Inception"): raise depth and width while holding compute constant by designing around the Hebbian intuition — cluster correlated activations — and multi-scale processing, so each module runs 1x1/3x3/5x5/pool branches with 1x1 reductions guarding the budget. The submitted 22-layer GoogLeNet set the classification and detection state of the art at the challenge. Summary grounded in the arXiv record plus the Tsang review and tutorial above; the PDF is the canonical full text.

- Design principle: more depth and width at constant compute via crafted sparsity.
- Inception module: parallel multi-scale branches with 1x1 compute guards.
- Result: ILSVRC 2014 classification and detection wins (GoogLeNet, 22 layers).
- Auxiliary classifiers and GAP are part of the same paper's training/inference package.
- Advanced-task link: the Inception architecture this paper defines.

## Highway Networks (2015)

URL: https://arxiv.org/pdf/1505.00387  ·  Date: 2015-05-03 (publication)  ·  Status: summary

Srivastava, Greff and Schmidhuber's ICML-2015-workshop paper: let information ride "highways" across layers through learned gating units that regulate how much of the transformed vs carried signal passes — LSTM-style gating unfolded from time into depth. Highway nets with hundreds of layers train directly with SGD across activation choices, reopening the study of extremely deep feedforward architectures. ResNet is its open-gate special case, and the follow-up literature (covered in the Variants overview) found the ungated shortcut trains at least as well — keeping the highway clear matters more than the extra gating capacity. Summary grounded in the arXiv record plus secondary sources; the PDF is canonical.

- Gated shortcuts: learned gates blend transform and carry paths per layer.
- Depth result: hundred-layer nets trainable with plain SGD.
- Lineage: LSTM gating → Highway gates → ResNet identity (gates fixed open).
- Lesson: extra gate parameters help less than an unimpeded gradient path.
- Historical slot: the direct 2015 precursor to the ResNet paper.

## Deep Residual Learning for Image Recognition (2015)

URL: https://arxiv.org/pdf/1512.03385  ·  Date: 2015-12-10 (publication)  ·  Status: summary

He, Zhang, Ren and Sun's ILSVRC-2015 paper (the spec paper for all three mandatory tasks): reformulate deep layers as residual functions F(x) = H(x) − x with identity shortcuts, train 152-layer nets (8x VGG depth at lower complexity), and win ILSVRC 2015 classification with a 3.57%-error ensemble, plus firsts in detection, localization, COCO detection and segmentation; CIFAR-10 analysis pushes to 100 and 1000 layers. Solely from extreme depth comes a 28% relative COCO detection gain. Our tasks implement its bottleneck block (1x1 reduce → 3x3 → 1x1 restore, the identity and projection variants) and assemble the [3,4,6,3] ResNet-50 from the paper's stage table. Summary grounded in the arXiv record plus He's talk and both reviews; the PDF is canonical.

- Residual reformulation: fit F(x), output F(x) + x; identity costs zero weights.
- Scale: 152 layers trainable; depth 8x VGG with lower complexity.
- Headline: 3.57% ImageNet top-5 (ensemble); sweep of 2015 detection/segmentation tracks.
- Bottleneck + stages: the exact block and stage layout our three tasks replicate.
- Spec authority: every init/seed/BN/ReLU demand in tasks 0–2 traces to this paper's setup.

## Aggregated Residual Transformations for Deep Neural Networks (2017)

URL: https://arxiv.org/pdf/1611.05431  ·  Date: 2016-11-16 (publication)  ·  Status: summary

Xie et al.'s ResNeXt paper (CVPR 2017, ILSVRC 2016 runner-up): repeat one block that aggregates C identical-topology transformations, exposing cardinality — the number of parallel paths — as a scaling dimension next to depth and width. Under matched complexity, more cardinality beats deeper or wider (ResNeXt-50 32×4d: 22.2% top-1 vs ResNet-50's 23.9%), and the design is homogeneous with a single hyperparameter, easier to adapt than Inception's per-branch tuning. Code and models were released publicly. Summary grounded in the arXiv record plus the Tsang review; the PDF is canonical.

- Cardinality: parallel identical paths summed — split-transform-merge with addition.
- Efficiency: cardinality scaling beats depth/width scaling at fixed FLOPs/params.
- Homogeneity: one repeated block, one hyperparameter — Inception's tidy successor.
- Result: ILSVRC 2016 2nd place (3.03% top-5), COCO and ImageNet-5K gains over ResNet.
- Advanced-task link: ResNeXt-style thinking, beyond the mandatory ResNet-50.

## Densely Connected Convolutional Networks (2018)

URL: https://arxiv.org/pdf/1608.06993  ·  Date: 2016-08-25 (publication)  ·  Status: summary

Huang, Liu, van der Maaten and Weinberger's DenseNet paper (CVPR 2017 Best Paper; intranet labels 2018): connect each layer to every other feed-forward — L(L+1)/2 direct connections — feeding every layer the concatenated feature maps of all predecessors and forwarding its own everywhere downstream. The design alleviates vanishing gradients, strengthens propagation, encourages reuse, and cuts parameters; ImageNet, CIFAR-10/100 and SVHN all improve at lower compute (code and pretrained models released). Bottleneck (DenseNet-B) and compression (DenseNet-C, θ) variants, combined as DenseNet-BC, control width across same-size dense blocks joined by transition layers. Summary grounded in the arXiv record plus Huang's talk and the Tsang review; the PDF is canonical.

- L(L+1)/2 connections vs L in plain nets: maximal feature flow, minimal redundancy.
- Concatenation (not addition) preserves every scale of feature for the classifier.
- Growth rate + bottleneck + compression: the three width controls.
- Best-paper recognition with released code/models — the reproducibility anchor.
- Advanced-task link: dense and transition blocks are optional-task blocks.

## Multi-Scale Dense Networks for Resource Efficient Image Classification (2018)

URL: https://arxiv.org/pdf/1703.09844  ·  Date: 2017-03-29 (publication)  ·  Status: summary

Huang et al.'s MSDNet paper: classify under test-time compute budgets via early-exit classifiers embedded in one network and wired with dense connectivity, over a two-dimensional multi-scale backbone that keeps coarse and fine features alive throughout. Two budgeted regimes are targeted — anytime classification (a valid prediction available at any moment, improving as compute continues) and budgeted-batch classification (spend unevenly, more on hard inputs). Experiments on three image-classification tasks beat the prior state of the art in both regimes. Summary grounded in the arXiv record (title, authors, abstract) plus Huang's DenseNet-talk preview of the same idea; the PDF is canonical.

- Early exits: intermediate classifiers let easy inputs leave the network early.
- Anytime + budgeted-batch: two formal settings for compute-limited inference.
- Multi-scale backbone: coarse and fine features maintained at every depth.
- Dense wiring reuses computation across exits instead of duplicating it.
- Context: efficiency research downstream of DenseNet; background reading, no task maps to it.

## Quiz Hooks

- Skip (residual) connection — identity shortcut carrying x past weight layers so gradients flow back unattenuated.
- Residual mapping — fit F(x) = H(x) − x; zero weights recover the identity for free.
- Degradation problem — deeper plain nets get higher training error; an optimizer failure, not overfitting.
- Bottleneck block — 1x1 reduce, 3x3 process, 1x1 restore; the ResNet-50 unit.
- Identity block — same-shape shortcut added directly (task 0: `identity_block(A_prev, filters)`).
- Projection block — 1x1 strided conv on the shortcut to match changed shapes (task 1: `projection_block(A_prev, filters, s=2)`).
- 1x1 convolution — per-pixel fully connected layer over channels; dials depth up/down at low cost.
- Batch norm + ReLU after every conv — the per-layer pattern and He-normal seed-0 init in all three tasks.
- Inception module — parallel 1x1/3x3/5x5/pool branches concatenated; multi-scale without choosing.
- Global average pooling — parameter-free FC replacement that regularizes.
- Auxiliary classifier — intermediate softmax head (0.3 weight) that supervises mid-network features in training.
- Cardinality — ResNeXt's count of parallel identical paths; a third scaling axis with depth and width.
- Growth rate k — channels each DenseNet layer adds; small k keeps dense nets thin and cheap.
- Concatenation vs addition — DenseNet preserves all maps; ResNet adds residuals to the stream.
- Ensemble view — a ResNet is 2^i implicit shallow paths; most gradient rides 9–18-layer effective paths.
- ResNet-50 stages — [3,4,6,3] bottleneck blocks over 64→256, 128→512, 256→1024, 512→2048 channels; ~25.6M params.
