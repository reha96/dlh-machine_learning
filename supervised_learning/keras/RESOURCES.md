# Tensorflow 2 & Keras — Resources

Intranet project 2300. Python 3.9, numpy 1.25.2, tensorflow 2.15, pycodestyle
2.11.1. Only import allowed unless stated: `import tensorflow.keras as K`.

## Read or watch

### TensorFlow 1 vs TensorFlow 2: Is the new TF better? (365 Data Science)

URL: https://www.youtube.com/watch?v=t48a_KOh0fQ
Status: summary (transcript)

TF1 was versatile but hard to learn: strange methods, unfamiliar coding
logic. Higher-level packages (PyTorch, Keras) rose because of this. Keras was
integrated into core TensorFlow in 2017 — its author calls it "an interface
for TensorFlow rather than a different library". In 2019 TF2.0 borrowed
Keras' high-level syntax wholesale; "TensorFlow 2 is basically Keras". TF2
simplified the API, removed duplicate/deprecated functions, and most
importantly adopted eager execution: standard Python semantics instead of
explicit computation graphs.

- TF2 = TF1's power + Keras' simplicity; Keras became the official high-level
  API.
- Eager execution: code runs line by line like NumPy; no graph building.
- TF1's pain points (sessions, placeholders) are gone in TF2.
- Keras was always an interface/abstraction, not a standalone engine.

### Differences Between Tensorflow 1.x and Tensorflow 2.0, Episode 3 (Lazy Programmer)

URL: https://www.youtube.com/watch?v=4NLlNx6wVaw
Status: summary (transcript)

The overarching theme: TF2 is simpler. Keras is an API specification — the
user-facing code is identical between TF1.14 and TF2.0. In TF1 you could
build layers many ways (tf.layers, tf.contrib, custom classes); in TF2 those
are removed and the Keras API is the official way to build networks.
`tf.contrib` is gone (its useful parts moved). Custom layers/models are built
by subclassing `tf.keras.layers.Layer` / `tf.keras.Model`. Sessions are gone:
TF1 built a graph, then fed values via `session.run(feed_dict=...)` with
placeholders; TF2 enables eager execution by default (like PyTorch, like
plain Python). For performance, `@tf.function` gives compiled-graph speed
without sessions.

- TF1 flow: define variables/placeholders -> build graph -> `session.run`
  with `feed_dict`; `C = A + B` does NOT compute 3.
- TF2 flow: eager by default; `a + b` computes immediately. No sessions, no
  placeholders, no `tf.global_variables_initializer`.
- Keras API = the standard for creating models; other pre-built layer APIs
  removed.
- Custom layers: subclass `K.layers.Layer`; custom models: subclass
  `K.Model`; compile to graphs with `@tf.function` only if you need speed.

### Keras Explained (Siraj Raval)

URL: https://www.youtube.com/watch?v=j_pJmXJwMLA (start at 3:48)
Status: summary (transcript)

Keras is a high-level interface that wraps multiple backends (TensorFlow,
Theano, CNTK) — same code no matter the backend; by 2017 Google chose it as
the official high-level API of TensorFlow. A deep network is a series of math
operations as layers: `input * weight + bias` then an activation, repeated.
Layers are modular building blocks (dense, conv, dropout, recurrent...).
Keras abstracts the magic numbers away. The workflow: define -> compile ->
fit -> evaluate -> predict. A Sequential model is a sequence of layers; add
layers one at a time or pass a list to the constructor; the first layer must
define the expected input size.

- Pipeline: define the model, compile it (choose optimizer + loss), fit on
  (X, y), evaluate, then predict on new data.
- Sequential = pipeline of layers; input at bottom, predictions at top.
- First layer must state its input size.
- Activations can be separate layers (e.g. `K.layers.Activation`).

### The Sequential model (TensorFlow Core guide)

URL: https://www.tensorflow.org/guide/keras/sequential_model
Status: summary

A Sequential model is right only for a plain stack of layers where each layer
has exactly one input tensor and one output tensor — wrong for multi-input /
multi-output, layer sharing, or non-linear topology (residuals, branches).
Create it by passing a list of layers to the constructor or by calling
`add()` incrementally (and `pop()` to remove); it behaves like a list of
layers. Layers create their weights only when first called on an input
("built"): until then `model.summary()` and `model.weights` fail. To build
with a known input shape from the start, pass an `Input` object or set
`input_shape`/`input_dim` on the first layer.

- `K.Sequential([...])` or `K.Sequential(); model.add(layer)`.
- Layers are weightless until first called; summary() needs a built model.
- Input shape comes from the first layer's `input_shape=(nx,)` or
  `input_dim=nx`, or an explicit `Input` layer.
- Task 0 forbids the `Input` class — use `input_dim`/`input_shape` instead.

### Keras vs. tf.keras: What's the difference in TensorFlow 2.0? (PyImageSearch)

URL: https://pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/
Status: summary

Keras (by Francois Chollet, 2015) was a set of abstractions sitting on a
computational backend (Theano first, TensorFlow as default since Keras
v1.1.0) — your code never changed regardless of backend. TensorFlow v1.10
introduced the `tf.keras` submodule; with TF2 (Sept 30, 2019) and Keras
v2.3.0, Keras became the official high-level API of TensorFlow. The two are
in sync, but the standalone `keras` package only gets bug fixes — everyone
should use `tf.keras`. Switching is just changing the import
(`from tensorflow.keras... import ...`). TF2.0 features: eager execution,
automatic differentiation via `GradientTape`, model/layer subclassing, and
better multi-GPU training (`MirroredStrategy`).

- Standalone `keras` and `tf.keras` are separate projects; use `tf.keras`
  going forward (the project's only allowed import).
- Keras = abstraction over a computational backend.
- TF2: eager execution by default; `GradientTape` for custom training loops.
- Three ways to build models: Sequential, functional, subclassing.

### Hierarchical Data Format (Wikipedia)

URL: https://en.wikipedia.org/wiki/Hierarchical_Data_Format
Status: summary

HDF is a set of file formats (HDF4, HDF5) for storing and organizing large
amounts of data, originally developed at NCSA, maintained by The HDF Group.
HDF5, the current version, has only two object types: datasets (typed
multidimensional arrays) and groups (containers that can hold datasets and
other groups) — a filesystem-like hierarchy where resources live at
paths like `/path/to/resource`. Metadata is stored as named attributes on
groups and datasets. Used widely for scientific data (NASA EOS, NetCDF4).

- HDF5 = groups + datasets + attributes; hierarchical, self-describing.
- Extension `.h5` / `.hdf5`; the format behind Keras model files.
- Good for big scientific arrays; fast bulk access vs SQL rows.

## References (link only)

- tf.keras — https://www.tensorflow.org/api_docs/python/tf/keras
- tf.keras.models — https://www.tensorflow.org/api_docs/python/tf/keras/models
- tf.keras.activations — https://www.tensorflow.org/api_docs/python/tf/keras/activations
- tf.keras.callbacks — https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
- tf.keras.initializers — https://www.tensorflow.org/api_docs/python/tf/keras/initializers
- tf.keras.layers — https://www.tensorflow.org/api_docs/python/tf/keras/layers
- tf.keras.losses — https://www.tensorflow.org/api_docs/python/tf/keras/losses
- tf.keras.metrics — https://www.tensorflow.org/api_docs/python/tf/keras/metrics
- tf.keras.optimizers — https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
- tf.keras.regularizers — https://www.tensorflow.org/api_docs/python/tf/keras/regularizers
- tf.keras.utils — https://www.tensorflow.org/api_docs/python/tf/keras/utils
- Serialization and saving — https://www.tensorflow.org/guide/keras/serialization_and_saving

## Quiz Hooks

- Sequential model — a linear stack of layers; one input, one output per layer.
- Eager execution — TF2 runs ops immediately; no sessions/placeholders/graphs.
- Backend — the computation engine a high-level API (Keras) delegates to.
- tf.keras vs keras — same API, but tf.keras is the maintained one in TF2.
- Model lifecycle — define, compile, fit, evaluate, predict.
- input_dim / input_shape — how the first layer declares its input size.
- L2 regularization — penalizes big weights to fight overfitting.
- Dropout — randomly drops nodes (rate = 1 - keep_prob) to fight overfitting.
- HDF5 — the file format (groups + datasets) Keras uses to save models.
- GradientTape — TF2's automatic differentiation for custom training loops.
