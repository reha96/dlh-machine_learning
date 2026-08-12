# Supervised Learning — Tensorflow 2 & Keras

MNIST digit classification with tf.keras: the Sequential and functional model APIs, L2 + dropout regularization, Adam optimization, training callbacks (early stopping, learning-rate decay, checkpointing), and every persistence format for models, weights, and configuration.

---

## Learning Objectives

| # | Concept |
|---|---------|
| 1 | What is Keras? |
| 2 | What is a model? |
| 3 | How to instantiate a model (2 ways) |
| 4 | How to build a layer |
| 5 | How to add regularization to a layer |
| 6 | How to add dropout to a layer |
| 7 | How to add batch normalization |
| 8 | How to compile a model |
| 9 | How to optimize a model |
| 10 | How to fit a model |
| 11 | How to use validation data |
| 12 | How to perform early stopping |
| 13 | How to measure accuracy |
| 14 | How to evaluate a model |
| 15 | How to make a prediction with a model |
| 16 | How to access the weights/outputs of a model |
| 17 | What is HDF5? |
| 18 | How to save and load a model's weights, a model's configuration, and the entire model |

---

## Task-by-Task Reference

Each task entry captures only what is new relative to all previous tasks — techniques from earlier tasks are not repeated.

---

### Task 0 — Sequential API (`0-sequential.py`)

**Challenge:** Build a three-layer network (256-256-10) as a linear stack of layers with Keras' Sequential API, adding L2 regularization and dropout.

**Approach:** Create an empty `K.Sequential()` and `add()` one `K.layers.Dense` per entry in `layers`. The first Dense declares the input size via `input_dim=nx` (the `Input` class is banned here). Each Dense takes `activation=activations[i]` and `kernel_regularizer=K.regularizers.L2(lambtha)`; a `K.layers.Dropout(rate=1 - keep_prob)` follows every hidden layer.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `K.Sequential()` + `model.add(layer)` | Build a model as a linear pipeline of layers |
| `K.layers.Dense(units, input_dim=nx, activation=..., kernel_regularizer=...)` | One layer = `W·x + b` plus activation; `input_dim` declares input size on the first layer |
| `K.regularizers.L2(lambtha)` | Penalize large weights to fight overfitting |
| `K.layers.Dropout(rate=1 - keep_prob)` | Randomly silence nodes during training; `rate` is the drop probability |
| Guard: `keep_prob is not None and i != len(layers) - 1` | Add dropout only when requested and never on the output layer |

> **Key takeaway:** Sequential is a list-like stack of layers; the first layer must state its input size, and dropout with `rate=1-keep_prob` belongs on hidden layers only.

---

### Task 1 — Functional API (`1-input.py`)

**Challenge:** Build the same network without `K.Sequential` — the functional API wires tensors explicitly.

**Approach:** Define `inputs = K.Input(shape=(nx,))`, then chain layers by calling them on the running tensor: `x = K.layers.Dense(...)(x)` and `x = K.layers.Dropout(...)(x)`. Each call returns a new tensor; the model is the input/output pair `K.Model(inputs, x)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `K.Input(shape=(nx,))` | Explicit input tensor declaring the feature count |
| Layer-as-function call: `Dense(...)(x)` | Layers are callables — calling one on a tensor returns the next tensor |
| `K.Model(inputs, outputs)` | Wrap any tensor graph into a model |
| `x = layer(x)` reassignment | Thread a single variable through the graph |

> **Key takeaway:** In the functional API a layer is a function on tensors, so any graph topology is expressible — Sequential is just the special case of a single chain.

---

### Task 2 — Compile & Optimize (`2-optimize.py`)

**Challenge:** Configure how the model will train — optimizer, loss, and metrics — before any data is seen.

**Approach:** Call `network.compile()` with `K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)`, the string `'categorical_crossentropy'` as loss, and `metrics=['accuracy']`. The function returns `None` — compile mutates the model in place.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.compile(...)` | The "settings" step: fix optimizer, loss, and metrics |
| `K.optimizers.Adam(learning_rate=, beta_1=, beta_2=)` | Adam optimizer; note the TF2 keyword is `learning_rate`, not `lr` |
| `loss='categorical_crossentropy'` | Loss for one-hot multi-class targets |
| `metrics=['accuracy']` | Extra quantities to report per epoch |

> **Key takeaway:** Compile does not train — it records the optimizer, loss, and metrics that `fit` will use; the metrics list's order defines `evaluate`'s output order.

---

### Task 3 — One-Hot Encoding (`3-one_hot.py`)

**Challenge:** Convert integer labels into the one-hot matrix format that categorical crossentropy expects.

**Approach:** A one-liner: `return K.utils.to_categorical(labels, classes)`, with `classes=None` defaulting to the number of distinct labels.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `K.utils.to_categorical(labels, classes=None)` | Turn an `(m,)` label vector into an `(m, classes)` matrix with a single 1 per row |

> **Key takeaway:** Categorical crossentropy needs one-hot targets; `to_categorical` is the standard converter, and `classes` only needs passing when you want to force the column count.

---

### Task 4 — Fit (`4-train.py`)

**Challenge:** Run the training loop — mini-batch gradient descent over epochs — and return the training record.

**Approach:** Delegate to `network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)` and return its result, a `History` object. `verbose=True` and `shuffle=False` are the defaults.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.fit(x=, y=, batch_size=, epochs=, verbose=, shuffle=)` | The full training loop: batches, passes, progress output |
| `History` return value | Per-epoch record of loss and metrics |
| `shuffle=` parameter | Whether to reorder batches between epochs |

> **Key takeaway:** `fit` is the entire training loop in one call — everything before it was configuration; everything after it reads the `History` it returns.

---

### Task 5 — Validation Data (`5-train.py`)

**Challenge:** Watch generalization during training, not just training loss.

**Approach:** Add `validation_data=validation_data` to the same `fit` call, defaulting to `None`. When given, it must be a `(X_val, y_val)` tuple.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `validation_data=(X_val, y_val)` | Hold-out set evaluated each epoch; adds `val_loss`/`val_accuracy` to `History` |

> **Key takeaway:** `validation_data` gives a per-epoch out-of-sample check — and every later callback (early stopping, decay, checkpointing) depends on it.

---

### Task 6 — Early Stopping (`6-train.py`)

**Challenge:** End training automatically when the validation loss stops improving, saving epochs on a converged model.

**Approach:** Start `callbacks = []`; only when `early_stopping and validation_data is not None`, set `callbacks = [K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]` and pass `callbacks=callbacks` to `fit`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)` | Stop training when the monitored quantity stops improving for `patience` epochs |
| `fit(..., callbacks=callbacks)` | The hook point where callbacks attach to training |
| Guard: `early_stopping and validation_data is not None` | Early stopping is meaningless without a monitored value |

> **Key takeaway:** Callbacks are the plug-in mechanism for training behavior; EarlyStopping only makes sense when `validation_data` exists, since it monitors `val_loss`.

---

### Task 7 — Learning-Rate Decay (`7-train.py`)

**Challenge:** Shrink the learning rate epoch by epoch with inverse-time decay, without hand-computing each rate.

**Approach:** When `learning_rate_decay and validation_data is not None`, define a nested `scheduler(epoch)` returning `alpha / (1 + decay_rate * epoch)` and append `K.callbacks.LearningRateScheduler(scheduler, verbose=1)` to the callbacks list. Note the switch from `callbacks = [...]` to `callbacks.append(...)` — callbacks now stack.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `K.callbacks.LearningRateScheduler(scheduler, verbose=1)` | Calls your function with the current epoch at each epoch start |
| `scheduler(epoch) = alpha / (1 + decay_rate * epoch)` | Inverse-time decay: large steps early, fine steps late |
| `callbacks.append(...)` | Compose multiple callbacks instead of replacing the list |

> **Key takeaway:** A learning-rate schedule is just a function `epoch → lr` that Keras invokes; the `epoch` argument comes from Keras, and `append` is how callbacks accumulate.

---

### Task 8 — Model Checkpoint (`8-train.py`)

**Challenge:** Persist the best version of the model found during training, not just the final one.

**Approach:** When `save_best and filepath is not None`, append `K.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True)` to the callbacks list.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `K.callbacks.ModelCheckpoint(filepath=, monitor='val_loss', save_best_only=True)` | Write the model file only when the monitored metric improves |

> **Key takeaway:** With `save_best_only=True` the checkpoint callback keeps the best-scoring model on disk — a safety net against the last epochs overfitting.

---

### Task 9 — Save/Load the Full Model (`9-model.py`)

**Challenge:** Persist and restore the entire model — architecture, weights, and optimizer state — in one file.

**Approach:** `save_model` calls `network.save(filename)` and returns `None`; `load_model` returns `K.models.load_model(filename)`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.save(filename)` | Serialize weights + architecture + optimizer state to one file |
| `K.models.load_model(filename)` | Rebuild the full model from that file; returns the model |

> **Key takeaway:** Full-model save/load round-trips everything including optimizer state, so training can resume exactly where it stopped; save returns `None`, load returns the model.

---

### Task 10 — Save/Load Weights (`10-weights.py`)

**Challenge:** Persist only the learned parameters, keeping architecture handling separate.

**Approach:** `network.save_weights(filename, save_format=save_format)` with default `save_format='keras'`; `network.load_weights(filename)` loads them into an already-built model. Both return `None`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.save_weights(filename, save_format=...)` | Weights only; `'keras'` is the TF2-native format, `'h5'` the legacy HDF5 one |
| `model.load_weights(filename)` | Restore weights into a model whose architecture you built yourself |

> **Key takeaway:** Weights-only persistence assumes the architecture already exists — you must rebuild the model, then `load_weights` fills in the numbers.

---

### Task 11 — Save/Load Configuration (`11-config.py`)

**Challenge:** Persist only the architecture, as human-readable JSON, and rebuild an untrained model from it.

**Approach:** `save_config` writes `network.to_json()` to a file with `with open(filename, 'w')`; `load_config` reads it back with `with open(filename, 'r')` and returns `K.models.model_from_json(f.read())`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.to_json()` | Serialize the architecture (layers, activations, shapes) as JSON |
| `K.models.model_from_json(json_string)` | Rebuild the architecture from JSON — weights not included |
| `with open(filename, 'w'/'r')` + `f.write` / `f.read` | Idiomatic file I/O with guaranteed closing |

> **Key takeaway:** There are three persistence levels — configuration (architecture JSON), weights, and the full model; `to_json`/`model_from_json` round-trips architecture only, so the loaded model is untrained.

---

### Task 12 — Evaluate (`12-test.py`)

**Challenge:** Score the finished model on data it never trained on.

**Approach:** `return network.evaluate(x=data, y=labels, verbose=verbose)` — a forward pass with loss and metrics, no weight updates. It returns a list `[loss, accuracy]` in the order given to `compile`.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.evaluate(x=, y=, verbose=)` | The "test" step: score with labels, never train |

> **Key takeaway:** `evaluate` is `fit` without learning — same forward math, but the returned list's entries follow the compile-time loss/metrics order.

---

### Task 13 — Predict (`13-predict.py`)

**Challenge:** Produce model outputs for new, unlabeled data.

**Approach:** `return network.predict(x=data, verbose=verbose)` with default `verbose=False`. The result is the raw output matrix — for a softmax classifier, one probability distribution per row.

**New techniques introduced:**

| Technique | Purpose |
|-----------|---------|
| `model.predict(x=, verbose=)` | Forward propagation without labels or scoring; returns the output tensor |

> **Key takeaway:** `predict` is forward propagation alone — row *i* of the result is the model's probability distribution over the 10 classes for input *i*.

---

## Technique Inventory

| Task | New technique summarized | Category |
|------|--------------------------|----------|
| 0 | `K.Sequential()` + `add()`, `Dense(input_dim=nx, kernel_regularizer=L2)`, `Dropout(rate=1-keep_prob)` | Model APIs |
| 1 | `K.Input(shape=)`, layer-as-function chaining, `K.Model(inputs, outputs)` | Model APIs |
| 2 | `compile()`, `K.optimizers.Adam`, loss/metrics strings | Model Configuration |
| 3 | `K.utils.to_categorical(labels, classes)` | Data Encoding |
| 4 | `fit(x=, y=, batch_size=, epochs=, verbose=, shuffle=)`, `History` | Training & Callbacks |
| 5 | `validation_data=(X_val, y_val)` tuple | Training & Callbacks |
| 6 | `EarlyStopping(monitor='val_loss', patience=)`, `callbacks=` arg | Training & Callbacks |
| 7 | `LearningRateScheduler`, inverse-time decay `alpha/(1+decay_rate*epoch)` | Training & Callbacks |
| 8 | `ModelCheckpoint(filepath=, monitor='val_loss', save_best_only=True)` | Training & Callbacks |
| 9 | `model.save()` / `K.models.load_model()` — full model | Serialization |
| 10 | `save_weights()` / `load_weights()`, `save_format='keras'` | Serialization |
| 11 | `to_json()` / `K.models.model_from_json()` — architecture only | Serialization |
| 12 | `evaluate()` → `[loss, accuracy]` | Evaluation & Prediction |
| 13 | `predict()`, default `verbose=False`, probability matrix | Evaluation & Prediction |

---

## Resources

- [TensorFlow 1 vs TensorFlow 2: Is the new TF better? (365 Data Science)](https://www.youtube.com/watch?v=t48a_KOh0fQ)
- [Differences Between Tensorflow 1.x and Tensorflow 2.0, Episode 3 (Lazy Programmer)](https://www.youtube.com/watch?v=4NLlNx6wVaw)
- [Keras Explained (Siraj Raval)](https://www.youtube.com/watch?v=j_pJmXJwMLA)
- [The Sequential model (TensorFlow Core guide)](https://www.tensorflow.org/guide/keras/sequential_model)
- [Keras vs. tf.keras: What's the difference in TensorFlow 2.0? (PyImageSearch)](https://pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/)
- [Hierarchical Data Format (Wikipedia)](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)
- [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [tf.keras.models](https://www.tensorflow.org/api_docs/python/tf/keras/models)
- [tf.keras.activations](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
- [tf.keras.callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)
- [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)
- [tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
- [tf.keras.metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
- [tf.keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
- [tf.keras.regularizers](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers)
- [tf.keras.utils](https://www.tensorflow.org/api_docs/python/tf/keras/utils)
- [Serialization and saving (TensorFlow guide)](https://www.tensorflow.org/guide/keras/serialization_and_saving)
