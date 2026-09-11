================================================================
WiDiT App
================================================================

.. image:: https://rbturnbull.github.io/WiDiT/_images/WiDiT-Banner.png
   :alt: WiDiT banner
   :align: center

.. start-badges

|pypi badge| |testing badge| |coverage badge| |docs badge| |black badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/widitapp.svg?color=blue
    :target: https://pypi.org/project/widitapp/

.. |testing badge| image:: https://github.com/rbturnbull/widitapp/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/widitapp/actions

.. |docs badge| image:: https://github.com/rbturnbull/widitapp/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/widitapp
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/8187b43915256acc806510b100cf15d2/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/widitapp/coverage/
    
.. end-badges

.. start-quickstart

A base class for building image and volume learning apps with the
`WiDiT <https://github.com/rbturnbull/widit.git>`_ model and U-Net. WiDiTApp
combines model creation, PyTorch data loaders, diffusion or supervised training,
and Cluey command-line interfaces. It supports 2D images and 3D volumes.

Subclass ``WiDiTApp`` to supply your datasets and prediction workflow. The base
class provides training, but its ``datasets()`` and ``predict()`` methods raise
``NotImplementedError`` until you implement them. This package does not install
a standalone command-line executable.

Installation
==================================

Requires Python 3.10–3.13. Install in a virtual environment using pip:

.. code-block:: bash

    python -m pip install widitapp

Or install from the repository:

.. code-block:: bash

    python -m pip install git+https://github.com/rbturnbull/WiDiTApp.git

The current project configuration fetches WiDiT from GitHub over SSH. Installing
these dependencies requires Git and working GitHub SSH authentication.

Training currently requires a CUDA-capable GPU and a CUDA-enabled PyTorch
installation; the training loop explicitly checks for CUDA. Model construction
and CPU inference can be used independently of training.

Usage
==================================

Create a model
----------------------------------

Build a small 2D regression model from Python:

.. code-block:: python

    import torch
    from widitapp import WiDiTApp

    app = WiDiTApp()
    model = app.model(
        dim=2,
        use_diffusion=False,
        hidden_size=64,
        depth=1,
        num_heads=4,
        patch_size=2,
        window_size=4,
        use_flash_attention=False,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros(1, 1, 32, 32))

Use ``dim=3`` for volumes, or ``unet=True`` to build a U-Net. Inputs use
channel-first shapes: ``(N, C, H, W)`` for batched 2D images and
``(N, C, D, H, W)`` for batched 3D volumes. Choose spatial sizes compatible with
the model's patch and window settings.

Create your own training app
----------------------------------

As in Supercat and Dosefusion, implement ``datasets()`` and decorate it with
Cluey's ``@method``. Return a pair of PyTorch datasets: training first, validation
second. The inherited ``train`` tool collects the options from ``model()``,
``dataloaders()``, and your ``datasets()`` method.

Save this example as ``myapp.py``. It reads tensor pairs from ``train.pt`` and
``validation.pt`` in a directory you provide:

.. code-block:: python

    from pathlib import Path

    import torch
    from cluey import method
    from torch.utils.data import TensorDataset
    from widitapp import WiDiTApp


    class MyApp(WiDiTApp):
        @method
        def datasets(self, data_dir: Path = None, **kwargs) -> tuple:
            """Load paired input and target tensors."""
            if data_dir is None:
                raise ValueError("Provide data_dir (CLI: --data-dir).")
            data_dir = Path(data_dir)

            def load_split(name):
                inputs, targets = torch.load(
                    data_dir / name, map_location="cpu", weights_only=True
                )
                return TensorDataset(inputs.float(), targets.float())

            return load_split("train.pt"), load_split("validation.pt")


    if __name__ == "__main__":
        MyApp.tools()

Each file should contain a tuple ``(inputs, targets)`` saved with
``torch.save((inputs, targets), path)``. For single-channel 2D data, both tensors
have shape ``(N, 1, H, W)``. Use separate samples for training and validation.
Preprocess and normalize your data consistently before saving it; WiDiTApp does
not read image formats or normalize intensities for you.

Inspect the generated options and train a small regression model:

.. code-block:: bash

    python myapp.py train --help
    python myapp.py train \
        --data-dir /path/to/tensor-pairs \
        --no-use-diffusion \
        --dim 2 \
        --hidden-size 64 \
        --depth 1 \
        --num-heads 4 \
        --no-use-flash-attention \
        --batch-size 4 \
        --epochs 40 \
        --results-dir results \
        --run-name example

You can also call ``MyApp().train(data_dir=Path("/path/to/tensor-pairs"),
use_diffusion=False, ...)`` from Python with the corresponding keyword arguments.

Dataset and training modes
----------------------------------

* Each dataset item must return ``(input, target)`` or
  ``(input, target, timestep)``. Individual images omit the batch dimension;
  the data loader adds it. Samples in a batch must have compatible shapes.
* With ``use_diffusion=True`` (the default), the target is noised and the input
  is passed to the model as ``conditioned=input``. The training loop samples
  diffusion timesteps; any dataset-provided timestep is replaced. WiDiT expects
  the conditioning input and target to have the same shape, so resize paired
  data to match before training.
* With ``use_diffusion=False``, the model predicts the target directly from the
  input using mean squared error. An optional dataset timestep is decremented
  by one before being passed to the model.
* Return ``(training_dataset, None)`` to omit validation. The current training
  loop saves checkpoints during validation only, so provide a validation
  dataset when you need saved models.

Training data is shuffled; validation data is not. Both loaders retain the last
incomplete batch. Override ``dataloaders()`` if you need custom sampling or
collation.

Common options
----------------------------------

Python parameter names use underscores; their CLI equivalents use hyphens,
for example ``batch_size`` becomes ``--batch-size``. Boolean options also have a
negative form, such as ``--no-use-diffusion``.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Purpose
   * - ``dim``
     - ``2``
     - Spatial dimensionality.
   * - ``use_diffusion``
     - ``True``
     - Conditional diffusion training or direct regression.
   * - ``in_channels``
     - ``1``
     - Model input channels.
   * - ``out_channels``
     - Automatic
     - Defaults to 2 for diffusion and 1 for regression.
   * - ``use_conditioning``
     - Automatic
     - Defaults to the value of ``use_diffusion``.
   * - ``preset``
     - Empty
     - Named WiDiT preset; inspect ``widit.PRESETS`` for available names.
   * - ``hidden_size``, ``depth``, ``num_heads``
     - ``768``, ``12``, ``12``
     - Transformer architecture when no preset is selected.
   * - ``patch_size``, ``window_size``
     - ``2``, ``4``
     - Patch and attention window sizes.
   * - ``unet``
     - ``False``
     - Select U-Net; configure with ``filters``, ``kernel``, and ``layers``.
   * - ``batch_size``, ``num_workers``
     - ``1``, ``4``
     - Data loader batch size and worker count.
   * - ``epochs``, ``learning_rate``
     - ``40``, ``0.0001``
     - Training duration and AdamW learning rate.
   * - ``log_every``
     - ``100``
     - Log every this many successful training steps.
   * - ``results_dir``, ``run_name``
     - ``./results``, automatic
     - Output root and experiment subdirectory name.
   * - ``wandb``
     - ``False``
     - Enable Weights & Biases logging; ``wandb_project`` defaults to the app class name.
   * - ``checkpoint``
     - ``None``
     - Load an existing model instead of constructing one.

The default output-channel counts assume single-channel targets. For
multi-channel regression set ``out_channels`` to the target channel count;
diffusion with learned variance needs twice that count. A preset defines its
own spatial dimensionality and core architecture. Loading a checkpoint uses
the saved model configuration rather than the model-building options.

For finer control, call ``widitapp.training.train`` directly with a model and
data loaders. It additionally accepts ``precision`` (``fp16`` by default,
``bf16``, ``fp32``, or ``no``), ``loss_fn`` (``mse``, ``smoothl1``/``huber``, or
a PyTorch loss module), ``wandb_config``, and ``wandb_log_artifacts``. These
arguments are not forwarded by ``WiDiTApp.train``. The custom loss applies to
regression; diffusion uses its own objective.

Checkpoints and prediction
----------------------------------

Training maintains an exponential moving average (EMA) of model weights and
evaluates that model after each epoch. With validation, outputs are written to::

    results/<run_name>/
        log.txt
        checkpoints/
            best.pt

``best.pt`` stores the EMA model with the best validation loss. If the initial
validation loss is non-finite, a fallback checkpoint is saved until a finite
improvement is available. There is currently no separate final checkpoint or
optimizer-state resume: ``checkpoint=...`` initializes a new training run with
saved model weights.

Load a checkpoint with ``WiDiTApp().model(checkpoint=Path(".../best.pt"))`` or
``widit.load_model(path)``. For regression, move the loaded model and inputs to
the same device, call ``model.eval()``, and predict under ``torch.no_grad()``.
Diffusion inference requires an iterative sampling loop, rather than a single
forward pass; ``widitapp.diffusion.create_diffusion`` supplies the diffusion
process and sampling methods.

To expose prediction in your app, override ``predict()`` with ``@main`` from
Cluey. Implement input loading, preprocessing, checkpoint loading, regression
or diffusion inference, and output writing for your data format.

Package your command-line app
----------------------------------

If your class lives in ``myapp/apps.py``, add these entry points to your app's
Poetry configuration and install that app with ``python -m pip install -e .``:

.. code-block:: toml

    [tool.poetry.scripts]
    myapp = "myapp.apps:MyApp.main"
    myapp-tools = "myapp.apps:MyApp.tools"

``myapp`` invokes the method decorated with ``@main`` (your prediction method),
while ``myapp-tools train`` invokes inherited training. The training-only
example above can use just the ``myapp-tools`` entry point until prediction is
implemented. Decorate additional commands with Cluey's ``@tool``.

Example applications
----------------------------------

The following projects demonstrate how to extend WiDiTApp:

* **Supercat** (``supercat/apps.py``): provides 2D and 3D super-resolution
  datasets, tiled prediction, and additional porosity tools. Its
  ``supercat/pretrain.py`` provides further subclasses for image and video
  pretraining.
* **Dosefusion** (``dosefusion/apps.py``): builds datasets from a CSV with
  partition selection and implements volume prediction with diffusion and
  fusion sampling options. Its data preparation and prediction code live in
  ``dosefusion/data.py`` and ``dosefusion/models.py``.

In both projects, ``pyproject.toml`` registers the app's ``.main`` and ``.tools``
entry points. Use their dataset and prediction implementations as examples
when adapting the base class to your own files and task.

.. end-quickstart

Development
==================================

Install the project and its development dependencies with Poetry, then run the
tests:

.. code-block:: bash

    git clone https://github.com/rbturnbull/WiDiTApp.git
    cd WiDiTApp
    poetry install
    poetry run pytest

The tests cover app configuration, data loader construction, training helpers,
and diffusion utilities. Running the training example itself requires CUDA.


Credits
==================================

.. start-credits

`Robert Turnbull <https://robturnbull.com>`_ - Melbourne Data Analytics Platform (MDAP), The University of Melbourne

Created using cluey (https://github.com/rbturnbull/cluey).

Based on https://github.com/chuanyangjin/fast-DiT

.. end-credits
