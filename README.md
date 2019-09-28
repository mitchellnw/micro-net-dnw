# Discovering Neural Wirings (dnw) Micro-Net Submission

We use a method for "discovering neural wirings".
We relax the typical notion of layers and instead enable channels
to form connections independent of each other.
This allows for a much larger space of possible networks.
The wiring of our network is not fixed during training --
as we learn the network parameters we also learn the structure itself.

We also invite you to check out our preprint [here](https://arxiv.org/pdf/1906.00586.pdf) for more information.

## Set Up
0. Clone this repository.
1. Using `python 3.6`, create a `venv` with  `python -m venv venv` and run `source venv/bin/activate`.
2. Install requirements with `pip install -r requirements.txt`.
3. Create a **data directory** `<data-dir>`.
If you wish to run ImageNet experiments there must be a folder `<data-dir>/imagenet`
that contains the ImageNet `train` and `val`.


## ImageNet Experiments and Pretrained Models

The experiment files for the ImageNet experiments in the paper may be found in `apps/large_scale`.
To train your own model you may run
```bash
python runner.py app:apps/large_scale/<experiment-file> --gpus 0 1 2 3 --data-dir <data-dir>
```
and to evaluate a pretrained model which matches the experiment file use.
```bash
python runner.py app:apps/large_scale/<experiment-file> --gpus 0 1 --data-dir <data-dir> --resume <path-to-pretrained-model> --evaluate
```

As you may see in the `apps/large_scale` directory, each model below corresponds to exactly one `<experiment-file>`
(other than the MobileNet and ShuffleNet baselines).

Click on one of the links below (other than the first four, which are baselines, to download a checkpoint).

Table 1:

| Model  | Params | FLOPs | Accuracy (ImageNet) |
| :-------------: | :-------------: | :-------------: | :-------------: |
| [MobileNet V1 (x 0.25)](https://arxiv.org/abs/1704.04861)  |  0.5M  | 41M  | 50.6  |
| [ShuffleNet V2 (x 0.5)](https://arxiv.org/abs/1807.11164)  |  1.4M | 41M  | 60.3 |
| [MobileNet V1 (x 0.5)](https://arxiv.org/abs/1704.04861)  |  1.3M | 149M  | 63.7 |
| [ShuffleNet V2 (x 1)](https://arxiv.org/abs/1807.11164)  |  2.3M | 146M  | 69.4 |
| [MobileNet V1 Random Graph (x 0.225)](https://prior-datasets.s3.us-east-2.amazonaws.com/dnw/pretrained-models/rg_x225.pt)  |  1.2M | 55.7M  | 53.3 |
| [MobileNet V1 DNW Small (x 0.15)](https://prior-datasets.s3.us-east-2.amazonaws.com/dnw/pretrained-models/dnw_small_x15.pt)  |  0.24M | 22.1M  | 50.3 |
| [MobileNet V1 DNW Small (x 0.225)](https://prior-datasets.s3.us-east-2.amazonaws.com/dnw/pretrained-models/dnw_small_x225.pt)  |  0.4M | 41.2M  | 59.9 |
| [MobileNet V1 DNW (x 0.225)](https://prior-datasets.s3.us-east-2.amazonaws.com/dnw/pretrained-models/dnw_x225.pt)  |  1.1M | 42.1M | 60.9 |
| [MobileNet V1 DNW (x 0.3)](https://prior-datasets.s3.us-east-2.amazonaws.com/dnw/pretrained-models/dnw_x3.pt)  | 1.3M | 66.7M | 65.0 |
| [MobileNet V1 Random Graph (x 0.49)](https://prior-datasets.s3.us-east-2.amazonaws.com/dnw/pretrained-models/rg_x49.pt)  |  1.8M | 170M  | 64.1 |
| [MobileNet V1 DNW (x 0.49)](https://prior-datasets.s3.us-east-2.amazonaws.com/dnw/pretrained-models/dnw_x49.pt)  | 1.8M  | 154M  | 70.4 |

This may not be relevant, but you may also add the flag `--fast_eval` to make the model smaller and speed up inference. Adding `--fast_eval` removes the neurons which _die_.
As a result, the first conv, last linear layer, and all operations throughout have much fewer input and output channels. You may add both
`--fast_eval` and `--use_dgl` to obtain a model for evaluation that matches the theoretical FLOPs by using a graph implementation via
[https://www.dgl.ai/](https://www.dgl.ai/). You must then install the version of `dgl` which matches your CUDA and Python version
(see [this](https://www.dgl.ai/pages/start.html) for more details). For example, we run
```bash
pip uninstall dgl
pip install https://s3.us-east-2.amazonaws.com/dgl.ai/wheels/cuda9.2/dgl-0.3-cp36-cp36m-manylinux1_x86_64.whl
```
and finally
```bash
python runner.py app:apps/large_scale/<experiment-file> --gpus 0 --data-dir <data-dir> --resume <path-to-pretrained-model> --evaluate --fast_eval --use_dgl --batch_size 256
```

## 1. Parameter Storage and Math Operations

These results are provided in Table 1 and computed using `genutil/model_profiling.py`

For completeness we provide the code here:

```python
from genutil.config import FLAGS


def model_profiling(model):
    n_macs = 0
    n_params = 0

    if FLAGS.skip_profiling:
        return n_macs, n_params

    # using n_macs for conv2d as
    # (ins[1] * outs[1] *
    #  self.kernel_size[0] * self.kernel_size[1] *
    #  outs[2] * outs[3] // self.groups) * outs[0]
    # or, when batch_size = 1
    # in_channels * out_channels * kernel_size[0] * kernel_size[1] * out_spatial[0] * out_spatial[1] // groups

    # conv1 has stride 2. layer 1 has stride 1.
    spatial = 224 // 2

    # to compute the flops for conv1 we need to know how many input nodes in layer 1 have an output.
    # this is the effective number of output channels for conv1
    layer1_n_macs, layer1_n_params, input_with_output, _ = model.layers[0].profiling(
        spatial
    )

    conv1_n_macs = (
        model.conv1.in_channels * input_with_output * 3 * 3 * spatial * spatial
    )
    conv1_n_params = model.conv1.in_channels * input_with_output * 3 * 3

    n_macs = layer1_n_macs + conv1_n_macs
    n_params = layer1_n_params + conv1_n_params

    for i, layer in enumerate(model.layers):
        if i != 0:
            spatial = spatial // 2  # stride 2 for all blocks >= 1
            layer_n_macs, layer_n_params, _, output_with_input = layer.profiling(
                spatial
            )
            n_macs += layer_n_macs
            n_params += layer_n_params

    # output_with_input is the effective number of output channels from the body of the net.

    # pool
    pool_n_macs = spatial * spatial * output_with_input
    n_macs += pool_n_macs

    if getattr(FLAGS, "small", False):
        linear_n_macs, linear_n_params, _ = model.linear.profiling()
    else:
        linear_n_macs = output_with_input * model.linear.out_features
        linear_n_params = output_with_input * model.linear.out_features

    n_macs += linear_n_macs
    n_params += linear_n_params

    print(
        "Pararms: {:,}".format(n_params).rjust(45, " ")
        + "Macs: {:,}".format(n_macs).rjust(45, " ")
    )

    return n_macs, n_params
```