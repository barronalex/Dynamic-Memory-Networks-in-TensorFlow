# Dynamic Memory Networks in TensorFlow

DMN+ implementation in TensorFlow for question answering with the bAbI 10k dataset.

Structure and parameters from  [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) which is henceforth referred to as Xiong et al.

Adapted from Stanford's [cs224d](http://cs224d.stanford.edu/) assignment 2 starter code  and using methods from [Dynamic Memory Networks in Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano) for importing the Babi-10k dataset.

## Repository Contents
| file | description |
| --- | --- |
| `dmn_plus.py` | contains the DMN+ model |
| `dmn_train.py` | trains the model on a specified (-b) babi task|
| `dmn_test.py` | tests the model on a specified (-b) babi task |
| `babi_input.py` | prepares bAbI data for input into DMN |
| `fetch_babi_data.sh` | shell script to fetch bAbI tasks (from [DMNs in Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano)) |

## Usage
Install [TensorFlow](https://www.tensorflow.org/versions/r0.10/get_started/index.html)

Run the included shell script to fetch the data

	./fetch_babi_data.sh

Use 'dmn_train.py' to train the DMN+ model contained in 'dmn_plus.py'

	python dmn_train.py --babi-task-id 2

Once training is finished, test the model on a specified task

	python dmn_test.py --babi-task-id 2

The l2 regularization constant can be set with -l2-loss (-l). All other parameters were specified by [Xiong et al](https://arxiv.org/abs/1603.01417) and can be found in the 'Config' class in 'dmn_plus.py'.

## Benchmarks

The TensorFlow DMN+ reaches close to state of the art performance on the 10k dataset when weakly supervised (no supporting facts).

| . | TensorFlow DMN+| Xiong et al DMN+ |
| :---: | :---: | :---: |
| Mean error | xx | 2.8 |
| Failed tasks (>5% error) | x | 1 |

Each task was trained on separately with l2 = 0.001. As the paper suggests, 10 training runs were used for tasks 2, 3, 17 and 18 (configurable with --num-runs), where the weights which produce the lowest validation loss in any run are used for testing.

The pre-trained weights which achieve these benchmarks are available in 'pretrained'.

I haven't yet had the time to fully optimize the l2 parameter which is not specified by the paper. My hypothesis is that fully optimizing l2 regularization would close the final XX% performance gap between the TensorFlow and original DMN+s. 

Below are the full results for each bAbI task:

| Task ID | TensorFlow DMN+| Xiong et al DMN+ |
| :---: | :---: | :---: |
| 1 | 0.0 | 0.0 |
| 2 | 0.9 | 0.3 |
| 3 |  | 1.1 |
| 4 | 0.0 | 0.0 |
| 5 | 0.5 | 0.5 |
| 6 | 0.0 | 0.0 |
| 7 | 2.8 | 2.4 |
| 8 | 0.5 | 0.0 |
| 9 | 0.1 | 0.0 |
| 10 | 0.0 | 0.0 |
| 11 | 0.0 | 0.0 |
| 12 | 0.0 | 0.0 |
| 13 | 0.0 | 0.0 |
| 14 | 0.0 | 0.2 |
| 15 | 0.0 | 0.0 |
| 16 | 46.2 | 45.3 |
| 17 | 5.0 | 4.2 |
| 18 | 2.2 | 2.1 |
| 19 | 0.0 | 0.0 |
| 20 | 0.0 | 0.0 |



