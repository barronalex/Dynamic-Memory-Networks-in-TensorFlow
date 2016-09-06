# Dynamic-Memory-Networks-in-TensorFlow
DMN+ implementation in TensorFlow

Structure and parameters from https://arxiv.org/abs/1603.01417 "Dynamic Memory Networks for Visual and Textual Question Answering".

Adapted from Stanford's cs224d assignment 2 starter code (http://cs224d.stanford.edu/) and using methods from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano for importing the Babi-10k dataset.

## Repository Contents
| file | description |
| --- | --- |
| `dmn_plus.py` | contains the DMN+ model |
| `dmn_train.py` | trains the model on a specified (-b) babi task|
| `dmn_test.py` | tests the model on a specified (-b) babi task |
| `babi_input.py` | prepares babi data for input into DMN |
| `fetch_babi_data.sh` | shell script to fetch bAbI tasks (from [DMNs in Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano)) |

