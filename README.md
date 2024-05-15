## SURE

This repo is the code implementation of paper "Fast and Fine-grained Autoscaler for Streaming Jobs with Reinforcement Learning (IJCAI'2022)".

#### Code Structure

- `run.py` is the entrance file.
- `env` is the implementation of stream processing system and a `gym` wrapper.
- `model` contains the implementation of our Neural Variational Subgraph Sampler and Mutual Information loss.
- `schedule_algo` is the RL algorithm.
- `utils` is the folder containing helper classes and functions.


#### Citation

If you find our paper or code useful, please cite the following BibTex:

```text
@inproceedings{ijcai2022p0080,
  title     = {Fast and Fine-grained Autoscaler for Streaming Jobs with Reinforcement Learning},
  author    = {Xing, Mingzhe and Mao, Hangyu and Xiao, Zhen},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
}

```