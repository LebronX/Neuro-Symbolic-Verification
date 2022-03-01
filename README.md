The supplementary material for IJCAI submission of `Neuro-Symbolic Verification of Deep Neural Networks`.

To run the code, please first install Marabou (https://github.com/NeuralNetworkVerification/Marabou) with Python version 3.6.0. Then you should use pytorch to train a network with two sets of input, which like `self.fc_input = nn.Linear(14 * 14, 12)` and `self.fc_input_prop = nn.Linear(14 * 14, 5)` and train with the network with corresponding property you would like to verify. Of course, you could also use the network that we provide in `networks` folder.

The experiment results are also provided in the `validate` folder.

Basic command:

Network training:
`python3 mnist_equivalence_training.py --epochs 1 --save-model --seed 1`

Verification:
`python3 run-test.py --network 1`
