# Introduction

Each risk in Amulet represents either an *attack*, a *defense*, or a *metric*.
To see the list of the risks and the features, please see the [Getting Started guide](https://github.com/ssg-research/amulet/blob/main/docs/GETTING_STARTED.md).

## Design Overview
Most attacks and defenses are designed such that they take the target model, and some additional information (such as data, hyperparameters, configuration, etc.) as input, run an algorithm, and output a result.
This result can then be passed onto the respective metrics modules to evaluate the attack or defense. A brief pipeline would look something like:
```python
data = load_data()

target_model = initialize_model(*model_architecture_parameters)
target_model = train_model(target_model, data.train, data.test, *training_parameters)

attack = AttackClass()
attack_output = attack.run_attack(target_model, data.test, *attack_parameters)

result = evalute_attack(attack_output)
```

The rest of the documents in this folder provide detailed usage guidance for each risk implemented in Amulet.
Please see the [example scripts](https://github.com/ssg-research/amulet/tree/main/examples) provided for each attack / defense.
These scripts provide an end-to-end pipeline that may be used to run experiments.
