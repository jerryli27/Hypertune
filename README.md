# Hypertune

This repo provides a simple helper function to tune the hyper-parameters of a tensorflow model.

## Prerequisite

`pip install bayesian-optimization`

## Example usage

```bash
# Assume that you would like to tune test_controller.py which has two hyper-parameters x and y.
# And assume that test_controller.py has a function called `hyper_tune()`, which will train the model and
# return a score representing the performance of the current set of hyper-parameters defined in the flags.
python hyper_tune.py --file_name=test_controller \
--params_and_constraints="[('x',-5,5),('y',-5,5),]"
```