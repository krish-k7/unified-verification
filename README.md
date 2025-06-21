# Unified Verification and Validation for Mountain Car and Synthetic Systems Case Studies

This repository provides code for the unified verification and validation of vision-based autonomous systems. This pipeline is implemented on the classic Mountain Car case study and a synthetic benchmark.

## Project Structure

- `mountain-car/`  
  - Scripts and models for the Mountain Car environment, including data generation, training, validating, and PRISM model generation.
  - Contains sample data, trained weights, and PRISM model files.
- `synthetic/`  
  - Scripts and models for a synthetic benchmark system, following a similar structure to the Mountain Car folder.

## What's in This Project?

- **Reinforcement Learning**: Trains a DQN policy for the Mountain Car environment.
- **Vision-Based State Estimation**: Uses a neural network to estimate the car's state from rendered images.
- **Data Generation**: Scripts to collect trajectories and estimation data.
- **Interval Generation**: Computes confidence intervals for state estimation errors.
- **PRISM Model Generation**: Automatically generates .pm files for the PRISM tool<sup>[1]</sup>.
- **Validation**: Scripts to validate the design-time abstractions on test-time trajectories.

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `torch`, `gym`, `matplotlib`, `opencv-python`
- PRISM model checker<sup>[1]</sup>

Install Python dependencies with:

```bash
pip install numpy scipy torch gym matplotlib opencv-python
```

## Mountain Car Study Usage

Execute scripts in the `mountain-car/` directory

1. **Optional: Train Policy**: adjust the hyperparameters of `trainPolicy.py` and execute; however, a pretrained model is saved as 'policy_2.pth'
2. **Optional: Test Policy**: test the DQN policy on the Mountain Car environment without intermediate vision-based state estimation ('testPolicy.py') or with state estimation ('testPolicyWithVis.py')
3. **Generate Data**: Run `getData.py` to collect training and validation data. Some sample trajectories are already provided in 'sample-data.pkl'
4. **Generate Intervals**: Use `genIntervals.py` to compute confidence intervals from the data. These are the intervals of the abstract interval Markov decision process.
5. **Generate PRISM Model**: Run `prismgen.py` to create a PRISM model of the IMDP.
6. **PRISM Model Checking**: From the terminal, navigate to the home directory, and build the model in PRISM with the command:
```bash
<PATH-TO-PRISM> --javamaxmem 10g --cuddmaxmem 10g <PATH-TO-.pm-MODEL> <PATH-TO-.props-PROPS>
```
Increase/decrease java and cudd memory as needed.

7. **Validate**: Use `validate.py` to test the validity of the IMDP on novel data.

## Synthetic Study Usage

1. **Generate Data**: Run `getData.py` to collect training and validation data. Some sample trajectories are already provided in 'sample-data.pkl'
2. **Generate Intervals**: Use `genIntervals.py` to compute confidence intervals from the data. These are the intervals of the abstract interval Markov decision process.
3. **Generate PRISM Model**: Run `prismgen.py` to create a PRISM model of the IMDP.
4. **PRISM Model Checking**: From the terminal, navigate to the home directory, and build the model in PRISM with the command:
```bash
<PATH-TO-PRISM> --javamaxmem 10g --cuddmaxmem 10g <PATH-TO-.pm-MODEL> <PATH-TO-.props-PROPS>
```
Increase/decrease java and cudd memory as needed.

5. **Validate**: Use `validate.py` to test the validity of the IMDP on novel data.

## Data and Models

- Sample data and intervals are stored in the `data/` subfolders.
- Trained weights for the estimator and policy are in `sample-weights/`.
- Generated PRISM models and properties are in `prism-models/` subfolders.

## Acknowledgements

- Mountain Car environment from OpenAI Gym.
- PRISM model checker: https://www.prismmodelchecker.org/

---

<sup>[1]</sup> Marta Kwiatkowska, Gethin Norman and David Parker. PRISM 4.0: Verification of Probabilistic Real-time Systems. In Proc. 23rd International Conference on Computer Aided Verification (CAV’11), volume 6806 of LNCS, pages 585-591, Springer, 2011.

For more details, see comments in the individual scripts.
