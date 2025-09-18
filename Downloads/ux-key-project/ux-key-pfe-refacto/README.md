# Bayesian Calculator

## Introduction

The Bayesian Calculator is a sophisticated tool designed to evaluate the intuitiveness, fluidity, and rapidity of websites. It inputs CSV data about website performance and employs Bayesian statistics to identify the best-fitting distribution for the data. This tool then returns estimations of the aforementioned metrics, complete with confidence intervals and a likelihood value that indicates how well the model fits the data.

## Project Structure

This project is organized into several directories and files, each serving a specific function:

- **distributions/**: Contains modules that define and handle different statistical distributions.
- **outputs/**: Directory where every output type is defined, including confidence intervals, likelihood, and probability distribution functions.
- **scores/**: Directory where we define the scoring functions used for our metrics.
- **main.py**: The main Python script that connects the input data, the YAML file, and processes the pipelines.
- **params.yaml**: A YAML file for setting and adjusting the parameters used by the calculator.

## Prerequisites

Before you begin, ensure you have Python 3.x installed on your system. The project also depends on several Python libraries which are listed in `requirements.txt`.

## Installation

The best way to set up the environment for the Bayesian Calculator is by creating a new Conda environment:

```bash
conda create --name bayesian_calculator python=3.9
conda activate bayesian_calculator
pip install -r requirements.txt
```

## Usage 

First, complete the params.yaml file with the necessary parameters. Then run the following command:

```bash

python main.py <path to data.csv> <path to params.yaml>

```
exemple : --> python bayesian_calculator/bayesian_calculator/main.py data/endsomethingthou.csv bayesian_calculator/params.yaml




## Contact 

If you have any questions, feedback, or would like to get involved further, please reach out through GitHub issues or by emailing us at [mohamed.elmejdani@tudent-cs.fr].

