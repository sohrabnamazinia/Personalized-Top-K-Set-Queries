# Personalized Top-k Set Queries Over Predicted Scores

This repository contains the source code for the experiments presented in *Personalized Top-k Set Queries Over Predicted Scores*. The project introduces a computational framework to efficiently identify personalized top-k sets using minimal oracle (e.g., LLM) calls.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
  - [Datasets](#datasets)
  - [Scoring Functions](#scoring-functions)
  - [Baselines](#baselines)
- [Results](#results)
  - [Cost Analysis](#cost-analysis)
  - [Scalability Analysis](#scalability-analysis)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
The code implements a framework for answering personalized top-k set queries with user-defined scoring functions over multi-modal data. It aims to:
- Minimize the number of oracle calls required for scoring.
- Leverage probabilistic models to identify the top-k sets efficiently.

## Features
- Decomposable scoring functions supporting relevance, diversity, and other constructs.
- Probabilistic model for efficient candidate scoring and pruning.
- Comparison with baseline and random methods for top-k queries.

## Requirements
- Python 3.11 or above
- Required libraries:
  - LangChain
  - NumPy
  - Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sohrabnamazinia/Personalized-Top-K-Set-Queries.git
   cd Personalized-Top-K-Set-Queries
