<div align="center">

# xhec-mlops-project-student

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

This repository has for purpose to industrialize the [Abalone age prediction](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) Kaggle contest.

<details>
<summary>Details on the Abalone Dataset</summary>

The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age.

**Goal**: predict the age of abalone (column "Rings") from physical measurements ("Shell weight", "Diameter", etc...)

You can download the dataset on the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

</details>

## Table of Contents

## Table of Contents

- [xhec-mlops-project-student](#xhec-mlops-project-student)
  - [Table of Contents](#table-of-contents)
  - [Deliverables and Evaluation](#deliverables-and-evaluation)
    - [Deliverables](#deliverables)
    - [Evaluation](#evaluation)
  - [Steps to reproduce to build the deliverable](#steps-to-reproduce-to-build-the-deliverable)
    - [Pull requests in this project](#pull-requests-in-this-project)
    - [Tips to work on this project](#tips-to-work-on-this-project)
- [Instructions](#instructions)
  - [Using Mlflow UI](#using-mlflow-ui)
  - [Deploying with Prefect](#deploying-with-prefect)
  - [Dockerization](#dockerization)


## Deliverables and notation

### Deliverables

The deliverable of this project is a copy of this repository with the industrialization of the Abalone age prediction model. We expect to see:

1. a workflow to train a model using Prefect
- The workflows to train the model and to make the inference (prediction of the age of abalone) are in separate modules and use Prefect `flow` and `task` objects
- The code to get the trained model and encoder is in a separate module and must be reproducible (not necessarily in a docker container)
2. a Prefect deployment to retrain the model regularly
3. an API that runs on a local app and that allows users to make predictions on new data
  - A working API which can be used to make predictions on new data
    - The API can run on a docker container
    - The API has validation on input data (use Pydantic)

### Evaluation

Each of your pull requests will be graded based on the following criteria:

- **Clarity** and quality of code
  - good module structure
  - naming conventions
  - use of docstrings and type hinting
- **Formatting**
  - respect of clear code conventions

  *P.S. you can use a linter and automatic code formatters to help you with that*

- Proper **Functioning** of the code
  - the code must run without bugs

Bseides the evaluation of the pull requests, we will also evaluate:
- **Reproducibility** and clarity of instructions to run the code (we will actually try to run your code)
  - Having a clear README.md with
    - the context of the project
    - the name of the participants and their github users
    - the steps to recreate the Python environment
    - the instructions to run all parts of the code
- Use of *Pull Requests* (see below) to coordinate your collaboration

## Steps to reproduce to build the deliverable

To help you with the structure and order of steps to perform in this project, we created different pull requests templates.
Each branch in this repository corresponds to a future pull request and has an attached markdown file with the instructions to perform the tasks of the pull request.
Each branch starts with a number.
You can follow the order of the branches to build your project and collaborate.

> [!NOTE]
> There are "TODO" in the code of the different branches. Each "TODO" corresponds to a task to perform to build the project.
> [!IMPORTANT]
> Remember to remove all code that is not used before the end of the project (including all TODO tags in the code).

**Please follow these steps**:

- If not done already, create a GitHub account
- If not done already, create a [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (so you can download the dataset)
- Fork this repository (one person per group)

**WARNING**: make sure to **unselect** the option "Copy the `master` branch only", so you have all the branches in the forked repository.

- Add the different members of your group as admin to your forked repository
- Follow the order of the numbered branches and for each branch:
  - Read the PR_i.md (where i is the number of the branch) file to understand the task to perform
   > [!NOTE]
   > Dont forget to integrate your work from past branches (except for when working on branch #1 obviously (!))
   > ```bash
   > git checkout branch_number_i
   > git pull origin master
   > # At this point, you might have a VIM window opening, you can close it using the command ":wq"
   > git push
   > ```
    - Read and **follow** all the instructions in the the PR instructions file
    - Do as many commits as necessary on the branch_number_i to perform the task indicated in the corresponding markdown file
    - Open **A SINGLE** pull request from this branch to the main branch of your forked repository
    - Once done, merge the pull request in the main branch of your forked repository

### Pull requests in this project

Github [Pull Requests](https://docs.github.com/articles/about-pull-requests) are a way to propose changes to a repository. They have for purpose to integrate the work of *feature branches* into the main branch of the repository, with a collaborative review process.

**PR tips:**

Make sure that you select your own repository when selecting the base repository:

![PR Wrong](assets/PR_wrong.png)

It should rather look like this:

![PR Right](assets/PR_right.png)

### Tips to work on this project

- Use a virtual environment to install the dependencies of the project (conda or virtualenv for instance)

- Once your virtual environment is activated, install pre-commit hooks to automatically format your code before each commit:

```bash
pip install pre-commit
pre-commit install
```

This will guarantee that your code is formatted correctly and of good quality before each commit.

- Use a `requirements.in` file to list the dependencies of your project. You can use the following command to generate a `requirements.txt` file from a `requirements.in` file:

```bash
pip-compile requirements.in
```


# Instructions

### Using Mlflow UI

Run the whole modelling.ipynb Notebook.

```bash
mlflow ui
```

You can see here the different models in our experiment:

![image3](images/image3.png)

and here is the best model of our last run (LR):

![image4](images/image4.png)


### Deploying with Prefect

```markdown
Illustrate the deployment process for retraining the model using Prefect.
```

```bash
prefect config set PREFECT_API_URL='http://localhost:4200/api'
```

#### Steps to Follow

1. **Run the Correct Command:** Use the command above to set the `PREFECT_API_URL`.

2. **Verify the Configuration:** After setting it, verify that the change took effect by running:

    ```bash
    prefect config view
    ```

   It should display:

    ```plaintext
    PREFECT_API_URL='http://localhost:4200/api'
    ```

3. **Start the Prefect Server:** If you haven’t already, start the Prefect server in a separate terminal:

    ```bash
    prefect server start
    ```

4. **Run Your Deployment Script:** Now that you’ve set the API URL correctly, run your `deployment.py` script:

    ```bash
    python \src\modelling\deployment.py
    ```

#### Outputs
When you navigate to http://0.0.0.0:4200/deployments, click on 'train-model' to view the scheduled retraining of the model, as shown below:

![image1](images/image1.png)


"Click on 'Quick Run' to train the model immediately, then scroll to the bottom of the page and select the most recent run:"
![image2](images/image2.png)

### Dockerization

#### Cleaning of the requirement.in

We used this command to clean the requirements.in in the last pull request
    ```bash
    pipreqs . --savepath requirements.in
    ```
#### Building & Running the Docker Image

This builds a Docker image from the Dockerfile.app, tagging it as api-with-prefect.

  ```bash
  docker build -t api-with-prefect -f Dockerfile.app .
  ```

This runs a container from the api-with-prefect image, exposing ports 8000 and 4200 for external access to the app and Prefect UI.

  ```bash
  docker run -p 8000:8000 -p 4200:4200 api-with-prefect
  ```
