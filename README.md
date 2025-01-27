# SysVar

Welcome to `SysVar`, a tool for ensuring consistency in treatment of systematics.

This documentation will guide the user to begin making use of it or even contribute to its development. Welcome onboard!

## Getting started


The best practice to follow for using/contributing to the package are as follows:

1. Fork the project [repository](https://gitlab.desy.de/itsaklid/sysvar): Click on the *Fork* button near the top of the
    page. This creates a copy of the code under your account.

2. Clone this copy to your local disk:

    ```bash
    git clone git@github.com:YourLogin/SysVar.git
    cd SysVar

    ```

<!-- 3. Once the repository has been cloned, a good habit is to create an isolated virtual environment to have consistency in the installed python packages and ensuring a smooth process while using `SysVar`. There exists a `environment.yaml` which hosts the dependencies required and a virtual  environment named `sysvar`. It can be created and activated via e.g. [Miniconda](https://docs.anaconda.com/miniconda/) -->
<!--
    ```bash
    conda env create -f environment.yaml
    conda activate sysvar
    ```
     -->
3. Now, you are ready to install and use the package with a simple:

    For developer mode:
    ```bash
    pip install -e .
    ```
    For end-users (not recommended currently):
    ```bash
    pip install .
    ```
4. (Optional: for developers) Install `pre-commit` ::

    ```bash
    pip install pre-commit
    pre-commit install
    ```

    This ensures that the code being written is not susceptible to any trivial styling issues.

## Usage

For getting familiar with the functioning and concepts of the package, a nice place to start is the example notebook located at `examples/minimal_example.ipynb`. It guides you through a prototype analysis from scratch where such a tool would be needed while explaining everything a front-end user needs to know about the package. Feel free to dig deeper into the code to access other cool stuff though!


## Contributing


There are various ways in which you could contribute by just discussing ideas, suggesting improvements, questions about usage/code or contributing to the codebase.

For this purpose, the first step would be to open an `Issue` with your idea/suggestion/question and choose an appropriate label(s) from the already existing ones (or open an issue to suggest a new label!):

-  `Urgent`: Priority task.

-  `bug`: An issue in the code, implementation of something etc.

- `doc`: Related to documentation.

- `easy-to-start`: A great pickup point for newcomers.

- `enhancement`: Improvement in a feature, structure etc.

- `feat`: Adding new features to the codebase.

- `help-needed`: Help in development.

- `performance`: Enhancing performance of the code.

- `question`: Questions regarding usage/code or general concept. (all questions are welcome and encouraged)

- `suggestion`: Suggestions related to code/concepts.



If you are interested in contributing by coding, the guidelines to follow are:

1. Create a branch and start making your changes. Never work on the main branch!

2. Don't forget to add docstrings to new functions, modules and classes. `Sysvar` uses [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

3. Add and commit your changes. A commit message helps in keeping track of the changes you have made and most importantly helps the reviewer understand the changes. Therefore, a good commit message goes a long way in making the editing process smoother and a few tips on writing [good commit messages](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/) are:
    - Labelling each commit message with a `type`. E.g. feat: a new feature is introduced with the changes. A few types that can be included in a commit message are: `feat, fix, chore, refactor, docs, style, test, perf, ci, build, revert`. These types are also recommended to be used for naming your branches.
    - A detailed commit message and not just keywords.

    - Each commit message should be [linked to the particular issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue) that you are working on. This can be done by including the relevant issue number followed by a keyword. E.g. `fix #11`.

4. Ensure that your commit passes the `pre-commit` hooks and fix any eventual issues using the styling fix recommended.

5. Push the local branch to your repository and open a pull request for review. Mark it as draft if there are more commits expected. Add @itsaklid and @s6agagga as reviewers.


We encourage you to contribute and help in developing `Sysvar` further!

