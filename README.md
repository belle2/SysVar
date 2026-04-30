# SysVar

Welcome to `SysVar`, a tool for ensuring consistency in the treatment of systematics.

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
    pip install -e .[dev]
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

## Setting the SysVar Path

**Please add the following two lines at the very beginning of every scipt that uses SysVar to point to the location where you have installed your SysVar fork**

```
import sys
sys.path.insert(0,'{path_where_you_pip_installed_sysvar}/SysVar/src')
```

This step is currently necessary because the basf2 developers have renamed their `pidvar` class to `sysvar`, anticipating an early merge of the SysVar package into basf2.

If you are running with an environment sourced from CVMFS and have not executed the code above, you will encounter the following error:

```
ModuleNotFoundError: No module named 'sysvar.utils'; 'sysvar' is not a package
```

## Further Reading

We do not yet have a dedicated Sphinx webpage (coming soon), so we refer interested users to the following Belle II internal material:

- [Technical talk at the Analysis Tools meeting](https://indico.belle2.org/event/12666/)
- [Recommendations talk for end users at the (S)L Working Group meeting](https://indico.belle2.org/event/12979/)
- [Presentation at the "Challenges in Semileptonic B Decays" outlining new gateways for measurement combinations](https://indico.cern.ch/event/1345421/contributions/6084737/)
- [Belle II note of the analysis that inspired the creation of SysVar (see especially Sections 3.1.4 and Appendix A)](https://docs.belle2.org/pub_data/documents/4547/)


## Contributing


There are various ways in which you could contribute by just discussing ideas, suggesting improvements, questions about usage/code or contributing to the codebase.

For this purpose, the first step would be to open an `Issue` with your idea/suggestion/question and choose an appropriate label(s) from the already existing ones (or open an issue to suggest a new label!):

-  `urgent`: Priority task.

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

5. Push the local branch to your repository and open a pull request for review. Mark it as draft if there are more commits expected. Add @itsaklid or @s6agagga as reviewer.


We encourage you to contribute and help in developing `Sysvar` further!

## 👥 Contributors

We gratefully acknowledge the following individuals for their code contributions to this project.

| Contributor               | Email                            | Commits | Contributions                                       |
|---------------------------|----------------------------------|---------|-----------------------------------------------------|
| Ilias Tsaklidis           | itsaklid@uni-bonn.de             | 304     | Original idea, main developer                       |
| Agrim Aggarwal            | s6agagga@uni-bonn.de             | 54      | Co-developer, documentation, testing                |
| Felix Metzner             | felixmetzner@outlook.com         | 4       | Feature additions, validation, feedback             |
| Georgios Alexandris       | galexand@uni-bonn.de             | 3       | CI/CD setup and maintenance, feedback               |
| Tristan Fillinger         | tristan.fillinger@kek.jp         | 2       | Plotting, Feauture additions, feedback              |
| Giacomo De Pietro         | giacomo.pietro@kit.edu           | 2       | CI/CD setup and maintenance                         |
| Maximilian Hoverath       | s6mahove@uni-bonn.de             | 2       | BF correction updates from PDG                      |
| Daniil Ivanov             | ivanovd@hepl.phys.nagoya-u.ac.jp | 2       | Bug-fixes, feedback                                 |
| Melisa-Melek Akdak        | makdag@uni-bonn.de               | 1       | Logo                                                |

> 📊 Commit counts are based on Git history (`git shortlog -sne`) and may include merge commits.
> They reflect authored commits; contributions via design, review, and discussion are also highly valued.


