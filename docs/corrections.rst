.. _corrections:

============
Corrections
============

Calling the eigedecompose helper function for the API returns an :mod:`sysvar.eigendecomposer` object. From this object, the associated corrections can be accessed via the ``Eigendecomposer.correction`` attribute. 

The corrections supported by **SysVar** follow the types recommended by the Belle II *Performance Group*.
These are provided for the different Monte-Carlo campaigns; at this stage, the available campaign support includes MC15rd.

In addition, several branching-fraction (BF) corrections are included (this is not an exhaustive list). SysVar also provides tools for defining custom corrections, so users can implement analysis-specific corrections when needed.

As analysts important information to know about the structure for each type of corrections is listed below:

- **Correction1D**: These depend on a single variable which the correction weights are binned in. The corrections which fall under this category are:

    - Charged and neutral slow pion, 
    - Tracking efficiency, and
    - FEI

    The important attributes are:

        - **dependent_variable** : The name of the variable in the dataframe that will be queried.

        - **min** and **max** : The variable will be queried in these ranges. For e.g. if the variable to be queried is ``tag_mode``, ``min`` is [1,2] and ``max`` is [2,3]. The queries will be ``1 < tag_mode < 2`` and ``2 < tag_mode < 3``.

        - **corrections** : This could either be an array or the the csv format table path can be specified. Each correction weight corresponds to the queries built above in order.

        - **cov_matrix_path** : A path to a covariance matrix can be specified. If this is not available, then individual uncertainties are looked for and a covariance matrix is built.

        - **uncertainties** : If **cov_matrix_path** is empty, the uncertainties are read from the CSV table specified above.

                - **fully_correlated**: Can be a dictionary with statistical and systematical components
                - **uncorrelated**: Can be a dictionary with statistical and systematical components

- **Correction2D**: These depend on two variables which the correction weights are binned in. The corrections which fall under this category are:

    - Neutral pion

    The important attributes are:

        - **dependent_variable_1** : The name of the first variable in the dataframe that will be queried.
        - **dependent_variable_2** : The name of the second variable in the dataframe that will be queried.
        - **corrections**: Table path of the corrections

- **Correction3D**: These depend on three variables which the correction weights are binned in. The corrections which fall under this category are:

    - Kshort efficiency

    The important attributes are:

        - **dependent_variable_1** : The name of the first variable in the dataframe that will be queried.
        - **dependent_variable_2** : The name of the second variable in the dataframe that will be queried.
        - **dependent_variable_3** : The name of the three variable in the dataframe that will be queried.
        - **corrections**: Table path of the corrections

- **CorrectionBF**: MC events are generated using the best available branching-fraction (BF) values at the time of generation. If updated BF values or uncertainties (e.g. from *PDG*) become available, events can be reweighted accordingly, using correction weights with the appropriate uncertainties. A basic structure — with example corrections — is provided in SysVar, but these are analysis-dependent and should be adjusted to match the needs of each analysis.
    The types of decays for which BF corrections are currently available in SysVar include:

    - :math:`B` meson semileptonic decays
    - :math:`D^{(*,**)}` meson decays
    - :math:`B \to H_{c} H_{c} (H_{s})` Double charm B Meson decays
    - :math:`B \to D^{*} n \cdot h_{u}` Prompt hadronic B meson decays
    - :math:`\tau` lepton decays

    The important attributes are:

        - **dependent_variable** : The name of the first variable in the dataframe that will be queried.
        - **modes** : Dictionary of dictionary for modes to be queried and reweighted

            - **mode_name**: This is a dictionary containing the information about this particular mode_name

                - **dmID**: List of modes where the queried variable looks for
                - **pdg_live**: List of PDG values. First item is the new branching fraction value and second its uncertainty.
                - **decay_dec**: Branching fraction value with which the MC was generated
                - **daughters**: List of decay chain PDG numbers for visualisation purposes.

- **CorrectionPID**: The particle identification efficiency and fake rate corrections as recommended by the performance group. The tables should be generated with the `Systematics Framework`_ and the default path in the below mentioned correction file can be modified according to the analyst. This is essentially a 3D corrections but we have implemented a dedicated class to handle it due to the special format that the `Systematics Framework`_ saves the correction tables

    - :math:`e/\mu` identification efficiency
    - :math:`e/\mu` fake rates 
    - :math:`K/\pi` identification efficiency
    - :math:`K/\pi` fake rates

    The important attributes are:

        - **momentum_variable** : The name of the first variable in the dataframe that will be queried.
        - **theta_variable** : The name of the second variable in the dataframe that will be queried.
        - **PDG_variable** : The name of the third variable in the dataframe that will be queried.
        - **mcPDG_variable** : The name of the fourth variable in the dataframe that will be queried.
        - **variable**: The threshold cut variable
        - **online_cut** : The threshold cut 
        - **table_path**: The path of the table generated using the `Systematics Framework`_


-  **Custom Correction**: Analysis specific corrections can be created and provided in the similar format as above in a dictionary format.

.. note::
    
    To all the variables, the prefix specified in the api configuration file is appended. The analyst should consider this and engineer 
    the variables in the ntuples accordingly.

The full module reference for the correction types currently supported can be found at :mod:`sysvar.corrections`. 

.. _Systematics Framework:  https://syscorrfw.readthedocs.io/en/latest/#
