# CM3070 Final Project

This is the code repository for the CM3070 Final Project module of UoL BSc in Computer Science. The project is a fake news detector, proposed within the scope of Natural Language Processing (NLP) techniques and theory. The project utilizes Jupyter notebooks for exploratory data analysis, visualization, and presenting results.

## Structure of the repository

```
CM3070-Final-Project
│
├── .gitignore
├── README.md
│
└── notebooks
    ├── exploratory
    └── hpc_cluster_scripts
        ├── myriad
        └── powerbook

```

- `notebooks/exploratory`: This directory stores the Jupyter notebooks. There are two main versions: v2 is the notebook to explore the statistical significance of sentiment analysis metadata and further exploration of the outcomes if they are used to enhance the measurement scores of traditional supervised ML algorithms with plots and graphs. This notebook replicated the same study as the **prototype notebook**but uses the **TruthSeeker 2023** dataset instead of the prototype **LIAR** dataset. The version **without "v2"** is the **general notebook that process the dataset in full as described in the report** without adding sentiment analysis metadata, i.e., pre-processing, classical ML training and scoring of different algorithms, and preparation, training and benchmark scoring of 4 different unsupervised BERT models.

- `notebooks/hpc_cluster_scripts`: This directory stores the scripts used **to study further the unsupervised BERT models** using the **UCL HPC cluster facility called Myriad** with Intel CPUs and NVidia GPUs. The limits of the local workstation, an Apple MacBook Pro with M1 Pro CPU, 32GN RAM and accelerated GPU, do not allow the training of the BERT models to the maximum values allowed for token length size and batch size (usually 512 and 32, respectively). The cluster also allowed the use of large versions of BERT.

- `notebooks/hpc_cluster_scripts/myriad`: This directory stores the **output of the different BERT models**. The hierarchy naming convention allows to find which model and which parameters were used.

- `notebooks/hpc_cluster_scripts/powerbook`: This directory stores the output of the **cluster code** but it was **executed locally** in the Apple MacBook Pro workstation. The same naming convention was used as the UCL HPC cluster.
