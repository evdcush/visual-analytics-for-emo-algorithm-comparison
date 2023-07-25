# VisSystem Backend

The backend server of comparative visual analytics for evolutionary processes. Programming in Python language.

## Environment Requirement

- **Python** version `>= 3.10.0` (Maybe you can run the backend on some Python versions that are out of date successfully, but we are not recommand for doing that since it could lead to some implicit problems.)
- **Windows OS** only (This is because some `dll` files used in sampling algorithm, if you want to run on other OS like Linus or MacOS, you can visit [LibSampling](https://github.com/thu-vis/libsampling) for possible alternative approaches)

## Deploy and Running

To start the backend, you need to perform the following steps:

1. Download the `data.zip` packet we provided and extract the files into the `./data` directory.
2. Start a terminal and go to the `backend` folder directory.
3. Ensure that you have a **correct** Python environment on your computer, and then execute `pip install -r requirement.txt` in the terminal.
4. Execute `python ./src/main.py` in the terminal.

## Data Extension

If you want to add data for a new algorithm or a new problem set, please refer to the following steps:

For **new problem sets**, create a new folder under the `./data` path and rename it with the dataset you wish to display in the frontend. Hereafter we will refer to this folder as `./data/NEW_PROBLEM`.

Under the path `./data/NEW_PROBLEM`, create a new file named `index.json`, which is the credentials used by the backend to recognize and load the data. The internal format of the file is as follows:

```json
{
  "dataSet": "NEW_PROBLEM",                         // The name of yout benchmark problem, equal to the folder name
  // Pleace create the following 4 folders in advance!
  "origin": "./origin",                             // Path of the original data
  "display": "./pca",                               // Path to save the calculated PCA data
  "distance": "./distance",                         // Path to save the calculated simularity data
  "attach": "./attach",                             // Path of the attached data (IGD distribution)
  "pca": true,                                      // Whether to run PCA while data preprocessing
                                                    // Please set to "true" if the number of objective is more than 2
  "reference": "NEW_PROBLEM_PF_set.json",           // The file name of the pareto front data (reference set)
  "info": {                                         // Information about the problem that will display
    "desc": "A fantastic EMO benchmarking problem", // The description of your problem
    "obj": 3,                                       // The number of objective in the problem
    "dec": 12                                       // The dimension of the decision space in the problem
  },
  "algorithms": {                                   // Original data files from different algorithms
    "Alg1": "Alg1_NEW_PROBLEM.json",                // The name of the data file from algorithm 1
    "Alg2": "Alg2_NEW_PROBLEM.json",                // The name of the data file from algorithm 2
    "Alg3": "Alg3_NEW_PROBLEM.json",                // The name of the data file from algorithm 3
    ...
  },
  "attachments": {                                  // Atteched data files from different algorithms
    "Alg1": "Alg1_NEW_PROBLEM.json",                // The name of the data file from algorithm 1
    "Alg2": "Alg2_NEW_PROBLEM.json",                // The name of the data file from algorithm 2
    "Alg3": "Alg3_NEW_PROBLEM.json",                // The name of the data file from algorithm 3
    ...
  }
}
```

Notice that all the algorithms you want to visualize in the frontend should be registered in the `index.json` file before.

The original data file is also a `json` file, the format is as follows, but how to convert the data from the output of your algorithm to such format is depend on you.

```json
{
  "result": {
    "obj": {
      "500": [            // Key for the evaluation or iteration number
        [8, 6, 4],        // Single line for an individual in the population
        [5, 1, 2],
        [7, 3, 9],
      ],
      ...
    },
  },
  "metric": {
    "HV": {
      "500": 3,           // Key for the evaluation or iteration number
      ...
    },
    "IGD": {
      "500": 3,
      ...
    },
    "MS": {
      "500": 3,
      ...
    },
    "SP": {
      "500": 3,
      ...
    },
  },
}
```

The attachment data files is only for the IGD disribution now, it is split from the data file is because the amount of data is too large and needs to be loaded dynamically. The format of the file is as follows:

```json
{
  "metric": {
    "VIS_IGD_distr": {
      "500": [3, 3, 3],   // Each number corresponds to the IGD value of an individual in the population
      ...
    },
  },
}
```

After creating the folder, configuring `index.json` and placing all the data files under the correct path, you can execute the scripts `convert.py` and `similarity.py` under the path `./data`, which will compute and generate the data after PCA dimensionality reduction as well as the similarity values between the iterations of all the algorithms.

After successfully completing all the above steps, you are ready to do visual analyze on your data in the frontend.