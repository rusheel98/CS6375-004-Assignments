# Assignment 1

## Instructions to run
1. Install all the necessary libraries using 
    ```
    pip install -r requirements.txt
    ```
2. To run individual model, you may run the respective part files, `part1` code corresponding to manual implementation of gradient descent, while `part2`
 contains code corresponding to using `sklearn` implementation for the same.
    ```
    python part1.py <path_to_data_file>
    ```
    ```
    python part2.py <path_to_data_file>
    ```
3. Execute the `experiments.py` file to run comparative experiments between both the models and generate all the plots attached in the report.
    ```
    python experiments.py <path_to_data_file>
    ```

**NOTE:** If the `path_to_data_file` isn't provided, then the script assumes the data is available at `./data/` as `abalone.data`.

For the `part 2` of this assignment the `SGDRegressor` implementation of linear regression model has been used.

## Data
The data used for training and testing has been saved over [here](https://github.com/chaitanya-basava/CS6375-004-Assignment-1-data/blob/main/abalone.data).
Please download this file and pass its path in your machine as instructed above to execute the scripts.
