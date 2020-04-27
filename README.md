# Machine Learning Final Project
### Navigation
- [Machine Learning Final Project](#machine-learning-final-project)
    - [Navigation](#navigation)
    - [Goal](#goal)
    - [Background](#background)
  - [How to Run](#how-to-run)
### Goal 
***
Using provided Training data, build models to accurately predict the labels for the Test Data  
[Refer Here For More Information](ProblemStatement.md)
### Background
***
I was provided with Training Data and Training Labels to use to build a model through a classifier of my choice. These models are to be used on the Test Data to predict labels. These labels are known only by the professor and will serve as a representation of the efficitivty of the models that were built and implemented.

## How to Run
- Clone this repository to your local machine
  - <pre><code>git clone https://github.com/ConnorLacy/Preprocess-and-Classify.git</code></pre>
- Driver: `test_samples.py`
  - Flags:
    - Required:
      - --dataset &lt;dataset_number&gt;
        - Specify which dataset you would like to use by number.
        - E.g. 'TestData1.txt'
          - `python3 test_samples.py --dataset 1`
    - Optional:
      - --knn &lt;number_of_neighbors&gt;
        - Specify the number of neighbors you wish to use in KNN Classifier
        - Default value: 3
        - E.g. K=3
          - `python3 test_samples.py --dataset 1 --knn 3`
      - --verbose
        - Specifiy output verbosity
        - Default value: False
        - E.g. Verbose=True
          - `python3 test_samples.py --dataset 1 --verbose`