# Problem 1

## Classification: 
Classification is to identify which category a new observation belongs to, on the basis of a training dataset. There are five datasets. For each dataset, we provide the training dataset, training label, and test dataset. Please use the training dataset and training label to build your classifier and predict the test label. A class label is represented by an integer. For example, in the 1st dataset, there are 4 classes where 1 represents the 1st class, 2 represents the 2nd class, etc. Note that, there exist some missing values in some of the dataset (a missing entry is filled by 1.00000000000000e+99), please fill the missing values before perform your classification algorithm.

```
- TrainData 1 contains 3312 features with 150 samples. 
- Testdata1 contains 3312 features with 53 samples. There are 5 classes in this dataset.

- TrainData 2 contains 9182 features with 100 samples. 
- Testdata2 contains 9182 features with 74 samples. There are 11 classes in this dataset.

- TrainData 3 contains 13  features with 6300 samples. 
- Testdata3 contains 13 features with 2693 samples. There are 9 classes in this dataset.

- TrainData 4 contains 112 features with 2547 samples. 
- Testdata4 contains 112 features with 1092 samples. There are 9 classes in this dataset.

- TrainData 5 contains 11 features with 1119 samples. 
- Testdata5 contains 11 features with 480 samples. There are 11 classes in this dataset.

- TrainData 6 contains 142 features with 612 samples. 
- Testdata5 contains 142 features with 262 samples. 
  - This is not a classification problem. You are asked to predict the real value. 
    (Graduate Students Only)
```

### Sample Data
***

#### Training data
|Row1|Row2|Row3|Row4|
| :---: | :---: | :---: | :---: |
|1.1|2.1|2.1|5.2|
|2.1|2.4|2.4|2.1|
|3.1|1.5|2.6|1.5|

#### Training label
| Row # | Prediction |
|:---:|:---:|
|1|1|
|2|1|
|3|2|

### Test data
***
|Row1|Row2|Row3|Row4|
| :---: | :---: | :---: | :---: |
|3.1|2.2|1.5|2.5|
|2.1|2.1|2.1|2.6|

Please use the training data and training label to predict the test label. For example, if your prediction for the test sample is 1, 2. That is, the first sample in the test dataset (first row) is predicted as 1 and second as 2. Then please return me the test result of each dataset as an individual files.

| Row # | Prediction |
|:---:|:---:|
|1|1|
|2|2|

# Problem 2
## Missing Value Estimation

Gene expression data often contain missing expression values and it is very important to estimate those missing value as accurate as possible. The first task of the course project is to estimate missing value in the Microarray Data.

```
- Dataset 1 contains 242 genes with 14 samples.

- Dataset 2 contains 758 genes with 50 samples.

- Dataset 3 contains 273 viruses with 79 samples. There are only 3815 observed values.
  (Bonus Questions for Undergraduate)
```

### Table 1
|Row1|Row2|Row3|
| :---: | :---: | :---: |
|1|1.00000000000000e+99|1.00000000000000e+99|
|1|1|1|
|2|2|2|


Note that the missing entry is filled by 1.00000000000000e+99.  For example, in the Table 1, the second and third entries in the first row are missing values. There are 4% missing values in the Dataset 1 and 10% missing values in the Dataset 2.  Please fill those missing entries with estimated values and return the complete dataset to me.