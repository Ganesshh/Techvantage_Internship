### Telecom Churn Prediction
**Objective**: Analyse the customer-level data of a leading telecom firm, build predictive models to identify the high  value customers at high risk of churn and identify the main indicators of churn.

Did exploratory data analysis (EDA) & Handled class imbalance using oversampling with ADASYN. Used PCA for dimensionality reduction & Carried out hyperparametric tuning Usin GridSearchCV.
Tried using different ML algorithms like logistic regression, SVM, Random forest classifier to build the mode for predicting whether high value customers will churn or not. Random forest classifer model is found to have an accuracy of 95% 

### Credit Card Default
**Objective**: To build a ML model that can predict whether a given customer defaults or not.

Did exploratory data analysis (EDA) & Handled class imbalance using oversampling with ADASYN. Used GridSearchCV for hyperparameter tuning & Standard scaler for feature scaling. Build model using random forest classifier with an accuracy of 83% & found important features

### Ecommerce Gender Prediction
**Objective**: To build a ML model that will predict the gender of the e-commerce consumers.

Tried predicting the genders of the e-commerce consumers using different classification algorithms. After pre-processing and building the models, compared their performance in terms of accuracy. Found that the dataset is not linearly separable, that’s why it doesn’t work well with Logistic Regression and SVC with linear kernel. It works well with SVC’s Radial basis function kernel with an accuracy of 93%.

### Smart Shelf
**Objective**: Determine a set of grocery 5-10 items. Identify the number of remaining items for each category of the grocery item present in the shelf. If the number of identified items is less than a particular threshold, a notification is sent to a procurement system to ask for more supplies of the particular item. 

Found datasets containing annotated images of single product on shelves. Used roboflow for creating dataset. Dataset contains 1200+ images of 5 different classes.
Used YoloV5 algorithm for the training the model & evaluated the model using mAP values. Got an overall mean average precision(mAP) score as 86%. Also created an Email alert system to ask for more supplies of the particular item using smtplib.
