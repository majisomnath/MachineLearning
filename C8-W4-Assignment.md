---
title: "Machine Learning - Prediction Assignment"
author: "Somnath Maji"
date: "January 15, 2018"
output: html_document
---

###Executive Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

###Data Source
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

###Data Loading

```r
trainData <- read.csv("pml-training.csv", sep = ",", header = TRUE, na.strings = c('#DIV/0', '', 'NA'))
testData <- read.csv("pml-testing.csv", sep = ",", header = TRUE, na.strings = c('#DIV/0', '', 'NA'))
```

###Exploitory Data Analysis
Let’s take a look at the dimensions of the training and testing datasets, and also check variable CLASSE to see how it is distributed on given data.


```r
dim(trainData)
```

```
## [1] 19622   160
```

```r
dim(testData)
```

```
## [1]  20 160
```

```r
summary(trainData$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
# There are many columns with NA values, these columns will be removed now
trainData <- trainData[, colSums(is.na(trainData)) == 0]
testData <- testData[, colSums(is.na(testData)) == 0]

# remove unwanted 7 columns from last
trainData = trainData[,-c(1:7)]
testData = testData[,-c(1:7)]

# recheck data dimension
dim(trainData)
```

```
## [1] 19622    53
```

```r
dim(testData)
```

```
## [1] 20 53
```

###Cross Validation
The Training dataset will be splitted into two parts, the 70% data set will be used as Traning data and rest will be used for Test data for prediction. Below we are going to to fit Dicision Tree(DT), Random Forest(RF) and Linear Discriminant Analysis(LDA) models to see how they perform.


```r
# Set the seed for reproducibility
set.seed(4321)

# Load necessary libraries 
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(rattle)

#Split the Training dataset into Train and Test 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
cvTrainData <- trainData[inTrain,]
cvTestData <- trainData[-inTrain,]
#dim(cvTrainData)
```

###Desicion Tree Algorithm
A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds a class label. The topmost node in a tree is the root node. Let us use this algorithm to determine Test data prediction.


```r
# Dicision tree algo application and fancy plot the outcome
fitDT <- rpart(classe ~ ., data = trainData, method = "class")
fancyRpartPlot(fitDT)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![plot of chunk DecisionTree](figure/DecisionTree-1.png)

```r
# predicting the outcome variable(classe) on test dataset
predDT <- predict(fitDT, cvTestData, type = "class")

# Create Confusion Matrix on redicted outcome on test dataset
confDT <- confusionMatrix(predDT, cvTestData$classe)
confDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1494  177   15   51   19
##          B   53  726  101   66   98
##          C   45  119  825  143  141
##          D   59   69   52  634   54
##          E   23   48   33   70  770
## 
## Overall Statistics
##                                           
##                Accuracy : 0.756           
##                  95% CI : (0.7448, 0.7669)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6909          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8925   0.6374   0.8041   0.6577   0.7116
## Specificity            0.9378   0.9330   0.9078   0.9524   0.9638
## Pos Pred Value         0.8508   0.6954   0.6481   0.7304   0.8157
## Neg Pred Value         0.9564   0.9147   0.9564   0.9342   0.9369
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2539   0.1234   0.1402   0.1077   0.1308
## Detection Prevalence   0.2984   0.1774   0.2163   0.1475   0.1604
## Balanced Accuracy      0.9151   0.7852   0.8559   0.8051   0.8377
```

```r
#plot the confusion matrix
plot(confDT$table, col = confDT$byClass, main = "Decision Tree Confusion Matrix")
```

![plot of chunk DecisionTree](figure/DecisionTree-2.png)


###Random Forest Model Prediction
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.


```r
# Fit Random Forest(RF) model
#fitRF <- train(classe ~ ., data = trainData, method = "rf")
fitRF <- randomForest(classe ~ ., data = trainData)
plot(fitRF)
```

![plot of chunk RandomForest](figure/RandomForest-1.png)

```r
# Predicting the outcome variable on test dataset
predRF <- predict(fitRF, cvTestData)

# Create Confusion Matrix on redicted outcome on test dataset
confRF <- confusionMatrix(predRF, cvTestData$classe)
confRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    0 1026    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9994, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
#plot the confusion matrix
plot(confRF$table, col = confRF$byClass, main = "Random Forest Confusion Matrix")
```

![plot of chunk RandomForest](figure/RandomForest-2.png)

###Linear Discriminant Aalysis Prediction
Logistic regression is a classification algorithm traditionally limited to only two-class classification problems. If you have more than two classes then Linear Discriminant Analysis is the preferred linear classification technique.


```r
# Fit Linear discriminant analysis (LDA) model
fitLDA <- train(classe ~ ., data = trainData, method = "lda")

# Predicting with Linear discriminant analysis(LDA) model
predLDA <- predict(fitLDA, cvTestData)

# Create Confusion Matrix on LDA outcome on test dataset
confLDA <- confusionMatrix(predLDA, cvTestData$classe)
confLDA
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1363  187  108   63   55
##          B   27  730  118   35  168
##          C  147  130  663  105  110
##          D  131   36  112  719  113
##          E    6   56   25   42  636
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6986          
##                  95% CI : (0.6867, 0.7103)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6183          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8142   0.6409   0.6462   0.7459   0.5878
## Specificity            0.9019   0.9267   0.8987   0.9203   0.9731
## Pos Pred Value         0.7675   0.6772   0.5740   0.6472   0.8314
## Neg Pred Value         0.9243   0.9149   0.9233   0.9487   0.9129
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2316   0.1240   0.1127   0.1222   0.1081
## Detection Prevalence   0.3018   0.1832   0.1963   0.1888   0.1300
## Balanced Accuracy      0.8581   0.7838   0.7725   0.8331   0.7805
```

```r
#plot the confusion matrix
plot(confLDA$table, col = confLDA$byClass, main = "Linear Discriminant Aalysis Confusion Matrix")
```

![plot of chunk LinerDisc](figure/LinerDisc-1.png)

###Conclusion
The Random Forest (RF) model has an excellent performance, with 99.94% accuracy on the training dataset.
The Linear Discriminant Analysis(LDA) model and Dicision Tree(DT) have an inferior performance compared to previous model, it has 70% and 75% accuracy respectively.

We have found the Random Forest(RF) model has best fitted model and it will be used to make the out of the sample prediction, in this case, the testing dataset with new 20 samples.

###Random Forest predictions for Testing dataset

```r
predict(fitRF,testData)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
