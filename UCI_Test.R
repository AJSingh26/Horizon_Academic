library(dplyr)
library(nnet)
library(clue)
library(class)
library(MASS)
library(e1071)
library(randomForest)
library(glmnet)

set.seed(6)
pc <- read.csv("~/Desktop/UCI_Heart_Disease/processed.cleveland.txt")
##Imputing values of CA and THAL into ca_data
predict(ca_model, pc[pc$CA=='?',])
pc[pc$CA=='?',]$CA <- predict(ca_model, pc[pc$CA=='?',])

predict(thal_model, pc[pc$THAL=='?',])
pc[pc$THAL=='?',]$THAL <- predict(thal_model, pc[pc$THAL=='?',])

pc$CA <- as.numeric(pc$CA)
pc$THAL <- as.numeric(pc$THAL)
pc$DIAGNOSIS <- as.numeric(pc$DIAGNOSIS)

M1 <- lm(pc$DIAGNOSIS, data=pc)
M2 <- lm(pc$DIAGNOSIS~Age+Sex+Chest_Pain_Type+Resting_Blood_Pressure+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak+Slope_ST_Segment+CA+THAL, data=processed.cleveland)
summary(M1)
predict(M2, data=processed.cleveland)
#class(processed.cleveland$DIAGNOSIS)
#processed.cleveland$DIAGNOSIS <- as.factor(processed.cleveland$DIAGNOSIS)
with(pc,table(DIAGNOSIS, Chest_Pain_Type))

fun <- function(x) {
  return(x^2)
}

with(pc,do.call(rbind,tapply(Resting_Blood_Pressure, DIAGNOSIS, function(x) c(m = mean(x), SD = sd(x)))))
#relevel(pc$diagnosis, ref="0") -- COME BACK LATER

M3 <- multinom(diagnosis~Chest_Pain_Type+Resting_Blood_Pressure+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak+Slope_ST_Segment+CA+THAL, data=pc)
summary(M3)$coefficients/summary(M3)$standard.errors


length(pc$CA[pc$CA=='?']) #Does it make more sense to include CA as a factor or number?
pc[pc$CA=='?',]


# finished created ca_data, which removes the question marks and removes the question mark as a factor level
pc$CA <- as.factor(pc$CA)
ca_data <- pc[-c(167, 193, 288, 303), ]
ca_data$CA <- ca_data$CA[ca_data$CA!="?"]
levels(ca_data$CA)
table(ca_data$CA)
ca_data$CA <- factor(ca_data$CA)
levels(ca_data$CA)


# Removing the question marks in the THAL variable
ca_data$THAL <- as.factor(ca_data$THAL)
# 88, 265 is missing
ca_data <- ca_data[-c(88, 265), ]
ca_data$THAL <- ca_data$THAL[ca_data$THAL!="?"]
levels(ca_data$THAL)
table(ca_data$THAL)
ca_data$THAL <- factor(ca_data$THAL)
levels(ca_data$THAL)

# Creating testing/training set
training <- nrow(pc)*0.7
rowIndex <- sample(1:nrow(pc), training)
testSet <- pc[-rowIndex,]
trainingSet <- pc[rowIndex,]

# Multinomial Regression
multinom_results <- rep(NA, 6)
pc$THAL <- as.numeric(pc$THAL)
pc$CA <- as.numeric(pc$CA)
for (w in 1:6) {
  test_set_multinom <- pc[which(folds==w), 1:14]
  train_set_multinom <- pc[which(folds!=w), 1:14]
  
  multinom_model <- multinom(DIAGNOSIS~
                               Chest_Pain_Type+Resting_Blood_Pressure+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak+Slope_ST_Segment+CA+THAL
                             , data=train_set_multinom)
  predictions_multinom <- as.numeric(predict(multinom_model, test_set_multinom)) - 1
  real_multinom <- as.numeric(test_set_multinom$DIAGNOSIS)
  fold_res <- rep(NA, length(real_multinom))
  #multinom_results[w] <- sum(predictions_multinom==real_multinom)/length(real_multinom)
  for (i in 1:length(fold_res)) {
    if ((predictions_multinom[i] == 0 & real_multinom[i] != 0) | predictions_multinom[i] != 0 & real_multinom[i] == 0) {
      fold_res[i] <- 0
    } else if (abs(predictions_multinom[i] - real_multinom[i]) == 0) {
      fold_res[i] <- 1
    } else if (abs(predictions_multinom[i] - real_multinom[i]) == 1) {
      fold_res[i] <- 0.75
    } else if (abs(predictions_multinom[i] - real_multinom[i]) > 1) {
      fold_res[i] <- 0.5
    }
  }
  multinom_results[w] <- mean(fold_res)
}

mean(multinom_results) #Cross-validated accuracy for multinom for binomial classification is 82.81046%, cross-validated for multinomial classification is 74.66176%


# Trying to replace the question marks with new values
#Creating linear regression model for predicing CA value
ca_model <- multinom(CA~Age+Sex+Chest_Pain_Type+Resting_Blood_Pressure+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak+Slope_ST_Segment+DIAGNOSIS, data=trainingSet)
predictions_ca <- predict(ca_model, testSet)
real_res_ca <- testSet$CA
val_acc_ca <- sum(predictions_ca==real_res_ca)/length(predictions_ca)



ca_model_removed <- multinom(CA~Age+Sex+DIAGNOSIS, data=trainingSet)
predictions_ca_removed <- predict(ca_model_removed, testSet)
acc_ca_removed <- sum(predictions_ca_removed==real_res_ca)/length(real_res_ca)

app <- apply(pc,2,function(x) which(x=='?'))
predSet <- pc[app$CA,]
predict(ca_model, predSet)

#Finding mean variation
ca_combined <- list(c(predictions_ca), c(real_res_ca))
means_ca <- lapply(ca_combined, mean)
mean_diff_ca <- abs(means_ca[[1]] - means_ca[[2]])


#Creating linear regression model for predicting THAL values
thal_model <- multinom(THAL~Age+Sex+Chest_Pain_Type+Resting_Blood_Pressure+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak+Slope_ST_Segment+DIAGNOSIS, data=trainingSet)
predictions_thal <- predict(thal_model, testSet)
real_res_thal <- testSet$THAL
val_acc_thal <- sum(predictions_thal==real_res_thal)/length(predictions_thal)

#Finding mean variation
thal_combined <- list(c(predictions_thal), c(real_res_thal))
means_thal <- lapply(thal_combined, mean)
mean_diff <- abs(means_thal[[1]] - means_thal[[2]])


thal_model_removed <- multinom(THAL~Age+Sex+DIAGNOSIS, data=trainingSet)
predictions_thal_removed <- predict(thal_model_removed, testSet)
acc_thal_removed <- sum(predictions_thal_removed==real_res_thal)/length(real_res_thal)

# going to use the app variable from line 81
predSet_thal <- pc[app$THAL,]
predict(thal_model, predSet_thal)

#Imputing values of CA and THAL into ca_data
predict(ca_model, pc[pc$CA=='?',])
pc[pc$CA=='?',]$CA <- predict(ca_model, pc[pc$CA=='?',])

predict(thal_model, pc[pc$THAL=='?',])
pc[pc$THAL=='?',]$THAL <- predict(thal_model, pc[pc$THAL=='?',])

summary(ca_data$DIAGNOSIS)

pc$DIAGNOSIS[pc$DIAGNOSIS!=0] <- 1
pc$DIAGNOSIS

error_vec <- rep(NA, 6)
pc <- pc[sample(nrow(pc)),]
folds <- cut(1:nrow(pc), breaks=6, labels=FALSE)
which(folds==2)

# Lasso regression
lasso_df <- as.data.frame(matrix(1:16524, nrow=51))
lasso_lambda_err <- as.data.frame(matrix(1:336, nrow=56))
ridge_lambda_err <- as.data.frame(matrix(1:600, nrow=100))
for (i in 1:6) {
  train_set <- pc[which(folds!=i), 1:14]
  test_set <- pc[which(folds==i), 1:14]
  train_set$THAL <- as.numeric(train_set$THAL)
  test_set$THAL <- as.numeric(test_set$THAL)
  x <- model.matrix(DIAGNOSIS~
                      Chest_Pain_Type+Old_Peak+CA+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina
                    , data=train_set)[,-1]
  y <- as.numeric(train_set$DIAGNOSIS)
  lasso1 <- glmnet(x, y, family='binomial', alpha=1, lamba=NULL)
  ridge1 <- glmnet(x, y, family='binomial', alpha=0, lamba=NULL)
  x.test <- model.matrix(DIAGNOSIS~
                           Chest_Pain_Type+Old_Peak+CA+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina
                         , test_set)[,-1]
  
  predictions1 <- round(lasso1 %>% predict(newx = x.test, type='response'))
  for (w in 1:ncol(predictions1)) {
    lasso_lambda_err[w,i] <- sum(predictions1[,w]==test_set$DIAGNOSIS)/length(test_set$DIAGNOSIS)
  }
  
  predictions2 <- round(ridge1 %>% predict(newx = x.test, type='response'))
  for (w in 1:ncol(predictions2)) {
    ridge_lambda_err[w,i] <- sum(predictions2[,w]==test_set$DIAGNOSIS)/length(test_set$DIAGNOSIS)
  }
}

rowMeans(lasso_lambda_err[-(56:57),]) # best lambda is 0.006959, 77.86928%
which.max(rowMeans(lasso_lambda_err[-(56:57),]))

rowMeans(ridge_lambda_err) # best lambda is at the 76th row of ridge1, 0.244, 78.20261%
which.max(rowMeans(ridge_lambda_err))



#Normalizing function for KNN implementation
normalize <- function(list) {
  return ((list-min(list))/(max(list)-min(list)))
}

pc$CA <- as.numeric(pc$CA)
pc$THAL <- as.numeric(pc$THAL)

#KNN Evaluation, doesn't have the DIAGNOSIS label
knn_avg <- vector(mode = "list", length = 6)

for (w in 1:6) {
  test_set <- pc[which(folds==w),1:13]
  train_set <- pc[which(folds!=w),1:13]
  
  for (x in 1:ncol(train_set)) {
    train_set[,x] <- normalize(train_set[,x])
  }
  
  for (x in 1:ncol(test_set)) {
    test_set[,x] <- normalize(test_set[,x])
  }
  
  pc_train_labels <- pc[which(folds!=w),14]
  test_labels <- pc[which(folds==w),14]
  errors <- rep(NA, 31)
  
  for (i in 5:35) {
    errors[i-4] <- sum(knn(train=train_set, test=test_set, cl=pc_train_labels, k=i)==test_labels)/length(test_labels)
  }
  knn_avg[w] <- list(errors)
}
kresults <- as.data.frame(t(matrix(1:186, nrow=6, byrow=TRUE)))
kresults$V1 <- knn_avg[[1]]
kresults$V2 <- knn_avg[[2]]
kresults$V3 <- knn_avg[[3]]
kresults$V4 <- knn_avg[[4]]
kresults$V5 <- knn_avg[[5]]
kresults$V6 <- knn_avg[[6]]
which.max(colMeans(t(kresults))) # Best k-value for binomial classification is 26, best for multinomial classification is 13
### Using k=26 to predict on binomial classification and evaluate the results
knn_best_avg <- rep(NA, 6)
for (w in 1:6) {
  test_set <- pc[which(folds==w),1:13]
  train_set <- pc[which(folds!=w),1:13]
  for (x in 1:ncol(train_set)) {
    train_set[,x] <- normalize(train_set[,x])
  }
  
  for (x in 1:ncol(test_set)) {
    test_set[,x] <- normalize(test_set[,x])
  }
  pc_train_labels <- pc[which(folds!=w),14]
  test_labels <- pc[which(folds==w),14]
  knn_best_avg[w] <- sum(knn(train=train_set, test=test_set, cl=pc_train_labels, k=26)==test_labels)/length(test_labels)
  
}
mean(knn_best_avg) #Cross-validated accuracy of KNN for binomial classification is 77.51634%
### End
### Using k=13 to predict on multinomial classification and evaluate the results
knn_best_avg <- rep(NA, 6)
for (w in 1:6) {
  test_set <- pc[which(folds==w),1:13]
  train_set <- pc[which(folds!=w),1:13]
  for (x in 1:ncol(train_set)) {
    train_set[,x] <- normalize(train_set[,x])
  }
  
  for (x in 1:ncol(test_set)) {
    test_set[,x] <- normalize(test_set[,x])
  }
  pc_train_labels <- pc[which(folds!=w),14]
  test_labels <- pc[which(folds==w),14]
  predictions <- as.numeric(knn(train=train_set, test=test_set, cl=pc_train_labels, k= 13)) - 1
  knn_weighted_results <- rep(NA, length(test_labels))
  for (i in 1:length(test_labels)) {
    if ((predictions[i] == 0 & test_labels[i] != 0) | predictions[i] != 0 & test_labels[i] == 0) {
      knn_weighted_results[i] <- 0
    } else if (abs(predictions[i] - test_labels[i]) == 0) {
      knn_weighted_results[i] <- 1
    } else if (abs(predictions[i] - test_labels[i]) == 1) {
      knn_weighted_results[i] <- 0.75
    } else if (abs(predictions[i] - test_labels[i]) > 1) {
      knn_weighted_results[i] <- 0.5
    }
  }
  knn_best_avg[w] <- mean(knn_weighted_results)
}
mean(knn_best_avg) #The cross-validated accuracy for KNN on multinomial classification is 71.92484%
### End

knn_lists <- rep(NA, 30)

knn_avg[[1]]

for (w in 1:31) {
  for (i in 1:6) {
    knn_lists[w] <- sum(knn_avg[[i]][w])  
  }
}



#Ordinal logistic regression
olg_results <- rep(NA, 6)
pc$DIAGNOSIS <- as.factor(pc$DIAGNOSIS)
pc$THAL <- as.numeric(pc$THAL)
pc$CA <- as.numeric(pc$CA)
for (w in 1:6) {
  test_set_olg <- pc[which(folds==w),1:14]
  train_set_olg <- pc[which(folds!=w),1:14]
  ord_model <- polr(DIAGNOSIS~Chest_Pain_Type+Resting_Blood_Pressure+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak+Slope_ST_Segment+CA+THAL, data=train_set_olg, Hess=TRUE)
  t_values <- unname(summary(ord_model)$coefficients[,"t value"])
  p_values <- pnorm(abs(t_values), lower.tail = FALSE)*2
  predictions <- as.numeric(predict(ord_model, newdata=test_set_olg)) - 1
  real_res <- as.numeric(test_set_olg$DIAGNOSIS) - 1
  olg_weighted_results <- rep(NA, length(real_res))
  for (i in 1:length(real_res)) {
    if ((predictions[i] == 0 & real_res[i] != 0) | predictions[i] != 0 & real_res[i] == 0) {
      olg_weighted_results[i] <- 0
    } else if (abs(predictions[i] - real_res[i]) == 0) {
      olg_weighted_results[i] <- 1
    } else if (abs(predictions[i] - real_res[i]) == 1) {
      olg_weighted_results[i] <- 0.75
    } else if (abs(predictions[i] - real_res[i]) > 1) {
      olg_weighted_results[i] <- 0.5
    }
  }
  olg_results[w] <- mean(olg_weighted_results) 
}

mean(olg_results) #Cross-validated accuracy for Ordinal Logistic Regression on multinom classification=72.69608%

#Support Vector Machine
svm_results <- rep(NA, 6)
svm_results_mean <- rep(NA, 6)

results_df <- as.data.frame(matrix(1:306, nrow=6))

for (w in 1:6) {
  
  test_set_SVM <- pc[which(folds==w),1:14]
  train_set_SVM <- pc[which(folds!=w),1:14]
  svm_model <- svm(DIAGNOSIS~
                     Chest_Pain_Type+Old_Peak+CA+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina
                   , data=train_set_SVM, type='C-classification', kernel='linear')
  # Resting_Blood_Pressure+Chest_Pain_Type+THAL+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak = 81.37255, mean = 66.66667
  # Chest_Pain_Type+THAL+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak = 80.88235%, mean= 68.62745
  predict(svm_model, newdata = test_set_SVM)
  mean(predict(svm_model, newdata = test_set_SVM)==test_set_SVM$DIAGNOSIS)
  
  predictions <- as.numeric(predict(svm_model, newdata = test_set_SVM)) - 1
  svm_results <- as.numeric(test_set_SVM$DIAGNOSIS)
  for (i in 1:nrow(test_set_SVM)) {
    if ((predictions[i] == 0 & svm_results[i] != 0) | predictions[i] != 0 & svm_results[i] == 0) {
      results_df[w, i] <- 0
    } else if (abs(predictions[i] - svm_results[i]) == 0) {
      results_df[w, i] <- 1
    } else if (abs(predictions[i] - svm_results[i]) == 1) {
      results_df[w, i] <- 0.75
    } else if (abs(predictions[i] - svm_results[i]) > 1) {
      results_df[w, i] <- 0.5
    }
  }
  svm_results_mean[w] <- mean(predictions==svm_results)
}

mean(svm_results_mean)
results_df_a <- results_df[c(1, 3, 6),]
results_df_b <- results_df[c(2, 4, 5),-51]
mean(c(rowMeans(results_df_a),rowMeans(results_df_b))) # Overall cross-validated performance for the SVM binomial, 83.15033%



## Alternate SVM evaluation that goes along the lines of all the other evaluations (a.k.a. doesn't use a data frame to store weighted accuracies)
svm_acc <- rep(NA, 6)

for (w in 1:6) {
  test_set_SVM <- pc[which(folds==w),1:14]
  train_set_SVM <- pc[which(folds!=w),1:14]
  svm_model <- svm(DIAGNOSIS~Resting_Blood_Pressure+Chest_Pain_Type+THAL+Serum_Cholesterol+Fasting_Blood_Sugar+Resting_Electrocardiographic_Results+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina+Old_Peak+CA, data=train_set_SVM, type='C-classification', kernel='linear')
  predictions <- as.numeric(predict(svm_model, newdata = test_set_SVM)) - 1
  svm_results <- as.numeric(test_set_SVM$DIAGNOSIS) - 1
  svm_fold_avg <- rep(NA, length(test_set_SVM))
  for (i in 1:length(test_set_SVM)) {
    if ((predictions[i] == 0 & svm_results[i] != 0) | predictions[i] != 0 & svm_results[i] == 0) {
      svm_fold_avg[i] <- 0
    } else if (abs(predictions[i] - svm_results[i]) == 0) {
      svm_fold_avg[i] <- 1
    } else if (abs(predictions[i] - svm_results[i]) == 1) {
      svm_fold_avg[i] <- 0.75
    } else if (abs(predictions[i] - svm_results[i]) > 1) {
      svm_fold_avg[i] <- 0.5
    }
  }
  
  svm_acc[w] <- mean(svm_fold_avg)
}

mean(svm_acc) #Overall cross-validated accuracy on SVM for multinomial classification is 74.10714%, binomial classification is 86.90476%

# Random Forest Implementation
results_rf <- as.data.frame(matrix(1:306, nrow=6))

rf_cross_results <- rep(NA, 6)

for (f in 1:6) {
  
  rf_train_set <- pc[which(folds!=f),1:14]
  rf_test_set <- pc[which(folds==f),1:14]
  rf_train_set$DIAGNOSIS <- as.factor(rf_train_set$DIAGNOSIS)
  rf_test_set$DIAGNOSIS <- as.factor(rf_test_set$DIAGNOSIS)
  rf <- randomForest(DIAGNOSIS~
                       Chest_Pain_Type+Old_Peak+CA+Maximum_Heart_Rate_Achieved+Exercise_Induced_Angina
                     , data = rf_train_set, ntree=100, mtry=3, importance=TRUE)
  rf_results <- as.numeric(unname(predict(rf, newdata = rf_test_set))) - 1 
  mean(rf_results==rf_test_set$DIAGNOSIS)
  rf_results <- as.numeric(rf_results)
  rf_test_set$DIAGNOSIS <- as.numeric(rf_test_set$DIAGNOSIS) - 1
  rf_res_vec <- rep(NA, 51)
  for (i in 1:nrow(rf_test_set)) {
    if ((rf_results[i] == 0 & rf_test_set$DIAGNOSIS[i] != 0) | rf_results[i] != 0 & rf_test_set$DIAGNOSIS[i] == 0) {
      results_rf[f, i] <- 0
    } else if (abs(rf_results[i] - rf_test_set$DIAGNOSIS[i]) == 0) {
      results_rf[f, i] <- 1
    } else if (abs(rf_results[i] - rf_test_set$DIAGNOSIS[i]) == 1) {
      results_rf[f, i] <- 0.75
    } else if (abs(rf_results[i] - rf_test_set$DIAGNOSIS[i]) > 1) {
      results_rf[f, i] <- 0.5
    }
  }
  
  rf_cross_results[f] <- mean(rf_res_vec)
  
}

mean(rf_results==rf_test_set$DIAGNOSIS)
results_rf_a <- results_rf[c(1, 3, 6),]
results_rf_b <- results_rf[c(2, 4, 5),-51]
mean(c(rowMeans(results_rf_a),rowMeans(results_rf_b))) #Overall cross-validated data from the random forest for multi-class classification: 70.88562%, for the 0-1 classification it was 81.81699%


#for the glm model, family=binomial, type='response' it was 74.26144%
#for the lm model (all specs stay the same, family gets disregarded), it was 73.60131%
# for the plogis mode, it was 45.9281%
# for the rlm (family and type removed), it was 74.91503%
#for the knn implementation, it was 86.27451%

# KNN Implementation
# test_set <- pc[which(folds==i),]
# train_set <- pc[which(folds!=i),]
# 
# #This is just for knn
# pc_train_labels <- pc[which(folds!=i),14]
# knn_model <- as.numeric(knn(train=train_set, test=test_set, cl=pc_train_labels, k=10))
# 
# error_vec[i] <- sum(test_set$DIAGNOSIS==round(knn_model, 0))/nrow(test_set)

# Evaluation Metric
#error_vec[i] <- sum(test_set$DIAGNOSIS==round(predict(glm(DIAGNOSIS~Resting_Blood_Pressure+
#Chest_Pain_Type, data=train_set, 
#family=binomial), newdata=test_set,
#type='response'), 0))/nrow(test_set)

