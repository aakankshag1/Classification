install.packages("caret")
library(caret)

# attach the iris dataset to the environment
data(iris)
View(iris)

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(iris$Species, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- iris[-validation_index,]

# use the remaining 80% of data to training and testing the models
iris <- iris[validation_index,]

# dimensions of dataset
dim(iris)

# list types for each attribute
sapply(iris, class)

# take a peek at the first 5 rows of the data
head(iris)

# list the levels for the class
levels(iris$Species)

# summarize the class distribution
percentage <- prop.table(table(iris$Species)) * 100
cbind(freq=table(iris$Species), percentage=percentage)

# summarize attribute distributions
summary(iris)

# split input and output
x <- iris[,1:4]
y <- iris[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

# barplot for class breakdown
plot(y)

install.packages("ellipse")
# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

install.packages("e1071")
# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=iris, method="lda", metric=metric, trControl=control)

# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=iris, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=iris, method="knn", metric=metric, trControl=control)

# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=iris, method="svmRadial", metric=metric, trControl=control)
install.packages("randomForest")
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=iris, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)
