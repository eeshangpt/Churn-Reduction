rm(list = ls())

## Reading data
read.csv("Train_data.csv", header = T) -> churn.data
read.csv("Test_data.csv", header = T) -> churn.test
str(churn.data)
head(churn.data)
###################### PREPROCESSING ######################

## Checking for missing values
for (i in colnames(churn.data)) {
    print(i)
    print(sum(is.na(churn.data[i])))
}

for (i in c("international.plan", "voice.mail.plan")) {
    as.character(churn.data[, i]) -> churn.data[, i]
    as.character(churn.test[, i]) -> churn.test[, i]
    churn.data[, i][churn.data[, i] == " no"] <- 0
    churn.test[, i][churn.test[, i] == " no"] <- 0
    churn.data[, i][churn.data[, i] == " yes"] <- 1
    churn.test[, i][churn.test[, i] == " yes"] <- 1
    factor(churn.data[, i]) -> churn.data[, i]
    factor(churn.test[, i]) -> churn.test[, i]
}
rm(i)

as.character(churn.data[, "Churn"]) -> churn.data[, "Churn"]
as.character(churn.test[, "Churn"]) -> churn.test[, "Churn"]
churn.data[, "Churn"][churn.data[, "Churn"] == " False."] <- 0
churn.test[, "Churn"][churn.test[, "Churn"] == " False."] <- 0
churn.data[, "Churn"][churn.data[, "Churn"] == " True."] <- 1
churn.test[, "Churn"][churn.test[, "Churn"] == " True."] <- 1
factor(churn.data[, "Churn"]) -> churn.data[, "Churn"]
factor(churn.test[, "Churn"]) -> churn.test[, "Churn"]

## Target variable separated from features
y <- churn.data[, 21] # Training target
y.test <- churn.test[, 21] # Testing target
X <- churn.data[, 1:20] # Training features
X.test <- churn.test[, 1:20] # Testing features

## Removing columns of no use
drop.list <- c('area.code', 'phone.number', "state")
X <- X[, !names(X) %in% (drop.list)]
X.test <- X.test[, !names(X.test) %in% (drop.list)]
rm(drop.list)

numerical <- colnames(X[, sapply(X, is.numeric)])
categorical <- colnames(X[, sapply(X, is.factor)])

###################### OUTLIER ANALYSIS ######################

for (i in numerical) {
    print(i)
    X[, i][X[, i] %in% boxplot.stats(X[, i])$out] -> val
    
    boxplot(X[, i])
    
    val[val < boxplot.stats(X[, i])$stats[1]] -> val.below
    val[val > boxplot.stats(X[, i])$stats[5]] -> val.above
    print(paste0("The outlier above the maximum : ", length(val.above)))
    print(paste0("The outlier below the minimum : ", length(val.below)))
    cat("\n\n")
    
    boxplot.stats(X[, i])$stats[5] ->
        X[, i][X[, i] > boxplot.stats(X[, i])$stats[5]]
    boxplot.stats(X[, i])$stats[1] ->
        X[, i][X[, i] < boxplot.stats(X[, i])$stats[1]]
    
}
rm(list = c("i", "val", "val.above", "val.below"))

for (i in numerical) {
    print(i)
    X[, i] = (X[, i] - mean(X[, i])) / sd(X[, i])
}

for (i in numerical) {
    print(i)
    X.test[, i] = (X[, i] - mean(X[, i])) / sd(X[, i])
}

library(corrgram)
corrgram(X[, numerical],
         order = F,
         upper.panel = panel.cor,
         text.panel = panel.txt)

###################### MODEL DEVELOPMENT ######################

XX <- cbind(X, y)

library(caret)
library(randomForest)
rf.model <- randomForest(y ~ ., data = X, ntree = 400)
summary(rf.model)
print(rf.model$confusion)
pred <- predict(rf.model, X.test)
confusionMatrix(table(y.test, pred))

library(C50)
c50.model <- C5.0(X, y, rules = T)
summary(c50.model)
predict(c50.model, X.test) -> pred
confusionMatrix(table(y.test, pred))

logicistic.model <- glm(y ~ ., data = XX, family = 'binomial')
summary(logicistic.mode)
pred <-
    ifelse((predict(logicistic.model, X.test, type = 'response')) > 0.5, 1, 0)
confusionMatrix(table(y.test, pred))
