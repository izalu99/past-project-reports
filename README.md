# Project 1
---
title: 'Project 1 : Predicting the weight of bananas'
author: "Iza"
date: "2/6/2022"
output:
  pdf_document: default
  word_document: default
  html_document:
    df_print: paged
---

```{r, setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(fig.width = 8)
knitr::opts_chunk$set(fig.show = T)
knitr::opts_chunk$set(fig.cap = T)
```

```{r wrap-hook, include=FALSE}
library(knitr)
hook_output = knit_hooks$get('output')
knit_hooks$set(output = function(x, options) {
  # this hook is used only when the linewidth option is not NULL
  if (!is.null(n <- options$linewidth)) {
    x = knitr:::split_lines(x)
    # any lines wider than n should be wrapped
    if (any(nchar(x) > n)) x = strwrap(x, width = n)
    x = paste(x, collapse = '\n')
  }
  hook_output(x, options)
})
```

```{r include=FALSE}
library(knitr)
library(ggplot2)
library(GGally)  # an extension to ggplot2
library(Metrics)
```

```{r, echo=FALSE}
# import the banana data
bdata = readxl::read_xlsx("Copy of Banana Data.xlsx")

colnames(bdata) = c("weight","length","radius")

```

```{r, echo=FALSE}
# split data into train and test subsets
# test data are rows:1,8,12,17,18,23,24,28,44,45,47,53,55,69,76,77
testrows = c(1,8,12,17,18,23,24,28,44,45,47,53,55,69,76,77)
banana_train = bdata[-testrows,] 
banana_test = bdata[testrows,]

```


## Summary
This report is about predicting the weight of bananas (given its radius and length), by using linear regression. A sample data of size, n= 78 is collected,then split into training dataset and test data set for cross validation of the model. The Mean Absolute Value and Mean Absolute Percent Error are also used for further evaluation of the models. A weighted linear model is generated to be the final linear model. 

## Introduction
### Background
A known formula for computing the weight of an object is 
$Weight = Density \times Volume.$ If a cylinder has the volume defined as $Volume = \pi \times radius^2 \times length$ then, we would have an approximate formula of weight as 
$Weight = Density \times \pi \times radius^2 \times length.$ \



### Purpose
If we assume that a banana is technically a cylinder then we can approximate its weight given its length and radius. Since the log function $log()$ has the property of $log(XY^m) = log(X) + mlog(X)$ we can use it to transform the weight equation above into a linear form. By doing so we can use a linear regression model to predict the weight of a banana. Applying the log function the new equation we use for linear regression is
$log(Weight) = log(Density) + log(\pi) + 2log(radius) + log(length).$ \

Then further modified into 
$log(Weight) = \beta_0 + \beta_1 log(radius) + \beta_2 log(length).$\

\
where we have the constants in $\beta_0.$ 
\

The linear regression will then estimate the values of $\beta_0, \beta_1, \beta_2$ 

## Methods

### Data collection
Each student is tasked to measure the radius, length and weight of six bananas which is compiled together into a dataset with 78 rows and 3 columns. 


### Exploratory data analysis
To get a glimpse of the data, a correlation plot is generated. From the pairwise plot below, its found that the variables have weak to moderate linear relationship with each other.

```{r, echo=FALSE, fig.cap= "Pairwise plot with the correlation of each variable in the data"}
ggpairs(data = banana_train, 
        axisLabels = "show")
```

### Model Building
There are three linear models to choose from: \

- $log(Weight) = \beta_0 + \beta_1 log(radius) + \beta_2 log(length) + \varepsilon$\
- $log(Weight) = \beta_0 + \beta_1 log(length) + \varepsilon$\
- $log(Weight) = \beta_0 + \beta_1 log(radius) + \varepsilon$ \
\
For each model, we use the *lm()* to generate the estimates of the coefficients, $\beta_0, \beta_1$, and the residuals,$\varepsilon$. But, before applying *lm()*, the original  dataset is split into a training and test set where, we use the train set to build the models. Once the models are built,the models are assessed by comparing their Mean Absolute Error(MAE) and Mean Absolute Percent Error(MAPE). The model that yields the lowest MAE and MAPE is chosen to be the model for prediction. The assumptions for linear regression have to be checked on each model, if any assumption is validated, the model is modified. \


## Results

Once the models are generated, the estimated coefficients are on the table below. \
```{r =FALSE}

linfit_full = lm(log(weight) ~ log(length) + log(radius),data=banana_train)   #all variables
linfit_l = lm(log(weight)~log(length), data=banana_train)                    #just length as x
linfit_r = lm(log(weight)~log(radius), data=banana_train)                    #just radius as x
linfit_none = lm(weight~1, data=banana_train)                   #empty; used for forward step, model selection

summary(linfit_full)
summary(linfit_l)
summary(linfit_r)
summary(linfit_none)
```
```{r echo=F}
linfit_b0 = c(linfit_full$coefficients[1], linfit_l$coefficients[1], linfit_r$coefficients[1])
linfit_b1 = c(linfit_full$coefficients[2], linfit_l$coefficients[2], linfit_r$coefficients[2])
linfit_b2 = c(linfit_full$coefficients[3], NA, NA)
linfit_coefs = 
  cbind.data.frame(
  c("linfit_full","linfit_l","linfit_r"),
  linfit_b0,linfit_b1,linfit_b2)
names(linfit_coefs)[1] ="model"
kable(linfit_coefs, format="pipe")
```
\
\
\
\
\
Thus, the linear models are: \

- **linfit_full**: $log(Weight) = 0.918 + (0.121) log(radius) + (1.182) log(length)$ \
- **linfit_l**: $log(Weight) = 3.492 + (0.281) log(length)$ \
- **linfit_r**: $log(Weight) = 1.294 + (1.278) log(radius)$ \

From just the adjusted R-squared values of each model, **linfit_r** had the highest value of $0.2092$. This means that the predictor in **linfit_r*** (radius), is responsible of the response variable's (weight) variablity of about 20.92%. When the Step wise method is used for Akaike Information Criterion (AIC) and Bayesian Information Criterion(BIC), the model produced is the as **linfit_r**. So by comparing the adjusted R-squared values and performing variable selection by AIC and BIC, **linfit_r** seems to be the best model among the three. \

```{r include=FALSE}
# backwards, using both length and radius variables
step(linfit_full, direction="backward", trace=F)

# forward...
step(linfit_none, direction="forward", scope = list(lower= linfit_none, upper= linfit_full), trace=F)

# Result is the model with radius as the predictor
```

### Results of cross validation

```{r include=FALSE}
# using the model with all the variables (l and  r)
pred.full = predict(object = linfit_full, newdata = banana_test)

# with just length (l)
pred.l = predict(object = linfit_l, newdata = banana_test)

# with just radius (r)
pred.r = predict(object = linfit_r, newdata = banana_test)
```

```{r include=FALSE}
# MAE - Mean absolute error
# MPAE - Mean Percent absolute error
# function to compute MAE
computeMae = function(measured,predicted,n){
  summed_term = 0
  for (i in 1:n){
    abs_term = abs(measured[i]-predicted[i])
    summed_term = summed_term + abs_term
  }
  
  MAE = summed_term/n
  
  return(MAE)
}
```

```{r include=FALSE}
#----MAE
MAE_train_full = computeMae(log(banana_train$weight), linfit_full$fitted.values, n=nrow(banana_train))
MAE_test_full = computeMae(log(banana_test$weight), log(pred.full), n= nrow(banana_test))
mae_full = c(MAE_train_full, MAE_test_full)

MAE_train_l = computeMae(log(banana_train$weight), linfit_l$fitted.values, n= nrow(banana_train))
MAE_test_l = computeMae(log(banana_test$weight), log(pred.l), n= nrow(banana_test))
mae_l = c(MAE_train_l, MAE_test_l)

MAE_train_r = computeMae(log(banana_train$weight), linfit_r$fitted.values, n= nrow(banana_train))
MAE_test_r = computeMae(log(banana_test$weight), log(pred.r), n= nrow(banana_test))
mae_r = c(MAE_train_r, MAE_test_r)

mae_df = cbind.data.frame(mae_full, mae_l, mae_r)

rownames(mae_df) = c("weight(train)","weight(test)")
colnames(mae_df) = c("predicted weight(full)", "predicted weight(length)", "predicted weight(radius)")

mae_table = kable(mae_df,format= "pipe", row.names = T)


#----MAPE
# use mape() in Metrics package
MAPE_train_full = mape(log(banana_train$weight), linfit_full$fitted.values)
MAPE_test_full = mape(log(banana_test$weight), log(pred.full))
mape_full = c(MAPE_train_full,MAPE_test_full)

MAPE_train_l = mape(log(banana_train$weight), linfit_l$fitted.values)
MAPE_test_l = mape(log(banana_test$weight), log(pred.l))
mape_l = c(MAPE_train_l, MAPE_test_l)

MAPE_train_r = mape(log(banana_train$weight), linfit_r$fitted.values)
MAPE_test_r = mape(log(banana_test$weight), log(pred.r))
mape_r = c(MAPE_train_r, MAPE_test_r)

#put in df
mape_df = cbind(mape_full, mape_l, mape_r)
rownames(mape_df) = c("obs.weight(train)"," obs.weight(test)")
colnames(mape_df) = c("pred.weight(full)", "pred.weight(length)", "pred.weight(radius)")

#then table
mape_table = kable(mape_df, row.names = T)
```

In order to perform a cross validation, the models are used to predict the weight of bananas using the test dataset. The resulting MAE and MAPE values are: \

```{r echo=TRUE, fig.cap="MAE table", paged.print=FALSE}
mae_table
# The lesser the percent value of mae/mape, the more accurate the model
```
*MAE values; The column name correspond to the which model generated the predicted values and the row name is where the observed values are from.*
\
```{r echo=TRUE, fig.cap="MAE table", paged.print=TRUE}
mape_table
```
*MAPE values; The column name correspond to the which model generated the predicted values and the row name is where the observed values are from.* \

From the MAE and MAPE tables, we can see that, the linear model  with radius as the sole predictor(**linfit_r**) and the model with length as the predictor(**linfit_l**),have the same values of MAE and MAPE when the test dataset is used to predict the weight. If we consider the MAE and MAPE values when predicted weight is generated by the train dataset we can see that **linfit_r** has a lower value for both measurements. Thus, MAE and MAPE confirms that **linfit_r** is the best among the three linear models.


### Validity of the model
In order for the predicted values to be valid and to make other inferences from the the **linfit_r** model, the assumptions for linear regression is checked. \

```{r echo=FALSE, fig.cap= "Diagnostic plots for the model:linfit.r"}
#radius
par(mfrow=c(2,2))
plot(linfit_r)
```

```{r include=FALSE}
# Shapiro-Wilk Normality Test
#H_0: The residuals are normally distributed
#H_1: The residuals are not normally distributed
shapiro.test(linfit_r$residuals)
```


- ***Linearity***: The *Residuals vs. Fitted* graph has an approximately horizontal line which means that there is no fitted pattern. Thus, we can assume that the radius has a linear relationship with the weight. \
 
- ***Residual's normality test***: After running a Shapiro-Wilk Normality Test with a significance level of 0.05, we fail to reject the null hypothesis since $p-value = 0.6425 > (2*0.05)$. This means that we have 95% confidence of the residuals being normally distributed. \
 
- **Homoscedasticity(equal variances)**: From the plot of Scale-Location we can see that the line is not horizontal which implies that variance of the residuals increases with the fitted values. This imply that **linfit_r** produces non-constant variance of the response variable (banana weights). Adding weights to the model done in the next section overcomes this violation. \
 
\
```{r, echo=FALSE, fig.cap= "Boxplots of the variables"}
#boxplot
par(mfrow=c(1,3))
boxplot(banana_train$weight, ylab = "Weight(g)", main = "Boxplot of Banana weights", col="lightyellow")
boxplot(banana_train$length, ylab = "length(mm)", main = "Boxplot of Banana length", col="lightgreen")
boxplot(banana_train$radius, ylab = "radius(mm)", main = "Boxplot of Banana radius", col="lightpink")
```

- **Outliers and influential points**: The *Residuals vs. Leverage* Graph, show that there is no point that crosses Cook's distance. Therefore, there are no influential points that we could omit. So even though there seemed to be some radius outliers shown on the boxplot, removing them would not have a significant effect on the model.

### Modifying the model


By adding the weight $w_i = \frac{1}{(st.dev(Y_i))^2}$ to the **linfit_r** model we get a new model (**fitr_wi**): \
$$log(Weight)_w = \hat{\beta_{0}} + \hat{\beta_{1}} log(radius)_w + \hat{\varepsilon},$$
where $_w$ represents the weights added on the observations[1]. \
Applying the weights on the linear model resulted in a small improvement of the Adjusted R-squared from $0.2092$ to $0.2928$. The figure below confirms small improvement as the scale location graph became a bit more horizontal.  
\
```{r}
# add a weight to stabilize the variance
# weight  = 1/(residuals)^2; we estimate the residuals via slr for it
resfit = lm(abs(linfit_r$residuals)~ linfit_r$fitted.values)

wi = 1/resfit$fitted.values^2
fitr_wi = lm(log(weight)~log(radius), data=banana_train, weights = wi)

summary(fitr_wi)

```
\
```{r}
par(mfrow=c(2,2))
plot(fitr_wi)

```
\
When **fitr_wi** is used to predict the banana weights with the test dataset, the resulting MAE and MAPE are: \
\
```{r echo=FALSE, table.cap="MAE and MAPE with the fitr_wi model"}
# predict weight using fitr_w8
pred.rwi = predict(object = fitr_wi, newdata = banana_test)

#MAE
mae_rwi = computeMae(log(banana_test$weight), pred.rwi, n= nrow(banana_test))

#MAPE
mape_rwi = mape(log(banana_test$weight), pred.rwi)

# to make a line plot of your regression model:
#ggplot(fitr_w8, aes(names(fitr_w8)[2], names(fitr_w8)[1])) + geom_abline()

maepe_table = data.frame(mae_rwi,mape_rwi)
names(maepe_table)[1] = "MAE"
names(maepe_table)[2] = "MAPE"
kable(maepe_table, format="pipe")
```
\
*The MAE and MAPE values above are computed from the test dataset.* \

### Final model 

The final model is **fitr_wi**:
$$log(Weight)_w = \hat{\beta_0} + \hat{\beta_1} log(radius)_w + \hat{\varepsilon},$$
where $\hat{\beta_0}= 1.395139, \hat{\beta_1}= 1.243427.$\


The following graph is the models **linfit_r**(purple), and **fitr_wi**(blue) overlaying the plot of banana weights from the test test data set.
\
```{r echo=FALSE, fig.cap="linfit_r(purple) and fitr_wi model(blue)"}
plot(x=banana_test$radius, y = banana_test$weight, type='p', col='red')

#overlay line plot of linfit_r predictions
lines(banana_test$radius, exp(pred.r), col='blue')

#overlay line plot of fitr_wi predictions
lines(banana_test$radius, exp(pred.rwi), col='purple')

#add legend
legend(1, 25, legend=c('Line 1', 'Line 2', 'Line 3'),
       col=c('red', 'blue', 'purple'), lty=1)
```
\

The predicted banana weights are: \

```{r echo=FALSE}
bweight_df = data.frame(banana_test$radius, banana_test$weight, exp(pred.r),exp(pred.rwi))
print(bweight_df)
```

While the 95% CI of the density is
```{r}
# confidence interval for the density
# density is in the intercept (beta_0) term of linfit_full
b0_full = linfit_r$coefficients[1]
# st error of intercept in linfit_r: 49.059
q.se = qnorm(1-(0.05/2))*49.059
confidence_i = b0_full + c(-1,1)*q.se
q.se2 = 25.78
confidence_i + c(-1,1)*q.se2
```
## Conclusion
Even though MAE and MAPE values are small the linear regression **fitr_wi** may not be the best model to predict the weight of bananas. That is because the Adjusted R-squared values are not high enough which means that the predictor variable (i.e., radius) does not predict the response (banana weight) very well. More predictor variables could possibly improve the linear model. For instance, the ripeness of the banana (ripe/unripe). Adding size (e.g.,small/medium/large) might also allow for separate prediction models based on each size. Moreover, variance could be minimized during data collection by implementing the same techniques and tools in measuring the radius, length and weight of the banana. Non linear models may also yield better predictions.


## Appendix
The following are the R codes used for this report.
\

```{r, linewidth=60}
#library(knitr)
##library(ggplot2)
#library(GGally)  # an extension to ggplot2
#library(Metrics)
```

```{r, linewidth=60}
# import the banana data
#bdata = readxl::read_xlsx(
#"Banana Data.xlsx")

#colnames(bdata) = c("weight","length","radius")

```

```{r, linewidth=60}
# split data into train and test subsets
# test data are rows:
#1,8,12,17,18,23,24,28,44,45,47,53,55,69,76,77
#testrows = c(1,8,12,17,18,
#23,24,28,44,45,47,53,55,69,76,77)
#banana_train = bdata[-testrows,] 
#banana_test = bdata[testrows,]

```

```{r, linewidth=60}
#ggpairs(data = banana_train, 
#        title = 
#"Pairwise plot of the three variables in the data", 
#        axisLabels = "show")
# https://www.
#rdocumentation.org/packages/GGally/versions/
#1.5.0/topics/ggpairs 
# stars signify significance at 10%, 5% and 1% levels. https://r-coder.com/correlation-plot-r/ 

```

```{r, fig.cap= "Histogram of weight distribution of the bananas in the training set.", linewidth=60}
# histogram of the weight's distribution
#hist(banana_train$weight, col = "lightyellow", 
#     border="black", prob= T, 
#     xlab=" banana weight (g)", 
#main = " Histogram of bananas' weight distribution")

#lines(density(banana_train$weight), lwd = 2.5, col = "orange")
```


```{r, linewidth=60}

#linfit_full = lm(log(weight) ~ log(length) + log(radius),data=banana_train)   #all variables
#linfit_l = lm(log(weight)~log(length), data=banana_train)                    #just length as x
#linfit_r = lm(log(weight)~log(radius), data=banana_train)                    #just radius as x
#linfit_none = lm(weight~1, data=banana_train)                   #empty; used for forward step, model selection

#summary(linfit_full)
#summary(linfit_l)
#summary(linfit_r)
#summary(linfit_none)
```
```{r}
#linfit_b0 = c(linfit_full$coefficients[1], linfit_l$coefficients[1], linfit_r$coefficients[1])
#linfit_b1 = c(linfit_full$coefficients[2], linfit_l$coefficients[2], linfit_r$coefficients[2])
#linfit_b2 = c(linfit_full$coefficients[3], NA, NA)
#linfit_coefs = cbind.data.frame(c("linfit_full","linfit_l",
#"linfit_r"),linfit_b0,linfit_b1,linfit_b2)
#names(linfit_coefs)[1] ="model"
#kable(linfit_coefs, format="simple")

```


*Not really need to do this, since there are only 2 possible predictors*).
\
```{r echo=TRUE}
# backwards, using both length and radius variables
#step(linfit_full, direction="backward", trace=F)

# forward...
#step(linfit_none, direction="forward", scope = list(lower= linfit_none, upper= linfit_full), trace=F)

# Result is the model with radius as the predictor
```

```{r}
# using the model with all the variables (l and  r)
#pred.full = predict(object = linfit_full, newdata = banana_test)

# with just length (l)
#pred.l = predict(object = linfit_r, newdata = banana_test)

# with just radius (r)
#pred.r = predict(object = linfit_r, newdata = banana_test)
```



Comparing the models's accuracy

```{r}
# MAE - Mean absolute error
# MPAE - Mean Percent absolute error
# function to compute MAE
#computeMae = function(measured,predicted,n){
#  summed_term = 0
# for (i in 1:n){
#    abs_term = abs(measured[i]-predicted[i])
#    summed_term = summed_term + abs_term
#  }
  
#  MAE = summed_term/n
  
#  return(MAE)
#}
```

```{r, linewdith=60}
# now compare all the maes of train and test on each model
#----MAE
#MAE_train_full = computeMae(log(banana_train$weight), linfit_full$fitted.values, n=nrow(banana_train))
#MAE_test_full = computeMae(log(banana_test$weight), 
#pred.full, n= nrow(banana_test))
#mae_full = c(MAE_train_full, MAE_test_full)

#MAE_train_l = computeMae(log(banana_train$weight), linfit_l$fitted.values, n= nrow(banana_train))
#MAE_test_l = computeMae(log(banana_test$weight), 
#pred.l, n= nrow(banana_test))
#mae_l = c(MAE_train_l, MAE_test_l)

#MAE_train_r = computeMae(log(banana_train$weight), linfit_r$fitted.values, n= nrow(banana_train))
#MAE_test_r = computeMae(log(banana_test$weight), pred.r, 
#n= nrow(banana_test))
#mae_r = c(MAE_train_r, MAE_test_r)

#mae_df = cbind.data.frame(mae_full, mae_l, mae_r)

#rownames(mae_df) = c("weight(train)","weight(test)")
#colnames(mae_df) = c("predicted weight(full)", "predicted weight(length)", "predicted weight(radius)")

#mae_table = kable(mae_df,format= "pipe", row.names = T)


#----MAPE
# use mape() in Metrics package
#MAPE_train_full = mape(log(banana_train$weight), linfit_full$fitted.values)
#MAPE_test_full = mape(log(banana_test$weight), pred.full)
#mape_full = c(MAPE_train_full,MAPE_test_full)

#MAPE_train_l = mape(log(banana_train$weight), linfit_l$fitted.values)
#MAPE_test_l = mape(log(banana_test$weight), pred.l)
#mape_l = c(MAPE_train_l, MAPE_test_l)

#MAPE_train_r = mape(log(banana_train$weight), linfit_r$fitted.values)
#MAPE_test_r = mape(log(banana_test$weight), pred.r)
#mape_r = c(MAPE_train_r, MAPE_test_r)

#put in df
#mape_df = cbind(mape_full, mape_l, mape_r)
#rownames(mape_df) = c("weight(train)","weight(test)")
#colnames(mape_df) = c("predicted weight(full)", "predicted weight(length)", "predicted weight(radius)")

#then table
#mape_table = kable(mape_df,format= "pipe", row.names = T)
```


```{r, echo=FALSE, fig.cap= "Mean Absolute Error(Banana weight observations minus the three models's predicted weight"}
#mae_table
# The lesser the percent value of mae/mape, the more accurate the model
```

```{r, echo=FALSE,fig.cap= "Mean Absolute Percent Error (Banana weight observations minus the three models's predicted weight"}
#mape_table
```

```{r, echo=FALSE,fig.cap= "Diagnostic plots for the model: linfit.full"}
#plot to check assumptions for linear regression
#full
#par(mfrow=c(2,2))
#plot(linfit_full)
```
```{r, echo=FALSE, fig.cap= "Diagnostic plots for the model:linfit.l"}
#length
#par(mfrow=c(2,2))
#plot(linfit_l)
```

```{r, echo=FALSE, fig.cap= "Diagnostic plots for the model:linfit.r"}
#radius
#par(mfrow=c(2,2))
#plot(linfit_r)
```

```{r}
# Shapiro-Wilk Normality Test
#H_0: The residuals are normally distributed
#H_1: The residuals are not normally distributed
#shapiro.test(linfit_r$residuals)
```

```{r, echo=FALSE, fig.cap= "Boxplots of the variables"}
#boxplot
#par(mfrow=c(1,3))
#boxplot(banana_train$weight, ylab = "Weight(g)", main = "Boxplot of Banana weights", col="lightyellow")
#boxplot(banana_train$length, ylab = "length(mm)", main = "Boxplot of Banana length", col="lightgreen")
#boxplot(banana_train$radius, ylab = "radius(mm)", main = "Boxplot of Banana radius", col="lightpink")
```

```{r include=FALSE}
# add a weight to stabilize the variance
# weight  = 1/(residuals)^2; we estimate the residuals via slr for it
#resfit = lm(abs(linfit_r$residuals)~ linfit_r$fitted.values)

#wi = 1/resfit$fitted.values^2
#fitr_wi = lm(log(weight)~log(radius), data=banana_train, weights = wi)
#summary(fitr_wi)
```
```{r echo=FALSE}
#par(mfrow=c(2,2))
#plot(fitr_wi)
```

```{r echo=FALSE, table.cap="MAE and MAPE with the fitr_wi model"}
# predict weight using fitr_w8
#pred.rwi = predict(object = fitr_wi, newdata = banana_test)

#MAE
#mae_rwi = computeMae(log(banana_test$weight), pred.rwi, n= nrow(banana_test))

#MAPE
#mape_rwi = mape(log(banana_test$weight), pred.rwi)

# to make a line plot of your regression model:
#ggplot(fitr_w8, aes(names(fitr_w8)[2], names(fitr_w8)[1])) + geom_abline()

#maepe_table = data.frame(mae_rwi,mape_rwi)
#names(maepe_table)[1] = "MAE"
#names(maepe_table)[2] = "MAPE"
#kable(maepe_table, format="pipe")
```

```{r}
#create plot of rdius vs weight in test data
#plot(x=banana_test$radius, y = banana_test$weight, type='p', col='red')

#overlay line plot of radius vs predicted weight by linfit_r
#lines(banana_test$radius, exp(pred.r), col='blue')

#overlay line plot of radius vs predicted by fitr_wi
#lines(banana_test$radius, exp(pred.rw8), col='purple')

#add legend
#legend(1, 25, legend=c('actual', 'linfit_r', 'fitr_wi'),
#       col=c('red', 'blue', 'purple'), lty=1)
```

```{r, linewidth=60}
#bweight_df = data.frame(banana_test$radius, 
#banana_test$weight, 
#exp(pred.r),exp(pred.rwi))
#print(bweight_df)
```

```{r}
# confidence interval for the density
# density is in the intercept (beta_0) term of linfit_full
b0_full = linfit_full$coefficients[1]
# st error of intercept in linfit_full: 49.39822
q.se = qnorm(1-(0.05/2))*49.39822
confidence_i = (pred.rwi / pred.r)+c(-1,1)*0.008949
min(confidence_i)
max(confidence_i)

confidence_i2 = (pred.rwi / pred.r)+c(-1,1)*0.5986
abs(min(confidence_i2))
abs(max(confidence_i2))
```


## References 

[1]  “13.1 - weighted least squares: Stat 501,” PennState: Statistics Online Courses. [Online]. Available: https://online.stat.psu.edu/stat501/lesson/13/13.1. [Accessed: 06-Feb-2022].  




# Project 2
---
title: "Calibrating Snow Gauge"
author: "Iza"
date: "3/24/2022"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## the data
```{r}
Density = c(0.686,0.604,0.508,0.412,
            0.318,0.223,0.148,0.080,0.001)

Gain1 = c(17.6,24.8,39.4,60.0,
          87.0,128,199,298,423)
Gain2 = c(17.3,25.9,37.6,58.3,
          92.7,130,204,298,421)
Gain3 = c(16.9,26.3,38.1,59.6,
          90.5,131,199,297,422)
Gain4 = c(16.2,24.8,37.7,59.1,
          85.8,129,207,288,428)
Gain5 = c(17.1,24.8,36.3,56.3,
          87.5,127,200,296,436)
Gain6 = c(18.5,27.6,38.7,55.0,
          88.3,129,200,293,427)
Gain7 = c(18.7,28.5,39.4,52.9,
          91.6,132,205,301,426)
Gain8 = c(17.4,30.5,38.8,54.1,
          88.2,133,202,299,428)
Gain9 = c(18.6,28.4,39.2,56.9,
          88.6,134,199,298,427)
Gain10 = c(16.8,27.7,40.3,56.0,
           84.7,133,199,293,429)

#df with only two columns
df1 = data.frame(
  rep(Density,10),
  c(Gain1,Gain2,Gain3,Gain4,Gain5,
    Gain6,Gain7,Gain8,Gain9,Gain10))
#chage column names
names(df1) = c("Density","Gain")

```

## How are density and gain related?
```{r}
library(ggplot2)

dxg = ggplot(df1,aes(x=Density, y =Gain))
dxg_df1 = dxg + ggtitle("Point Plot each Polyethylene Density and its 10 Gains reading") + geom_point(aes(colour = factor(Density)))
```

# Relationship of Gain and Density
$Gain = e^{b*Density}$ or $ln(Gain) = b*Density$


# model 0: just a linear 
```{r}
model0 = lm(Gain~Density, data=df1)
summary(model0)
coefs1 = model0$coefficients
plot(model0)
```



# Model fitting using trasformation and linear regression
In order to use "lm()" we have to make the relationship between the two variables to be linear. Hence the log transformation of the Gain variable.
```{r}
model1 = lm(log(Gain)~Density, data=df1)
summary(model1)
coefs1 = model1$coefficients
plot(model1)
```

# the physical model representin
$$Gain = e^{b*Density}$$
$$log(Gain) = b*Density$$
using the data, the model with coefficients is
$$log(Gain) = b*Density + a + \epsilon$$

,where $b = -4.60594$ and $a = 5.99727$. 

If apply exponent
$$Gain = e^{b*Density + a} = e^{b*Density} *e^{a}$$

# Predict the Density using the models


```{r}
# without error term
# predict Gains using model1
#G.pred1 = exp(predict(model1, df1, level=0.95, interval="confidence"))

# but we want density, inverse estimating... using the equation of the model
D.pred1 = (log(df1$Gain) - model1$coefficients[1]) /  model1$coefficients[2]
# notice that they give the same point estimates of density

num = (D.pred1 - mean(df1$Gain))^2
denom = sum((df1$Gain - mean(df1$Gain))^2)

n = nrow(df1)
s.hat = 0.06792 # is this value right?
lowerb = D.pred1-(qnorm(0.975)*s.hat*sqrt((1/n) + num/denom))
upperb = D.pred1+(qnorm(0.975)*s.hat*sqrt((1/n) + num/denom))

lowerb < df1$Density & df1$Density < upperb
table(lowerb < df1$Density & df1$Density < upperb)


D.pred1.pt = (log(df1$Gain) - model1$coefficients[1] - model1$residuals) /  model1$coefficients[2]
lowerb.pt = D.pred1.pt - (qnorm(0.975)*s.hat*sqrt((1/n) + num/denom))
upperb.pt = D.pred1.pt + (qnorm(0.975)*s.hat*sqrt((1/n) + num/denom))
table(lowerb.pt < df1$Density & df1$Density < upperb.pt)
# with error term, all the interval estimates encompass all obs. densities

```



#plotting model1
$$Gain = c* e^{b*Density},$$
where $c = e^{a}$, $b = -4.60594$ and $a = 5.99727$.
\
```{r}

func = function(x){
  exp(model1$coefficients[2]*x)*exp(model1$coefficients[1])
}
dxg_df1
dxg_df1 + geom_function(fun =func) 
```



# modelling without extreme cases

## model2 - same as model1 but 0.001 density data removed from training set
```{r}
df2 = subset(df1, Density!= 0.001)
#model without 0.001
model2.l = lm(log(Gain)~Density, data=df2)
summary(model2.l)
coefs2.l = model2.l$coefficients
#plot(model2.l)

# but we want density
D.pred2.l = (log(df2$Gain) - model2.l$coefficients[1]) /  model2.l$coefficients[2]
# notice that they give the same point estimates of density

num2.l = (D.pred2.l - mean(df2$Gain))^2
denom2.l = sum((df2$Gain - mean(df2$Gain))^2)

n2.l = nrow(df2)
s.hat2.l = 0.0694 # is this value right?
lowerb2.l = D.pred2.l-(qnorm(0.975)*s.hat2.l*sqrt((1/n2.l) + num2.l/denom2.l))
upperb2.l = D.pred2.l+(qnorm(0.975)*s.hat2.l*sqrt((1/n2.l) + num2.l/denom2.l))

#check if observed density is within the interval
table(lowerb2.l < df2$Density & df2$Density < upperb2.l)


#predicting point est of density with error term
D.pred2.l.pt = (log(df2$Gain) - model2.b$coefficients[1] - model2.l$residuals) /  model2.b$coefficients[2]
lowerb2.l.pt = D.pred2.l.pt-(qnorm(0.975)*s.hat2.l*sqrt((1/n2.l) + num2.l/denom2.l))
upperb2.l.pt = D.pred2.l.pt+(qnorm(0.975)*s.hat2.l*sqrt((1/n2.l) + num2.l/denom2.l))

table(lowerb2.l.pt < df2$Density & df2$Density < upperb2.l.pt)

# with error term, all the interval estimates encompass all obs. densities

```
#model3 - same techniques as model1 but with 0.686 density data removed from training set
```{r}
df3 = subset(df1,  Density!= 0.686)
#model without 0.686
model2.u = lm(log(Gain)~Density, data=df3)
summary(model2.u)
coefs2.u = model2.u$coefficients
#plot(model2.u)

# but we want density
D.pred2.u = (log(df3$Gain) - model2.u$coefficients[1]) /  model2.u$coefficients[2]
# notice that they give the same point estimates of density

num2.u = (D.pred2.u - mean(df3$Gain))^2
denom2.u = sum((df3$Gain - mean(df3$Gain))^2)
n2.u = nrow(df3)
s.hat2.u = 0.06633 # is this value right?
lowerb2.u = D.pred2.u-(qnorm(0.975, nrow(df3)-2)*s.hat2.u*sqrt(1+ (1/n2.u) + num2.u/denom2.u))
upperb2.u = D.pred2.u+(qnorm(0.975, nrow(df3)-2)*s.hat2.u*sqrt(1+(1/n2.u) + num2.u/denom2.u))

lowerb2.u < df3$Density & df3$Density < upperb2.u
table(lowerb2.u < df3$Density & df3$Density < upperb2.u)

#predicting density with error term
D.pred2.u.pt=(log(df3$Gain) - model2.b$coefficients[1] - model2.u$residuals) /  model2.b$coefficients[2]
lowerb2.u.pt = D.pred2.u.pt-(qt(0.975, nrow(df3)-2)*s.hat2.u*sqrt(1+(1/n2.u) + num2.u/denom2.u))
upperb2.u.pt = D.pred2.u.pt+(qt(0.975, nrow(df3)-2)*s.hat2.u*sqrt(1+(1/n2.u) + num2.u/denom2.u))
table(lowerb2.u.pt < df3$Density & df3$Density < upperb2.u.pt)
# with error term, all the interval estimates encompass all obs. densities

```

#model4 - same techniques as model1 but both 0.001 0.686 density data removed from training set
```{r}
df4 = subset(df1,  Density>0.001 & Density<0.686)

#model without 0.686 and 0.001
model2.b = lm(log(Gain)~Density, data=df4)
summary(model2.b)
coefs2.b = model2.b$coefficients
#plot(model2.b)

# but we want density
D.pred2.b = (log(df4$Gain) - model2.b$coefficients[1]) /  model2.b$coefficients[2]



num2.b = (D.pred2.b - mean(df4$Gain))^2
denom2.b = sum((df4$Gain - mean(df4$Gain))^2)
n2.b = nrow(df4)
s.hat2.b = 0.06868 # is this value right?
lowerb2.b = D.pred2.b-(qt(0.975,df = nrow(df4)-2)*s.hat2.b*sqrt((1/n2.b) + num2.b/denom2.b))
upperb2.b = D.pred2.b+(qt(0.975, df = nrow(df4)-2)*s.hat2.b*sqrt((1/n2.b) + num2.b/denom2.b))

lowerb2.b < df4$Density & df4$Density < upperb2.b
table(lowerb2.b < df4$Density & df4$Density < upperb2.b)

#predicting density with error term
D.pred2.b.pt = (log(df4$Gain) - model2.b$coefficients[1] - model2.b$residuals) /  model2.b$coefficients[2]
lowerb2.b.pt = D.pred2.b.pt-(qnorm(0.975)*s.hat2.b*sqrt((1/n2.b) + num2.b/denom2.b))
upperb2.b.pt = D.pred2.b.pt+(qnorm(0.975)*s.hat2.b*sqrt((1/n2.b) + num2.b/denom2.b))
table(lowerb2.b.pt < df4$Density & df4$Density < upperb2.b.pt)
# with error term, all the interval estimates encompass all obs. densities

```

# error
truth = reported value + measurement error
```{r}
e= model1$residuals
qqnorm(e)
qqline(e)
shapiro.test(e)
# residuals are normally distributed
df1$Density 
```
# confidence interval of Density estimate by model
```{r}
# function to find lower and upper bounds for a model
CI = function(model, alpha){
  if(model == 1){
    num = (D.pred1 - mean(df1$Gain))^2
    denom = sum((df1$Gain - mean(df1$Gain))^2)
    n = nrow(df1)
    s.hat = 0.06792 # is this value right?
    lowerb = D.pred1-(qnorm(1-(alpha/2))*s.hat*sqrt((1/n) + num/denom))
    upperb = D.pred1+(qnorm(1-(alpha/2))*s.hat*sqrt((1/n) + num/denom))
  }
  else if(model==2){
    num2.l = (D.pred2.l - mean(df2$Gain))^2
    denom2.l = sum((df2$Gain - mean(df2$Gain))^2)
    n2.l = nrow(df2)
    s.hat2.l = 0.0694 # is this value right?
    lowerb = D.pred2.l-(qnorm(alpha/2)*s.hat2.l*sqrt((1/n2.l) + num2.l/denom2.l))
    upperb = D.pred2.l+(qnorm(alpha/2)*s.hat2.l*sqrt((1/n2.l) + num2.l/denom2.l))
  }
  else if(model==3){
    num2.u = (D.pred2.u - mean(df3$Gain))^2
    denom2.u = sum((df3$Gain - mean(df3$Gain))^2)
    n2.u = nrow(df3)
    s.hat2.u = 0.06633 # is this value right?
    lowerb = D.pred2.u-(qnorm(alpha/2)*s.hat2.u*sqrt((1/n2.u) + num2.u/denom2.u))
    upperb = D.pred2.u+(qnorm(alpha/2)*s.hat2.u*sqrt((1/n2.u) + num2.u/denom2.u))
  }
  else if(model==4){
    num2.b = (D.pred2.b - mean(df4$Gain))^2
    denom2.b = sum((df4$Gain - mean(df4$Gain))^2)
    n2.b = nrow(df4)
    s.hat2.b = 0.06868 # is this value right?
    lowerb = D.pred2.b-(qnorm(alpha/2)*s.hat2.b*sqrt((1/n2.b) + num2.b/denom2.b))
    upperb = D.pred2.b+(qnorm(alpha/2)*s.hat2.b*sqrt((1/n2.b) + num2.b/denom2.b))
  }
  else{
    print("model not found...")
    lowerb = NA
    upperb = NA
  }
  return(as.data.frame(cbind(lowerb,upperb)))
}

# 95% confidence interval of model1
CI(1,0.05)

# 95% confidence interval of model2.l 
# note: model2.l has log(Gain) as y and Density as x; data with low density:0.001 is removed 
CI(2,0.05)

# 95% confidence interval of model2.u
# note: model2.u has log(Gain) as y and Density as x; data with high density:0.686 is removed 
CI(3,0.05)

# 95% confidence interval of model2.b
# note: model2.b has log(Gain) as y and Density as x; both low density:0.001 and 0.686 are removed 
CI(4,0.05)
```

# Given a gain, what is it's estimated density and its interval, depending on the model used?
```{r}
estimateDensity = function(gain, model, alpha){
  if(model==1){
    pointEstimate = (log(gain) - model1$coefficients[1]) /  model1$coefficients[2]
    num = (pointEstimate - mean(df1$Gain))^2
    denom = sum((df1$Gain - mean(df1$Gain))^2)
    n = nrow(df1)
    s.hat = 0.06792 # is this value right?
    ci = pointEstimate + c(-1,1)*(qt(1-(alpha/2), nrow(df1)-2)*s.hat*sqrt(1+ (1/n) + num/denom))
  }
  else if(model==2){
    pointEstimate = (log(gain) - model2.l$coefficients[1]) /  model2.l$coefficients[2]
    num2.l = (pointEstimate - mean(df2$Gain))^2
    denom2.l = sum((df2$Gain - mean(df2$Gain))^2)
    n2.l = nrow(df2)
    s.hat2.l = 0.0694 # is this value right?
    ci = pointEstimate + c(-1,1)*(qt(1-(alpha/2), nrow(df2)-2)*s.hat2.l*sqrt(1+ (1/n2.l) + num2.l/denom2.l))
    
  }
  else if(model==3){
    pointEstimate = (log(gain) - model2.u$coefficients[1]) /  model2.u$coefficients[2]
    num2.u = (pointEstimate - mean(df3$Gain))^2
    denom2.u = sum((df3$Gain - mean(df3$Gain))^2)
    n2.u = nrow(df3)
    s.hat2.u = 0.06633 # is this value right?
    ci = pointEstimate + c(-1,1)*(qt(1-(alpha/2), nrow(df3)-2)*s.hat2.u*sqrt(1+ (1/n2.u) + num2.u/denom2.u))
  }
  else if(model==4){
    pointEstimate = (log(gain) - model2.b$coefficients[1]) /  model2.b$coefficients[2]
    num2.b = (pointEstimate - mean(df4$Gain))^2
    denom2.b = sum((df4$Gain - mean(df4$Gain))^2)
    n2.b = nrow(df4)
    s.hat2.b = 0.06868 # is this value right?
    ci = pointEstimate + c(-1,1)*(qt(1-(alpha/2), nrow(df4)-2)*s.hat2.b*sqrt(1+(1/n2.b) + num2.b/denom2.b))
    
  }
  else{
    print("model not found...")
    pointEstimate = NA
    ci = NA
  }
  
  df = data.frame(cbind(
    rep(c("point estimate ", "lowerb","upperb"),length(gain)),
    c(pointEstimate, ci)))
  names(df) = c("label", "value")
  return(df)
}
```


/plot the conf intervals for each model

```{r}
# put model type: 1,2,3,4, gains, fit.value of density and their, se.fit. in a df then add on the plot

m1 = rep(1, length(D.pred1))
m2 = rep(2, length(D.pred2.l))
m3 = rep(3, length(D.pred2.u))
m4 = rep(4, length(D.pred2.b))
model = c(m1,m2,m3,m4)
p.estimates = c(D.pred1, D.pred2.l, D.pred2.u, D.pred2.b)
est.df = data.frame(model,p.estimates)

func = function(x){
  exp(model1$coefficients[2]*x)*exp(model1$coefficients[1])
}
dxg_df1
dxg_df1 + geom_function(fun =func) 
```


