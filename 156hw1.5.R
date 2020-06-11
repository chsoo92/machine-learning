rm(list=ls())
#declare weight vectors
b0 <- vector()
b1 <- vector()

for (val in seq(from = 50, to=1000, by = 50)) # for different n, obtain b0 and b1
{
n = val
x1 = matrix(1,n,1) #x1 is n-by-1 matrix
x1 = c(x1, 2) # x1 is now (n+1)-by-1 matrix
e = rnorm(n+1,0,1) # a vector of n+1 normally distributed noise
y = 2+3*x1 + e 
simplefit = lm(y~x1)#least squares linear regression
coeff = coef(simplefit)#get coefficients

#store coeffcients for the current n
b0 = rbind(b0,coeff[1])
b1 = rbind(b1,coeff[2])
}
plot(seq(from = 50, to=1000, by = 50),b0)
plot(seq(from = 50, to=1000, by = 50),b1)
meanb0 = paste("mean =",toString(mean(b0)), sep = " ", collapse = NULL)
meanb1 =  paste("mean =",toString(mean(b1)), sep = " ", collapse = NULL)
hist(b0, main = meanb0)
hist(b1, main = meanb1)
correlation = cor(b0,b1)
plot(b0,b1)



# problem 5
library(MASS)
library(glmnet)
zeros_j = c(0,0,0,0,
            0,0,0,0,0,0)
betas = t(t(c(-4,-3,-2,-1,0,0,1,2,3,4)))
rho = 0.5
sigma = 2

#covariates with autocorrection
gamma_vals = c(1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512,
                1/2, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 
                1/4, 1/2, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128,
                1/8, 1/4, 1/2, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64,
                1/16, 1/8, 1/4, 1/2, 1, 1/2, 1/4, 1/8, 1/16, 1/32,
                1/32, 1/16, 1/8, 1/4, 1/2, 1, 1/2, 1/4, 1/8, 1/16,
                1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 1/2, 1/4, 1/8,
                1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 1/2, 1/4,
                1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 1/2,
                1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1)
#gamma cov matrix
Gamma = matrix(gamma_vals, nrow = 10, ncol = 10)

x_learning= mvrnorm(n = 100, mu = zeros_j, Sigma = Gamma)#x learning set
y_learning= vector()
for (n in 1:100)
{
  yval = rnorm(1, x_learning[n,]%*%betas, 4)#obtain y for each row in x learning set
  y_learning = c(y_learning,yval) #construct y leanring set
}

x_testing <- mvrnorm(n = 1000, mu = zeros_j, Sigma = Gamma)# x testin set
y_testing = vector()
for (n in 1:1000)
{
  yval = rnorm(1, x_testing[n,]%*%betas, 4)#obtain y for each row in x testing set
  y_testing = c(y_testing,yval)#construct y testing set
}

hist(x_learning,xlab = "x")
hist(y_learning,xlab = "y")
hist(x_testing, xlab = "x")
hist(y_testing, xlab = "y")

#problem b
lambdas <- seq(0,100)#lambdas from 0 to 100

fit.lasso <- glmnet(x_learning, y_learning,family="gaussian", lambda=lambdas, alpha=1)
fit.ridge <- glmnet(x_learning, y_learning,family="gaussian", lambda=lambdas, alpha=0)
fit.elnet <- glmnet(x_learning, y_learning,family="gaussian", lambda=lambdas, alpha=0.5)


plot(as.numeric(unlist(fit.lasso[5])),as.numeric(unlist(fit.lasso[3])), xlab = "lambdas",ylab = "degree of freedom",main = "LASSO")
plot(as.numeric(unlist(fit.ridge[5])),as.numeric(unlist(fit.ridge[3])), xlab = "lambdas",ylab = "degree of freedom",main = "Ridge")
plot(as.numeric(unlist(fit.elnet[5])),as.numeric(unlist(fit.elnet[3])), xlab = "lambdas",ylab = "degree of freedom",main = "Elastic Net")

plot(fit.lasso, xvar="lambda", main="LASSO")
plot(fit.ridge, xvar="lambda", main="Ridge")
plot(fit.elnet, xvar="lambda", main="Elastic Net")

fit.lasso.mse <- cv.glmnet(x_learning, y_learning, type.measure = "mse",family="gaussian", lambda=lambdas, alpha=1)
fit.ridge.mse <- cv.glmnet(x_learning, y_learning, type.measure = "mse",family="gaussian", lambda=lambdas, alpha=0)
fit.elnet.mse <- cv.glmnet(x_learning, y_learning, type.measure = "mse", lambda=lambdas, alpha=0.5,family="gaussian")

coef.lasso <- coef(fit.lasso.mse, s='lambda.1se')
coef.ridge <- coef(fit.ridge.mse, s='lambda.1se')
coef.elnet <- coef(fit.elnet.mse, s='lambda.1se')


# Plot solution paths:
par(mfrow=c(3,2))


plot(fit.lasso.mse, main ="LASSO")
plot(fit.ridge.mse, main="Ridge")
plot(fit.elnet.mse, main="Elastic Net")


#problem c)

#instantiate mse vectors to store mse values
mse0= vector()
mse1= vector()
mse2= vector()

#obtain predictions y for each lambda
for (n in lambdas)
{
yhat0 <- predict.cv.glmnet(fit.lasso.mse, s=n, newx=x_testing)
yhat1 <- predict.cv.glmnet(fit.ridge.mse, s=n, newx=x_testing)
yhat2 <- predict.cv.glmnet(fit.elnet.mse, s=n, newx=x_testing)

#store mse for each lambda
mse0 = c(mse0, mean((y_testing - yhat0)^2))
mse1 <- c(mse1, mean((y_testing - yhat1)^2))
mse2 <- c(mse2, mean((y_testing - yhat2)^2))

}

plot(lambdas,mse0,xlab="lambda",ylab="MSE", main="LASSO",xlim = c(0,15))
plot(lambdas,mse1,xlab="lambda",ylab="MSE", main="Ridge",xlim = c(0,15))
plot(lambdas,mse2,xlab="lambda",ylab="MSE", main="Elastic Net",xlim= c(0,15))

yhat00 <- predict(fit.lasso,  s=fit.lasso$lambda.1se, newx=x_testing)
yhat11 <- predict(fit.ridge,  s=fit.ridge$lambda.1se, newx=x_testing)
yhat22 <- predict(fit.elnet,  s=fit.elnet$lambda.1se, newx=x_testing)

mse00 <- mean((y_testing - yhat00)^2)
mse11 <- mean((y_testing - yhat11)^2)
mse22 <- mean((y_testing - yhat22)^2)