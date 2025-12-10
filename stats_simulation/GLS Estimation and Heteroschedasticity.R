# Generialized Least Square Estimation
# with a focus on the heteroskedasticity issue
library(sandwich)
#(1) model set-up
# The variance of error term follows poisson with mean of 99, unconditionally
# The real regression equation is y = 2∗x1 + 3∗x2 + epsilon

num <- 1000
set.seed(109)
x1 <- rpois(num, 3);x1
x2 <- rgamma(num, 2);x2

cond_var <- rpois(num, 99)
error_term <- sapply(cond_var, function(i) rnorm(1, 0, sqrt(i))) 

y <- 2*x1 + 3*x2 + error_term 

dat <- cbind(y, x1, x2, error_term, cond_var) ; dat
dat <- as.data.frame(dat[sample(1:num,500,replace = TRUE),]) # 500サンプルを無作為抽出

# without any treatment
reg <- lm(y ~ x1 + x2, data  = dat);summary(reg)
vcov(reg)

#(2)Treatment 1 - Weighted least sqaure (WLS)
# for the formula, see　https://online.stat.psu.edu/stat501/lesson/13/13.1
# this treatment is basically infeasible, 

design_m <- model.matrix(reg)
y_m <- as.matrix(dat[,"y"])
weight_m <- matrix(0,500,500) # 500*500行列
diag(weight_m) <- dat$cond_var # 
beta_wls <- solve(t(design_m) %*% solve(weight_m) %*% design_m) %*% (t(design_m) %*% solve(weight_m) %*% y_m)
beta_wls

reg_wls <- lm(y ~ x1 + x2, data  = dat, weights = 1/cond_var) ;summary(reg_wls)

# (3) Treatment 2 - Robust Standard Error
# The mechanism of HC estimator is hard to summarize. 
# ways to find an asymptotically consistent analogue of X′σX.
# It is an improvement on the OLS, rather than GLS
vcovHC(reg, type = 'HC')

# (4) Treatment 3 -Feasible GLS (FGLS)
# the feasible version of the first treatment.
# we dont know the variance of error terms, so we resort to some exogeneous vector and treat it as weight
fgls <- lm(y ~ x1 + x2, data  = dat, weights = 1/reg$residuals^2);summary(fgls)
vcov(fgls)

# (5) Treatment 4 - Iterative Feasible GLS (IFGLS)
test <- vector('list', 50)
test[[1]] <- list(coef0 = coef(reg), res0 = reg$residuals)

wls_func_lm <- function(obj){
  weight <- obj[['res0']]^2
  lm_reg <- lm(y ~ x1 + x2, data = dat, weights = 1/weight)
  lm_reg_res <- lm(lm_reg$residuals^2 ~ x1 + x2, data = dat)
  return(list(coef0 = coef(lm_reg), res0 = lm_reg_res$fitted.values)) 
}

for (i in 1:10){
test[[i+1]] <- wls_func_lm(test[[i]])
dif <- sum(test[[i+1]][['coef0']] - test[[i]][['coef0']])^2
cat(i, test[[i]][['coef0']], dif , '\n')
}

