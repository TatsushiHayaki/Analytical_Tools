####### 1. iterate calculating correlations between x1 and X2-x1 #######
library(tidyverse)
set.seed(100)
S <- 100  #iteration
N <- 100  #sample size
DF <- do.call(rbind, lapply(1:S, function(i) {
  # assuming normality
  # x1 <- rnorm(N)
  # x2 <- rnorm(N)
    # we can also assume more realistic distributions: rating on a scale from 1 to 10 with probability assigned to each box
 		boxes <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
		prob  <- c(0.5, 0.5, 2, 2, 5, 20, 25, 20, 15, 10)/100
    x1 <-  sample(x=boxes, size=N, prob=prob, replace=T)
    x2 <-  sample(x=boxes, size=N, prob=prob, replace=T)
  data.frame(iteration=i, correlation=cor.test(x1, x2-x1)$estimate)})
)
with(DF, hist(correlation, breaks=100, main=paste("histogram of correlation between x1 and x2-x1; mean=", round(mean(DF$correlation),4))))
abline(v=mean(DF$correlation))


####### 2. iterate calculating correlations between x1 and X2-x1 #######
library(dplyr)
set.seed(100)
S <- 1000  #iteration
N <- 1000  #sample size
DF <- do.call(rbind, lapply(1:S, function(i) {
  # assuming normality
  # x1 <- rnorm(N)
  # x2 <- rnorm(N)
    # we can also assume more realistic distributions: rating on a scale from 1 to 10 with probability assigned to each box
		boxes <- c(1,2,3,4,5,6,7,8,9,10)
		prob  <- c(0.5, 0.5, 2, 2, 5, 20, 25, 20, 15, 10)/100
    x1 <-  sample(x=boxes, size=N, prob=prob, replace=T)
    x2 <-  sample(x=boxes, size=N, prob=prob, replace=T)
  x1_ave <- mean(x1)
  x2_ave <- mean(x2)
  data.frame(base=x1_ave, diff=x2_ave - x1_ave)})
)
with(DF, plot(base, diff,  main="base vs. diff"))
with(DF,cor.test(base, diff))
with(DF,cor.test(base, diff))
-1/sqrt(2)

####### 3. simulation for J.D. Power's index model  #######
## one-factor model with three attributes
library(dplyr)
set.seed(100)
par(mfrow=c(2,3))
par(oma=c(0,0,2,0))

for(k in 1:6){
#number of brands (with market share either large, medium, or small)
	if(k == 1) {brand_L <- 1; brand_M <- 3; brand_S <- 0}
	if(k == 2) {brand_L <- 1; brand_M <- 3; brand_S <- 4}
	if(k == 3) {brand_L <- 2; brand_M <- 6; brand_S <- 10}
	if(k == 4) {brand_L <- 4; brand_M <- 10; brand_S <- 16}
	if(k == 5) {brand_L <- 10; brand_M <- 15; brand_S <- 25}
	if(k == 6) {brand_L <- 20; brand_M <- 30; brand_S <- 50}
	
	S <- 10000  #iteration
	brands <- c(paste("brand", seq(sum(brand_L, brand_M, brand_S)), sep="_"));brands #generating brand names
	samples <- c(rep(130,brand_S), rep(300,brand_M), rep(500,brand_L));samples  #sample size for each brand
	N <- sum(samples);N  #total sample size for each year 
	ID <- seq(1,N)
	boxes <- c(1,2,3,4,5,6,7,8,9,10)
	prob  <- c(0.5, 0.5, 2, 2, 5, 20, 25, 20, 15, 10)/100
	importance_weights <- c(25,25,50)/100 
	
	result <- do.call(rbind, lapply(1:S, function(i){
	  ############### First-year ############### 
	  attr01 <-  sample(x=boxes, size=N, prob=prob, replace=T)
	  attr02 <-  sample(x=boxes, size=N, prob=prob, replace=T)
	  attr03 <-  sample(x=boxes, size=N, prob=prob, replace=T)
	  csi <- as.vector(cbind(attr01, attr02, attr03) %*% importance_weights * 100)
	
	  ## ranking table##
	  DF1 <- data.frame(ID, brand=rep(brands, samples), attr01, attr02, attr03, csi); DF1
	  ranking01 <- DF1 %>% 
	    group_by(brand) %>% 
	    summarize(csi_year01   =mean(csi)) %>% 
	    arrange(desc(csi_year01));ranking01
	  
	  ##  t-test for a pair of brands with largest csi and smallest csi ##
	  pair <- c(with(ranking01, brand[which.max(csi_year01)]),with(ranking01, brand[which.min(csi_year01)]));pair
	  sub <- DF1 %>%  filter(brand %in% pair);sub
	  pvalue_year01 <- t.test(csi ~ brand, data=sub)$p.value; pvalue_year01
	  
	  ############### Second-year ############### 
	  attr01 <-  sample(x=boxes, size=N, prob=prob, replace=T)
	  attr02 <-  sample(x=boxes, size=N, prob=prob, replace=T)
	  attr03 <-  sample(x=boxes, size=N, prob=prob, replace=T)
	  csi <- as.vector(cbind(attr01, attr02, attr03) %*% importance_weights * 100)
	  
	  ## ranking table ##
	  DF2 <- data.frame(ID, brand=rep(brands,samples), attr01, attr02, attr03, csi); DF2
	  ranking02 <- DF2 %>% 
	    group_by(brand) %>% 
	    summarize(csi_year02=mean(csi)) %>% 
	    arrange(desc(csi_year02));ranking02
	  
	  ##  t-test for a pair of brands with largest csi and smallest csi ##
	  pair <- c(with(ranking02, brand[which.max(csi_year02)]),with(ranking02, brand[which.min(csi_year02)]));pair
	  sub <- DF2 %>%  filter(brand %in% pair);sub
	  pvalue_year02 <- t.test(csi ~ brand, data=sub)$p.value; pvalue_year02
	
	  ############### combine two results ############### 
	  ranking_all <- full_join(ranking01, ranking02, by="brand")
	  ranking_all <- ranking_all %>% 
	  	mutate(diff_csi=csi_year02 - csi_year01,
	  				 rank_year01=dense_rank(desc(csi_year01)),
	  				 rank_year02=dense_rank(desc(csi_year02)));ranking_all
		data.frame(iteration=i, 
	             correlation=with(ranking_all, cor.test(diff_csi, csi_year01)$estimate),
	             pvalue_year01,
	             pvalue_year02)
	}))

	# result %>% mutate(YR01=ifelse(pvalue_year01<0.05,"significant","non-significant"),
	# 						      YR02=ifelse(pvalue_year02<0.05,"significant","non-significant")) %>% count(YR01)
	
	###### plot result with kernel density estimation ########
	# Histogram of Correlation;
	with(result, 
	  hist(correlation, breaks=100, prob=TRUE, xlim=c(-1,1),
	       main=paste(c("# of large brands=","# of medium brands=","# of small brands="),c(brand_L, brand_M, brand_S))
	      )
	  )
	abline(v=mean(result$correlation))
	d <- with(result, density(correlation, from=-1, to=1))
	lines(d, col="blue", lwd=2)
	box(lty=1) 
	legend("topright", legend= c(paste(" mean=",round(mean(result$correlation),2)),"kernel estimation"), 
	       col=c("black","blue"), lty=1, lwd=c(1,2), cex =.8)
}

title ("Histogram of Correlation", outer=T)

##### compute area above v #####
xx <- d$x
yy <- d$y
dx <- xx[2L] - xx[1L] #bin size
v <- -0.5
f <- approxfun(xx, yy)
C <- integrate(f, min(xx), max(xx))$value
p.unscaled <- integrate(f, v, max(xx))$value
p.scaled <- p.unscaled / C
p.scaled
