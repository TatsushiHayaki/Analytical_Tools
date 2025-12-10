# Central Limit Theorem
# Gamma distributions assumed
N <- 10 # Sample size
S <- 100000 # Iterations
hist(rgamma(N, shape = 1), breaks = 100)
DF2 <- do.call(rbind, lapply(1:S, function(i){
  x     <- rgamma(N, shape = 1)
  y     <- rgamma(N, shape = 1)
  x_ave <- mean(x)
  y_ave <- mean(y)
  data.frame(x_ave, y_ave)
}))
print(DF2)
with(DF2, hist(x_ave))