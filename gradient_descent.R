set.seed(1234)

x <- matrix(rnorm(1000), ncol = 5)
y <- rnorm(200)

X <- cbind(rep(1, 200), x)
theta <- rep(0, 6)

cost <- function(x, y, theta) {
  J <- sum((X %*% theta - y)^2)/(2 * length(y))
}

gradient <- function(x, y, theta, alpha, iter) {
  n <- length(y)
  old_cost <- rep(0, iter)
  for(i in 1:iter) {
    theta <- theta - (1/n) * alpha * (t(X) %*% (X %*% theta - y))
    old_cost[i] <- cost(X, y, theta)
  }
  
  result = list(theta, old_cost)
  return(result)
}

train <- gradient(X, y, theta, alpha = 0.01, iter = 1000)

plot(train[[2]])
