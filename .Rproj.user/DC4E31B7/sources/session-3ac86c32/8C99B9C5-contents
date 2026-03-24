## Vectorized Exponentiated Gradient for KLD
## sum( (y - yhat) * log(y / y_hat) )
kld <- function(y, x, tol = 1e-8, maxit = 50000, alpha = 0.1) {
  eps <- 1e-10
  p <- dim(x)[2]
  # be <- rep(1/p, p)
  be <- as.vector( .skl.irls(y, x, tol = tol, maxit = maxit)$coefficients ) 
  tx <- t(x)

  for ( iter in 1:maxit ) {
    y_hat <- pmax(x %*% be, eps)
    grad <- tx %*% (log(y_hat/y) + 1 - y/y_hat)
    be_new <- be * exp(-alpha * grad)
    be_new <- be_new / sum(be_new)
    y_hat_new <- pmax(x %*% be_new, eps)
    obj_new <- sum( (y - y_hat_new) * log(y / y_hat_new) )
    obj_old <- sum( (y - y_hat) * log(y / y_hat) )
    # If not improving, reduce step size
    if ( obj_new > obj_old ) {
      alpha <- alpha * 0.5
      next
    }
    # Check convergence
    if ( max(abs(be_new - be) ) < tol) {
      be <- be_new
      break
    }
    be <- be_new
    alpha <- min(alpha * 1.05, 0.5)  # Slowly increase step size
  }

  # Final objective
  y_hat <- pmax(x %*% be, eps)
  obj <- sum( (y - y_hat) * log(y / y_hat) )

  list( coefficients = round(be, 12), value = obj, iterations = iter )
}


.skl.irls <- function(y, x, tol = 1e-8, maxit = 100) {

  p <- dim(x)[2]
  be <- rep(1/p, p)
  eps <- 1e-12   # positivity safeguard
  pi_hat <- drop(x %*% be)
  dev1 <- sum( (y - pi_hat) * log(y / pi_hat), na.rm = TRUE )

  Amat <- diag(p)
  Amat <- rbind(1, Amat)
  Amat <- t(Amat)
  bvec <- c(1, numeric(p) )

  s <-  - y / pi_hat + (1 - y) / (1 - pi_hat) + log( (pi_hat * (1 - y) ) / ( y * (1 - pi_hat) ) )
  w <- y / (pi_hat^2) + (1 - y) / (1 - pi_hat)^2 + 1/pi_hat + 1/(1 - pi_hat)
  w[w < eps] <- eps
  z <- pi_hat - s / w
  wx <- w * x  # weight each row
  Dmat <- crossprod(x, w * x)
  dvec <- crossprod(x, w * z)
  be <- quadprog::solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
  pi_hat <- drop(x %*% be)
  dev2 <- sum( (y - pi_hat) * log(y / pi_hat), na.rm = TRUE )
  i <- 2

  while ( abs(dev1 - dev2) > tol  &  i < maxit ) {
    i <- i + 1
    dev1 <- dev2
    s <-  - y / pi_hat + (1 - y) / (1 - pi_hat) + log( (pi_hat * (1 - y) ) / ( y * (1 - pi_hat) ) )
    w <- y / (pi_hat^2) + (1 - y) / (1 - pi_hat)^2 + 1/pi_hat + 1/(1 - pi_hat)
    w[w < eps] <- eps
    z <- pi_hat - s / w
    wx <- w * x  # weight each row
    Dmat <- crossprod(x, w * x)
    dvec <- crossprod(x, w * z)
    be <- quadprog::solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
    pi_hat <- drop(x %*% be)
    dev2 <- sum( (y - pi_hat) * log(y / pi_hat), na.rm = TRUE )
  }

  list( coefficients = as.matrix( round(be, 12) ), value = dev2, iterations = i )
}
