## irls logistic regression with identity link
## sum( y * log(y / y_hat) )
lr <- function(y, x, tol = 1e-8, maxit = 100) {
  p <- dim(x)[2]
  be <- rep(1/p, p)
  pi_hat <- drop(x %*% be)
  dev1 <- sum( y * log(y / pi_hat), na.rm = TRUE )

  Amat <- diag(p)
  Amat <- rbind(1, Amat)
  Amat <- t(Amat)
  bvec <- c(1, numeric(p) )

  # Working weights and response
  w <- 1 / ( pi_hat * (1 - pi_hat) )
  z <- y
  # Weighted least squares update: beta = (X'WX)^(-1) X'Wz
  Dmat <- crossprod(x, w * x)
  dvec <- crossprod(x, w * z)
  be <- quadprog::solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
  pi_hat <- drop(x %*% be)
  dev2 <- sum( y * log(y / pi_hat) )
  i <- 2

  # IRLS iteration
  while ( dev1 - dev2 > tol  &  i < maxit ) {
    i <- i + 1
    dev1 <- dev2
    w <- 1 / ( pi_hat * (1 - pi_hat) )
    Dmat <- crossprod(x, w * x)
    dvec <- crossprod(x, w * z)
    be <- quadprog::solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
    pi_hat <- drop(x %*% be)
    dev2 <- sum( y * log(y / pi_hat) )
  }

  list( coefficients = as.matrix( round(be, 12) ), value = dev2, iterations = i )
}
