mlr <- function(Y, x, tol = 1e-8, maxit = 100) {
  p <- dim(x)[2]
  Amat <- diag(p)
  Amat <- rbind(1, Amat)
  Amat <- t(Amat)
  bvec <- c(1, numeric(p) )

  be <- rep(1/p, p)
  pi_hat1 <- drop(x %*% be)
  w1 <- 1 / ( pi_hat1 * (1 - pi_hat1) )
  Dmat1 <- crossprod(x, w1 * x)

  d <- dim(Y)[2]
  B <- matrix(0, p, d)
  obj <- iterations <- numeric(d)

  for ( j in 1:d ) {
    y <- Y[, j]
    z <- y
    dev1 <- sum( y * log(y / pi_hat1), na.rm = TRUE )
    dvec <- crossprod(x, w1 * z)
    be <- quadprog::solve.QP(Dmat1, dvec, Amat, bvec, meq = 1)$solution
    pi_hat <- drop(x %*% be)
    dev2 <- sum( y * log(y / pi_hat) )
    i <- 2
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
    obj[j] <- dev2
    B[, j] <- round(be, 12)
    iterations[j] <- i
  }

  list( coefficients = round(B, 12), value = obj, iterations = iterations )
}
