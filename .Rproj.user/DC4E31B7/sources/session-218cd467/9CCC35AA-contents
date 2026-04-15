jsd <- function(y, x, tol = 1e-8, maxit = 300000, 
                                    alpha = 0.01) {
  eps <- 1e-10
  p <- ncol(x)
  be <- rep(1/p, p)
  tx <- t(x)
  
  for ( iter in 1:maxit ) {
    y_hat <- pmax(drop(x %*% be), eps)
    m <- 0.5 * (y + y_hat)
    # Gradient
    grad <- 0.5 * tx %*% log(y_hat / m)
    grad_natural <- grad - mean(grad)
    be_new <- be * exp(-alpha * grad_natural)
    be_new <- be_new / sum(be_new)  
    if (max(abs(be_new - be)) < tol) {
      be <- be_new
      break
    }
    be <- be_new
  }
  
  y_hat_final <- pmax(drop(x %*% be), eps)
  m_final <- 0.5 * (y + y_hat_final)
  obj <- 0.5 * sum(y * log(y / m_final)) + 0.5 * sum(y_hat_final * log(y_hat_final / m_final))
  list( coefficients = round(be, 12), value = obj, iterations = iter )
}



