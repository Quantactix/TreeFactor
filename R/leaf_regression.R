leaf_regression <- function(y, X, alpha, beta, xi_normal, xi_spike, xi_slab, p_normal_prior, p_spike_slab, R = 1000, keep = 1, intercept = TRUE) {
    # note that Inverse-Chisq(nu, ssq) = Inverse-Gamma(nu / 2, nu * ssq / 2)
    # therefore Inverse-Gamma(alpha, beta) = Inverse-Chisq(2 * alpha, beta / alpha)
    nu <- 2 * alpha
    ssq <- beta / alpha

    posterior <- leaf_regression_cpp(y, X, nu, ssq, R, keep, xi_normal, xi_spike, xi_slab, p_normal_prior, p_spike_slab, intercept)

    return(posterior)
}