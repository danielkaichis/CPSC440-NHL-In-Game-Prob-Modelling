import pymc as pm


def run_advi_inference(n_iter=30000, learning_rate=0.01, n_samples=1000, start_msg=None):
    """Run ADVI on the active PyMC model and return posterior samples."""
    if start_msg:
        print(start_msg)

    approx = pm.fit(
        n=n_iter,
        # Mean-field ADVI optimizes ELBO for posterior inference.
        method='advi',
        # Use adam for convergence.
        obj_optimizer=pm.adam(learning_rate=learning_rate),
        # Stop early when parameter updates become numerically small.
        callbacks=[pm.callbacks.CheckParametersConvergence(diff='absolute', tolerance=1e-3)]
    )
    # Draw posterior samples from the variational approximation for downstream MC.
    return approx.sample(n_samples)
