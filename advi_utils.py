import pymc as pm


def run_advi_inference(n_iter=30000, learning_rate=0.01, n_samples=1000, start_msg=None):
    """Run ADVI on the active PyMC model and return posterior samples."""
    if start_msg:
        print(start_msg)

    approx = pm.fit(
        n=n_iter,
        method='advi',
        obj_optimizer=pm.adam(learning_rate=learning_rate),
        callbacks=[pm.callbacks.CheckParametersConvergence(diff='absolute', tolerance=1e-3)]
    )
    return approx.sample(n_samples)
