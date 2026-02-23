import numpy as np

def validate_inputs(is_correct, confidence):
    """Checks if inputs are valid for BAS calculation."""
    if len(is_correct) != len(confidence):
        raise ValueError(f"Length mismatch: is_correct ({len(is_correct)}) and "
                         f"confidence ({len(confidence)}) must be the same size.")
    
    conf_array = np.array(confidence)
    if np.any((conf_array < 0) | (conf_array > 1)):
        raise ValueError("Confidence values must be in the range [0, 1]. "
                         "Please normalize your scores before evaluation.")

def bas_score(is_correct, confidence, prior='uniform', epsilon=1e-4):
    """
    Computes the Behavioral Alignment Score (BAS).
    Matches Equation 6 in the BAS preprint.
    """
    validate_inputs(is_correct, confidence)
    
    # Section 2.4: Numerical stability clipping
    s = np.clip(np.array(confidence), epsilon, 1.0 - epsilon)
    z = np.array(is_correct).astype(bool)
    
    if prior == 'uniform':
        # Eq 4: Realized utility for uniform risk
        return np.where(z, s, s + np.log(1 - s))
    
    elif prior == 'linear':
        # Analytical integral for w(t) = 2t
        return np.where(z, s**2, s**2 + 2*s + 2*np.log(1 - s))
    
    elif prior == 'quadratic':
        # Analytical integral for w(t) = 3t^2 (Safety-critical)
        return np.where(z, s**3, s**3 + 1.5*s**2 + 3*s + 3*np.log(1 - s))
    
    else:
        raise ValueError(f"Unsupported prior: {prior}. Choose 'uniform', 'linear', or 'quadratic'.")