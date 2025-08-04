from . import hanger_resonator

import numpy as np
import scipy.signal as sps

def hanger_ccw_fit(paramsVec, res, residual=True, **kwargs):
    '''
    '''    
    # Repackage resonator data in 1D vector form
    data = np.concatenate((res.I, res.Q), axis=0)

    if residual:
        if (res.sigmaI is not None) and (res.sigmaQ is not None):
            cmplxSigma = np.concatenate((res.sigmaI, res.sigmaQ), axis=0)
        else:
            # Calculate eps from stdev of first 10 pts of data if not supplied
            epsI = np.std(sps.detrend(res.I[0:10]))
            epsQ = np.std(sps.detrend(res.Q[0:10]))
            cmplxSigma = np.concatenate((np.full_like(res.I, epsI), np.full_like(res.Q, epsQ)))

    # Use the clockwise hanger_resonator model - returns 1D array of concatenated [I, Q]
    model = hanger_resonator.hanger_fit(paramsVec, res, residual=False, **kwargs)

    # For counterclockwise convention we take the conjugate of the model (flip the sign of Q)  
    model[model.size // 2:] *= -1

    # Return model or residual
    if residual == True:
        return (model - data) / cmplxSigma
    else:
        return model

def hanger_ccw_params(res, **kwargs):
    '''
    '''
    params = hanger_resonator.hanger_params(res, **kwargs)

    # For the CCW convention we flip the sign of the pgain parameters
    params.get('pgain0').value *= -1
    params.get('pgain1').value *= -1
    params.get('pgain2').value *= -1

    return params