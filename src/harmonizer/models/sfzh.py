'''
Wrapper for the SFH and ZDist modules in synthesizer.
Expands to add various SFH models that observers might want to use, 
such as continuity SFHs, or SFHs with stochastic bursts.

'''


from synthesizer.parametric import ZDist, SFH

DeltaConstant = ZDist.DeltaConstant
Normal = ZDist.Normal



def get(name):
    '''
    Get the SFH or ZDist model by name
    '''
    if name == 'DeltaConstant':
        return DeltaConstant

    elif name == 'constant':
        return SFH.Constant

    else:
        raise ValueError(f"Unknown model: {name}")