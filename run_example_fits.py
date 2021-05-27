import utils

def zero_eclipse_GP():
    """
    This shows how you can fit the lightcurve well with a GP
    even though the eclipse depth here is 0.0  !!
    
    """
    sb = utils.starry_basemodel(descrip='Orig_001',map_type='fixed',amp_type='fixedAt0')
    sb.plot_lc(point='mxap')
    
    sb = utils.starry_basemodel(descrip='Orig_001',map_type='fixed',amp_type='fixedAt1e-3')
    sb.plot_lc(point='mxap')
    