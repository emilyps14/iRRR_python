import numpy as np

from iRRR.iRRR_normal import _objValue, _softThres

def test_objValue():
    pass

def test_softThres():
    d = np.array([-1,-3,1,3])
    dout = _softThres(d,2)
    assert(np.allclose(dout,[0,-1,0,1],rtol=0,atol=0))


