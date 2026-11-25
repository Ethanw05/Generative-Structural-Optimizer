"""Shared depth and color behavior for headless engineering comparisons."""
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
from gso_workflows.orthographic import rasterizer

draw = rasterizer(Path(sys.argv[1]))
v = np.array([[0.,0.,1.],[6.,0.,1.],[0.,6.,1.],[0.,0.,0.],[6.,0.,0.],[0.,6.,0.]])
for f, colors in [([[0,1,2],[5,4,3]],[[10,20,30],[210,220,230]]),([[3,4,5],[2,1,0]],[[210,220,230],[10,20,30]])]:
    pixels=np.empty((8,8,4),dtype=np.uint8)
    assert draw(6,v,2,np.array(f,dtype=np.int32),np.array(colors,dtype=np.uint8),8,8,pixels) == 0
    assert list(pixels[1,2]) == [10,20,30,255]
    assert list(pixels[7,7]) == [0,0,0,0]

from gso_workflows.orthographic import color_positions
values=np.array([0.,1.,8.,64.])
mapped=color_positions(values,64.,1.)
assert mapped[0]==0. and mapped[-1]==1. and np.all(np.diff(mapped)>0.)
np.testing.assert_allclose(np.expm1(mapped*np.log(65.)),values)
