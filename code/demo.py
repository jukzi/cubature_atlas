import numpy as np
from cubature_atlas import search, certificate, margin

name, n, N = "triangle", 5, 7           # degree 5 on the triangle with 7 nodes
X, W = search(name, n, N, seed=1)
print("nodes inside, min weight:", margin(name, X) > 0, W.min() > 0)
c = certificate(name, n, X, W)
print("certified:", c["certified"], " h =", c["h"], " radius =", c["radius"], " inside:", c["inside"])

f = lambda P: np.sqrt(P.sum(axis=1))    # integral of sqrt(x+y) over the triangle = 2/5
print("Q =", W @ f(X), " exact = 0.4")
