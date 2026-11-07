import numpy as np, ctypes, os
_lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fastjac.so'))
_dp=ctypes.POINTER(ctypes.c_double)
def install(ProbG):
    def jac(self,x,N):
        d=self.d; Q=x[:d*N].reshape(N,d); w=np.ascontiguousarray(x[d*N:(d+1)*N])
        P,Jm=self.dom.to_phys(Q); P=np.ascontiguousarray(P); Jm=np.ascontiguousarray(Jm)
        J=np.empty((self.dom.M,(d+1)*N)); nm=np.ascontiguousarray(self.dom.norm)
        _lib.fastjac(ctypes.c_int(d),ctypes.c_int(self.dom.deg),ctypes.c_int(N),ctypes.c_int(self.dom.M),*(a.ctypes.data_as(_dp) for a in (P,Jm,w,nm,J)))
        return J
    ProbG._old_jac=ProbG.jac; ProbG.jac=jac
