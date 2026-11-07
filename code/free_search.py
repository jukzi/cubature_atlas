import sys,time,json,numpy as np
import kubatur as k
class Free(k.Problem):
    def bounds(self,N):
        d=self.d
        return np.r_[np.full(d*N,-0.7),np.full(N,0.0)],np.r_[np.full(d*N,1.7),np.full(N,50.0)]
name=sys.argv[1]; degs=[int(a) for a in sys.argv[2].split(',')]; starts=int(sys.argv[3]); k.SCALE_COLUMNS=True
inner=json.load(open('allrules30.json'))['rules'][name]
out={}
for deg in degs:
    Z=k.counting_value(name,deg); best=int(inner[str(deg)]['n'] if 'n' in inner[str(deg)] else len(inner[str(deg)]['pts']))
    best=len(inner[str(deg)]['pts'])
    D=k.Domain(name,deg); P=Free(D); found=None; t0=time.time()
    for N in range(Z,best):
        for s in range(starts):
            rng=np.random.default_rng(3000+s); C,W=k.heuristic_start(name,N,rng); x0=np.r_[D.to_parameters(C).ravel(),W]
            x,c=P.solve_with_restarts(x0,N,1500)
            if c<1e-24:
                X,Wt=P.points_and_weights(x,N)
                if Wt.min()>1e-13:
                    X2,W2=P.points_and_weights(x,N)
                    cert=k.certificate(name,deg,X2,W2)
                    print('  N',N,'start',s,'certificate h=%.2e certified=%s'%(cert['h'],cert['certified']),flush=True)
                    if cert['certified']:
                        found=(X,Wt); break
        if found: break
    if found:
        X,Wt=found; np.save('outer_%s_d%d_n%d.npy'%(name,deg,len(Wt)),np.c_[X,Wt])
        bar=np.c_[X,1-X.sum(1)].min(1)
        print(name,deg,'Z',Z,'inner',best,'outer',len(Wt),'nodes outside',(bar<-1e-9).sum(),'min bary %.2f'%bar.min(),'%.0fs'%(time.time()-t0),flush=True)
    else:
        print(name,deg,'Z',Z,'inner',best,'outer none below inner','%.0fs'%(time.time()-t0),flush=True)
