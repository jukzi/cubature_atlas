#!/usr/bin/env python3
"""C3-reduced homotopy from the proven degree-12 PO31 rule to PI31.

The PO rule has one 3-node C3 orbit outside the triangle and very small positive
weights. Project that orbit into the triangle in two ways, pull it a small
distance inside, and re-solve in the C3-reduced space. Compare to a full
31-node asymmetric solve from the same projected starts.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import tri_search as TS

ROOT=Path(__file__).resolve().parents[1]
START=ROOT/"rules/minimal_pointcount_for_degree_n_PO/triangle/CA-T2-12-31-PO-MPC.json"
DEGREE=12
N=31
K=10
MARGINS=(1e-6,1e-4,1e-3,5e-3,1e-2)
PROJECTIONS=("simplex","centroid_ray")
SEEDS=tuple(range(12001,12009))
MAXFEV=1000
ACCEPT=1e-13
MIN_WEIGHT=1e-8

def extract_base(rule):
    X=np.asarray(rule["nodes"],float); W=np.asarray(rule["weights"],float)
    center=np.array([1/3,1/3])
    ci=int(np.argmin(np.linalg.norm(X-center,axis=1)))
    if np.linalg.norm(X[ci]-center)>1e-10: raise RuntimeError("Missing C3 centroid")
    reps=list(range(0,30,3))
    weights=np.asarray([W[i] for i in reps]+[W[ci]])
    return X[reps].copy(),weights

def bary_project(b,mode):
    b=np.asarray(b,float)
    if mode=="simplex":
        u=np.sort(b)[::-1]
        cssv=np.cumsum(u)-1
        rho=np.nonzero(u-cssv/np.arange(1,len(b)+1)>0)[0][-1]
        theta=cssv[rho]/(rho+1)
        return np.maximum(b-theta,0.0)
    if mode=="centroid_ray":
        c=np.full(3,1/3); alpha=1.0
        for z in b:
            if z<0: alpha=min(alpha,(1/3)/(1/3-z))
        return c+alpha*(b-c)
    raise ValueError(mode)

def make_q(Xbase,weights,projection,margin):
    Xp=Xbase.copy()
    for i in range(K):
        x,y=Xp[i]
        b=np.array([1-x-y,x,y])
        bp=bary_project(b,projection)
        bp=(1-3*margin)*bp+margin
        Xp[i]=[bp[1],bp[2]]
    s,t=TS.xy_to_st(Xp[:,0],Xp[:,1])
    q=np.r_[s,t,weights[:K],weights[K]]
    return q,Xp

def assess(v):
    s,t,w=TS.unpack(v,N); x,y=TS.to_xy(s,t)
    X=np.column_stack([x,y])
    bary=np.column_stack([1-x-y,x,y])
    r,_=TS.residual_and_jac(v,DEGREE,N)
    rinf=float(np.linalg.norm(r,np.inf))
    minw=float(w.min()); minb=float(bary.min())
    return {"residual":rinf,"min_weight":minw,"min_barycentric":minb,
            "outside_nodes":int((bary.min(axis=1)<-1e-9).sum()),
            "accepted":bool(rinf<ACCEPT and minw>MIN_WEIGHT and minb>=-1e-13)},X,w

def solve_c3(q0):
    q,_=TS.solve_c3(q0,K,True,DEGREE,max_nfev=MAXFEV)
    return TS.c3_to_full(q,K,True)

def solve_full(q0,margin,seed):
    v=TS.c3_to_full(q0,K,True)
    s,t,w=TS.unpack(v,N)
    rng=np.random.default_rng(seed)
    s=np.clip(s+rng.normal(0,2e-4,N),1e-10,1-1e-10)
    t=np.clip(t+rng.normal(0,2e-4,N),1e-10,1-1e-10)
    v=np.r_[s,t,w]
    v,_=TS.solve(v,DEGREE,N,max_nfev=MAXFEV)
    return v

def main():
    rule=json.loads(START.read_text())
    if rule.get("id")!="CA-T2-12-31-PO-MPC" or len(rule["nodes"])!=31:
        raise RuntimeError("Unexpected PO source rule")
    Xbase,weights=extract_base(rule)
    results=[]
    for projection in PROJECTIONS:
      for margin in MARGINS:
        for seed in SEEDS:
          q0,Xp=make_q(Xbase,weights,projection,margin)
          pair={}
          methods=("c3","full") if seed%2==0 else ("full","c3")
          for method in methods:
            t0=time.perf_counter()
            v=solve_c3(q0) if method=="c3" else solve_full(q0,margin,seed)
            quality,X,W=assess(v)
            row={"projection":projection,"margin":margin,"seed":seed,"method":method,
                 "seconds":time.perf_counter()-t0,"quality":quality,
                 "projected_outer_base":Xp[0].tolist(),
                 "rule":({"degree":DEGREE,"N":N,"nodes":X.tolist(),"weights":W.tolist()}
                         if quality["accepted"] else None)}
            results.append(row); pair[method]=row
            print(f"END projection={projection} margin={margin:g} seed={seed} method={method} "
                  f"res={quality['residual']:.3e} wmin={quality['min_weight']:.3e} "
                  f"minb={quality['min_barycentric']:.3e} outside={quality['outside_nodes']} "
                  f"success={quality['accepted']} seconds={row['seconds']:.2f}",flush=True)
          if pair["c3"]["quality"]["accepted"] != pair["full"]["quality"]["accepted"]:
            print(f"PAIRED DIFFERENCE projection={projection} margin={margin} seed={seed}",flush=True)
    summary={"target":"triangle degree12 PI N31 (current PI32, PO31)",
      "start_rule":"CA-T2-12-31-PO-MPC","outside_orbit_weight":float(rule["weights"][0]),
      "projections":PROJECTIONS,"margins":MARGINS,"seeds":SEEDS,"maxfev":MAXFEV,
      "accept":ACCEPT,"results":results,"candidates":[
        {"projection":r["projection"],"margin":r["margin"],"seed":r["seed"],
         "method":r["method"],**r["quality"],**r["rule"]}
        for r in results if r["quality"]["accepted"] and r["rule"]]}
    summary["methods"]={}
    for projection in PROJECTIONS:
      summary["methods"][projection]={}
      for method in ("c3","full"):
        part=[r for r in results if r["projection"]==projection and r["method"]==method]
        summary["methods"][projection][method]={"trials":len(part),
          "successes":sum(r["quality"]["accepted"] for r in part),
          "best_residual":min(r["quality"]["residual"] for r in part),
          "median_residual":float(np.median([r["quality"]["residual"] for r in part])),
          "total_seconds":sum(r["seconds"] for r in part)}
    Path("degree12_c3_po_to_pi_results.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("FINAL SUMMARY")
    print(json.dumps(summary["methods"],indent=2))
    print("Candidates:",len(summary["candidates"]),flush=True)

if __name__=="__main__": main()
