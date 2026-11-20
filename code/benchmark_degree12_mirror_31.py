#!/usr/bin/env python3
"""Mirror-symmetric elimination of one axis node from the degree-12 PI32 rule.

The current PI32 rule is mirror-symmetric with 14 pairs + 4 axis nodes.
Deleting one axis node gives N=31 while preserving the mirror group:
14 pairs + 3 axis nodes. We test every axis-node deletion, equal and
inverse-distance redistribution, four perturbation levels, and paired
mirror-reduced vs full-asymmetric solves.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import tri_search as TS

ROOT=Path(__file__).resolve().parents[1]
START=ROOT/"rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-12-32-PI-MPC.json"
DEGREE=12
TARGET_N=31
A=14
B=3
NOISES=(0.0,5e-4,5e-3,3e-2)
REDISTRIBUTIONS=("equal","inverse_distance")
SEEDS=tuple(range(14001,14009))
CONTINUATION=(8,10,11)
CONT_MAXFEV=120
MAXFEV=900
ACCEPT=1e-13
MIN_WEIGHT=1e-8

def extract_rule(rule):
    X=np.asarray(rule["nodes"],float); W=np.asarray(rule["weights"],float)
    used=set(); pairs=[]; axis=[]
    for i,(x,y) in enumerate(X):
        if i in used: continue
        if abs(x-y)<1e-9:
            axis.append((float(x),float(W[i]),int(i)));used.add(i)
        else:
            target=np.array([y,x])
            ds=np.linalg.norm(X-target,axis=1)
            j=int(np.argmin(ds))
            if ds[j]>1e-9 or j in used: raise RuntimeError("Missing mirror pair")
            if x>y: i,j=j,i; x,y=X[i]
            pairs.append((float(x),float(y),float(W[i]),int(i),int(j)))
            used.add(i);used.add(j)
    pairs.sort(key=lambda p:(p[0],p[1]))
    axis.sort(key=lambda p:p[0])
    if len(pairs)!=14 or len(axis)!=4: raise RuntimeError((len(pairs),len(axis)))
    return X,W,pairs,axis

def build_start(X,W,pairs,axis,drop_axis,redistribution):
    removed=axis[drop_axis]
    kept_axis=[p for i,p in enumerate(axis) if i!=drop_axis]
    coords=[np.array([p[0],p[1]]) for p in pairs]+[np.array([p[0],p[0]]) for p in kept_axis]
    if redistribution=="equal":
        inc=np.full(TARGET_N,removed[1]/TARGET_N)
        pair_inc=inc[:A]; axis_inc=inc[A:]
    else:
        inv=[]
        for p in pairs:
            d2=np.sum((np.array([p[0],p[1]])-np.array([removed[0],removed[0]]))**2)
            inv.append(1.0/max(d2,1e-10))
        for p in kept_axis:
            d2=2.0*(p[0]-removed[0])**2
            inv.append(1.0/max(d2,1e-10))
        inv=np.asarray(inv,float)
        # Pair nodes occur twice physically, each with the same increment.
        denom=2*inv[:A].sum()+inv[A:].sum()
        pair_inc=removed[1]*inv[:A]/denom
        axis_inc=removed[1]*inv[A:]/denom
    pair_x=np.array([p[0] for p in pairs])
    pair_y=np.array([p[1] for p in pairs])
    s,t=TS.xy_to_st(pair_x,pair_y)
    pair_w=np.array([p[2] for p in pairs])+pair_inc
    u=np.array([p[0] for p in kept_axis])
    axis_w=np.array([p[1] for p in kept_axis])+axis_inc
    return np.r_[s,t,pair_w,u,axis_w],removed

def perturb(q0,noise,seed):
    q=q0.copy();rng=np.random.default_rng(seed)
    if noise:
        q[:2*A]+=rng.normal(0,noise,2*A)
        q[:2*A]=np.clip(q[:2*A],1e-10,1-1e-10)
        q[3*A:3*A+B]+=rng.normal(0,noise,B)
        q[3*A:3*A+B]=np.clip(q[3*A:3*A+B],1e-10,0.5-1e-10)
        q[2*A:3*A]*=np.exp(rng.normal(0,noise,A))
        q[3*A+B:3*A+2*B]*=np.exp(rng.normal(0,noise,B))
    return q

def solve_mirror(q0,noise,seed):
    q=perturb(q0,noise,seed)
    for d in CONTINUATION:
        q,_=TS.solve_mirror(q,A,B,d,max_nfev=CONT_MAXFEV)
    q,_=TS.solve_mirror(q,A,B,DEGREE,max_nfev=MAXFEV)
    return TS.mirror_to_full(q,A,B)

def solve_full(q0,noise,seed):
    q=perturb(q0,noise,seed)
    v=TS.mirror_to_full(q,A,B)
    s,t,w=TS.unpack(v,TARGET_N)
    rng=np.random.default_rng(seed+987654)
    if noise:
        s=np.clip(s+rng.normal(0,noise,TARGET_N),1e-10,1-1e-10)
        t=np.clip(t+rng.normal(0,noise,TARGET_N),1e-10,1-1e-10)
    v=np.r_[s,t,w]
    for d in CONTINUATION:
        v,_=TS.solve(v,d,TARGET_N,max_nfev=CONT_MAXFEV)
    v,_=TS.solve(v,DEGREE,TARGET_N,max_nfev=MAXFEV)
    return v

def assess(v):
    s,t,w=TS.unpack(v,TARGET_N);x,y=TS.to_xy(s,t)
    X=np.column_stack([x,y])
    bary=np.column_stack([1-x-y,x,y])
    res,_=TS.residual_and_jac(v,DEGREE,TARGET_N)
    rinf=float(np.linalg.norm(res,np.inf));minw=float(w.min());minb=float(bary.min())
    quality={"residual":rinf,"min_weight":minw,"min_barycentric":minb,
             "accepted":bool(rinf<ACCEPT and minw>MIN_WEIGHT and minb>=-1e-13)}
    return quality,X,w

def main():
    rule=json.loads(START.read_text())
    X,W,pairs,axis=extract_rule(rule)
    results=[]
    for drop in range(4):
      for redistribution in REDISTRIBUTIONS:
        q0,removed=build_start(X,W,pairs,axis,drop,redistribution)
        for noise in NOISES:
          for seed in SEEDS:
            methods=("mirror","full") if seed%2==0 else ("full","mirror")
            pair={}
            for method in methods:
              t0=time.perf_counter()
              v=solve_mirror(q0,noise,seed) if method=="mirror" else solve_full(q0,noise,seed)
              quality,Xf,Wf=assess(v)
              row={"drop_axis":drop,"removed_axis_u":removed[0],"removed_axis_weight":removed[1],
                   "redistribution":redistribution,"noise":noise,"seed":seed,"method":method,
                   "seconds":time.perf_counter()-t0,"quality":quality,
                   "nodes":Xf.tolist(),"weights":Wf.tolist()}
              results.append(row);pair[method]=row
              print(f"END drop={drop} redist={redistribution} noise={noise:g} seed={seed} "
                    f"method={method} res={quality['residual']:.3e} "
                    f"wmin={quality['min_weight']:.3e} margin={quality['min_barycentric']:.3e} "
                    f"success={quality['accepted']} seconds={row['seconds']:.2f}",flush=True)
            if pair["mirror"]["quality"]["accepted"]!=pair["full"]["quality"]["accepted"]:
              print("PAIRED DIFFERENCE",drop,redistribution,noise,seed,flush=True)
    summary={"target":"triangle degree12 PI N31 (current PI count 32, PO count31)",
       "start_rule":"CA-T2-12-32-PI-MPC","starting_mirror_structure":{"pairs":14,"axis_nodes":4},
       "target_mirror_structure":{"pairs":14,"axis_nodes":3},
       "redistributions":REDISTRIBUTIONS,"noise_levels":NOISES,"seeds":SEEDS,
       "continuation_degrees":CONTINUATION,"continuation_maxfev":CONT_MAXFEV,"maxfev":MAXFEV,
       "accept":ACCEPT,"methods":{},"results":results}
    for drop in range(4):
      case=f"drop_axis_{drop}";summary["methods"][case]={}
      for method in ("mirror","full"):
        part=[r for r in results if r["drop_axis"]==drop and r["method"]==method]
        summary["methods"][case][method]={"trials":len(part),
          "successes":sum(r["quality"]["accepted"] for r in part),
          "best_residual":min(r["quality"]["residual"] for r in part),
          "median_residual":float(np.median([r["quality"]["residual"] for r in part])),
          "total_seconds":sum(r["seconds"] for r in part)}
    summary["candidates"]=[r for r in results if r["quality"]["accepted"]]
    Path("degree12_mirror_32_to_31_results.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("FINAL SUMMARY");print(json.dumps(summary["methods"],indent=2))
    print("Candidates:",len(summary["candidates"]),flush=True)

if __name__=="__main__":main()
