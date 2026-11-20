#!/usr/bin/env python3
"""Mirror-symmetric and full asymmetric searches for degree-9 PI rules N=18/17.

Start from the newly proven S3-symmetric 19-node rule. Removing its centroid
leaves exactly 7 mirror pairs and 4 nodes on x=y. This is the natural mirror
parameterization for N=18 (2a+b=18, a=7, b=4), which had not been tested in
the previous C3 and full-asymmetric searches.

For N=17, additionally remove one of the four axis nodes (a=7,b=3), redistribute
its weight symmetrically, and re-solve. The same physical starts are sent to a
mirror-reduced LM and a full asymmetric bounded LM.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import tri_search as TS

ROOT=Path(__file__).resolve().parents[1]
START=ROOT/"rules/minimal_pointcount_for_degree_n_PI/triangle/CA-T2-09-19-PI-MPC-a.json"
DEGREE=9
NOISES=(0.0,1e-3,1e-2,5e-2)
SEEDS=tuple(range(13001,13009))
MAXFEV=700
CONTINUATION=(5,7,8)
CONT_MAXFEV=100
ACCEPT=1e-13
MIN_WEIGHT=1e-8

def decompose_s3_rule(rule):
    X=np.asarray(rule["nodes"],float); W=np.asarray(rule["weights"],float)
    centroid=np.array([1/3,1/3])
    ci=int(np.argmin(np.linalg.norm(X-centroid,axis=1)))
    if np.linalg.norm(X[ci]-centroid)>1e-10: raise RuntimeError("No centroid")
    keep=set(range(len(W)))-{ci}
    axis=[]; pairs=[]
    for i in sorted(list(keep)):
        if i not in keep: continue
        x,y=X[i]
        if abs(x-y)<1e-9:
            axis.append((float(x),float(W[i]),i)); keep.remove(i)
        else:
            target=np.array([y,x])
            ds=np.linalg.norm(X-target,axis=1)
            j=int(np.argmin(ds))
            if ds[j]>1e-9 or j not in keep: raise RuntimeError("Mirror pair missing")
            # mirror_expand expects one point (x,y) and its swapped pair
            if x>y: i,j=j,i; x,y=X[i]
            pairs.append((float(x),float(y),float(W[i]),i,j))
            keep.remove(i);keep.remove(j)
    pairs.sort(key=lambda p:(p[0],p[1]))
    axis.sort(key=lambda p:p[0])
    if len(pairs)!=7 or len(axis)!=4: raise RuntimeError((len(pairs),len(axis)))
    return pairs,axis,float(W[ci]),X,W


def build_start(pairs,axis,centroid_w,drop_axis=None):
    P=list(pairs)
    A=list(axis)
    if drop_axis is not None:
        removed=A.pop(drop_axis)
        total=centroid_w+removed[1]
    else:
        total=centroid_w
    N=2*len(P)+len(A)
    increment=total/N
    s=[];t=[];w=[]
    for x,y,wt,_,_ in P:
        si,ti=TS.xy_to_st(np.array([x]),np.array([y]))
        s.append(float(si[0]));t.append(float(ti[0]));w.append(wt+increment)
    u=[p[0] for p in A]
    wu=[p[1]+increment for p in A]
    q=np.r_[s,t,w,u,wu]
    return q,len(P),len(A),total


def perturb_mirror(q,a,b,noise,seed):
    rng=np.random.default_rng(seed)
    v=q.copy()
    # q layout: s(a), t(a), w(a), u(b), wu(b)
    if noise:
        v[:2*a]+=rng.normal(0,noise,2*a)
        v[:2*a]=np.clip(v[:2*a],1e-10,1-1e-10)
        v[3*a:3*a+b]+=rng.normal(0,noise,b)
        v[3*a:3*a+b]=np.clip(v[3*a:3*a+b],1e-10,0.5-1e-10)
        v[2*a:3*a+b]*=np.exp(rng.normal(0,noise, a+b))
    return v


def solve_mirror(q0,a,b,noise,seed):
    q=perturb_mirror(q0,a,b,noise,seed)
    for d in CONTINUATION:
        q,_=TS.solve_mirror(q,a,b,d,max_nfev=CONT_MAXFEV)
    q,_=TS.solve_mirror(q,a,b,DEGREE,max_nfev=MAXFEV)
    return TS.mirror_to_full(q,a,b)


def solve_full(q0,a,b,noise,seed):
    q=perturb_mirror(q0,a,b,noise,seed)
    v=TS.mirror_to_full(q,a,b)
    s,t,w=TS.unpack(v,2*a+b)
    rng=np.random.default_rng(seed+700000)
    if noise:
        s=np.clip(s+rng.normal(0,noise,2*a+b),1e-10,1-1e-10)
        t=np.clip(t+rng.normal(0,noise,2*a+b),1e-10,1-1e-10)
    v=np.r_[s,t,w]
    for d in CONTINUATION:
        v,_=TS.solve(v,d,2*a+b,max_nfev=CONT_MAXFEV)
    v,_=TS.solve(v,DEGREE,2*a+b,max_nfev=MAXFEV)
    return v


def assess(v,N):
    s,t,w=TS.unpack(v,N);x,y=TS.to_xy(s,t)
    X=np.column_stack([x,y])
    bary=np.column_stack([1-x-y,x,y])
    r,_=TS.residual_and_jac(v,DEGREE,N)
    rinf=float(np.linalg.norm(r,np.inf)); minw=float(w.min()); minb=float(bary.min())
    return {"residual":rinf,"min_weight":minw,"min_barycentric":minb,
            "accepted":bool(rinf<ACCEPT and minw>MIN_WEIGHT and minb>=-1e-13)},X,w


def main():
    rule=json.loads(START.read_text())
    pairs,axis,wc,X0,W0=decompose_s3_rule(rule)
    results=[]
    cases=[]
    # N=18: remove centroid only.
    cases.append(("N18",None))
    # N=17: additionally remove each of the four axis nodes.
    cases.extend((f"N17_drop_axis_{i}",i) for i in range(4))
    t_all=time.perf_counter()
    for case,drop in cases:
      q0,a,b,redistributed=build_start(pairs,axis,wc,drop)
      N=2*a+b
      for noise in NOISES:
       for seed in SEEDS:
        methods=("mirror","full") if seed%2==0 else ("full","mirror")
        pair={}
        for method in methods:
          t0=time.perf_counter()
          v=solve_mirror(q0,a,b,noise,seed) if method=="mirror" else solve_full(q0,a,b,noise,seed)
          quality,X,W=assess(v,N)
          row={"case":case,"N":N,"drop_axis":drop,"noise":noise,"seed":seed,
               "method":method,"seconds":time.perf_counter()-t0,"quality":quality,
               "rule":({"degree":DEGREE,"N":N,"nodes":X.tolist(),"weights":W.tolist()}
                       if quality["accepted"] else None)}
          results.append(row);pair[method]=row
          print(f"END case={case} N={N} method={method} noise={noise:g} seed={seed} "
                f"res={quality['residual']:.3e} wmin={quality['min_weight']:.3e} "
                f"margin={quality['min_barycentric']:.3e} success={quality['accepted']} "
                f"seconds={row['seconds']:.2f}",flush=True)
        if pair["mirror"]["quality"]["accepted"] != pair["full"]["quality"]["accepted"]:
          print(f"PAIRED DIFFERENCE {case} noise={noise} seed={seed}",flush=True)
    summary={"target":"triangle degree9 PI N=18 and N=17; current proven N=19",
      "start_rule":"CA-T2-09-19-PI-MPC-a",
      "start_structure":{"mirror_pairs":7,"axis_nodes":4,"centroid_weight":wc},
      "continuation_degrees":CONTINUATION,"noise_levels":NOISES,"seeds":SEEDS,
      "maxfev":MAXFEV,"accept":ACCEPT,"results":results,"methods":{}}
    for case,_ in cases:
      summary["methods"][case]={}
      for method in ("mirror","full"):
        part=[r for r in results if r["case"]==case and r["method"]==method]
        summary["methods"][case][method]={
          "trials":len(part),"successes":sum(r["quality"]["accepted"] for r in part),
          "best_residual":min(r["quality"]["residual"] for r in part),
          "median_residual":float(np.median([r["quality"]["residual"] for r in part])),
          "total_seconds":sum(r["seconds"] for r in part)}
    summary["candidates"]=[
      {"case":r["case"],"method":r["method"],"noise":r["noise"],"seed":r["seed"],
       **r["quality"],**r["rule"]}
      for r in results if r["quality"]["accepted"] and r["rule"]]
    summary["elapsed_seconds"]=time.perf_counter()-t_all
    Path("degree9_mirror_18_17_results.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("FINAL SUMMARY")
    print(json.dumps(summary["methods"],indent=2))
    print("Candidates:",len(summary["candidates"]),flush=True)


if __name__=="__main__":main()
