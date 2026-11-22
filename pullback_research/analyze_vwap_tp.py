"""VWAP 익절 vs 고정 7~10% 익절 대조 (예상 저점 진입, 두 시기).
  python analyze_vwap_tp.py ../stock-data
"""
import sys,warnings; warnings.filterwarnings("ignore"); import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); DATA = sys.argv[1] if len(sys.argv) > 1 else "../stock-data"
import numpy as np, pandas as pd, data, engine as en, run_integrated as ri, run_entry2 as r2, run_vwap as rv
pd.set_option("display.width",220)
A=ri.prepare(data.load_pack(DATA,"krx_")); B=ri.prepare(data.load_pack(DATA,""))
preds=ri.predict_past_only(A,B); lq=r2.predict_low(A,B,0.6); P=en.PRE; rows=[]
for era,D in (("A",A),("B",B)):
    pr,thr=preds[era]; w,ev=D["w"],D["ev"]; sd,ed=r2.ERAS[era]
    inper=(ev.date>=sd).values&((ev.date<ed).values if ed else True); sel=np.isfinite(pr)&(pr>=thr[0.7])
    gate=D["up20"][np.clip(ev.t.values+1,0,len(D["up20"])-1)]
    e=en.limit_entry(w,w.c0*np.clip(1+lq[era],0.70,0.97),11,"q","x",start=2); r_=np.arange(len(w.c0)); K=w.K
    el=np.where(e.col>=0,w.L[r_,np.clip(e.col,0,K-1)],np.nan); ce=np.broadcast_to(el[:,None],(len(el),K))
    vwD,_=rv.avwap(w,P); evw=np.where(e.col>=0,vwD[r_,np.clip(e.col,0,K-1)],np.nan)
    for nm,tp,tpx in (("고정7%",0.07,None),("고정8%",0.08,None),("고정9%",0.09,None),("고정10%",0.10,None),("VWAP익절(아래면7%)",0.07,evw),("VWAP익절(아래면9%)",0.09,evw)):
        ret,hold,xcol,valid=en.simulate(w,e,tp,np.inf,10,0.003,None,ce,tpx)
        m=valid&sel&inper; mg=m&gate
        pf,cs=en.portfolio_trades(w.dates,(w.t0+e.col-P)[mg],(w.t0+xcol-P)[mg],ret[mg],slots=10,start=sd,end=ed,max_new_per_day=2)
        rows.append(dict(시기=era,익절=nm,거래당=en.wmean(ret[m]),승률=(ret[m]>0).mean(),계좌연=pf["cagr"],낙폭=pf["mdd"]))
    above=(evw/e.price-1)[sel&inper&(e.col>=0)]
    print(era,"진입 시 VWAP이 매수가보다 1% 이상 위인 비율",np.nanmean(above>0.01).round(3),"| VWAP 익절폭 10/50/90%",np.nanquantile(above[above>0.01],[0.1,0.5,0.9]).round(3))
R=pd.DataFrame(rows); R["calmar"]=R["계좌연"]/R["낙폭"].abs()
print(R.pivot_table(index="익절",columns="시기",values=["거래당","승률","계좌연","낙폭","calmar"]).round(4).to_string())
