"""매수 가격: 현재(급등일 종가×0.85) vs 예상 저점(분위 0.6) 연도별 계좌 수익.
  python analyze_entry_years.py ../stock-data
"""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); DATA = sys.argv[1] if len(sys.argv) > 1 else "../stock-data"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, data, engine as en, run_integrated as ri, run_entry2 as r2
A=ri.prepare(data.load_pack(DATA,"krx_")); B=ri.prepare(data.load_pack(DATA,""))
preds=ri.predict_past_only(A,B); lq=r2.predict_low(A,B,0.6); P=en.PRE; rows=[]
for era,D in (("A",A),("B",B)):
    pr,thr=preds[era]; w,ev=D["w"],D["ev"]; sd,ed=r2.ERAS[era]
    sel=np.isfinite(pr)&(pr>=thr[0.7]); gate=D["up20"][np.clip(ev.t.values+1,0,len(D["up20"])-1)]
    for nm,lv in (("현재 0.85",w.c0*0.85),("예상저점 0.6",w.c0*np.clip(1+lq[era],0.70,0.97))):
        e=en.limit_entry(w,lv,11,"x","x",start=2); rr=np.arange(len(lv))
        el=np.where(e.col>=0,w.L[rr,np.clip(e.col,0,w.K-1)],np.nan)
        ret,hold,xcol,valid=en.simulate(w,e,0.07,np.inf,10,0.003,None,np.broadcast_to(el[:,None],(len(el),w.K)))
        mg=valid&sel&gate&(ev.date>=sd).values&((ev.date<ed).values if ed else True)
        pf,cs=en.portfolio_trades(w.dates,(w.t0+e.col-P)[mg],(w.t0+xcol-P)[mg],ret[mg],slots=10,start=sd,end=ed,max_new_per_day=2)
        ann=cs.resample("YE").last(); ar=ann/ann.shift(1).fillna(1.0)-1
        for d,v in ar.items(): rows.append(dict(방식=nm,연도=d.year,수익=v))
Y=pd.DataFrame(rows).pivot(index="연도",columns="방식",values="수익"); print(Y.round(3).to_string())
print("플러스 연도:", (Y>0).sum().to_dict(), "/", len(Y))
