"""통합 전략 청산 대조 분석: 예측 익절 vs 고정 7·8·10%, 진입일 저가·급등 전 가격 손절 조합.
  python analyze_exit_controls.py ../stock-data
"""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); DATA = sys.argv[1] if len(sys.argv) > 1 else "../stock-data"
import numpy as np, pandas as pd, data, engine as en, run_integrated as ri, run_exit2 as rx
pd.set_option("display.width",220)
A=ri.prepare(data.load_pack(DATA,"krx_")); B=ri.prepare(data.load_pack(DATA,""))
preds=ri.predict_past_only(A,B); P=en.PRE; INF=np.inf
ctx={}
for era,D in (("A",A),("B",B)):
    pr,thr=preds[era]; w,ev=D["w"],D["ev"]; e=en.limit_entry(w,w.c0*0.85,11,"E4","i",start=2)
    ctx[era]=dict(D=D,e=e,sel=np.isfinite(pr)&(pr>=thr[0.7]),gate=D["up20"][np.clip(ev.t.values+1,0,len(D["up20"])-1)])
mp=rx.predict_mfe(A,B,ctx["A"]["e"],ctx["B"]["e"])
rows=[]
for era,c in ctx.items():
    D,e,sel,gate=c["D"],c["e"],c["sel"],c["gate"]; w=D["w"]; sd,ed=rx.ERAS[era]; p=e.price; K=w.K
    b=lambda lv: np.broadcast_to(lv[:,None],(len(lv),K)); rr=np.arange(len(p))
    el=np.where(e.col>=0,w.L[rr,np.clip(e.col,0,K-1)],np.nan)
    tp6=p*(1+np.clip(0.6*np.nan_to_num(mp[era],nan=0.10),0.05,0.30))
    print(era,"0.6×예측 익절폭 분포: 10/50/90% =",np.round(np.nanquantile((tp6/p-1)[sel&(e.col>=0)],[0.1,0.5,0.9]),3))
    cfg={"기준 고정10%":(0.10,None,None),"고정7%":(0.07,None,None),"고정8%":(0.08,None,None),
         "예측익절0.6":(INF,None,tp6),"예측익절0.6+진입일저가손절":(INF,b(el),tp6),"예측익절0.6+급등전가격손절":(INF,b(w.p0),tp6),
         "고정7%+진입일저가손절":(0.07,b(el),None),"고정10%+진입일저가손절":(0.10,b(el),None)}
    for k,(tp,ce,tpx) in cfg.items():
        r=en.simulate(w,e,tp,INF,10,0.003,None,ce,tpx); rows.append(dict(청산=k,시기=era,**rx.evaluate(D,e,sel,gate,sd,ed,*r)))
R=pd.DataFrame(rows); R["calmar"]=R["계좌연"]/R["낙폭"].abs()
print(R.pivot_table(index="청산",columns="시기",values=["거래당","승률","계좌연","낙폭","calmar"]).round(4).to_string())
