"""KRX 2005~2014 에서 반등 예측 모델(키움 2015~2026 학습) 상세 분석: 10분위, 연도별, 매수법별, 계좌.
  python analyze_model_krx.py ../stock-data
"""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); DATA = sys.argv[1] if len(sys.argv) > 1 else "../stock-data"
import numpy as np, pandas as pd, data, engine as en, screen as sc, capit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
pd.set_option("display.width",220)
kp=data.load_pack(DATA,""); kev=en.find_events(kp,min_value=1e9); kw=en.build_windows(kp,kev)
KX,KY=sc.features(kp,kw,kev),sc.targets(kw); ok=KY.y_bar.notna().values
mdl=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=200,min_samples_leaf=200,l2_regularization=1.0,random_state=0).fit(KX.values[ok],KY.y_bar.values[ok].astype(int))
thr=np.quantile(mdl.predict_proba(KX.values[ok])[:,1],0.8)
x=data.load_pack(DATA,"krx_"); ev=en.find_events(x,min_value=1e9); w=en.build_windows(x,ev); P=en.PRE
X=sc.features(x,w,ev); Y=sc.targets(w); pr=mdl.predict_proba(X.values)[:,1]
inper=((ev.date>="2005-01-01")&(ev.date<"2015-01-01")).values; yr=ev.date.dt.year.values; dates=ev.date.values
m=inper&Y.y_bar.notna().values
print(f"AUC(2005~2014) = {roc_auc_score(Y.y_bar[m],pr[m]):.3f}, 선별 비율 {(pr[inper]>=thr).mean():.1%}")
d=pd.DataFrame({"p":pr[m],"y":Y.y_bar[m],"r":Y.y_ret10[m]}); d["dec"]=pd.qcut(d.p.rank(method="first"),10,labels=False)+1
print(d.groupby("dec").agg(성공률=("y","mean"),수익10일=("r","mean"),중앙=("r","median")).round(4).T.to_string())
top=pr>=thr
lk=en.locked_up(w,np.arange(len(ev)),np.full(len(ev),P+1)); C1=w.C[:,P+1]; O2=w.O[:,P+2]
E={"D+1종가·익절15%·10일":(en.Entry("a",np.where(lk|~np.isfinite(C1),-1,P+1),C1,"close"),0.15,10,np.inf),
   "D+1종가·익절10%·10일":(en.Entry("a",np.where(lk|~np.isfinite(C1),-1,P+1),C1,"close"),0.10,10,np.inf),
   "D+2시가·익절15%·10일":(en.Entry("b",np.where(np.isfinite(O2),P+2,-1),O2,"open"),0.15,10,np.inf),
   "급등일종가-15%지정가·익절10%·10일":(en.limit_entry(w,w.c0*0.85,11,"c","s",start=2),0.10,10,np.inf),
   "급등일종가-25%지정가·익절15%·20일":(en.limit_entry(w,w.c0*0.75,21,"d","s",start=2),0.15,20,np.inf)}
rows=[];yrows=[]
for k,(e,tp,H,sl) in E.items():
    ret,hold,xcol,valid=en.simulate(w,e,tp,sl,H,0.003)
    for g,gm in (("모델상위20%",top),("전체",np.ones(len(ev),bool))):
        mm=valid&gm&inper; mc,se=capit.clustered_mean_se(pd.Series(ret[mm]),pd.Series(dates[mm]))
        pf,_=en.portfolio_trades(w.dates,(w.t0+e.col-P)[mm],(w.t0+xcol-P)[mm],ret[mm],slots=10,start="2005-01-01",end="2015-01-01",max_new_per_day=2)
        rows.append(dict(매수법=k,그룹=g,n=mm.sum(),거래당=en.wmean(ret[mm]),승률=(ret[mm]>0).mean(),t묶음=mc/se,계좌연=pf["cagr"],낙폭=pf["mdd"]))
        if k=="D+1종가·익절15%·10일":
            for y in range(2005,2015):
                q=valid&gm&(yr==y); yrows.append(dict(y=y,g=g,v=en.wmean(ret[q])))
print(pd.DataFrame(rows).round(4).to_string(index=False))
Yt=pd.DataFrame(yrows).pivot(index="y",columns="g",values="v"); Yt["차이"]=Yt["모델상위20%"]-Yt["전체"]
print(Yt.round(4).to_string()); print("상위20%가 더 좋은 해:", int((Yt["차이"]>0).sum()),"/",len(Yt))

print("\n=== 시장 상승국면 필터 + 동시보유/하루 진입 변화 (모델 상위20%) ===")
idx=sc.market_index(x); up20=(idx>idx.rolling(20).mean()).values[np.clip(ev.t.values+1,0,len(idx)-1)]
rows=[]
for k in ("D+1종가·익절10%·10일","D+2시가·익절15%·10일","급등일종가-15%지정가·익절10%·10일"):
    e,tp,H,sl=E[k]; ret,hold,xcol,valid=en.simulate(w,e,tp,sl,H,0.003)
    for gate in (False,True):
        for slots,cap in ((10,2),(20,1)):
            mm=valid&top&inper&(up20 if gate else True)
            pf,_=en.portfolio_trades(w.dates,(w.t0+e.col-P)[mm],(w.t0+xcol-P)[mm],ret[mm],slots=slots,start="2005-01-01",end="2015-01-01",max_new_per_day=cap)
            rows.append(dict(매수법=k,시장상승국면만=gate,동시보유=slots,하루신규=cap,거래=pf["trades"],계좌연=pf["cagr"],낙폭=pf["mdd"]))
print(pd.DataFrame(rows).round(4).to_string(index=False))
