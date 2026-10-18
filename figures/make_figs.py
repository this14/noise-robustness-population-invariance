"""make_figs.py -- Figures 1 and 2. Usage: python make_figs.py <results51/analysis> [out_dir]"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
R = (sys.argv[1].rstrip("/") + "/") if len(sys.argv) > 1 else "../results51/analysis/"  # analyze51.py output folder
OUT = (sys.argv[2].rstrip("/") + "/") if len(sys.argv) > 2 else "./"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#8a8a85","axes.labelcolor":"#2b2b29","xtick.color":"#55554f","ytick.color":"#55554f"})
ARCH=["resnet18","resnet34","inception1d","xresnet1d101","bilstm"]
LAB={"resnet18":"ResNet-18","resnet34":"ResNet-34","inception1d":"Inception1d","xresnet1d101":"xresnet1d101","bilstm":"BiLSTM"}
COL=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"]; MK=["o","s","^","D","v"]
GRID="#e6e5e0"
def panel(ax, df, xs, xlab, m, lo, hi, title, ylab):
    off=np.linspace(-0.28,0.28,len(ARCH))
    for i,a in enumerate(ARCH):
        for j,x in enumerate(xs):
            r=df[(df.k==x)&(df.architecture==a)]
            if r.empty or pd.isna(r[m].iloc[0]): continue
            v=r[m].iloc[0]; e=[[v-r[lo].iloc[0]],[r[hi].iloc[0]-v]]
            ax.errorbar(j+off[i],v,yerr=e,fmt=MK[i],color=COL[i],ms=5.5,mec="white",mew=0.8,elinewidth=1.3,capsize=0,
                        label=LAB[a] if j==0 else None,zorder=3)
    ax.axhline(0,color="#8a8a85",lw=0.8,zorder=1)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xlab)
    ax.yaxis.grid(True,color=GRID,lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title,loc="left",fontsize=9.5,fontweight="bold",color="#2b2b29"); ax.set_ylabel(ylab)

# Figure 1
t=pd.read_csv(R+"table2_population.csv").rename(columns={"cohort":"k"})
co=["ptbxl","chapman","georgia","mimic","echonext"]; cl=["PTB-XL\nGermany","Chapman\nChina","Georgia\nU.S.","MIMIC-IV\nU.S.","EchoNext\nU.S."]
fig,axs=plt.subplots(1,2,figsize=(7.6,3.7),sharey=False)
panel(axs[0],t,co,cl,"n2","n2_lo","n2_hi","A  Tier 2 (patch-type wearable)","Normalized degradation\n(fraction of above-chance AUROC lost)")
panel(axs[1],t,co,cl,"n3","n3_lo","n3_hi","B  Tier 3 (high-motion wearable)","")
for ax in axs:
    ax.axvspan(1.5,4.5,color="#f3f2ee",zorder=0)
for ax in axs:
    ax.text(3,0.02,"U.S. cohorts",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5,color="#55554f")
    ax.tick_params(axis="x",labelsize=7.8)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=5,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.07,1,1))
fig.savefig(OUT + "Figure1.png",dpi=300); fig.savefig(OUT + "Figure1.tiff",dpi=300)

# Figure 2
d=pd.read_csv(R+"table3_channel_did.csv").rename(columns={"cohort":"k"})
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
for ax,mode,ttl in zip(axs,["shared","independent"],["A  Shared noise (primary)","B  Independent noise (sensitivity)"]):
    panel(ax,d[d.noise_mode==mode],["ptbxl","chapman"],["PTB-XL","Chapman"],"did_d3","did_d3_lo","did_d3_hi",ttl,
          "Difference in Tier 3 degradation\n(3-channel − Lead I, AUROC)" if mode=="shared" else "")
    ax.set_xlim(-0.6,1.6)
axs[0].annotate("3-channel degrades less ↑",xy=(0.02,0.97),xycoords="axes fraction",fontsize=7.5,color="#55554f",va="top")
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=5,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.08,1,1))
fig.savefig(OUT + "Figure2.png",dpi=300); fig.savefig(OUT + "Figure2.tiff",dpi=300)
print("ok")
