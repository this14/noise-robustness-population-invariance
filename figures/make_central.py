"""make_central.py -- Central Illustration. Usage: python make_central.py <results51/analysis> [out_dir]"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys
R = (sys.argv[1].rstrip("/") + "/") if len(sys.argv) > 1 else "../results51/analysis/"  # analyze51.py output folder
OUT = (sys.argv[2].rstrip("/") + "/") if len(sys.argv) > 2 else "./"
t=pd.read_csv(R+"table2_population.csv"); d=pd.read_csv(R+"table3_channel_did.csv")
CNN=["resnet18","resnet34","inception1d","xresnet1d101"]
t=t[t.architecture.isin(CNN)]
CO=["ptbxl","chapman","georgia","mimic","echonext"]
NM={"ptbxl":"PTB-XL","chapman":"Chapman","georgia":"Georgia","mimic":"MIMIC-IV","echonext":"EchoNext"}
REF,US="#55554f","#eb6834"
INK,MUTED,GRID="#2b2b29","#6b6b65","#e6e5e0"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5})
fig=plt.figure(figsize=(7.6,5.4))
fig.text(0.02,0.965,"Is Noise Robustness Population-Invariant?",fontsize=13,fontweight="bold",color=INK,va="top")
fig.text(0.02,0.925,"Five cohorts (China, Germany, 3 U.S. institutions) · 4 convolutional architectures × 10 seeds · simulated device-attachment noise",
         fontsize=8,color=MUTED,va="top")
layers=[("t1","LAYER 1  Baseline discrimination","Tier 1 AUROC","Differs by\npopulation and task",(0.70,1.0)),
        ("n2","LAYER 2  Robustness to patch-type noise","Fraction of above-chance AUROC lost (Tier 2)","Equivalent across\npopulations\n(10/10 within ±0.05)",(-0.06,0.03)),
        ("n3","LAYER 3  Robustness to high-motion noise","Fraction of above-chance AUROC lost (Tier 3)","Not equivalent (0/10);\nlarger loss for\nHF and SHD tasks",(-0.30,0.0))]
top=0.86; h=0.215; gap=0.045
for i,(col,title,xl,verdict,xlim) in enumerate(layers):
    y0=top-(i+1)*h-i*gap
    ax=fig.add_axes([0.125,y0,0.42,h*0.62])
    for j,c in enumerate(CO):
        g=t[t.cohort==c][col]; m=g.mean()
        colr=US if c in ("georgia","mimic","echonext") else REF
        ax.plot([g.min(),g.max()],[j,j],color=colr,lw=2.2,solid_capstyle="round",alpha=0.55)
        ax.plot(m,j,"o",color=colr,ms=7,mec="white",mew=1.0,zorder=3)
    ax.set_yticks(range(5)); ax.set_yticklabels([NM[c] for c in CO],fontsize=7.5,color=INK)
    ax.invert_yaxis(); ax.set_xlim(*xlim)
    ax.xaxis.grid(True,color=GRID,lw=0.6); ax.set_axisbelow(True)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#8a8a85"); ax.tick_params(axis="x",labelsize=7,colors=MUTED); ax.tick_params(axis="y",length=0)
    ax.set_xlabel(xl,fontsize=7.2,color=MUTED,labelpad=2)
    if col!="t1": ax.axvline(0,color="#8a8a85",lw=0.8)
    fig.text(0.02,y0+h*0.62+0.022,title,fontsize=9,fontweight="bold",color=INK)
    fig.text(0.575,y0+h*0.31,verdict,fontsize=7.8,color=INK,va="center",
             bbox=dict(boxstyle="round,pad=0.45",fc="#f3f2ee",ec="none"))
# 3-channel box
ax=fig.add_axes([0.80,0.10,0.185,0.76]); ax.axis("off")
ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0.0,rounding_size=0.04",fc="#eef4fc",ec="#2a78d6",lw=1.0,transform=ax.transAxes))
cn=d[(d.architecture.isin(CNN))&(d.noise_mode=="shared")]
g_ptb=cn[cn.cohort=="ptbxl"].t1_diff_3ch_minus_1ch; g_ch=cn[cn.cohort=="chapman"].t1_diff_3ch_minus_1ch
txt=("3-CHANNEL\n(I, II, V1)\nvs Lead I\n\n"
     "Baseline AUROC\n▲ higher\n"+f"PTB-XL\n+{g_ptb.min():.3f} to +{g_ptb.max():.3f}\nChapman\n+{g_ch.min():.3f} to +{g_ch.max():.3f}\n\n"
     "High-motion\ndegradation\n≈ unchanged\n|difference| ≤ 0.011\nno consistent\ndirection")
ax.text(0.5,0.96,txt,ha="center",va="top",fontsize=7.4,color=INK,linespacing=1.3,transform=ax.transAxes)
fig.text(0.02,0.045,"●  mean of 4 architectures      —  range across architectures",fontsize=7.2,color=MUTED)
fig.text(0.45,0.045,"■ reference cohorts",fontsize=7.2,color=REF); fig.text(0.60,0.045,"■ U.S. cohorts",fontsize=7.2,color=US)
fig.text(0.02,0.018,"HF = heart failure (MIMIC-IV); SHD = structural heart disease (EchoNext). Layer 2 equivalence: pooled U.S. vs each reference cohort, per architecture.",fontsize=6.8,color=MUTED)
fig.savefig(OUT + "Central_Illustration.png",dpi=300); fig.savefig(OUT + "Central_Illustration.tiff",dpi=300)
print("ok")
