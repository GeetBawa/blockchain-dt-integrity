import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
C={"blue":"#2a78d6","orange":"#eb6834","ink":"#0b0b0b","muted":"#52514e","fill":"#eef3fb","fill2":"#fdf0ea"}
fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
def box(x,y,w,h,t,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8.5,color=C["ink"])
def arr(a,b,dash=False,col=None,txt=None,off=(0,0.18)):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=11,lw=1.4,color=col or C["blue"],ls="--" if dash else "-"))
    if txt: ax.text((a[0]+b[0])/2+off[0],(a[1]+b[1])/2+off[1],txt,ha="center",fontsize=7.5,color=C["muted"])
box(0.2,3.3,1.9,1.1,"Physical asset\n(engine, sensors)",C["fill"],C["blue"])
box(2.9,3.3,1.9,1.1,"Gateway\nencode, batch,\nMerkle root",C["fill"],C["blue"])
box(5.6,3.3,1.9,1.1,"Twin store\nrecords, models,\ndecision log",C["fill"],C["blue"])
box(7.9,3.3,1.9,1.1,"Twin model\n(LSTM, RUL)",C["fill"],C["blue"])
box(2.2,0.7,2.3,1.1,"Registry contract\nroots, hash chain,\nmodels, decisions",C["fill2"],C["orange"])
box(5.9,0.7,2.3,1.1,"Verifier\nrecompute, compare,\nflag records",C["fill2"],C["orange"])
arr((2.1,3.85),(2.9,3.85)); arr((4.8,3.85),(5.6,3.85)); arr((7.5,3.85),(7.9,3.85))
arr((3.85,3.3),(3.5,1.8),True,C["orange"],"root per batch",(-0.75,0))
arr((4.5,1.25),(5.9,1.25),True,C["orange"],"read roots",(0,0.2))
arr((6.6,1.8),(6.6,3.3),False,C["blue"],"re-read records",(-0.85,0))
arr((7.6,1.8),(8.6,3.3),True,C["orange"],"verified / flagged",(1.0,-0.1))
ax.text(0.2,0.15,"solid: data path     dashed: commitment and verification path",fontsize=7.5,color=C["muted"],va="center")
fig.savefig("figures/fig_architecture.png",dpi=300,bbox_inches="tight")
