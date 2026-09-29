import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import os
out = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(out, "..", "hit_and_trial_results.csv"))
df["short"] = df.model_id
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
C = {"frozen":"#9aa0a6","simple":"#4c78a8","3zone":"#e45756"}
col = lambda s: C["frozen"] if "frozen" in s else (C["simple"] if "simple" in s else C["3zone"])

# 1 accuracy per experiment
fig, ax = plt.subplots(figsize=(11,5))
cols = [col(s) for s in df.fine_tune_strategy]
ax.bar(df.short, df.overall_accuracy*100, color=cols)
for i,v in enumerate(df.overall_accuracy*100): ax.text(i, v+1, f"{v:.1f}".rstrip("0").rstrip("."), ha="center", fontsize=8)
ax.set_ylabel("Overall accuracy (%)"); ax.set_title("All 14 logged experiments: accuracy (grey=frozen, blue=fine_tune_at, red=3-zone)")
plt.xticks(rotation=45, ha="right"); ax.set_ylim(0,108)
ax.annotate("domain shift:\nPBC->Raabin", xy=(1,85), xytext=(1.6,55), arrowprops=dict(arrowstyle="->"))
ax.annotate("wbc20 best", xy=(11,99), xytext=(8.3,72), arrowprops=dict(arrowstyle="->"))
ax.annotate("unseen 100-img set\n(only ~61%)", xy=(12,61), xytext=(12.2,30), arrowprops=dict(arrowstyle="->"))
plt.tight_layout(); plt.savefig(f"{out}/01_accuracy_all_experiments.png"); plt.close()

# 2 the story: per-class F1 along the main journey
journey = ["baseline_raabin_test","wbc_merged_frozen","wbc_finetune100","wbc_finetune110","wbc14","wbc20"]
labels = ["Baseline\non Raabin","+Stain norm\n+merge\n(frozen)","fine_tune\n=100","fine_tune\n=110","wbc14\n(best simple)","wbc20\n(3-zone+\nproper split)"]
cls = ["basophils_f1","eosinophils_f1","lymphocytes_f1","monocytes_f1","neutrophils_f1"]
d = df.set_index("model_id").loc[journey]
fig, ax = plt.subplots(figsize=(10,5.5))
for c in cls: ax.plot(range(len(journey)), d[c], marker="o", label=c.replace("_f1","").title())
ax.set_xticks(range(len(journey))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("F1 score"); ax.set_ylim(0,1.05); ax.legend(ncol=5, loc="lower right"); ax.grid(alpha=.3)
ax.set_title("Per-class F1 along the improvement journey")
plt.tight_layout(); plt.savefig(f"{out}/02_f1_journey.png"); plt.close()

# 3 heatmap all experiments x classes
fig, ax = plt.subplots(figsize=(9,7))
m = df[cls].values
im = ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels([c.replace("_f1","") for c in cls])
ax.set_yticks(range(len(df))); ax.set_yticklabels(df.model_id)
for i in range(m.shape[0]):
    for j in range(5): ax.text(j,i,f"{m[i,j]:.2f}",ha="center",va="center",fontsize=8)
ax.set_title("F1 heat-map: every experiment x every class"); plt.colorbar(im)
plt.tight_layout(); plt.savefig(f"{out}/03_f1_heatmap.png"); plt.close()

# 4 baseline collapse
b = df.set_index("model_id").loc[["baseline_pbc_only","baseline_raabin_test"]]
x = np.arange(5); fig, ax = plt.subplots(figsize=(9,5))
ax.bar(x-.2, b.loc["baseline_pbc_only",cls], .4, label="Tested on PBC (99.57%)", color="#54a24b")
ax.bar(x+.2, b.loc["baseline_raabin_test",cls], .4, label="Tested on Raabin (84.89%)", color="#e45756")
for i in range(5):
    ax.text(i-.2,b.iloc[0][cls[i]]+.02,f"{b.iloc[0][cls[i]]:.2f}",ha="center",fontsize=8)
    ax.text(i+.2,b.iloc[1][cls[i]]+.02,f"{b.iloc[1][cls[i]]:.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([c.replace("_f1","").title() for c in cls]); ax.set_ylim(0,1.15)
ax.set_ylabel("F1"); ax.legend(); ax.set_title("THE PROBLEM: same model, different microscope")
plt.tight_layout(); plt.savefig(f"{out}/04_problem_domain_shift.png"); plt.close()

# 5 architecture-tuning effect: kernels/CAB (fine-tune-at experiments)
s = df[df.model_id.isin(["wbc_finetune110","wbc11","wbc12","wbc14","wbc15"])]
fig, ax = plt.subplots(figsize=(10,5)); w=.13
metrics = ["overall_accuracy","basophils_f1","eosinophils_f1","monocytes_f1","macro_avg_f1"]
for k,mname in enumerate(metrics):
    ax.bar(np.arange(len(s))+(k-2)*w, s[mname], w, label=mname.replace("_f1","").replace("_"," "))
ax.set_xticks(range(len(s)))
ax.set_xticklabels([f"{a}\nFT@{b}\nCAB:{c}" for a,b,c in zip(s.model_id,s.fine_tune_at,s.cab_dw_kernels)], fontsize=7)
ax.set_ylim(.5,1.02); ax.legend(ncol=5,fontsize=8,loc="lower center"); ax.set_title("Kernel / CAB depth trials (pre-3-zone): wide CAB kernels win, extra layers hurt")
plt.tight_layout(); plt.savefig(f"{out}/05_kernel_cab_trials.png"); plt.close()

# 6 params
fig, ax = plt.subplots(figsize=(10,4.5))
ax.plot(df.model_id, df.trainable_params/1e6, "o-", label="trainable"); ax.plot(df.model_id, df.total_params/1e6, "s--", label="total")
ax.set_ylabel("Parameters (millions)"); plt.xticks(rotation=45, ha="right"); ax.legend()
ax.set_title("Model size per experiment (unfreezing backbone raises trainable params)")
plt.tight_layout(); plt.savefig(f"{out}/06_parameters.png"); plt.close()

# 7 MobileNetV2 zones diagram
fig, ax = plt.subplots(figsize=(12,3.6))
ax.add_patch(Rectangle((0,0),155,1,fc="#dddddd",ec="k"))
for (a,b),n in [((11,27),"Zone 1"),((38,54),"Zone 2"),((90,117),"Zone 3")]:
    ax.add_patch(Rectangle((a,0),b-a,1,fc="#e45756",ec="k")); ax.text((a+b)/2,.5,n,ha="center",va="center",fontsize=8,color="w")
ax.add_patch(Rectangle((110,-1.2),45,.8,fc="#4c78a8",ec="k")); ax.text(132,-.8,"simple fine_tune_at=110\n(layers 110-154)",ha="center",va="center",fontsize=7,color="w")
for L,n in [(26,"block_3_expand\n(feature1)"),(53,"block_6_expand\n(feature2)"),(116,"block_13_expand\n(feature3)")]:
    ax.annotate(n,xy=(L,1),xytext=(L,2.0),ha="center",fontsize=8,arrowprops=dict(arrowstyle="->"))
ax.axvspan(117,155,ymin=.0,ymax=.3,alpha=0); ax.text(136,1.25,"layers after 116:\nnever used by the head\n-> no gradient",ha="center",fontsize=7,color="#a00")
ax.set_xlim(-2,157); ax.set_ylim(-1.5,3); ax.set_yticks([]); ax.set_xlabel("MobileNetV2 layer index (155 layers). Red = 3-zone trainable, grey = frozen")
ax.set_title("Technique 4: where fine-tuning happens"); ax.spines["left"].set_visible(False)
plt.tight_layout(); plt.savefig(f"{out}/07_3zone_layers.png"); plt.close()

# 8 Reinhard demo: synthetic LAB distributions
rng = np.random.default_rng(0)
src = rng.normal(150,25,20000); tgt_m,tgt_s = 181.25,40.7
norm = (src-src.mean())/src.std()*tgt_s+tgt_m
fig, ax = plt.subplots(1,3,figsize=(12,3.6),sharey=True)
for a,(v,t,c) in zip(ax,[(src,"1. Source (Raabin-like) L channel","#e45756"),((src-src.mean())/src.std(),"2. Subtract mean, divide std","#9aa0a6"),(norm,"3. x target_std + target_mean (=PBC style)","#54a24b")]):
    a.hist(v,60,color=c); a.set_title(t,fontsize=9)
fig.suptitle("Technique 1: Reinhard normalisation (illustrative synthetic data, L channel; target mean=181.25, std=40.7)",fontsize=9)
plt.tight_layout(); plt.savefig(f"{out}/08_reinhard_stain_norm.png"); plt.close()

# 9 class imbalance + weights (numbers quoted in the scripts' docstrings)
fig, ax = plt.subplots(1,2,figsize=(11,4.2))
ax[0].bar(["Neutrophils","Monocytes","Basophils"],[9050,1772,1215],color=["#4c78a8","#f58518","#e45756"])
ax[0].set_title("Images per class before fix (as stated in script)")
for i,v in enumerate([9050,1772,1215]): ax[0].text(i,v+100,v,ha="center")
x=np.arange(3); ax[1].bar(x-.2,[.42,3.14,3.14],.4,label="weight quoted, imbalanced",color="#9aa0a6")
ax[1].bar(x+.2,[.49,1.47,1.47],.4,label="weight quoted, after oversampling to 3000",color="#54a24b")
ax[1].set_xticks(x); ax[1].set_xticklabels(["Neutrophils","Monocytes*","Basophils"]); ax[1].legend(fontsize=8)
ax[1].set_title("Class weights (*Monocyte pre-value not stated; Basophil value reused)")
plt.suptitle("Technique 3: oversample minorities + class weights"); plt.tight_layout(); plt.savefig(f"{out}/09_imbalance_oversampling.png"); plt.close()

# 10 split diagram
fig, ax = plt.subplots(figsize=(11,3.4))
ax.barh([2],[80,20],color=["#4c78a8","#e45756"],ec="k")
ax.text(40,2,"Train 80%",ha="center",color="w"); ax.text(90,2,"Test 20%\n(also used as validation = LEAK)",ha="center",va="center",color="w",fontsize=7)
ax.barh([0],[70,10,20],color=["#4c78a8","#f58518","#e45756"],left=[0,70,80],ec="k")
for xx,t in [(35,"Train 70% (oversample ONLY here)"),(75,"Val 10%\n2,391"),(90,"Test 20%\n4,781 (locked)")]: ax.text(xx,0,t,ha="center",va="center",fontsize=8,color="w")
ax.set_yticks([0,2]); ax.set_yticklabels(["Technique 5\n70/10/20","Technique 2 / earlier\n80/20"])
ax.set_xlim(0,100); ax.set_ylim(-.7,2.7); ax.set_xlabel("% of 23,904 merged images"); ax.set_title("Fixing evaluation leakage")
plt.tight_layout(); plt.savefig(f"{out}/10_data_split.png"); plt.close()

# 11 pipeline flow
fig, ax = plt.subplots(figsize=(13,3.2)); ax.axis("off")
steps=["Load PBC+Raabin\n(T2: merge, fix\nclass names)","Stain-normalise\nto LAB target\n(T1)","3-way split\nBEFORE\noversampling (T5)","Oversample\nBaso+Mono to 3000\n+ class weights (T3)","tf.data BigAug\nfloat16->float32\n(T5)","MobileNetV2\n3-zone unfreeze\n+ SK/MSFF/CAB (T4)","Train w/ val set\nCheckpoint,\nReduceLR, EarlyStop","Final eval on\nlocked test set"]
for i,s in enumerate(steps):
    ax.add_patch(FancyBboxPatch((i*1.6,0),1.4,1,boxstyle="round,pad=0.03",fc="#cfe2f3",ec="k")); ax.text(i*1.6+.7,.5,s,ha="center",va="center",fontsize=7)
    if i<len(steps)-1: ax.annotate("",xy=((i+1)*1.6-.02,.5),xytext=(i*1.6+1.42,.5),arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-.1,13); ax.set_ylim(-.1,1.1); ax.set_title("End-to-end production pipeline (how the 5 techniques fit)")
plt.tight_layout(); plt.savefig(f"{out}/11_pipeline_flow.png"); plt.close()

# 12 technique impact
fig, ax = plt.subplots(figsize=(9,4.5))
items=[("T1 Stain norm\nEosinophil F1",.10,.97),("T3 Oversample\nBasophil F1",.15,.98),("T4 3-zone\nBasophil F1",.17,.98),("T2+T3 Merge+\nweights Overall acc",.85,.91)]
for i,(n,a,b) in enumerate(items):
    ax.plot([i,i],[a,b],"k-"); ax.scatter(i,a,c="#e45756",s=60,zorder=3); ax.scatter(i,b,c="#54a24b",s=60,zorder=3)
    ax.text(i+.05,a,f"{a:.2f}",fontsize=8); ax.text(i+.05,b,f"{b:.2f}",fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels([n for n,_,_ in items],fontsize=8); ax.set_ylim(0,1.1)
ax.set_title("Impact claimed per technique (red=before, green=after)\nNote: claims come from the README; effects are combined, not isolated ablations")
plt.tight_layout(); plt.savefig(f"{out}/12_technique_impact_claims.png"); plt.close()
print(sorted(os.listdir(out)))
