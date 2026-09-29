"""Storyboard figures: PROBLEM -> CODE STEP -> EFFECT, one per technique + overview.
All numbers come from hit_and_trial_results.csv or from figures quoted in the technique scripts."""
import os, textwrap, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(out, "..", "hit_and_trial_results.csv")).set_index("model_id")
BLUE, ORANGE, AQUA, YEL, MAG = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BEFORE, AFTER = "#8a8984", BLUE
plt.rcParams.update({"figure.dpi":140,"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":MUTED,"axes.labelcolor":MUTED,"xtick.color":MUTED,"ytick.color":MUTED,"text.color":INK,
    "axes.facecolor":"#fcfcfb","figure.facecolor":"#fcfcfb"})
CLS = ["basophils_f1","eosinophils_f1","lymphocytes_f1","monocytes_f1","neutrophils_f1"]
NAMES = ["Basophil","Eosinophil","Lymphocyte","Monocyte","Neutrophil"]

def text_panel(ax, title, body, code=False, color=INK):
    ax.axis("off"); ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    ax.text(0, .97, body, va="top", ha="left", fontsize=8 if code else 9,
            family="monospace" if code else None, color=color, wrap=True, transform=ax.transAxes,
            linespacing=1.4)

def before_after(ax, labels, b, a, blab, alab, ylim=(0,1.12), fmt="{:.2f}"):
    x = np.arange(len(labels)); w = .38
    ax.bar(x-w/2, b, w, color=BEFORE, label=blab); ax.bar(x+w/2, a, w, color=AFTER, label=alab)
    for i,(vb,va) in enumerate(zip(b,a)):
        ax.text(i-w/2, vb+.02*ylim[1], fmt.format(vb), ha="center", fontsize=7, color=INK)
        ax.text(i+w/2, va+.02*ylim[1], fmt.format(va), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5,-.12), ncol=2)

def story(fname, title, problem, code, eff_title, draw, note, sub=None):
    fig = plt.figure(figsize=(15, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.35], wspace=.12)
    text_panel(fig.add_subplot(gs[0]), "1  THE PROBLEM", problem)
    text_panel(fig.add_subplot(gs[1]), "2  WHAT THE CODE DOES", code, code=True)
    ax = fig.add_subplot(gs[2]); ax.set_title("3  EFFECT ON RESULTS: " + eff_title, loc="left", fontweight="bold", fontsize=10)
    draw(ax)
    fig.suptitle(title, x=.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(.01, .01, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.subplots_adjust(top=.85, bottom=.22, left=.02, right=.985)
    fig.savefig(f"{out}/{fname}"); plt.close(fig)

W = lambda s, n=44: "\n".join(textwrap.wrap(s, n))

# ---- T1 stain normalisation
b, a = df.loc["baseline_raabin_test", CLS].values, df.loc["wbc_merged_frozen", CLS].values
def d1(ax):
    before_after(ax, NAMES, b, a, "Before: PBC-only model on Raabin", "After: stain norm + merge (+T3), backbone frozen")
    ax.set_ylabel("F1 score per class")
story("s1_stain_normalization.png", "Technique 1 - Reinhard stain normalisation  (technique1_stain_normalization.py)",
 W("PBC images are pink-purple; Raabin images (other microscope + stain) are darker purple. A CNN treats colour as a clue, so a model trained on PBC failed on Raabin: accuracy 99.57% -> 84.89%, Eosinophil F1 1.00 -> 0.10.\n\nColour is a lab artefact, not a property of the cell."),
 "img_lab = cv2.cvtColor(img, BGR2LAB)\nsrc_mean, src_std = per-image mean/std\n\nfor i in range(3):   # L, A, B\n  x = (x - src_mean[i]) / src_std[i]\n  x = x * TARGET_STD[i] + TARGET_MEAN[i]\n\nTARGET_MEAN=[181.25,142.24,131.07]\nTARGET_STD =[ 40.72,  9.86, 13.54]\n(average over 20 imgs/class of\n the merged train set)\n\n-> every image gets the SAME colour\n   statistics before the model sees it",
 "Eosinophil F1 0.10 -> 0.97", d1,
 "Caveat: this row also includes dataset merging (T2) and oversampling/class weights (T3), so the gain is not isolated to stain normalisation. The deployed app (orchestration.py) only divides by 255 - no LAB step.")

# ---- T2 merging
def d2(ax):
    ax.bar([0], [9390], color=BLUE, width=.5); ax.bar([0], [14514], bottom=[9390], color=ORANGE, width=.5)
    ax.text(0, 4700, "PBC\n9,390", ha="center", color="w", fontsize=9); ax.text(0, 16600, "Raabin\n14,514", ha="center", color=INK, fontsize=9)
    ax.text(0, 24300, "23,904 merged", ha="center", fontsize=9)
    ax.set_xlim(-.6, 3.1); ax.set_ylim(0, 27500); ax.set_xticks([0, 1.5, 2.5]); ax.set_xticklabels(["Training pool", "Raabin test\nacc (%)", "Merged test\nacc (%)"], fontsize=8)
    ax.bar([1.5], [84.89*250], color=BEFORE, width=.5); ax.bar([2.5], [91*250], color=AFTER, width=.5)
    ax.text(1.5, 84.89*250+300, "84.89", ha="center", fontsize=8); ax.text(2.5, 91*250+300, "91 (frozen)\n99 (final)", ha="center", fontsize=8)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.text(1.2, 26200, "left bar = image count;\nright bars = accuracy, own scale", fontsize=7, color=MUTED, ha="left")
story("s2_multi_source_merging.png", "Technique 2 - Merge PBC + Raabin into one training set  (technique2_multi_source_merging.py)",
 W("Trained on one source, the model only knows one microscope's look. It has never seen visual diversity, so any new lab breaks it.\n\nAlso: Raabin's test folders spell class names differently (Esinophils, Monocyte, Lymphocyte), so a naive merge would create fake extra classes."),
 "NAME_MAP = {\n 'Esinophils':'Eosinophils',\n 'Monocyte':'Monocytes',\n 'Lymphocyte':'Lymphocytes', ...}\n\nfor pattern in [PBC/*, Raabin/Train/*,\n                Raabin/TestA/*]:\n   label = NAME_MAP.get(folder, folder)\n   collect(path, label)\n\ntrain_test_split(paths, labels,\n   test_size=0.20, stratify=labels,\n   random_state=42)\n\n-> same class ratios in train and test",
 "sees both domains", d2,
 "84.89% -> 91% is the merged+stain-normalised+frozen row; 99% is the final model (also T4, T5). This split reuses test as validation - fixed in T5.")

# ---- T3 oversampling
def d3(ax):
    cl = ["Neutrophil", "Monocyte", "Basophil"]; n0 = [9050, 1772, 1215]
    x = np.arange(3); w = .38
    ax.bar(x-w/2, n0, w, color=BEFORE, label="Images before"); ax.bar(x+w/2, [9050, 3000, 3000], w, color=AFTER, label="After oversampling (target 3000)")
    for i,(u,v) in enumerate(zip(n0, [9050, 3000, 3000])):
        ax.text(i-w/2, u+150, f"{u:,}", ha="center", fontsize=7); ax.text(i+w/2, v+150, f"{v:,}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cl); ax.set_ylabel("training images (full data, as stated in script)")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5,-.1), ncol=2)
    ins = ax.inset_axes([.58, .42, .38, .42]); ins.spines[["top", "right"]].set_visible(False)
    ins.bar([0, 1], [df.loc["baseline_raabin_test","basophils_f1"], df.loc["wbc20","basophils_f1"]], color=[BEFORE, AFTER])
    ins.set_xticks([0, 1]); ins.set_xticklabels(["before", "final"], fontsize=7); ins.set_ylim(0, 1.15); ins.set_title("Basophil F1", fontsize=8)
    for i,v in enumerate([.15, .98]): ins.text(i, v+.03, f"{v:.2f}", ha="center", fontsize=7)
story("s3_oversampling_class_weights.png", "Technique 3 - Oversample rare classes + class weights  (technique3_oversampling_class_weights.py)",
 W("Neutrophils outnumber Basophils 7.4 to 1 (9,050 vs 1,215; Monocytes 1,772). With plain cross-entropy the cheapest way to lower loss is to guess Neutrophil, so Basophil F1 was only 0.15."),
 "MINORITY_AUGMENTOR = ImageDataGenerator(\n  rotation_range=30,\n  brightness_range=[0.6,1.4],  # stain intensity\n  channel_shift_range=30,      # colour shift\n  zoom 0.2, flips, shear 0.15)\n\nfor cls in ['Basophils','Monocytes']:\n   generate until count == 3000\n   store as float16 (RAM)\n\nclass_weight = compute_class_weight(\n    'balanced', ...)\nmodel.fit(..., class_weight=class_weights)\n\n-> more minority examples AND their\n   mistakes cost more",
 "Basophil F1 0.15 -> 0.98", d3,
 "Basophil F1 change is the whole journey (baseline -> wbc20), not T3 alone; the CSV never isolates oversampling. Doc-quoted weights (3.14 vs 0.42) differ from what sklearn gives for these counts.")

# ---- T4 3-zone fine tuning
seq = ["wbc_merged_frozen", "wbc_finetune100", "wbc_finetune110", "wbc14", "wbc17", "wbc20"]
lab = ["frozen\nbackbone", "fine_tune\n_at=100", "fine_tune\n_at=110", "wbc14\n(=90)", "wbc17\n3-zone", "wbc20\n3-zone+T5"]
def d4(ax):
    y = df.loc[seq, "overall_accuracy"].values*100; bf = df.loc[seq, "basophils_f1"].values
    cols = [BEFORE, BLUE, BLUE, BLUE, ORANGE, ORANGE]
    ax.bar(range(6), y, color=cols); ax.set_ylim(80, 101); ax.set_ylabel("overall accuracy (%)  [axis starts at 80]")
    for i,v in enumerate(y): ax.text(i, v+.3, f"{v:.0f}%", ha="center", fontsize=8)
    for i,v in enumerate(bf): ax.text(i, 81, f"Baso\nF1 {v:.2f}", ha="center", fontsize=7, color="w")
    ax.set_xticks(range(6)); ax.set_xticklabels(lab, fontsize=7); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.text(.02, .97, "grey = frozen  blue = fine_tune_at  orange = 3-zone", transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
story("s4_3zone_finetuning.png", "Technique 4 - 3-zone selective fine-tuning of MobileNetV2  (technique4_3zone_finetuning.py)",
 W("With the backbone frozen, the ImageNet features (cats, cars) are not WBC-specific; training stalled (README: ~20% train accuracy). fine_tune_at=N unfreezes everything after layer N, but the model only reads layers 26, 53 and 116, so layers after 116 get no gradient and the two earlier taps stay frozen."),
 "for layer in base.layers:\n    layer.trainable = False\n\nfor layer in base.layers[11:27]:   # -> block_3_expand\n    layer.trainable = True\nfor layer in base.layers[38:54]:   # -> block_6_expand\n    layer.trainable = True\nfor layer in base.layers[90:117]:  # -> block_13_expand\n    layer.trainable = True\n\n-> ~59 layers adapt, all 3 feature\n   scales learn WBC patterns\n   (see 07_3zone_layers.png)",
 "accuracy by strategy", d4,
 "Honest reading: 3-zone alone (wbc17, 95%) is NOT better than wbc14 (96%); the jump to 99% (wbc20) arrives together with the proper split (T5) and a different kernel/CAB set-up.")

# ---- T5 proper split
def d5(ax):
    labs = ["wbc17\n(3-zone,\nno proper split)", "wbc20\n(+ proper\nsplit)", "wbc20 on\nnew_dataset2\n(100 imgs)"]
    v = [df.loc["wbc17","overall_accuracy"]*100, df.loc["wbc20","overall_accuracy"]*100, df.loc["wbc20_on_new_dataset2","overall_accuracy"]*100]
    ax.bar(range(3), v, color=[ORANGE, BLUE, MAG], width=.55); ax.set_ylim(0, 112); ax.set_ylabel("accuracy (%)")
    for i,t in enumerate(v): ax.text(i, t+2, f"{t:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=8); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.text(1, 82, "val 97.95% (2,391)\ntest 99% (4,781)", ha="center", fontsize=8, color="w")
    ax.text(2, 30, "still weak on a\nnew domain", ha="center", fontsize=8, color="w")
story("s5_proper_split_bigaug.png", "Technique 5 - Proper 70/10/20 split + BigAug tf.data pipeline  (technique5_proper_split_bigaug_pipeline.py)",
 W("Earlier runs used validation_data=test_dataset, so the best checkpoint was chosen on the very data used to report accuracy (leakage). Also: float32 everything = 8.7 GB RAM -> GPU out-of-memory, and datagen.flow() mismatched the tf.data pipeline."),
 "# split BEFORE oversampling\ntrain_test_split(test_size=0.20)   # locked\ntrain_test_split(val_size=0.125)   # = 10%\n\nDataset.from_tensor_slices(x_float16)\n .shuffle(2000)\n .map(cast->float32, flip LR/UD,\n      brightness +-0.2, contrast .8-1.2)\n .batch(16).prefetch(AUTOTUNE)\n\nmodel.fit(validation_data=val_dataset,\n  callbacks=[ModelCheckpoint(val_acc),\n   ReduceLROnPlateau, EarlyStopping])\n\n-> test set touched once, at the end",
 "trustworthy accuracy", d5,
 "The 99% is honest for the merged PBC+Raabin distribution; on a third, unseen 100-image set the same model gets 61%, so domain shift is reduced, not solved.")

# ---- overview
stages = [("PBC-only\ntested on Raabin", "baseline_raabin_test", "The problem:\ndomain shift"),
          ("+ T1 stain norm\n+ T2 merge\n+ T3 oversample\n(frozen)", "wbc_merged_frozen", None),
          ("+ T4 simple\nfine_tune_at=110", "wbc_finetune110", None),
          ("wbc14\nbest simple\n(ft=90)", "wbc14", None),
          ("+ T4 3-zone\n(wbc17)", "wbc17", None),
          ("+ T5 proper\nsplit (wbc20)", "wbc20", "production"),
          ("wbc20 on\nnew 100-img set", "wbc20_on_new_dataset2", "remaining gap")]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={"height_ratios": [1, 1], "hspace": .5})
x = np.arange(len(stages)); acc = [df.loc[m, "overall_accuracy"]*100 for _, m, _ in stages]
ax.plot(x, acc, "-", color=MUTED, lw=2, zorder=1); ax.scatter(x, acc, s=90, color=[MAG if i in (0, 6) else BLUE for i in x], zorder=3, edgecolor="#fcfcfb", linewidth=2)
for i,(l,m,n) in enumerate(stages):
    ax.text(i, acc[i]+4, f"{acc[i]:.0f}%", ha="center", fontsize=10, fontweight="bold")
    if n: ax.text(i, acc[i]-11, n, ha="center", fontsize=8, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in stages], fontsize=8); ax.set_ylim(40, 110); ax.set_ylabel("overall accuracy (%)")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("From 85% on a new microscope to 99% on merged data - and 61% on a third source", loc="left", fontweight="bold")
mk = ["o", "s", "^", "D", "v"]; cl = [BLUE, ORANGE, AQUA, YEL, MAG]
for k,(c, nm) in enumerate(zip(CLS, NAMES)):
    y = [df.loc[m, c] for _, m, _ in stages]
    ax2.plot(x, y, marker=mk[k], color=cl[k], lw=1.8, ms=7, markeredgecolor="#fcfcfb", label=nm)
    ax2.text(x[-1]+.12, y[-1]+{"Basophil":.03,"Eosinophil":-.03}.get(nm,0), nm, va="center", fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels([s[0] for s in stages], fontsize=8); ax2.set_ylim(0, 1.05); ax2.set_ylabel("F1 score per class")
ax2.set_xlim(-.3, len(stages)+.4); ax2.yaxis.grid(True, color=GRID); ax2.set_axisbelow(True)
ax2.legend(frameon=False, ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(.5, -.42))
ax2.set_title("Per-class F1 at each stage: Eosinophil recovers first (colour fix), Basophil last (needs backbone tuning)", loc="left", fontweight="bold")
fig.savefig(f"{out}/s0_overview_steps_vs_results.png", bbox_inches="tight"); plt.close(fig)
print(sorted(f for f in os.listdir(out) if f.startswith("s")))
