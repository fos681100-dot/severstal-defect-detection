import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, add_config_args, config_from_args
from dataset import SteelDataset, build_dataframe
from model import build_model
from utils import set_seed

# Sinif renkleri (RGB)
COLORS = np.array([
    [255, 60, 60],    # Class 1 - kirmizi
    [60, 220, 60],    # Class 2 - yesil
    [60, 140, 255],   # Class 3 - mavi
    [255, 220, 40],   # Class 4 - sari
], dtype=np.float32)

CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4"]


def overlay_mask(img, mask, alpha=0.55):
    """Griye yakin fotografin uzerine renkli maske bindirir."""
    vis = img.astype(np.float32).copy()
    for c in range(mask.shape[0]):
        sel = mask[c] > 0
        if sel.any():
            vis[sel] = (1 - alpha) * vis[sel] + alpha * COLORS[c]
    return np.clip(vis, 0, 255).astype(np.uint8)


def dice_single(pred, true):
    """Tek goruntu icin sinif ortalamali Dice."""
    scores = []
    for c in range(pred.shape[0]):
        p, t = pred[c].sum(), true[c].sum()
        if p + t == 0:
            scores.append(1.0)          # ikisi de bos -> dogru
        else:
            inter = (pred[c] * true[c]).sum()
            scores.append(2.0 * inter / (p + t))
    return float(np.mean(scores)), scores


@torch.no_grad()
def predict_one(model, dataset, idx, device, threshold, min_size, tta=True):
    """Tek goruntu icin tahmin uretir. Donen: (gercek maske, tahmin maske)"""
    image, mask, _ = dataset[idx]
    x = image.unsqueeze(0).to(device)

    logits, _ = model(x)
    probs = torch.sigmoid(logits.float())
    if tta:
        for dims in ([3], [2]):
            lg, _ = model(torch.flip(x, dims=dims))
            probs = probs + torch.sigmoid(torch.flip(lg.float(), dims=dims))
        probs = probs / 3.0

    pred = (probs[0] > threshold).cpu().numpy().astype(np.float32)
    for c in range(pred.shape[0]):
        if pred[c].sum() < min_size:        # kucuk lekeleri temizle
            pred[c] = 0
    return mask.numpy(), pred


def choose_images(df, n, mode, seed=42):
    """Gosterilecek goruntuleri secer.

    mode='cesitli' : kasitli olarak farkli durumlar (kusursuz, yaygin sinif,
                     nadir sinif, coklu kusur) -> sunum icin en bilgilendirici
    mode='rastgele': tamamen rastgele
    """
    rng = np.random.default_rng(seed)
    if mode == "rastgele":
        return list(rng.choice(len(df), size=min(n, len(df)), replace=False))

    picks, used = [], set()

    def take(cond, etiket):
        nonlocal picks
        adaylar = df.index[cond].tolist()
        adaylar = [i for i in adaylar if i not in used]
        if adaylar:
            secim = int(rng.choice(adaylar))
            used.add(secim)
            picks.append((secim, etiket))

    take(df["n_defects"] == 0, "defect-free image")
    take(df["has3"] == 1, "Class 3 (most frequent class)")
    take(df["has2"] == 1, "Class 2 (rarest class)")
    take(df["n_defects"] > 1, "multiple defect types")
    take(df["has1"] == 1, "Class 1")
    take(df["has4"] == 1, "Class 4")

    # Yeterli olmadiysa rastgele tamamla
    while len(picks) < n:
        i = int(rng.integers(0, len(df)))
        if i not in used:
            used.add(i)
            picks.append((i, ""))
    return picks[:n]


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--n", type=int, default=5, help="Kac goruntu gosterilsin")
    parser.add_argument("--mode", default="cesitli", choices=["cesitli", "rastgele"])
    parser.add_argument("--tta", type=int, default=1)
    args = parser.parse_args()
    cfg = config_from_args(args)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.out_dir, cfg.run_name)

    # --- Modeli yukle ---
    ckpt = torch.load(os.path.join(out_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    saved = Config(**ckpt["cfg"])
    saved.data_dir, saved.out_dir, saved.run_name = cfg.data_dir, cfg.out_dir, cfg.run_name
    mcfg = saved

    model = build_model(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model yuklendi: epoch {ckpt['epoch']}, val dice {ckpt['val_dice']:.4f}")

    # --- Son islem ayarlarini oku (evaluate.py bunlari bulmustu) ---
    threshold, min_size = 0.5, 0
    mpath = os.path.join(out_dir, "metrics_test.json")
    if os.path.exists(mpath):
        import json
        m = json.load(open(mpath))
        threshold = m.get("threshold", 0.5)
        min_size = m.get("min_size", 0)
    print(f"Son islem ayarlari: threshold={threshold}, min_size={min_size}")

    # --- TEST setini yukle (model bu goruntuleri hic gormedi) ---
    df = build_dataframe(mcfg.data_dir, mcfg.num_classes)
    ids = set(pd.read_csv(os.path.join(out_dir, "split_test.csv"))["ImageId"])
    sub = df[df["ImageId"].isin(ids)].reset_index(drop=True)
    ds = SteelDataset(sub, mcfg.data_dir, mcfg, train=False)
    print(f"Test setinde {len(sub)} goruntu var, {args.n} tanesi gosterilecek.")

    secimler = choose_images(sub, args.n, args.mode, mcfg.seed)
    if secimler and not isinstance(secimler[0], tuple):
        secimler = [(i, "") for i in secimler]

    # --- Her goruntu icin uc panelli gorsel ---
    fig, axes = plt.subplots(len(secimler), 3, figsize=(19, 2.5 * len(secimler)))
    if len(secimler) == 1:
        axes = np.expand_dims(axes, 0)

    satirlar = []
    for satir, (idx, etiket) in enumerate(secimler):
        row = sub.iloc[idx]
        gercek, tahmin = predict_one(model, ds, idx, device, threshold, min_size,
                                     bool(args.tta))
        skor, sinif_skorlari = dice_single(tahmin, gercek)

        img = cv2.cvtColor(
            cv2.imread(os.path.join(ds.img_dir, row["ImageId"])), cv2.COLOR_BGR2RGB)

        axes[satir, 0].imshow(img)
        axes[satir, 1].imshow(overlay_mask(img, gercek))
        axes[satir, 2].imshow(overlay_mask(img, tahmin))

        for k in range(3):
            axes[satir, k].axis("off")

        if satir == 0:
            axes[satir, 0].set_title("Input image", fontsize=13, pad=10)
            axes[satir, 1].set_title("Ground truth (human annotation)", fontsize=13, pad=10)
            axes[satir, 2].set_title("Model prediction", fontsize=13, pad=10)

        gercek_siniflar = [CLASS_NAMES[c] for c in range(4) if gercek[c].sum() > 0] or ["no defect"]
        tahmin_siniflar = [CLASS_NAMES[c] for c in range(4) if tahmin[c].sum() > 0] or ["no defect"]

        sol_yazi = f"{row['ImageId']}"
        if etiket:
            sol_yazi += f"\n({etiket})"
        axes[satir, 0].set_xlabel(sol_yazi, fontsize=9)
        axes[satir, 0].axis("off")
        axes[satir, 0].text(0.01, -0.12, sol_yazi, transform=axes[satir, 0].transAxes,
                            fontsize=9, va="top")
        axes[satir, 1].text(0.01, -0.12, "Ground truth: " + ", ".join(gercek_siniflar),
                            transform=axes[satir, 1].transAxes, fontsize=9, va="top")

        renk = "#1a7f37" if skor >= 0.9 else ("#9a6700" if skor >= 0.7 else "#a40e26")
        axes[satir, 2].text(0.01, -0.12,
                            f"Prediction: {', '.join(tahmin_siniflar)}   |   Dice = {skor:.4f}",
                            transform=axes[satir, 2].transAxes, fontsize=9, va="top",
                            color=renk, fontweight="bold")

        satirlar.append({
            "ImageId": row["ImageId"],
            "durum": etiket,
            "gercek_siniflar": ", ".join(gercek_siniflar),
            "tahmin_siniflar": ", ".join(tahmin_siniflar),
            "dice": round(skor, 4),
            **{f"dice_{CLASS_NAMES[c]}": round(sinif_skorlari[c], 4) for c in range(4)},
        })
        print(f"  {row['ImageId']:<20} gercek={gercek_siniflar} "
              f"tahmin={tahmin_siniflar} dice={skor:.4f}")

    plt.tight_layout()
    png = os.path.join(out_dir, "demo_predictions.png")
    plt.savefig(png, dpi=130, bbox_inches="tight")
    plt.close()

    # --- Ozet tablo ---
    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(os.path.join(out_dir, "demo_table.csv"), index=False)

    md = ["## Demo: predictions on unseen test images", "",
          "| Image | Case | Ground truth | Prediction | Dice |",
          "|---|---|---|---|---|"]
    for r in satirlar:
        md.append(f"| {r['ImageId']} | {r['durum']} | {r['gercek_siniflar']} | "
                  f"{r['tahmin_siniflar']} | **{r['dice']:.4f}** |")
    md.append("")
    md.append(f"Ortalama Dice (bu {len(satirlar)} goruntu): "
              f"**{np.mean([r['dice'] for r in satirlar]):.4f}**")
    md.append("")
    md.append("> Bu goruntuler test setinden secildi; model egitim sirasinda hicbirini gormedi.")

    with open(os.path.join(out_dir, "demo_table.md"), "w") as f:
        f.write("\n".join(md))

    print("\n".join(md))
    print(f"\nGorsel: {png}")
    print(f"Tablo : {os.path.join(out_dir, 'demo_table.md')}")


if __name__ == "__main__":
    main()
