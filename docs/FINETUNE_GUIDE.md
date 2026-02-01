# LightOnOCR-2 Magyar Fine-tuning Útmutató

## Áttekintés

A LightOnOCR-2 modell fine-tuning-ja magyar karakterek (különösen ő és ű) jobb felismeréséhez.

## Követelmények

- Vast.ai fiók + ~$5 kredit
- SSH kulcs a Mac-en
- ~20 perc időd

---

## Vast.ai Használata (Lépésről Lépésre)

### 1. Regisztráció és Feltöltés

1. Menj a https://vast.ai oldalra
2. Kattints **"Sign Up"** → regisztrálj email-lel vagy GitHub-bal
3. Bejelentkezés után kattints a nevedre (jobb felső sarok) → **"Billing"**
4. **"Add Credit"** → tölts fel **$5-10**-et (kártyával vagy crypto)

### 2. SSH Kulcs Beállítása

1. Kattints a nevedre → **"Account"**
2. Görgess le az **"SSH Keys"** részhez
3. Mac-en nyisd meg a Terminált és futtasd:
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```
   (Ha nincs kulcsod: `ssh-keygen -t rsa -b 4096` majd Enter mindenhol)
4. Másold be a teljes kimenetet a Vast.ai "SSH Keys" mezőbe
5. Kattints **"Add SSH Key"**

### 3. GPU Bérlése

1. Kattints **"Search"** (bal oldali menü)
2. Szűrők beállítása:
   - **GPU Type**: H100 (vagy A100 ha olcsóbbat akarsz)
   - **GPU RAM**: 80 GB+
   - **Disk Space**: 50 GB+
3. Válassz egy gépet az árlistából ($/hr oszlop)
   - H100: ~$1.5-2.5/óra
   - A100 80GB: ~$1-1.5/óra
4. Kattints **"RENT"** a választott gépnél
5. A felugró ablakban:
   - **Docker Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
   - **Disk Space to Allocate**: 50 GB
   - Kattints **"RENT"**

### 4. Csatlakozás SSH-val

1. Várd meg, amíg a gép elindul (1-2 perc) - zöld lesz
2. Kattints **"Instances"** (bal menü) → látod a futó gépedet
3. Kattints a **">_ Connect"** gombra
4. Másold ki az SSH parancsot, pl:
   ```
   ssh -p 12345 root@123.45.67.89
   ```
5. Mac Terminálban futtasd ezt a parancsot
6. Kérdésre válaszolj: `yes`

### 5. Training Futtatása

A GPU szerveren futtasd sorban:

```bash
# Projekt klónozása
git clone https://github.com/hkmt-sw/HKMT-OCR.git
cd HKMT-OCR/scripts/finetune_h100

# Training indítása
chmod +x run.sh
./run.sh
```

**Ha nincs git repo**, másold fel a fájlokat a Mac-edről:

```bash
# Mac-en (ÚJ terminál ablak, ne zárd be a másikat):
cd /Users/martont/WORK/HKMT-OCR
scp -P PORTSZÁM -r scripts/finetune_h100 root@IP_CÍM:~/
```
(A PORTSZÁM és IP_CÍM az SSH parancsoból)

```bash
# Majd vissza a GPU szerveren:
cd ~/finetune_h100
chmod +x run.sh
./run.sh
```

### 6. Várakozás (~15-20 perc)

A script kiírja a progress-t:
```
Installing uv...
Creating virtual environment...
Installing dependencies...
Downloading fonts...
  ✓ DejaVuSans
  ✓ LiberationSans
  ...
Generating 1500 images (1000x500)...
  100/1500
  200/1500
  ...
Loading model: lightonai/LightOnOCR-2-1B-base
Training: 1500 images, 5 epochs, batch=8
  Step 10, loss=2.xxx
  Step 20, loss=1.xxx
  ...
✓ Training complete!
✓ Saved to ./merged
```

### 7. Modell Letöltése

Amikor végzett, töltsd le a Mac-edre (ÚJ terminál ablak):

```bash
# Mac-en:
scp -P PORTSZÁM -r root@IP_CÍM:~/HKMT-OCR/scripts/finetune_h100/merged ~/Downloads/lighton-hun-merged
```

Vagy ha közvetlenül töltötted fel:
```bash
scp -P PORTSZÁM -r root@IP_CÍM:~/finetune_h100/merged ~/Downloads/lighton-hun-merged
```

### 8. ⚠️ Gép Leállítása (FONTOS!)

1. Vast.ai weboldalon kattints **"Instances"**
2. Kattints a **"DESTROY"** gombra (piros)
3. **Erősítsd meg a törlést**

**⚠️ Ha nem állítod le, folyamatosan számláz! ⚠️**

### 9. MLX Konverzió (Mac-en)

```bash
cd ~/Downloads
mlx_vlm convert --hf-path lighton-hun-merged --mlx-path lighton-hun-mlx -q --q-bits 4
```

### 10. Használat

```bash
cd /Users/martont/WORK/HKMT-OCR
uv run python run_ocr.py dokumentum.pdf --engine lighton --model ~/Downloads/lighton-hun-mlx
```

---

## Költség Összesítés

| Tétel | Költség |
|-------|---------|
| H100 ~20 perc | ~$0.50-0.80 |
| **Összesen** | **< $1** |

---

## Konfiguráció Testreszabása

A `train.py` fájlban módosíthatod:

```python
@dataclass
class Config:
    num_images: int = 1500      # Képek száma (több = jobb, de lassabb)
    img_width: int = 1000       # Kép méret
    img_height: int = 500
    lora_r: int = 32            # LoRA rank (nagyobb = több kapacitás)
    batch_size: int = 8         # H100-hoz 8, A100-hoz 4
    num_epochs: int = 5         # Epoch-ok száma
```

---

## Hibaelhárítás

### "Permission denied" SSH-nál
```bash
# Mac-en ellenőrizd a kulcsot:
ls -la ~/.ssh/id_rsa.pub
# Ha nincs, generálj:
ssh-keygen -t rsa -b 4096
# Majd add hozzá újra a Vast.ai-hoz
```

### CUDA Out of Memory
Módosítsd a `train.py`-ban:
```python
batch_size: int = 4  # csökkentsd
gradient_accumulation: int = 4  # növeld
```

### Font letöltési hiba
```bash
mkdir -p fonts && cd fonts
wget 'https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip'
unzip dejavu-fonts-ttf-2.37.zip
cp dejavu-fonts-ttf-2.37/ttf/*.ttf .
cd ..
python train.py
```

### uv nem található
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

---

## Elvárt Eredmény

Fine-tuning előtt:
```
erő → erö ❌
hűtő → hütö ❌
működik → müködik ❌
```

Fine-tuning után:
```
erő → erő ✓
hűtő → hűtő ✓
működik → működik ✓
```

---

## Gyors Referencia

```bash
# 1. Vast.ai-n: bérelj H100-at, másold ki az SSH parancsot

# 2. Mac Terminal #1 - csatlakozás:
ssh -p PORT root@IP

# 3. GPU szerveren - training:
git clone https://github.com/hkmt-sw/HKMT-OCR.git
cd HKMT-OCR/scripts/finetune_h100
chmod +x run.sh && ./run.sh

# 4. Mac Terminal #2 - letöltés (amikor kész):
scp -P PORT -r root@IP:~/HKMT-OCR/scripts/finetune_h100/merged ~/Downloads/lighton-hun-merged

# 5. Vast.ai-n: DESTROY a gépet!

# 6. Mac-en - konverzió:
mlx_vlm convert --hf-path ~/Downloads/lighton-hun-merged --mlx-path ~/Downloads/lighton-hun-mlx -q --q-bits 4

# 7. Használat:
uv run python run_ocr.py doc.pdf --engine lighton --model ~/Downloads/lighton-hun-mlx
```
