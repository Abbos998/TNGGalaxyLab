#!/bin/bash
cd ~/paper_I_tng100
python -c "
import numpy as np
ids = [l.strip() for l in open('ids50f.txt') if l.strip()]
for k, ch in enumerate(np.array_split(ids, 2)): open(f'ids50h_{k}.txt','w').write('\n'.join(ch))"
for AP in "3.5 6.0" "5.0 8.0"; do
  T=$(echo $AP | tr ' ' '-')
  for k in 0 1; do
    python -u downsample_resolution_test.py --catalog batch_tng50/batch_output/catalog.csv \
      --ids-file ids50h_$k.txt --n-real 2 --ap-kpc $AP --out ap${T}_t50_$k.csv > ap${T}_t50_$k.log 2>&1 &
  done
  wait
  python -u downsample_resolution_test.py --basepath /home/tnguser/sims.TNG/TNG100-1/output \
    --box 75000 --f-ds 1.0 --catalog batch_tng50/batch_output_tng100/catalog.csv \
    --ids-file ids100f.txt --ap-kpc $AP --out ap${T}_t100.csv > ap${T}_t100.log 2>&1
done
echo ALL DONE
