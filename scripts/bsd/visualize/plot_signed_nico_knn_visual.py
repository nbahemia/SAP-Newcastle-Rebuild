#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
SHORTLIST=ROOT/'data'/'processed'/'bsd'/'shortlist'
FOCUS=SHORTLIST/'final_recruitment_focus'
VIS_DIR=ROOT/'visualizations'
OUTPUT=VIS_DIR/'nico_knn_signed_nejasmic.png'
BACKUP_CSV=FOCUS/'top_15_nico_gonzalez_backup_targets.csv'
SIGNED='Darko Nejašmić'

def norm(v):
    s=unicodedata.normalize('NFKD',str(v)); s=''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()

def main():
    VIS_DIR.mkdir(parents=True,exist_ok=True)
    if not BACKUP_CSV.exists(): raise FileNotFoundError(f'Run find_final_cb_rw_targets.py first: {BACKUP_CSV}')
    df=pd.read_csv(BACKUP_CSV)
    req=['Player','Nico Similarity Index','Nico Backup Fit','Nico KNN Distance','Defensive Score','Aerial Score']
    missing=[c for c in req if c not in df.columns]
    if missing: raise ValueError(f'Missing columns: {missing}')
    sm=df.Player.astype(str).map(norm).eq(norm(SIGNED))
    if not sm.any(): raise ValueError(f'{SIGNED} not found in {BACKUP_CSV.name}')
    shown=pd.concat([df.head(10),df.loc[sm].head(1)]).drop_duplicates('Player').copy()
    shown=shown.sort_values('Nico Similarity Index',ascending=True).reset_index(drop=True)
    signed=shown.Player.map(norm).eq(norm(SIGNED))
    neutral=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    colors=['#D4AF37' if s else neutral for s in signed]
    fig,ax=plt.subplots(figsize=(13.5,8))
    bars=ax.barh(shown.Player,shown['Nico Similarity Index'],color=colors,alpha=.9)
    ax.set_xlim(0,105); ax.set_xlabel('Nico Similarity Index',fontsize=10)
    ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
    for b,(_,r),s in zip(bars,shown.iterrows(),signed):
        txt=f"{r['Nico Similarity Index']:.1f}"
        if s: txt+=f"  SIGNED · Backup Fit {r['Nico Backup Fit']:.1f}"
        ax.text(b.get_width()+1,b.get_y()+b.get_height()/2,txt,va='center',fontsize=9,fontweight='bold' if s else 'normal')
    for lab in ax.get_yticklabels():
        if norm(lab.get_text())==norm(SIGNED): lab.set_fontweight('bold')
    darko=shown.loc[signed].iloc[0]
    ax.text(.985,.08,
        f"SIGNED SOLUTION · {darko['Player']}\n"
        f"KNN distance {darko['Nico KNN Distance']:.2f} · DEF {darko['Defensive Score']:.1f} · AER {darko['Aerial Score']:.1f}",
        transform=ax.transAxes,ha='right',va='bottom',fontsize=9,fontweight='bold')
    ax.set_title('NICO GONZÁLEZ BACKUP SEARCH · KNN',loc='left',fontsize=18,fontweight='bold',pad=18)
    ax.text(0,1.01,'Similarity = standardized Euclidean distance across Transition Protection · Deep Progression · Possession Security · Deep Involvement',transform=ax.transAxes,fontsize=9)
    ax.text(.5,-.09,'Defensive and aerial requirements are applied after KNN similarity · GOLD = SIGNED BY NEWCASTLE',transform=ax.transAxes,ha='center',fontsize=9,fontweight='bold')
    plt.tight_layout(); fig.savefig(OUTPUT,dpi=240,bbox_inches='tight'); plt.close(fig)
    print(f'[OK] {OUTPUT}')

if __name__=='__main__': main()
