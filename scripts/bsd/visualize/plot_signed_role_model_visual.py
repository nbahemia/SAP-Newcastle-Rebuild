#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re, unicodedata
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SHORTLIST = ROOT / 'data' / 'processed' / 'bsd' / 'shortlist'
VIS_DIR = ROOT / 'visualizations'
OUTPUT = VIS_DIR / 'signed_players_role_model.png'

SIGNED_PLAYERS = {
    'Kenan Yildiz','Nikola Krstovic','Rayan Ait-Nouri','Lazar Samardzic',
    'Bart van Rooij','Filip Lissah','Leopold Querfeld','Ludovic Blas',
    'Danilov Kirill Igorevich','Darko Nejasmic'
}

ROLE_MODELS = [
    ('HIGH-LINE CB','center_back_jaissle_detailed.csv',
     ['Jaissle CB','jaissle_cb_score','Role Score','Young CB Recruitment Score','Young CB Score'],
     ['Leopold Querfeld','Danilov Kirill Igorevich']),
    ('LEFT FULL-BACK','left_wingback_recruitment_progressive.csv',
     ['Jaissle LB','Jaissle Fullback','Role Score','Recruitment Score','Progressive Wingback Score','Wingback Score','Final Score'],
     ['Rayan Ait-Nouri']),
    ('RIGHT FULL-BACK','right_wingback_recruitment_progressive.csv',
     ['Jaissle RB','Jaissle Fullback','Role Score','Recruitment Score','Progressive Wingback Score','Wingback Score','Final Score'],
     ['Bart van Rooij','Filip Lissah']),
    ('HOLDING CONTROLLER','holding_controller_jaissle_detailed.csv',
     ['Jaissle Controller','Role Score','Controller Score','Final Score'],
     ['Darko Nejasmic']),
    ('CREATIVE 10','creative_10_jaissle_detailed.csv',
     ['Jaissle 10','Creative 10','Role Score','Creative 10 Score','Final Score'],
     ['Lazar Samardzic']),
    ('WINGER','winger_jaissle_archetypes.csv',
     ['Winger Recruitment Score','Best Score','overall_winger_score','Hybrid','Role Score','Final Score'],
     ['Kenan Yildiz','Ludovic Blas']),
    ('FOCAL STRIKER','focal_striker_jaissle_detailed.csv',
     ['Jaissle Striker','Focal Striker','Role Score','Focal Striker Score','Final Score'],
     ['Nikola Krstovic']),
]

def norm(v):
    s=unicodedata.normalize('NFKD',str(v)); s=''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()

SIGNED_N={norm(x) for x in SIGNED_PLAYERS}

def col(df,names):
    lookup={norm(c):c for c in df.columns}
    for n in names:
        if norm(n) in lookup: return lookup[norm(n)]
    return None

def panel(title, filename, score_names, signed_names, top_n=7):
    p=SHORTLIST/filename
    if not p.exists(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    pc=col(df,['Player','player_name','Name']); sc=col(df,score_names)
    if pc is None: raise ValueError(f'{title}: no player column')
    if sc is None:
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and any(k in norm(c) for k in ['score','jaissle','fit'])]
        if not nums: raise ValueError(f'{title}: no score column; columns={list(df.columns)}')
        sc=nums[0]
    v=df[[pc,sc]].copy(); v.columns=['Player','Score']; v['Score']=pd.to_numeric(v['Score'],errors='coerce')
    v=v.dropna().drop_duplicates('Player')
    top=v.sort_values('Score',ascending=False).head(top_n)
    wanted={norm(x) for x in signed_names}
    signed=v[v.Player.map(norm).isin(wanted)]
    return pd.concat([top,signed]).drop_duplicates('Player').sort_values('Score'), sc

def main():
    VIS_DIR.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(4,2,figsize=(16,13)); axes=axes.flatten()
    neutral=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    for ax,cfg in zip(axes,ROLE_MODELS):
        df,score_col=panel(*cfg)
        signed=df.Player.map(norm).isin(SIGNED_N)
        bars=ax.barh(df.Player,df.Score,color=['#D4AF37' if s else neutral for s in signed],alpha=.9)
        ax.set_title(cfg[0],loc='left',fontsize=12,fontweight='bold'); ax.set_xlabel(score_col,fontsize=8)
        ax.tick_params(axis='both',labelsize=8); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
        xmax=max(100,float(df.Score.max())*1.15); ax.set_xlim(0,xmax)
        for b,(_,r),s in zip(bars,df.iterrows(),signed):
            ax.text(b.get_width()+xmax*.012,b.get_y()+b.get_height()/2,f"{r.Score:.1f}"+('  SIGNED' if s else ''),va='center',fontsize=7.5,fontweight='bold' if s else 'normal')
        for lab in ax.get_yticklabels():
            if norm(lab.get_text()) in SIGNED_N: lab.set_fontweight('bold')
    for ax in axes[len(ROLE_MODELS):]: ax.axis('off')
    fig.suptitle('JAISSLE ROLE MODEL · FINAL RECRUITMENT',fontsize=19,fontweight='bold',y=.995)
    fig.text(.5,.973,'Role-specific scores rank players against the tactical responsibilities of each position',ha='center',fontsize=10)
    fig.text(.5,.012,'GOLD = SIGNED BY NEWCASTLE',ha='center',fontsize=10,fontweight='bold')
    plt.tight_layout(rect=[.03,.035,.98,.95]); fig.savefig(OUTPUT,dpi=240,bbox_inches='tight'); plt.close(fig)
    print(f'[OK] {OUTPUT}')

if __name__=='__main__': main()
