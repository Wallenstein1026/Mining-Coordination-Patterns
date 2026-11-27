"""Build the final vector poster and 9-page main report plus disclosures/references.

Uses only public aggregate results. Does not rerun experiments or require private data.
"""
from pathlib import Path
import html, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Flowable, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
import pdfrw.toreportlab as pdfrw_rl

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/experiments'; V=ROOT/'results/final_review'; F=ROOT/'figures/final'; OUT=ROOT/'output/pdf'
INK='#142D45'; BLUE='#185C85'; TEAL='#157E75'; GOLD='#C59132'; GRAY='#607488'; LIGHT='#EDF3F4'; PALE='#F7F8F6'; RED='#B65B47'
TITLE='Mining Coordination Patterns in Emerging Social Media Events'
NAMES='Jiaxin DONG  |  Qixiang HUA  |  Xiaofeng LI  |  Chang XU'

def load(name):return json.loads((R/name).read_text(encoding='utf-8'))
def read(name):return pd.read_csv(R/name)
def color(value):return colors.HexColor(value)
def install_fonts():
    for name,weight in [('Body','normal'),('BodyBold','bold'),('BodyItalic','normal')]:
        style='italic' if name=='BodyItalic' else 'normal'
        p=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight=weight,style=style))
        pdfmetrics.registerFont(TTFont(name,p))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    # Preserve pdfrw PDF syntax as bytes with ReportLab 4.
    pdfrw_rl._makestr=lambda doc,obj:str(getattr(obj,'encoded',None) or obj).encode('latin-1')

def save_plot(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(F/(name+'.png'),dpi=190,bbox_inches='tight')
    plt.close(fig)

def figures():
    F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#BBC9D1',
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    a=read('cib_informed_ablation.csv').set_index('representation')
    keys=['Fused baseline','Fused + HMI','Fused + user-balanced style','CIB-informed w=0.25']
    a=a.loc[keys];labels=['Fused baseline','Add HMI','Add user-balanced style','Joint signals (w = 0.25)']
    fig,axes=plt.subplots(1,2,figsize=(9,3.55),sharey=True,layout='constrained')
    y=np.arange(4);cs=[BLUE,'#8DAAB8','#8DAAB8',TEAL]
    axes[0].barh(y,a.common_fused_silhouette,color=cs,height=.6)
    axes[0].set_yticks(y,labels,fontsize=11);axes[0].invert_yaxis()
    axes[0].set(xlim=(0,.185),xlabel='Common-space silhouette',title='Cluster separation')
    for i,v in enumerate(a.common_fused_silhouette):axes[0].text(v+.004,i,f'{v:.3f}',va='center',fontsize=11)
    axes[1].barh(y,a.subsample_ari_mean,xerr=a.subsample_ari_std,color=cs,height=.6,
                  error_kw={'ecolor':INK,'capsize':3,'lw':1})
    axes[1].set(xlim=(0,1.02),xlabel='Subsample ARI (mean +/- SD)',title='Conditional stability')
    axes[1].tick_params(axis='y',length=0)
    for ax in axes:ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    save_plot(fig,'01_coordination_comparison')

    c=read('contextual_motif_comparison.csv');motifs=['LLHH','LHHL','LLLH']
    fig,(left,right)=plt.subplots(1,2,figsize=(9,3.5),gridspec_kw={'width_ratios':[1,1.25]},layout='constrained')
    from matplotlib.colors import ListedColormap
    matrix=np.array([[int(x=='H') for x in motif] for motif in motifs])
    left.pcolormesh(np.arange(5)-.5,np.arange(4)-.5,matrix,cmap=ListedColormap([LIGHT,TEAL]),vmin=0,vmax=1,rasterized=False)
    left.set_xlim(-.5,3.5);left.set_ylim(2.5,-.5)
    for i in range(3):
        for j in range(4):left.text(j,i,motifs[i][j],ha='center',va='center',fontsize=19,color='white' if matrix[i,j] else INK)
    left.set_xticks(range(4),['00-06','06-12','12-18','18-24'],fontsize=10)
    left.set_yticks(range(3),motifs,fontsize=12);left.set(xlabel='Beijing local hour',title='Recurring six-hour motifs')
    for s in left.spines.values():s.set_visible(False)
    part=c[c.motif.eq('LHHL')].set_index('context').loc[['Low HMI','High HMI']]
    vals=part.topic_weighted_share.to_numpy()*100
    bars=right.barh([0,1],vals,color=[BLUE,TEAL],height=.5)
    right.set_yticks([0,1],['Low HMI\n319 topics','High HMI\n356 topics'],fontsize=11);right.invert_yaxis()
    right.set(xlim=(0,47),xlabel='Mean within-topic LHHL share (%)',title='Daytime concentration differs')
    for i,v in enumerate(vals):right.text(v+1,i,f'{v:.1f}%',va='center',fontweight='bold',fontsize=14)
    right.grid(axis='x',alpha=.16);right.set_axisbelow(True)
    save_plot(fig,'02_temporal_patterns')

    m=read('forecast_metrics.csv');m=m[m.split.eq('test')].set_index('model')
    repeat=pd.read_csv(V/'forecast_repeat_metrics.csv');repeat=repeat[repeat.split.eq('test')].set_index('model')
    keys=['Recent volume','Activity RF','Activity + HMI RF','Activity + HMI + patterns/clusters RF']
    labels=['Recent volume','Activity RF','Activity + HMI RF','Add patterns / clusters']
    fig,(left,right)=plt.subplots(1,2,figsize=(10,3.55),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
    cs=['#B4C4CB',BLUE,'#7BABA7',TEAL];y=np.arange(4)
    left.barh(y,m.loc[keys,'average_precision'],color=cs,height=.57,label='Primary Windows run')
    left.scatter(repeat.loc[keys,'average_precision'],y,color=INK,marker='D',s=21,zorder=3,label='Independent macOS repeat')
    left.set_yticks(y,labels,fontsize=11);left.invert_yaxis()
    left.set(xlim=(0,.29),xlabel='Average precision (AP)',title='Next-day ranking on 35 test days')
    for i,v in enumerate(m.loc[keys,'average_precision']):left.text(v+.006,i,f'{v:.3f}',va='center',fontsize=11)
    left.legend(loc='upper left',bbox_to_anchor=(0,-.18),fontsize=8,frameon=False,ncol=2)
    keys2=['Activity RF','Activity + HMI + patterns/clusters RF']
    right.bar([0,1],m.loc[keys2,'reviewed_positives'],color=[BLUE,TEAL],width=.55)
    right.scatter([0,1],repeat.loc[keys2,'reviewed_positives'],marker='D',color=INK,s=26,zorder=3)
    right.set_xticks([0,1],['Activity RF','Full RF'],fontsize=11)
    right.set(ylim=(0,102),ylabel='True hotspots in 175 reviewed topic-days',title='Same daily review budget: five topics')
    for i,k in enumerate(keys2):
        v=m.loc[k,'reviewed_positives'];right.text(i,v+5,f'{int(v)} / 175',ha='center',fontsize=13,fontweight='bold')
    for ax in [left,right]:ax.grid(axis='x' if ax==left else 'y',alpha=.15);ax.set_axisbelow(True)
    save_plot(fig,'03_forecast_utility')

    poster_figures()

def draw_pdf(c,path,x,y,w,h):
    pg=PdfReader(str(path)).pages[0];obj=pagexobj(pg)
    bx=list(map(float,obj.BBox));pw,ph=bx[2]-bx[0],bx[3]-bx[1];scale=min(w/pw,h/ph)
    c.saveState();c.translate(x+(w-pw*scale)/2,y+(h-ph*scale)/2);c.scale(scale,scale);c.translate(-bx[0],-bx[1]);c.doForm(makerl(c,obj));c.restoreState()

def poster_figures():
    F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#BBC9D1',
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    topics = read('topic_tsne_display.csv')
    c3 = topics[topics.cluster.eq(3)]
    other = topics[topics.cluster.ne(3)]
    high = c3[c3.upper_hmi_quartile]
    fig, ax = plt.subplots(figsize=(3.8, 3.0), layout='constrained')
    ax.scatter(other.tsne_1, other.tsne_2, s=9, color='#A7B8C8', alpha=.6, linewidths=0,
               label='Other topics')
    ax.scatter(c3.tsne_1, c3.tsne_2, s=14, color=TEAL, linewidths=0, label=f'C3 ({len(c3)} topics)')
    ax.scatter(high.tsne_1, high.tsne_2, s=26, facecolors='none', edgecolors=GOLD, linewidths=.95,
               label='C3, top HMI quartile')
    ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2')
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(labelsize=9, colors=GRAY)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc='upper left', ncol=2,
              bbox_to_anchor=(-.02, 1.14), columnspacing=1, handletextpad=.15)
    ax.invert_yaxis()
    save_plot(fig, '04_topic_tsne')
    author_figure()


def author_figure():
    a=read('author_influence_summary.csv').set_index(['cohort','top_authors_removed'])
    fig,axes=plt.subplots(2,1,figsize=(3.8,4.5),layout='constrained')
    for ax,col,scale,title,fmt in [
        (axes[0],'median_post_share',100,'Post concentration','percent'),
        (axes[1],'median_absolute_score_change',1,'Score sensitivity','score')]:
        for cohort,label,clr,offset in [('small_gap','Small gap', '#90A9BB',-.18),('large_gap','Large gap', TEAL,.18)]:
            vals=[a.loc[(cohort,k),col]*scale for k in [1,5]]
            bars=ax.barh(np.arange(2)+offset,vals,height=.30,color=clr,label=label)
            ax.bar_label(bars,labels=[f'{v:.1f}%' if fmt=='percent' else f'{v:.4f}' for v in vals],padding=3,fontsize=10)
        ax.set_yticks([0,1],['Top 1','Top 5'],fontsize=10)
        ax.invert_yaxis();ax.set_title(title,fontsize=12)
        ax.set_xlim(0,a[col].max()*scale*1.38)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.tick_params(axis='x',labelsize=9)
    axes[0].set_xlabel('Median post share (%)',fontsize=10)
    axes[1].set_xlabel('Median absolute change',fontsize=10)
    axes[1].legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.48,-.32),ncol=2)
    save_plot(fig,'05_author_influence')


def poster():
    """Three columns following the course's Template C hierarchy."""
    w,h=80*cm,100*cm
    c=canvas.Canvas(str(OUT/'Poster.pdf'),pagesize=(w,h),pageCompression=1)
    c.setTitle('Mining Coordination Patterns - Three Columns')
    c.setAuthor('Group 28');c.scale(w/800,h/1000)
    blue='#2D5DB5'; navy='#163653'; pale='#EDF3FA'; muted='#526678'; teal='#157E75'
    cw=230; xs=[32,285,538]
    def box(x,y,w,h,fill):
        c.setFillColor(color(fill));c.rect(x,1000-y-h,w,h,fill=1,stroke=0)
    def line(x,y,ex,ey,clr='#CBD8E6',weight=.7,dash=False):
        c.setStrokeColor(color(clr));c.setLineWidth(weight);c.setDash([3,3] if dash else [])
        c.line(x,1000-y,ex,1000-ey);c.setDash([])
    def text(x,y,t,size=10,font='Body',fill=navy):
        c.setFillColor(color(fill));c.setFont(font,size);c.drawString(x,1000-y,t)
    def para(x,y,t,size=9.2,lead=12.5,fill=navy,limit=None,width=cw):
        p=Paragraph(t,ParagraphStyle('p',fontName='Body',fontSize=size,leading=lead,textColor=color(fill)))
        _,h=p.wrap(width,1000);p.drawOn(c,x,1000-y-h)
        if limit is not None:assert y+h<=limit,(t[:60],y+h,limit)
        return y+h
    def chart(name,x,y,w,h):draw_pdf(c,F/(name+'.pdf'),x,1000-y-h,w,h)
    def heading(x,y,title,num=None):
        text(x,y,title,16,'BodyBold',blue)
        line(x,y+10,x+cw,y+10,'#BCD0E8',.8)
    def bullet(x,y,t,limit=None):
        text(x,y+9,'•',9,'BodyBold',blue)
        return para(x+11,y,t,9.0,12.5,limit=limit,width=cw-11)
    box(0,0,800,1000,'#FFFFFF');box(0,0,800,148,blue)
    text(32,25,'AIAA 3111  /  DATA MINING COURSE PROJECT',9,'BodyBold','#D4E7FA')
    text(32,61,'Mining Coordination Patterns',28,'BodyBold','#FFFFFF')
    text(32,94,'in Emerging Social Media Events',25,'Body','#FFFFFF')
    text(32,124,NAMES,9.5,'Body','#FFFFFF')
    box(697,24,68,76,'#FFFFFF')
    c.drawImage(str(ROOT/'assets/template_d_hkust_gz.png'),713,1000-91,width=36,height=57,preserveAspectRatio=True,mask='auto')
    text(700,116,'HKUST(GZ)',10,'BodyBold','#FFFFFF');text(698,133,'GROUP 28',10,'BodyBold','#D4E7FA')
    box(0,148,800,4,'#65B5DB')
    line(273,176,273,931,'#B7C6D7',.8,True);line(526,176,526,931,'#B7C6D7',.8,True)

    # Left column: question and data lead directly into the cluster discovery.
    x=xs[0];heading(x,190,'Introduction')
    para(x,211,'We study recurring patterns and unusual posting behaviors in emerging Weibo topics, '
         'and whether past patterns help rank tomorrow\'s hotspots.',9.6,13.2,limit=283)
    heading(x,309,'Data')
    para(x,330,'<b>Official Weibo-COV 2.0 [1]</b><br/>September-November 2020 (91 days).<br/>'
         'Fixed 1/32 post-ID hash sample.',9.1,12.4,limit=370)
    box(x,382,cw,60,pale)
    for yy,n,label in [(400,'254,624','posts'),(419,'199,851','users'),(438,'1,422','hashtag topics')]:
        text(x+10,yy,n,14,'BodyBold',blue)
        text(x+119,yy,label,8.5,fill=muted)
    para(x,452,'Topic inclusion: <b>at least 30 posts and 5 authors</b>. Topics can share posts.',8.7,11.6,muted,limit=479)
    heading(x,507,'Pattern Mining')
    para(x,528,'<b>Shared posts explain an isolated cluster.</b> Selected: fused KMeans, k = 7; '
         't-SNE [3] displays the topic groups.',9.1,12.2,limit=579)
    chart('04_topic_tsne',x-4,583,cw+8,183)
    para(x,770,'<b>Fig. 1.</b> C3: 155 topics; gold: 18 top-quartile HMI topics. Apriori [4] and FP-growth [5] '
         'agree on <b>54 itemsets at 0.3% support</b>.',7.9,10.5,muted,limit=813)
    box(x,821,cw,42,pale)
    text(x+9,840,'0.797 vs. 0.078',17,'BodyBold',blue)
    text(x+9,855,'Mean 57-D silhouette: C3 vs. remaining topics',7.4,fill=muted)
    para(x,875,'<b>35 hashtags share exactly 103 posts</b>; another 35 share 102 of them. '
         'These 103 posts come from <b>5 authors</b>, with a median of <b>135 tags per post</b>.',8.9,12,limit=940)

    # Middle column: methodology then one portrait author-influence figure.
    x=xs[1];heading(x,190,'Methods')
    para(x,211,'Fuse text and behavior into <b>57 features</b>; compare KMeans and Ward. '
         'MisBot [2] provides auxiliary text-style scores.',9.2,12.5,limit=264)
    text(x,285,'Human-Manipulation Index (HMI)',10,'BodyBold')
    box(x,296,cw,31,pale)
    para(x+8,302,'<b>HMI = [rank(D) + rank(U)<br/>+ rank(B) + rank(S)] / 4</b>',8.9,11.3,width=cw-16,limit=326)
    para(x,338,'Percentile ranks: D duplicate-text share; U author concentration (HHI); '
         'B peak-day share; S same-minute post share.',8.5,11.3,muted,limit=384)
    heading(x,409,'Anomaly Detection')
    para(x,430,'<b>Measure author influence.</b> Compare post-weighted and equal-author style scores. '
         'Remove the top 1 or 5 authors by post count.',9.0,12,limit=482)
    chart('05_author_influence',x-2,491,cw+4,281)
    para(x,780,'<b>Fig. 2.</b> Small gap: &lt;0.001 (n=1,031); large gap: top 10% (n=143). '
         'Medians exclude empty topics for score changes: top-5 n=900 / 143.',7.9,10.5,muted,limit=824)
    box(x,834,cw,41,pale)
    text(x+9,862,'91.6%',22,'BodyBold',teal)
    para(x+93,841,'<b>131/143</b> large-gap scores move closer to original equal-author scores after top-1 removal.',7.6,9.5,width=cw-102,limit=874)
    para(x,889,'Large-gap topics have higher top-5 post share: <b>26.7% vs. 10.9%</b>. '
         'This tests sensitivity of an aggregate score, not account authenticity.',8.5,11.2,limit=934)

    # Right column: prediction, conclusion, future directions, and citations.
    x=xs[2];heading(x,190,'Hot Topic Prediction')
    para(x,211,'<b>Next-day ranking with past-only inputs.</b> Train Sep 08-Oct 09; validate Oct 11-24; '
         'test Oct 26-Nov 29, 2020 (35 issue days).',8.9,12.2,limit=263)
    para(x,253,'Hotspot: ≥4 next-day posts, ≥3 authors and ≥2× smoothed share vs. prior 7 days.',7.5,10,muted,limit=276)
    box(x,279,cw,90,pale)
    text(x+10,299,'Activity RF → Full RF [6]',9.3,'BodyBold',blue)
    text(x+10,325,'0.148 → 0.237',18,'BodyBold',blue)
    text(x+10,340,'Average precision (AP)',8,fill=muted)
    text(x+10,359,'44 → 84 hits / 175 reviews',10.3,'BodyBold')
    para(x,381,'Full RF adds rolling HMI, motifs and <b>six train-fitted activity/coordination groups</b>; '
         'it excludes MisBot. The HMI block alone adds <b>0.0013 AP</b>. Candidate coverage: <b>30.7%</b>; '
         'overall recall: <b>3.38%</b> (84/2,485). Five reviews per day.',8.6,11.5,limit=461)
    heading(x,472,'Conclusion')
    y=493
    for t in ['Shared-post overlap reveals an interpretable cluster; highly active authors can shift topic-level style scores.',
              'Past temporal patterns help next-day ranking in this sample; HMI alone adds little predictive value.',
              'HMI, t-SNE and style scores support manual review. They do not confirm malicious coordination or <b>Coordinated Inauthentic Behavior (CIB)</b>.']:
        y=bullet(x,y,t,limit=672)+9
    heading(x,697,'Future Work')
    y=718
    for t in ['Validate review candidates with independent labels of coordination and inauthentic behavior.',
              'Test later periods and new events with a fresh chronological holdout.',
              'Expand candidate coverage; assess sampling, topic overlap and author-balanced aggregation.']:
        y=bullet(x,y,t,limit=860)+8
    heading(x,879,'References')
    para(x,898,'[1] Hu et al., Weibo-COV, NLP4COVID 2020.<br/>'
         '[2] Wan et al., How Do Social Bots Participate in Misinformation Spread?, EMNLP 2025.<br/>'
         '[3] van der Maaten &amp; Hinton, t-SNE, JMLR 2008.<br/>'
         '[4] Agrawal &amp; Srikant, VLDB 1994. [5] Han et al., SIGMOD 2000. [6] Breiman, Random Forests, ML 2001.',6.3,8,muted,limit=949)

    # A slim brand footer, as in Template C.
    box(0,958,800,42,blue)
    text(32,977,'HKUST(GZ)',15,'BodyBold','#FFFFFF')
    text(32,991,'The Hong Kong University of Science and Technology (Guangzhou)',7.5,'Body','#D4E7FA')
    text(563,977,'AIAA 3111 Data Mining · Fall 2026',8.2,'Body','#FFFFFF')
    text(563,991,'Course Project Poster · Group 28',8.2,'Body','#FFFFFF')
    c.showPage();c.save()


class VectorFigure(Flowable):
    def __init__(self,path,width,height):Flowable.__init__(self);self.path=path;self.width=width;self.height=height
    def draw(self):draw_pdf(self.canv,self.path,0,0,self.width,self.height)

def report():
    OUT.mkdir(parents=True,exist_ok=True)
    story=[];width=A4[0]-96
    styles={
        'body':ParagraphStyle('body',fontName='Body',fontSize=10.2,leading=13.6,textColor=color(INK),spaceAfter=8),
        'small':ParagraphStyle('small',fontName='Body',fontSize=8.8,leading=12,textColor=color(GRAY),spaceAfter=7),
        'h':ParagraphStyle('h',fontName='BodyBold',fontSize=17.5,leading=22,textColor=color(INK),spaceAfter=13),
        'sub':ParagraphStyle('sub',fontName='BodyBold',fontSize=12,leading=16,textColor=color(TEAL),spaceBefore=7,spaceAfter=6),
        'title':ParagraphStyle('title',fontName='BodyBold',fontSize=23,leading=29,textColor=color(INK),spaceAfter=13),
        'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.3,leading=11,textColor=color(INK)),
        'ref':ParagraphStyle('reference',fontName='Body',fontSize=9.1,leading=11.7,
            textColor=color(INK),leftIndent=22,firstLineIndent=-22,spaceAfter=3.5),
        'caption':ParagraphStyle('caption',fontName='Body',fontSize=8.8,leading=12,textColor=color(GRAY),spaceAfter=10),
    }
    def p(t,kind='body'):story.append(Paragraph(t,styles[kind]))
    def sub(t):p(t,'sub')
    def new(t):
        if story:story.append(PageBreak())
        p(t,'h')
    def table(headers,rows,colwidths=None):
        data=[[Paragraph(str(x),styles['cell']) for x in row] for row in [headers,*rows]]
        tab=Table(data,colWidths=colwidths or [width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),color(LIGHT)),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),.6,color('#ACBEC8')),
            ('LINEBELOW',(0,1),(-1,-1),.3,color('#E0E7EA'))]))
        story.extend([tab,Spacer(1,10)])
    def fig(name,height=210):story.append(VectorFigure(F/(name+'.pdf'),width,height))
    def caption(t):p(t,'caption')
    primary=read('forecast_metrics.csv');test=primary[primary.split.eq('test')].set_index('model')
    repeat=pd.read_csv(V/'forecast_repeat_metrics.csv');rt=repeat[repeat.split.eq('test')].set_index('model')
    audit=json.loads((V/'audit.json').read_text());source=json.loads((ROOT/'results/weibo_data_audit.json').read_text())
    # Nine deliberately paginated pages of main text.
    p('AIAA 3111 COURSE PROJECT REPORT | GROUP 28','small')
    p(TITLE,'title')
    p('CIB-Informed Hot-Topic Clustering and Pattern Mining','sub')
    p(NAMES)
    p('26 November 2026 | Official Weibo-COV 2.0 research subset','small')
    sub('Abstract')
    p('We study how emerging topics attract attention, how observable coordination varies across topics, and whether repeated temporal patterns help prioritize next-day review. A fixed post-ID hash sample of official Weibo-COV 2.0 contains 254,624 posts, 199,851 users and 91 days. We compare text, behavior and fused clustering, define an interpretable Human-Manipulation Index (HMI), and mine hashtag itemsets and six-hour activity motifs. Official MisBot data provide auxiliary text-style evidence for retrospective interpretation.')
    p('Fused KMeans selects seven clusters with common-space silhouette 0.156. Adding coordination signals changes the grouping but does not establish CIB validity. In a chronological test, a pattern-augmented random forest reaches AP 0.237 versus 0.148 for activity alone; 84 of 175 reviewed topic-days are hotspots in the primary run. An independent macOS replay finds 83, with similar AP. Candidate eligibility covers only 30.7% of all sample hotspots. The useful result is improved ranking within a restricted candidate set, not general event discovery or verified manipulation detection.')
    sub('1. Introduction')
    p('High activity can reflect public interest, repeated reporting, automation, or coordinated promotion. Volume alone cannot distinguish these mechanisms. Our proposal therefore joins two primary data-mining tasks: topic clustering and pattern mining. CIB (Coordinated Inauthentic Behavior) provides the analytical context, while a supervised MisBot benchmark supplies an auxiliary source-domain signal.')
    table(['Research question','Evidence used'],[
        ['How is attention organized?','Text/behavior representations; KMeans versus Ward; compactness and stability.'],
        ['Which coordination patterns recur?','HMI profiles, itemsets, temporal motifs, matched descriptive comparisons.'],
        ['Do mined patterns help a practical task?','Past-only next-day ranking against volume, growth and activity baselines.']], [175,width-175])
    p('Our contribution is a reproducible workflow that makes the boundary between coordination proxies, retrospective findings and predictive evidence explicit. No account is identified as an inauthentic actor.', 'small')

    new('2. Problem Formulation')
    sub('2.1 Inputs, units and outputs')
    p('Each post supplies an ID, user ID, timestamp and text. Hashtags define overlapping topic memberships; broad collection tags such as general COVID terms are excluded as topics. The descriptive corpus retains topics with at least 30 posts and five users over the observation period. Outputs are topic groups, coordination profiles, recurring itemsets/motifs and a ranked list for later sample-defined growth.')
    sub('2.2 A sample-relative hotspot')
    p('Let n(t,d) be topic t\'s post count on day d and N(d) the total number of sample posts. Growth is G(t,d) = [(n(t,d)+1)/(N(d)+1)] / [(sum of n(t,j)+1)/(sum of N(j)+1)], where both sums cover the preceding seven days. A hotspot requires G at least 2, at least four posts, at least three users, and at least 20 sample posts in the reference period. This defines sampled attention growth, not an official Weibo trend label.')
    sub('2.3 Human-Manipulation Index (HMI)')
    p('<b>HMI = [r(D) + r(U) + r(B) + r(S)] / 4.</b> Each r is the percentile rank of that component among eligible topics. Ranks of 0.80, 0.60, 0.90 and 0.70 give an HMI of 0.75. We round raw values to 12 decimal places and assign tied values their average rank.')
    table(['Component','Operational definition'],[
        ['D: duplicate fraction','1 minus distinct normalized texts / topic posts.'],
        ['U: user concentration','HHI = sum of squared user participation shares.'],
        ['B: burst concentration','Largest daily count / topic posts in the observation window.'],
        ['S: minute synchrony','Fraction of posts in clock-minute groups containing two or more posts.']], [135,width-135])
    p('The proposal\'s HMI name is retained, but the score is a relative index of observable behavior. It does not measure human intent or manipulation probability. CIB additionally requires evidence of coordination and inauthentic identities or sources; neither HMI nor bot labels provides this truth.')
    sub('2.4 Forecasting already observed topics')
    p('At day d end, a topic is eligible with at least five cumulative posts, three cumulative users and one post in the past seven days; the sample reference total must be at least 20. We predict the hotspot label on d+1. Newly emerging or otherwise ineligible topics are not ranked. Their positives remain in the full hotspot census so candidate coverage and end-to-end recall are reported.')
    p('Descriptive HMI uses the full period; forecast HMI uses only the past/current 28-day window. These quantities and their topic populations are deliberately kept separate.', 'small')

    new('3. Methodology')
    sub('3.1 Official Weibo-COV 2.0 intake')
    table(['Month','Original posts','Retained posts','Original bytes'],[
        ['2020-09','3,672,914','114,820','2,040,438,498'],['2020-10','2,362,340','73,677','1,276,271,699'],
        ['2020-11','2,114,673','66,127','1,250,403,608'],['Total','8,149,927','254,624','4,567,113,805']], [80,110,110,width-300])
    p('We use Weibo-COV [1] from September to November 2020. Before downstream analysis, SHA-256 of each stripped post ID selects remainder zero modulo 32 using the first eight digest bytes as a big-endian integer. The same ID is always selected consistently. A streaming CSV reader and SQLite global deduplication preserve quoted/multiline text and string IDs; output order is timestamp then ID. Timestamps are naive Beijing local time. Bad widths, dates, missing values, bucket selection and duplicates are separately counted.')
    p('All 254,624 selected records pass required-field checks, with no duplicates in this release. There are 199,851 users and posts on all 91 calendar days, with 1,219-11,376 retained posts per day. The normalized file is 150,213,935 bytes. Independent final review checked its SHA-256 and the reader\'s counts, dates, users and reference accounting. Original monthly files were audited by the data owner; they were not independently downloaded again during final integration.')
    sub('3.2 Provenance and available propagation evidence')
    p('Normalized SHA-256: <font size="8">99aa5cfb20806185327f19c787a0fe0378b747a8fba43892ca4a387f6a9ee4ea</font>. Input manifests, fixed selection and aggregate audits are included. The earlier four-field public pilot is excluded from the final experiments.')
    p('209,524 posts contain a root reference (82.29%). Only 604 references match a post in the sampled subset: 0.288% of nonempty references. The data owner found 27,494 matches when searching all three raw months (13.12% with the same denominator). Root existence does not establish intermediate parent edges. We report coverage, not cascade depth or a reconstructed propagation tree.')
    sub('3.3 Official MisBot input and transfer')
    p('The MisBot [2] active-account file and the earlier mirror are byte-identical: 68,846,699 bytes, SHA-256 01d8170eb494367694c3403bb33b702eca19c35469c56df529c3e025778b7e16. It contains 48,536 accounts, 242,680 posts, 18,132 bot and 30,404 genuine labels. The final review replays the baselines from cached bytes matching that audited digest; it does not claim a second official download.')
    p('Raw posts, account IDs, per-account predictions and model files remain local. The public release contains aggregate outputs and data/download instructions only.', 'small')

    new('3. Methodology (continued)')
    sub('3.4 Topic representation and clustering')
    p('A topic document concatenates its posts after stripping hashtag strings. Character 2-3 gram TF-IDF [3] uses min_df=3, max_df=0.9, at most 16,000 features and sublinear term frequency. Randomized truncated SVD [4] gives 16 normalized text dimensions. Behavior contains seven summary measures, a 24-bin hour histogram and a ten-bin normalized lifecycle histogram (41 dimensions), standardized then normalized. Text and behavior blocks receive equal weight in the fused representation.')
    p('We compare KMeans [5] (n_init=20) with Ward linkage [6] across k=3-8. KMeans seeds 28-32 expose optimization sensitivity. We select the seed-28 fused model with the highest silhouette among models with at least ten topics per cluster. The resulting baseline is seven-cluster KMeans.')
    p('<b>Common-space silhouette</b> [7] compares how close each topic is to its own cluster versus the nearest other cluster. We score every partition using distances in the same original text-and-behavior space. Higher values indicate clearer separation in that space.')
    p('<b>Conditional stability</b> uses 20 random subsamples of 80% of topics with fixed features. The adjusted Rand index (ARI) [8] compares each refit with its own full-data grouping restricted to the sampled topics. ARI adjusts for chance agreement and equals 1 for identical partitions. A separate six-repeat check also refits the features. We use scikit-learn 1.5.1 [9].')
    sub('3.5 CIB-informed auxiliary representations')
    p('Figure 1 defines the feature additions and their weights. The sensitivity analysis also checks joint weights of 0.50 and 1.00.')
    p('The MisBot text-style logistic regression uses 12 style measures and account labels. Source-domain evaluation averages post scores within each held-out account. For each COVID topic, we first average each user\'s post scores and then give each user equal weight. A post-weighted alternative and influential-author sensitivity are supplementary controls. COVID scores are uncalibrated transfer signals.')
    sub('3.6 Itemsets and temporal motifs')
    p('Each post is a hashtag transaction. All retained posts count toward support. We compare Apriori [10] and FP-growth [11] using mlxtend 0.23.1 [12] at 0.3%, 0.5% and 1.0% support. Rules require confidence at least 0.5 and lift at least 1.5. Shared itemset counts are algorithm agreement, not a count of independent validated events.')
    p('For topic-days with at least four posts, four six-hour bucket counts are coded H if above that day\'s median and L otherwise. We compare topic-weighted motif proportions across HMI quartiles and growth contexts. Same-topic comparisons reduce topic composition differences; timing overlap with HMI and retrospective hotspot selection still prevents causal interpretation.')
    sub('3.7 Past-only prediction design')
    p('Ten activity features summarize counts at 1/3/7/28 days, seven-day users, age, recency, active days, current share and current growth. Four rolling coordination components plus HMI form the next block. The full block adds four train-selected motif proportions, the count of eligible motif-days and six train-fitted activity/coordination cluster indicators. These six forecast groups differ from the seven retrospective text/behavior groups. The forecast never uses full-period topic screening, full-period HMI or a future-trained MisBot model.')

    new('4. Experiments')
    p('<b>4.1 Clustering and auxiliary signals.</b> The 1,422 eligible topics contain 211,370 memberships in 117,255 distinct posts. Fused KMeans has k=7, silhouette 0.1564 and cluster sizes 121-255. At fixed k=7, text-only KMeans has the highest mean subsample ARI (0.8864), compared with 0.7609 for fused KMeans.')
    fig('01_coordination_comparison',175)
    caption('Figure 1. The poster\'s “Topic discovery and coordination” panel uses these four vertical-axis labels. Each row groups the same 1,422 topics into seven clusters with KMeans. Left: separation in the original fused space. Right: agreement with each setup\'s full-data grouping (mean ARI and standard deviation across 20 subsamples).')
    table(['Poster axis label','Features used to fit KMeans','Silhouette','Mean ARI +/- SD'],[
        ['Fused baseline','Text and posting-behavior features.','0.1564','0.7609 +/- 0.1287'],
        ['Add HMI','Baseline plus standardized HMI weighted by 0.50.','0.1232','0.7375 +/- 0.0895'],
        ['Add user-balanced style','Baseline plus the standardized MisBot style score (weight 0.50). Each author counts equally within their topic.','0.1413','0.7126 +/- 0.0938'],
        ['Joint signals (w = 0.25)','Baseline plus both standardized signals, each scaled by 0.25 / sqrt(2).','0.1540','0.8271 +/- 0.1178']], [106,235,59,width-400])
    p('The baseline already includes the four raw HMI components. Adding their rank average gives them extra influence on distances: unrelated topics with similar bursts can appear closer. HMI alone lowers silhouette to 0.1232 and gives an ARI of 0.3953 against the baseline. The joint setup raises observed subsample stability while slightly lowering silhouette. End-to-end feature refits give the baseline a mean ARI of 0.751 and a minimum of 0.548.')
    sub('MisBot source-domain benchmark')
    table(['Model','Account test AUC','Account AP','F1'],[
        ['Text-style LR','0.6893','0.5657','0.5574'],['Metadata LR','0.6102','0.4624','0.5222'],['Metadata RF','0.6211','0.4900','0.5155']], [200,100,100,width-400])
    p('Account-stratified train/validation/test sizes are 29,121 / 9,707 / 9,708. No UID overlaps across splits. Thresholds are selected on validation balanced accuracy; all three models select 0.37. Exact/normalized text overlap exists across accounts and is reported; removing test posts matching normalized training text changes post-level AUC from 0.6388 to 0.6364. Account isolation is not text independence. Independent final replay reproduces the source-domain metrics and thresholds to numerical tolerance. No target-domain AUC is available.')

    new('4. Experiments: Repeated Patterns')
    fig('02_temporal_patterns',178)
    caption('Figure 2. Six-hour motif definitions and the topic-weighted LHHL share in low/high full-period HMI groups. The groups provide descriptive contexts, not organic/CIB labels.')
    sub('4.2 Agreement across pattern-mining methods')
    table(['Minimum support','Apriori itemsets','FP-growth itemsets'],[
        ['0.3%','54','54'],['0.5%','31','31'],['1.0%','14','14']], [170,160,width-330])
    p('At 0.3% support, five repeated timings give medians of 0.162 s for Apriori and 0.618 s for FP-growth in the submitted environment. This small, sparse problem favors Apriori under that implementation and hardware; it is not a claim about asymptotic superiority. Timing measurements are not expected to match across machines. Deduplicated-text, distinct-user and weekly rule sensitivity are supplied as supplementary tables.')
    sub('4.3 Context-specific rhythms')
    p('There are 5,762 eligible topic-day motif windows. LLHH is most common (1,433; 24.87%), followed by LHHL (1,185; 20.57%). LHHL concentrates activity in 06:00-18:00. Its mean within-topic share is 20.09% for 319 eligible low-HMI topics and 39.47% for 356 high-HMI topics. Raw window shares differ from these averages because every topic receives equal weight.')
    p('For 554 topics with both hotspot and other eligible days, the within-topic LLHH difference is +28.08 percentage points; the topic-resampling interval is [24.22, 32.00] points. The direction persists at six- and eight-post day thresholds. This is a retrospective association: both the hotspot definition and the grouping reuse activity information. It does not establish that the rhythm causes growth or that high-HMI topics are manipulated.')
    sub('4.4 Case interpretation')
    p('Three high/low HMI pairs matched approximately on topic size and active days show higher duplicate-text shares in the high group. Concentrated reporting and shared public-event language are compatible explanations. Six public-event cases are not independent CIB truth, and AI-assisted title-pair judgments are excluded as primary semantic-validation evidence. Root-reference coverage remains too sparse for reliable cascade reconstruction.')

    new('4. Experiments: Chronological Forecast Protocol')
    table(['Forecast issue date','Rows','Hotspots','Role'],[
        ['Sep 08-Oct 09','36,617','1,051','Training'],['Oct 11-Oct 24','18,947','309','Validation'],
        ['Oct 26-Nov 29','45,547','763','Test (35 days)']], [165,80,80,width-325])
    p('All dates are in 2020 and identify prediction issue dates. Outcomes are next-day events. Oct 10 and Oct 25 are boundary embargo dates, so earlier labels do not reach later issue-date windows. Fitted scalers, KMeans centers, motif choice and supervised weights use training rows; the model family is selected from activity-only validation AP. Feature sets and model parameters are fixed for the reported run. Source configuration matches the saved outputs, but an independently timestamped freeze from before the first test was not verified. We therefore describe this as a chronological retrospective evaluation, not preregistered prospective validation.')
    sub('4.5 Baselines and controlled ablations')
    p('Two ranking rules use current post volume and current growth. Logistic regression uses C=1, balanced class weights and max_iter=2000. Random forests [13] use 250 trees, min_samples_leaf=15, max_features=sqrt, balanced_subsample class weights and seed 28. Both families compare activity, activity + coordination/HMI, and the full temporal-pattern/cluster block. RF is selected by validation activity-only AP; its activity AP is 0.1722 versus LR 0.1312 in the primary run.')
    rows=[]
    for name in test.index:
        rr=test.loc[name];short=name.replace('Activity + HMI + patterns/clusters','Full').replace('Activity + HMI','Activity + HMI')
        rows.append([short,f'{rr.average_precision:.4f}',f'{100*rr.precision_top5_per_day:.1f}%',str(int(rr.reviewed_positives))])
    table(['Test comparator','AP','Daily top-5 precision','Hits / 175'],rows,[245,65,110,width-420])
    p('We prioritize precision-recall measures for the imbalanced task [14]. Average precision (AP) covers all eligible test topic-days. Daily top-five precision uses five reviews per day. Candidate recall uses the 763 eligible positive topic-days. End-to-end recall uses all 2,485 sample hotspot topic-days in the same outcome window, including topics absent from the candidate list. Candidate prevalence is 1.675%.')

    new('4. Experiments: Predictive Utility and Replication')
    fig('03_forecast_utility',175)
    caption('Figure 3. Primary test results (bars) and independent macOS repeat (diamonds), using identical input and configuration. The review budget is five topics each day for 35 days.')
    p('The full RF achieves primary AP 0.2369 versus activity-only RF 0.1476: an absolute gain of 0.0893. Primary daily top-five precision rises from 25.14% (44/175) to 48.00% (84/175). Paired bootstrap resampling of whole weeks [15] gives a 400-draw AP gain interval of [0.0604, 0.1204]; all five observed test-week AP differences are positive. With only five blocks, this interval is weak evidence for broad temporal or platform generalization, not a deployment guarantee.')
    p('HMI/coordination alone adds only 0.0013 AP and has negative weekly changes in two weeks. A post-test diagnostic finds motif proportions alone, added to activity/coordination, attain AP 0.2352; cluster indicators alone attain 0.1505. The observed gain is therefore mainly associated with past temporal motifs, not established manipulation detection. These supplementary comparisons reuse the test window and are explanatory, not a fresh model-selection result.')
    table(['Metric','Primary Windows','Independent macOS'],[
        ['Activity RF AP',f'{test.loc["Activity RF","average_precision"]:.4f}',f'{rt.loc["Activity RF","average_precision"]:.4f}'],
        ['Full RF AP',f'{test.loc["Activity + HMI + patterns/clusters RF","average_precision"]:.4f}',f'{rt.loc["Activity + HMI + patterns/clusters RF","average_precision"]:.4f}'],
        ['Full RF hits / 175','84','83'],['Full RF top-five precision','48.00%','47.43%'],['AP gain interval','[0.0604, 0.1204]','[0.0611, 0.1202]']], [240,130,width-370])
    p('Across all validation/test RF comparisons the largest AP drift is 0.00330 and the largest top-five hit-count difference is two. No model was retuned for the repeat. Small floating-point, tied-rank and tree differences across environments are disclosed; the review does not assert bitwise reproducibility. Non-timing descriptive and auxiliary comparisons agree within 1e-9. The normalized input, candidate support, outcomes and review denominators match.')
    p('<b>Coverage is the practical bottleneck.</b> Candidates contain 763/2,485 = 30.70% of all test hotspots. Primary end-to-end recall is only 84/2,485 = 3.38% at the five-per-day budget. Useful candidate ranking must not be described as detecting 48% of all emerging events. A relaxed candidate rule was explored separately and requires new-period confirmation before adoption.')

    new('5. Conclusion and Limitations')
    sub('5.1 What the project establishes')
    p('The project produces interpretable groups of sampled topics and discovers recurring attention rhythms. Joint coordination and style features provide a descriptive alternative grouping; they do not consistently improve common-space compactness. Temporal motif histories provide incremental ranking information beyond basic activity for already observed topics in this study. This conclusion survives an independent execution environment, with small exact-score differences recorded. The coordination/HMI block has little predictive gain.')
    p('The practical use is to prioritize human review of known topics with changing attention. Evidence is insufficient for real-time platform-wide trend prediction, identifying inauthentic actors, attributing causality, or replacing human investigation. Those distinctions are central to interpreting a CIB-informed workflow responsibly and scientifically.')
    sub('5.2 Limits and next steps')
    p('Sampling: the fixed 1/32 post-ID subset suppresses rare events and root endpoint matches. Official provenance does not make it representative of every Weibo discussion. Topic unit: hashtag-defined memberships omit untagged discussions and overlap across topics. Coordination validity: HMI is endogenous to timing/participation measures and lacks independent CIB labels. MisBot transfer: a modest source-domain account AUC and source text overlap do not validate COVID-domain account judgments.')
    p('Temporal evaluation: only five test weeks, no verified external pre-test timestamp, and supplementary test reuse limit confirmatory claims. Coverage: the cold-start candidate bottleneck excludes most positive topic-days. Propagation: root references do not reveal intermediate parents, and sparse matching prevents reliable cascade metrics. Replication: exact RF numbers vary slightly across environments; preserve the full environment as well as seed and hashes.')
    p('Next steps should use a new temporal holdout, broaden and audit candidate eligibility, validate against independently obtained hotspot/CIB labels, evaluate human review costs and false alarms, and obtain trustworthy parent relations before propagation-tree analysis. Any new model or candidate rule should be frozen before evaluating that new holdout.')
    sub('5.3 Independent final review')
    table(['Review item','What was actually verified'],[
        ['Inputs','Normalized Weibo SHA-256; decoded counts/dates/users; MisBot digest; private score-table handoff and aggregates.'],
        ['Execution','10 unit tests; 22 descriptive, 35 retrospective, 29 validation, 38 test and 20 handoff checks, independently rerun.'],
        ['Comparison','28 additional aggregate checks; source-domain baselines replayed; primary/repeat RF drift disclosed.'],
        ['Not claimed','A second download/hash of original monthly CSVs, independent CIB truth, pre-test preregistration or human semantic annotation.']], [120,width-120])
    p('The README documents data preparation, canonical commands and public-only PDF rebuilding. Data/download instructions and the declared manifest are provided; social-media records, account predictions and trusted-local models are excluded from GitHub and the submission ZIP. The report has nine main-text pages; disclosures and references follow on a separate page.', 'small')

    new('6. Disclosures')
    sub('6.1 Team contributions and AIGC')
    table(['Member','Contribution'],[
        ['Jiaxin DONG','Implemented the basic framework, tested it on a small public dataset, coordinated integration and final checking, and prepared the poster/report.'],
        ['Qixiang HUA','Obtained official Weibo-COV files; fixed the study subset; implemented and audited intake, hashes, coverage and root-reference fields.'],
        ['Xiaofeng LI','Verified official MisBot against the mirror, reproduced account baselines, and prepared post/topic auxiliary style scores and handoff evidence.'],
        ['Chang XU','Delivered clustering, HMI/pattern ablations, chronological prediction, sensitivity analyses, figures and execution evidence.']], [110,width-110])
    p('OpenAI Codex assisted with code, exploratory runs, audits, independent reruns, figures and drafting. It reviewed the integration under Jiaxin\'s direction. The team excluded AI title-pair judgments from primary validation. The course guide [16] requires AIGC disclosure and prohibits submitting directly AI-generated work.')
    p('References','h')
    references = [
        'Hu, Y., Huang, H., Chen, A., and Mao, X.-L. (2020). Weibo-COV: A large-scale COVID-19 social media dataset from Weibo. <i>Proceedings of the 1st Workshop on NLP for COVID-19 (Part 2) at EMNLP 2020.</i> ACL.',
        'Wan, H., Luo, M., Ma, Z., Dai, G., and Zhao, X. (2025). How do social bots participate in misinformation spread? A comprehensive dataset and analysis. <i>Proceedings of EMNLP 2025</i>, 31493-31516. ACL.',
        'Salton, G., and Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. <i>Information Processing &amp; Management</i>, 24(5), 513-523.',
        'Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. <i>SIAM Review</i>, 53(2), 217-288.',
        'Lloyd, S. (1982). Least squares quantization in PCM. <i>IEEE Transactions on Information Theory</i>, 28(2), 129-137.',
        'Ward, J. H., Jr. (1963). Hierarchical grouping to optimize an objective function. <i>Journal of the American Statistical Association</i>, 58(301), 236-244.',
        'Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. <i>Journal of Computational and Applied Mathematics</i>, 20, 53-65.',
        'Hubert, L., and Arabie, P. (1985). Comparing partitions. <i>Journal of Classification</i>, 2(1), 193-218.',
        'Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.',
        'Agrawal, R., and Srikant, R. (1994). Fast algorithms for mining association rules. <i>Proceedings of the 20th International Conference on Very Large Data Bases</i>, 487-499.',
        'Han, J., Pei, J., and Yin, Y. (2000). Mining frequent patterns without candidate generation. <i>Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data</i>, 1-12.',
        'Raschka, S. (2018). MLxtend: Providing machine learning and data science utilities and extensions to Python’s scientific computing stack. <i>Journal of Open Source Software</i>, 3(24), 638.',
        'Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5-32.',
        'Saito, T., and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. <i>PLOS ONE</i>, 10(3), e0118432.',
        'Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. <i>Annals of Statistics</i>, 17(3), 1217-1241.',
        'AIAA 3111 Teaching Team. (2026). <i>Course Project Guide and poster submission notice.</i> Course materials, HKUST (Guangzhou).',
    ]
    reference_urls = {
        1: 'https://aclanthology.org/2020.nlpcovid19-2.34/',
        2: 'https://aclanthology.org/2025.emnlp-main.1604/',
        9: 'https://jmlr.org/papers/v12/pedregosa11a.html',
        12: 'https://doi.org/10.21105/joss.00638',
    }
    for number, reference in enumerate(references, 1):
        if number in reference_urls:
            href = reference_urls[number]
            reference = f'<link href="{href}" color="{BLUE}">{reference}</link>'
        p(f'[{number}] {reference}', 'ref')

    def decorate(c,doc):
        c.saveState();c.setFont('Body',8);c.setFillColor(color(GRAY))
        c.drawString(48,24,'AIAA 3111 | Group 28 | Coordination patterns in emerging events')
        c.drawRightString(A4[0]-48,24,str(doc.page));c.setStrokeColor(color('#D6E0E5'));c.line(48,39,A4[0]-48,39)
        c.restoreState()
    doc=SimpleDocTemplate(str(OUT/'Report.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=52,
        title=TITLE,author='Group 28',pageCompression=1)
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    pages=len(PdfReader(str(OUT/'Report.pdf')).pages)
    assert pages==10,('Expected nine main pages plus disclosures/references',pages)

def main():
    OUT.mkdir(parents=True,exist_ok=True);install_fonts();figures();poster();report()
    for name in ['Poster.pdf','Report.pdf']:shutil.copy2(OUT/name,ROOT/name)
    print('Built vector 80 x 100 cm Poster.pdf and 10-page Report.pdf (9 main + 1 disclosures/references).')

if __name__=='__main__':main()
