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
URL='https://github.com/Wallenstein1026/Mining-Coordination-Patterns'

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

def draw_pdf(c,path,x,y,w,h):
    pg=PdfReader(str(path)).pages[0];obj=pagexobj(pg)
    bx=list(map(float,obj.BBox));pw,ph=bx[2]-bx[0],bx[3]-bx[1];scale=min(w/pw,h/ph)
    c.saveState();c.translate(x+(w-pw*scale)/2,y+(h-ph*scale)/2);c.scale(scale,scale);c.translate(-bx[0],-bx[1]);c.doForm(makerl(c,obj));c.restoreState()

def poster():
    w,h=80*cm,100*cm;c=canvas.Canvas(str(OUT/'Poster.pdf'),pagesize=(w,h),pageCompression=1)
    c.setTitle('AIAA 3111 Group 28 Course Project Poster');c.setAuthor('Group 28')
    c.setFillColor(color(PALE));c.rect(0,0,w,h,fill=1,stroke=0)
    def text(x,y,t,size=25,bold=False,fg=INK):
        c.setFillColor(color(fg));c.setFont('BodyBold' if bold else 'Body',size);c.drawString(x,y,t)
    def para(x,y,width,t,size=25,leading=None,fg=INK):
        p=Paragraph(t,ParagraphStyle('p',fontName='Body',fontSize=size,leading=leading or size*1.35,textColor=color(fg)))
        _,height=p.wrap(width,h);p.drawOn(c,x,y-height);return y-height
    def panel(x,top,width,height,num,title):
        c.setFillColor(colors.white);c.roundRect(x,top-height,width,height,18,fill=1,stroke=0)
        text(x+28,top-48,f'{num}  {title}',31,True)
        c.setStrokeColor(color('#DBE5E8'));c.setLineWidth(1.5);c.line(x+28,top-66,x+width-28,top-66)
        return top-91
    margin=65;gap=35;cw=(w-2*margin-gap)/2;right=margin+cw+gap
    c.setFillColor(color(INK));c.rect(0,h-360,w,360,fill=1,stroke=0)
    text(margin,h-64,'AIAA 3111 COURSE PROJECT POSTER',27,True,'#D0B47C')
    c.setFillColor(color(TEAL));c.roundRect(w-margin-245,h-99,245,62,11,fill=1,stroke=0)
    text(w-margin-217,h-77,'GROUP 28',32,True,'#FFFFFF')
    text(margin,h-151,'Mining Coordination Patterns',65,True,'#FFFFFF')
    text(margin,h-227,'in Emerging Social Media Events',59,True,'#FFFFFF')
    text(margin,h-279,'CIB-informed clustering, interpretable HMI and temporal pattern mining',26,False,'#DCE8EC')
    text(margin,h-327,NAMES,25,False,'#FFFFFF')
    c.setFillColor(color(TEAL));c.roundRect(margin,h-484,w-2*margin,94,14,fill=1,stroke=0)
    text(margin+28,h-432,'Past temporal patterns improved next-day topic ranking.',38,True,'#FFFFFF')
    text(margin+28,h-467,'Evidence is limited to already observed topics in a sampled COVID-19 corpus.',25,False,'#E3F2EF')
    top=h-513
    y=panel(margin,top,cw,308,'01','Problem and data')
    y=para(margin+28,y,cw-56,'Which topics are growing, how is their attention organized, and do recurring patterns help prioritize follow-up?',27)
    text(margin+28,top-226,'254,624 posts   |   1,422 topics   |   91 days',29,True,TEAL)
    para(margin+28,top-248,cw-56,'Official Weibo-COV 2.0, Sep-Nov 2020. A fixed 1/32 post-ID hash sample of 8.15M posts.',23,31)
    y=panel(right,top,cw,308,'02','CIB and the Human-Manipulation Index')
    y=para(right+28,y,cw-56,'<b>HMI = [rank(D) + rank(U) + rank(B) + rank(S)] / 4</b>',27)
    y=para(right+28,y-14,cw-56,'D: repeated text &nbsp;&nbsp; U: user concentration (HHI)<br/>B: peak-day share &nbsp;&nbsp; S: same-minute posting',25,35)
    para(right+28,y-16,cw-56,'Coordinated Inauthentic Behavior (CIB) involves coordination plus inauthentic identities or sources. HMI and MisBot style are proxies, not proof.',21,28,GRAY)
    top2=top-340
    y=panel(margin,top2,cw,697,'03','Topic discovery and coordination')
    y=para(margin+28,y,cw-56,'Text + behavior features give <b>seven groups</b>. Low-weight joint signals preserve separation while changing the stability profile.',25,34)
    draw_pdf(c,F/'01_coordination_comparison.pdf',margin+20,top2-505,cw-40,min(335,y-18-(top2-505)))
    y=para(margin+28,top2-520,cw-56,'Fused silhouette: <b>0.156</b>. Joint signals (w=0.25): <b>0.154</b>; mean conditional ARI: <b>0.761 to 0.827</b>. This is a descriptive comparison, not proof of CIB.',24,33)
    para(margin+28,y-14,cw-56,'MisBot: 48,536 accounts; text-style AUC 0.689. Source-domain account labels are not COVID-domain CIB truth.',21,29,GRAY)
    y=panel(right,top2,cw,697,'04','Recurring patterns in attention')
    y=para(right+28,y,cw-56,'<b>54 frequent itemsets</b> agreed across Apriori and FP-growth at 0.3% support. Six-hour motifs describe repeated daily rhythms.',25,34)
    draw_pdf(c,F/'02_temporal_patterns.pdf',right+20,top2-505,cw-40,min(335,y-18-(top2-505)))
    y=para(right+28,top2-520,cw-56,'LHHL is daytime concentration. Its average within-topic share is <b>39.5% in high-HMI</b> versus <b>20.1% in low-HMI</b> topics.',24,33)
    para(right+28,y-14,cw-56,'H/L means above / at-or-below the day\'s median bucket count. Full-period grouping and shared timing features make this descriptive, not causal.',21,29,GRAY)
    top3=top2-731
    y=panel(margin,top3,w-2*margin,820,'05','Does the mined structure help? A chronological prediction test')
    para(margin+28,y,w-2*margin-56,'At day d end, rank previously observed topics for day d+1. Train to Oct 09; validate Oct 11-24; test Oct 26-Nov 29. Boundary days are separated; features use only past observations.',25,34)
    draw_pdf(c,F/'03_forecast_utility.pdf',margin+50,top3-655,w-2*margin-100,465)
    text(margin+30,top3-694,'AP gain: +0.089; 95% weekly-block interval [0.060, 0.120].',30,True,TEAL)
    para(margin+30,top3-717,w-2*margin-60,'Primary run: 84/175 reviews find a hotspot (48.0%); independent repeat: 83/175 (47.4%). The coordination/HMI block adds 0.0013 AP. Candidates contain 763 of 2,485 test hotspots (30.7%); primary end-to-end recall is only 3.38%.',25,34)
    bottomtop=top3-854
    c.setFillColor(color(INK));c.roundRect(margin,75,w-2*margin,bottomtop-75,14,fill=1,stroke=0)
    text(margin+29,bottomtop-47,'CONCLUSION',29,True,'#D0B47C')
    y=para(margin+29,bottomtop-66,w-2*margin-58,'Temporal patterns support <b>targeted review of known, growing topics</b>. The gain is not evidence that HMI detects manipulation or that the system forecasts all emerging events.',27,37,'#FFFFFF')
    y=para(margin+29,y-14,w-2*margin-58,'Limits: one 1/32 research subset; five test-week blocks; no verified pre-test external timestamp; supplementary test reuse; sparse root matches (0.29%); no CIB truth or full propagation trees.',22,30,'#D8E4E9')
    y=para(margin+29,y-15,w-2*margin-58,'Reproducible code, source audits, cross-platform repeat and full report:',20,27,'#D8E4E9')
    text(margin+29,y-26,URL,20,False,'#FFFFFF')
    assert y-26>=94,('Poster footer overflow',y)
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

    new('3. Methodology and Data Preparation')
    sub('3.1 Official Weibo-COV 2.0 intake')
    table(['Month','Original posts','Retained posts','Original bytes'],[
        ['2020-09','3,672,914','114,820','2,040,438,498'],['2020-10','2,362,340','73,677','1,276,271,699'],
        ['2020-11','2,114,673','66,127','1,250,403,608'],['Total','8,149,927','254,624','4,567,113,805']], [80,110,110,width-300])
    p('The declared study window is September-November 2020. Before downstream analysis, SHA-256 of each stripped post ID selects remainder zero modulo 32 using the first eight digest bytes as a big-endian integer. The same ID is always selected consistently. A streaming CSV reader and SQLite global deduplication preserve quoted/multiline text and string IDs; output order is timestamp then ID. Timestamps are naive Beijing local time. Bad widths, dates, missing values, bucket selection and duplicates are separately counted.')
    p('All 254,624 selected records pass required-field checks, with no duplicates in this release. There are 199,851 users and posts on all 91 calendar days, with 1,219-11,376 retained posts per day. The normalized file is 150,213,935 bytes. Independent final review checked its SHA-256 and the reader\'s counts, dates, users and reference accounting. Original monthly files were audited by the data owner; they were not independently downloaded again during final integration.')
    sub('3.2 Provenance and available propagation evidence')
    p('Normalized SHA-256: <font size="8">99aa5cfb20806185327f19c787a0fe0378b747a8fba43892ca4a387f6a9ee4ea</font>. Input manifests, fixed selection and aggregate audits are included. The earlier four-field public pilot is excluded from the final experiments.')
    p('209,524 posts contain a root reference (82.29%). Only 604 references match a post in the sampled subset: 0.288% of nonempty references. The data owner found 27,494 matches when searching all three raw months (13.12% with the same denominator). Root existence does not establish intermediate parent edges. We report coverage, not cascade depth or a reconstructed propagation tree.')
    sub('3.3 Official MisBot input and transfer')
    p('The official active-account file and the earlier mirror are byte-identical: 68,846,699 bytes, SHA-256 01d8170eb494367694c3403bb33b702eca19c35469c56df529c3e025778b7e16. It contains 48,536 accounts, 242,680 posts, 18,132 bot and 30,404 genuine labels. The final review replays the baselines from cached bytes matching that audited digest; it does not claim a second official download.')
    p('Raw posts, account IDs, per-account predictions and model files remain local. The public release contains aggregate outputs and data/download instructions only.', 'small')

    new('3. Methodology (continued)')
    sub('3.4 Topic representation and clustering')
    p('A topic document concatenates its posts after stripping hashtag strings. Character 2-3 gram TF-IDF uses min_df=3, max_df=0.9, at most 16,000 features and sublinear term frequency. Truncated SVD gives 16 normalized text dimensions. Behavior contains seven summary measures, a 24-bin hour histogram and a ten-bin normalized lifecycle histogram (41 dimensions), standardized then normalized. Text and behavior blocks receive equal weight in the fused representation.')
    p('KMeans (n_init=20) and Ward linkage are compared across k=3-8. KMeans seeds 28-32 expose optimization sensitivity. We select the seed-28 fused model with the highest silhouette among models with at least ten topics per cluster. The resulting baseline is seven-cluster KMeans.')
    p('<b>Common-space silhouette</b> compares how close each topic is to its own cluster versus the nearest other cluster. We score every partition using distances in the same original text-and-behavior space. Higher values indicate clearer separation in that space.')
    p('<b>Conditional stability</b> uses 20 random subsamples of 80% of topics with fixed features. The adjusted Rand index (ARI) compares each refit with its own full-data grouping restricted to the sampled topics. ARI adjusts for chance agreement and equals 1 for identical partitions. A separate six-repeat check also refits the features.')
    sub('3.5 CIB-informed auxiliary representations')
    p('Figure 1 defines the feature additions and their weights. The sensitivity analysis also checks joint weights of 0.50 and 1.00.')
    p('The MisBot text-style logistic regression uses 12 style measures and account labels. Source-domain evaluation averages post scores within each held-out account. For each COVID topic, we first average each user\'s post scores and then give each user equal weight. A post-weighted alternative and influential-author sensitivity are supplementary controls. COVID scores are uncalibrated transfer signals.')
    sub('3.6 Itemsets and temporal motifs')
    p('Each post is a hashtag transaction; all retained posts remain in the support denominator. Apriori and FP-growth are compared at support 0.3%, 0.5% and 1.0%. Rules require confidence at least 0.5 and lift at least 1.5. Shared itemset counts are algorithm agreement, not a count of independent validated events.')
    p('For topic-days with at least four posts, four six-hour bucket counts are coded H if above that day\'s median and L otherwise. We compare topic-weighted motif proportions across HMI quartiles and growth contexts. Same-topic comparisons reduce topic composition differences; timing overlap with HMI and retrospective hotspot selection still prevents causal interpretation.')
    sub('3.7 Past-only prediction design')
    p('Ten activity features summarize counts at 1/3/7/28 days, seven-day users, age, recency, active days, current share and current growth. Four rolling coordination components plus HMI form the next block. The full block adds four train-selected motif proportions, the count of eligible motif-days and six train-fitted activity/coordination cluster indicators. These six forecast groups differ from the seven retrospective text/behavior groups. The forecast never uses full-period topic screening, full-period HMI or a future-trained MisBot model.')

    new('4. Experiments: Clustering and Auxiliary Signals')
    p('The 1,422 eligible topics contain 211,370 memberships in 117,255 distinct posts. The chosen fused KMeans has k=7, silhouette 0.1564 and cluster sizes 121-255. Text-only KMeans has the highest conditional mean ARI (0.8864) at fixed k=7; fused KMeans has 0.7609. Thus compactness in the fused space and perturbation stability favor different representations.')
    fig('01_coordination_comparison',175)
    caption('Figure 1. The poster\'s “Topic discovery and coordination” panel uses these four vertical-axis labels. Each row groups the same 1,422 topics into seven clusters with KMeans. Left: separation in the original fused space. Right: agreement with each setup\'s full-data grouping (mean ARI and standard deviation across 20 subsamples).')
    table(['Poster axis label','Features used to fit KMeans','Silhouette','Mean ARI +/- SD'],[
        ['Fused baseline','Text and posting-behavior features.','0.1564','0.7609 +/- 0.1287'],
        ['Add HMI','Baseline plus standardized HMI weighted by 0.50.','0.1232','0.7375 +/- 0.0895'],
        ['Add user-balanced style','Baseline plus the standardized MisBot style score (weight 0.50). Each author counts equally within their topic.','0.1413','0.7126 +/- 0.0938'],
        ['Joint signals (w = 0.25)','Baseline plus both standardized signals, each scaled by 0.25 / sqrt(2).','0.1540','0.8271 +/- 0.1178']], [106,235,59,width-400])
    p('The baseline already includes the four raw HMI components. Adding their rank average gives them extra influence on distances: unrelated topics with similar bursts can appear closer. HMI alone lowers silhouette to 0.1232 and gives an ARI of 0.3953 against the baseline. The joint setup raises observed subsample stability while slightly lowering silhouette. End-to-end feature refits give the baseline a mean ARI of 0.751 and a minimum of 0.548.')
    sub('4.1 MisBot source-domain benchmark')
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
    p('Two ranking rules use current post volume and current growth. Logistic regression uses C=1, balanced class weights and max_iter=2000. Random forests use 250 trees, min_samples_leaf=15, max_features=sqrt, balanced_subsample class weights and seed 28. Both families compare activity, activity + coordination/HMI, and the full temporal-pattern/cluster block. RF is selected by validation activity-only AP; its activity AP is 0.1722 versus LR 0.1312 in the primary run.')
    rows=[]
    for name in test.index:
        rr=test.loc[name];short=name.replace('Activity + HMI + patterns/clusters','Full').replace('Activity + HMI','Activity + HMI')
        rows.append([short,f'{rr.average_precision:.4f}',f'{100*rr.precision_top5_per_day:.1f}%',str(int(rr.reviewed_positives))])
    table(['Test comparator','AP','Daily top-5 precision','Hits / 175'],rows,[245,65,110,width-420])
    p('AP is computed over all eligible test topic-days; daily top-five precision uses exactly five reviews per day. Candidate recall uses the 763 eligible positive topic-days. End-to-end recall uses all 2,485 sample hotspot topic-days in the same outcome window, including topics absent from the candidate list. ROC AUC is supplementary because the candidate prevalence is only 1.675%.')

    new('4. Experiments: Predictive Utility and Replication')
    fig('03_forecast_utility',175)
    caption('Figure 3. Primary test results (bars) and independent macOS repeat (diamonds), using identical input and configuration. The review budget is five topics each day for 35 days.')
    p('The full RF achieves primary AP 0.2369 versus activity-only RF 0.1476: an absolute gain of 0.0893. Primary daily top-five precision rises from 25.14% (44/175) to 48.00% (84/175). The paired 400-draw weekly-block bootstrap interval for the AP gain is [0.0604, 0.1204]; all five observed test-week AP differences are positive. With only five blocks, this interval is weak evidence for broad temporal or platform generalization, not a deployment guarantee.')
    p('HMI/coordination alone adds only 0.0013 AP and has negative weekly changes in two weeks. A post-test diagnostic finds motif proportions alone, added to activity/coordination, attain AP 0.2352; cluster indicators alone attain 0.1505. The observed gain is therefore mainly associated with past temporal motifs, not established manipulation detection. These supplementary comparisons reuse the test window and are explanatory, not a fresh model-selection result.')
    table(['Metric','Primary Windows','Independent macOS'],[
        ['Activity RF AP',f'{test.loc["Activity RF","average_precision"]:.4f}',f'{rt.loc["Activity RF","average_precision"]:.4f}'],
        ['Full RF AP',f'{test.loc["Activity + HMI + patterns/clusters RF","average_precision"]:.4f}',f'{rt.loc["Activity + HMI + patterns/clusters RF","average_precision"]:.4f}'],
        ['Full RF hits / 175','84','83'],['Full RF top-five precision','48.00%','47.43%'],['AP gain interval','[0.0604, 0.1204]','[0.0611, 0.1202]']], [240,130,width-370])
    p('Across all validation/test RF comparisons the largest AP drift is 0.00330 and the largest top-five hit-count difference is two. No model was retuned for the repeat. Small floating-point, tied-rank and tree differences across environments are disclosed; the review does not assert bitwise reproducibility. Non-timing descriptive and auxiliary comparisons agree within 1e-9. The normalized input, candidate support, outcomes and review denominators match.')
    p('<b>Coverage is the practical bottleneck.</b> Candidates contain 763/2,485 = 30.70% of all test hotspots. Primary end-to-end recall is only 84/2,485 = 3.38% at the five-per-day budget. Useful candidate ranking must not be described as detecting 48% of all emerging events. A relaxed candidate rule was explored separately and requires new-period confirmation before adoption.')

    new('5. Conclusion, Limitations and Reproducibility')
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

    new('6. Disclosures and References')
    sub('6.1 Team contributions and AIGC')
    table(['Member','Contribution'],[
        ['Jiaxin DONG','Implemented the basic framework, tested it on a small public dataset, coordinated integration and final checking, and prepared the poster/report.'],
        ['Qixiang HUA','Obtained official Weibo-COV files; fixed the study subset; implemented and audited intake, hashes, coverage and root-reference fields.'],
        ['Xiaofeng LI','Verified official MisBot against the mirror, reproduced account baselines, and prepared post/topic auxiliary style scores and handoff evidence.'],
        ['Chang XU','Delivered clustering, HMI/pattern ablations, chronological prediction, sensitivity analyses, figures and execution evidence.']], [110,width-110])
    p('These roles follow the team\'s stated work and repository handoffs. OpenAI Codex assisted the framework, code, exploratory execution, audits, independent final reruns, statistical/visual summaries and English poster/report drafting. The final integration review recorded here was executed by Codex under Jiaxin\'s direction; it does not assert an unrecorded independent human review. AI title-pair judgments are not used as primary scientific validation. No external human collaborator beyond the listed team is asserted. The course requires members to inspect, understand, verify and revise AI-assisted material before Canvas submission; GitHub publication does not substitute for that responsibility.')
    sub('6.2 References')
    p('[1] Hu, Y.; Huang, H.; Chen, A.; Mao, X.-L. (2020). <i>Weibo-COV: A Large-Scale COVID-19 Social Media Dataset from Weibo.</i> NLP4COVID, EMNLP 2020. DOI: 10.18653/v1/2020.nlpcovid19-2.34. Official data documentation: https://github.com/nghuyong/weibo-cov.', 'small')
    p('[2] Wan, H.; Luo, M.; Ma, Z.; Dai, G.; Zhao, X. (2025). <i>How Do Social Bots Participate in Misinformation Spread? A Comprehensive Dataset and Analysis.</i> EMNLP 2025, pp. 31481-31504. https://aclanthology.org/2025.emnlp-main.1604/. Official repository: https://github.com/whr000001/MisBot.', 'small')
    p('[3] scikit-learn documentation. Clustering, text feature extraction, logistic regression, random forests and evaluation metrics; implementation version 1.5.1. https://scikit-learn.org/stable/.', 'small')
    p('[4] mlxtend documentation. Frequent-pattern algorithms: Apriori and FP-growth; implementation version 0.23.1. https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/ and https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/.', 'small')
    p('[5] AIAA 3111 Teaching Team. (2026). <i>Course Project Guide</i> and poster submission notice, supplied course materials. The team proposal specifies the CIB/HMI research scope.', 'small')
    p('Code and public aggregate evidence: '+URL+'. Data version, model provenance, original delivered execution and final independent audit are recorded in the repository. Online references were checked against the supplied official project documentation and committed source audits.', 'small')

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
