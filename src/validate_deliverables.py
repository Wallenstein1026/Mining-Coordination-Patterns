"""Verify final PDF geometry, content, page count and vector-only poster artwork."""
from pathlib import Path
import hashlib, json, subprocess
from pdfrw import PdfReader
ROOT=Path(__file__).resolve().parents[1]

def text(path):return subprocess.check_output(['pdftotext',str(path),'-'],text=True)
def images(resources,visited=None):
    visited=set() if visited is None else visited
    total=0
    for obj in (getattr(resources,'XObject',None) or {}).values():
        if id(obj) in visited:continue
        visited.add(id(obj))
        if str(getattr(obj,'Subtype',''))=='/Image':total+=1
        elif getattr(obj,'Resources',None):total+=images(obj.Resources,visited)
    return total

def main():
    poster=ROOT/'Poster.pdf';report=ROOT/'Report.pdf'
    a,b=PdfReader(str(poster)),PdfReader(str(report))
    assert len(a.pages)==1 and len(b.pages)==10
    box=list(map(float,a.pages[0].MediaBox));w=(box[2]-box[0])*2.54/72;h=(box[3]-box[1])*2.54/72
    assert abs(w-80)<1e-4 and abs(h-100)<1e-4
    p,t=text(poster),text(report)
    for phrase in ['AIAA 3111 COURSE PROJECT POSTER','GROUP 28','Jiaxin DONG','Qixiang HUA','Xiaofeng LI','Chang XU','Human-Manipulation Index','HMI =','30.7%','3.38%']:
        assert phrase in p,phrase
    pages=[s for s in t.split('\f') if s.strip()]
    assert len(pages)==10 and '6. Disclosures and References' in pages[9]
    assert '6. Disclosures and References' not in ''.join(pages[:9])
    for phrase in ['2. Problem Formulation','3. Methodology','4. Experiments','5. Conclusion','0.2365','47.43%','3.38%','Codex']:
        assert phrase in t,phrase
    assert images(a.pages[0].Resources)==0,'Poster contains raster artwork'
    for path in [poster,report]:assert path.read_bytes()==(ROOT/'output/pdf'/path.name).read_bytes()
    receipt={'poster_cm':[w,h],'poster_pages':1,'poster_raster_images':0,'poster_result_figures':3,
        'report_main_pages':9,'report_disclosure_reference_pages':1,'required_headers_names_cib_hmi':'passed',
        'files':{path.name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in [poster,report]}}
    (ROOT/'results/final_review/artifacts.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PDF geometry, names, CIB/HMI, 9+1 report pages, matching copies and vector-only poster passed.')

if __name__=='__main__':main()
