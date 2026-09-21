// Native editable draw.io shapes and a matching vector preview.
import fs from 'node:fs';
const shapes=[],edges=[];let id=2;
function box(x,y,w,h,label='',fill='#FFFFFF',stroke='#30343B',font=20,extra={}){shapes.push({id:id++,x,y,w,h,label,fill,stroke,font,...extra});}
function text(x,y,w,h,label,font=20,color='#30343B',extra={}){box(x,y,w,h,label,'none','none',font,{color,...extra});}
function arrow(pts,color='#30343B',dash=false){edges.push({id:id++,pts,color,dash});}
box(525,62,1035,570,'','#FFFFFF','#9299A3',20,{dash:true,radius:22});
box(525,685,1035,390,'','#FFFFFF','#9299A3',20,{dash:true,radius:22});
text(550,15,960,40,'Stage 1 · Parallel probabilistic prediction',27,'#202733',{bold:true});
text(550,638,960,40,'Stage 2 · Tokenized Multi-class Brier tuning',26,'#202733',{bold:true});
box(36,118,443,172,'','#E9F4E3','#30343B',20,{radius:16});
text(58,138,402,135,'Question: What is the capital of Australia?\nA · Sydney       B · Melbourne\nC · Canberra     D · Brisbane',19);
arrow([[255,302],[255,352]]);
text(35,357,445,40,'Before fine-tuning',24,'#30343B',{bold:true});
box(36,407,443,126,'Decision: A  (incorrect)\np = [0.72, 0.08, 0.14, 0.06]','#E8EEFA','#30343B',21,{radius:14});
arrow([[255,548],[255,607]]);
text(276,558,195,45,'Brier fine-tuning',18,'#2B63B8');
text(36,625,443,40,'JevTuner',27,'#30343B',{bold:true});
box(36,679,443,126,'Decision: C  (correct)\np = [0.03, 0.04, 0.91, 0.02]','#E8EEFA','#30343B',21,{radius:14});
text(43,837,425,98,'Illustrative predictions.\nCalibration is evaluated across a dataset.',17,'#667080');
[['State',590,170],['Question',775,180],['Choices: A B C D',970,290],['Decision:',1275,220]].forEach(([l,x,w])=>box(x,92,w,47,l,'#ECEEF0','#ECEEF0',20));
arrow([[1040,143],[1040,171]]);
box(598,178,884,75,'LLM · one forward pass','#E0E8F9','#30343B',27,{radius:12,bold:true});
arrow([[1040,254],[1040,280]]);
text(650,283,790,28,'Full vocabulary logits — |V| entries',20);
const xs={};
['t₁','…','A','…','tᵢ','B','…','tⱼ','C','…','tₖ','D','…','t|V|'].forEach((l,i)=>{const x=574+i*66;box(x,326,64,45,l,['A','B','C','D'].includes(l)?'#FFE1C8':'#E9EBED','#FFFFFF',20);if(['A','B','C','D'].includes(l))xs[l]=x+32;});
['A','B','C','D'].forEach((l,i)=>{const dest=948+i*82;arrow([[xs[l],372],[xs[l],410+i*5],[dest,410+i*5],[dest,443]],'#2B63B8');box(dest-40,449,78,39,'z'+l,'#FFF0E3','#DCC6B7',20);});
text(585,443,300,46,'Gather candidate token IDs\nSelected logits',17,'#2B63B8');
arrow([[1070,489],[1070,522]]);
text(585,501,320,38,'Softmax over A/B/C/D',19,'#2B63B8');
function bars(y,values){['A','B','C','D'].forEach((l,i)=>{const x=597+214*i;box(x,y,190,84,'','#F4F5F6','#E0E3E7',18,{radius:4});text(x+10,y+3,170,26,l+'   '+values[i].toFixed(2),19);box(x+14,y+37,162,30,'','#FCE8D8','none');box(x+14,y+37,162*values[i],30,'','#F38C43','none');});}
bars(534,[.72,.08,.14,.06]);
arrow([[1040,621],[1460,621],[1460,709],[1214,709],[1214,728]]);
box(565,735,348,86,'Correct choice: C\ny = [0, 0, 1, 0]','#E7F4ED','#36906B',22,{radius:9});
box(988,735,452,86,'Tokenized Multi-class Brier Loss\nL = Σₖ (pₖ − yₖ)²','#FFF3D8','#30343B',22,{radius:9});
arrow([[915,778],[982,778]],'#36906B');
arrow([[1440,778],[1525,778],[1525,215],[1489,215]],'#2B63B8',true);
text(1248,827,248,30,'Backpropagate to LLM',17,'#2B63B8');
text(595,861,880,29,'Illustrative distribution after fine-tuning',18,'#667080');
bars(901,[.03,.04,.91,.02]);
['A','B','C','D'].forEach((l,i)=>text(597+214*i,992,190,30,l==='C'?'↑ increase pC':'↓ decrease p'+l,18,l==='C'?'#20865C':'#B14A39'));
text(576,1034,910,29,'Supervision requires only the correct choice; no manually specified probability targets.',17,'#57606F');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll('\n','&#10;');
let xml='<mxfile host="app.diagrams.net"><diagram id="jevtuner-method" name="JevTuner Method"><mxGraphModel grid="1" gridSize="10" page="1" pageWidth="1600" pageHeight="1110"><root><mxCell id="0"/><mxCell id="1" parent="0"/>';
let svg='<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1110" viewBox="0 0 1600 1110"><rect width="1600" height="1110" fill="white"/><defs>';
for(const c of ['#30343B','#2B63B8','#36906B'])svg+='<marker id="a'+c.slice(1)+'" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="'+c+'"/></marker>';
svg+='</defs>';
for(const s of shapes){
 xml+='<mxCell id="'+s.id+'" value="'+esc(s.label)+'" style="rounded='+(s.radius?1:0)+';whiteSpace=wrap;html=0;fillColor='+s.fill+';strokeColor='+s.stroke+';strokeWidth=2;dashed='+(s.dash?1:0)+';fontFamily=Arial;fontSize='+s.font+';fontColor='+(s.color||'#30343B')+';fontStyle='+(s.bold?1:0)+';align=center;verticalAlign=middle;" vertex="1" parent="1"><mxGeometry x="'+s.x+'" y="'+s.y+'" width="'+s.w+'" height="'+s.h+'" as="geometry"/></mxCell>';
 svg+='<rect x="'+s.x+'" y="'+s.y+'" width="'+s.w+'" height="'+s.h+'" rx="'+(s.radius||0)+'" fill="'+s.fill+'" stroke="'+s.stroke+'" stroke-width="2" '+(s.dash?'stroke-dasharray="11 8"':'')+'/>';
 const lines=s.label.split('\n');lines.forEach((l,i)=>{svg+='<text x="'+(s.x+s.w/2)+'" y="'+(s.y+s.h/2+(i-(lines.length-1)/2)*s.font*1.45)+'" dominant-baseline="middle" text-anchor="middle" font-family="Arial, sans-serif" font-size="'+s.font+'" font-weight="'+(s.bold?700:400)+'" fill="'+(s.color||'#30343B')+'">'+esc(l)+'</text>';});
}
for(const e of edges){const a=e.pts[0],b=e.pts.at(-1);
 xml+='<mxCell id="'+e.id+'" style="edgeStyle=none;endArrow=block;endFill=1;strokeColor='+e.color+';strokeWidth=2;dashed='+(e.dash?1:0)+';" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="'+a[0]+'" y="'+a[1]+'" as="sourcePoint"/><mxPoint x="'+b[0]+'" y="'+b[1]+'" as="targetPoint"/><Array as="points">'+e.pts.slice(1,-1).map(p=>'<mxPoint x="'+p[0]+'" y="'+p[1]+'"/>').join('')+'</Array></mxGeometry></mxCell>';
 svg+='<polyline points="'+e.pts.map(p=>p.join(',')).join(' ')+'" fill="none" stroke="'+e.color+'" stroke-width="2" marker-end="url(#a'+e.color.slice(1)+')" '+(e.dash?'stroke-dasharray="8 5"':'')+'/>';
}
fs.writeFileSync(new URL('jevtuner_method.drawio',import.meta.url),xml+'</root></mxGraphModel></diagram></mxfile>');
fs.writeFileSync(new URL('jevtuner_method.svg',import.meta.url),svg+'</svg>');
