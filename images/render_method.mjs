// Reconstruct the accepted 1774×887 reference as editable draw.io primitives.
// SVG and PNG previews use the same geometry, typography and labels.
import fs from 'node:fs';
const W=1774,H=887,ink='#282927',blue='#2159C7',green='#29A56B',red='#CC2B1C';
const nodes=[],edges=[];let nextId=2;
const esc=v=>String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll('\n','&#10;');
function box(x,y,w,h,fill='#E3E3E3',stroke='none',opts={}){nodes.push({id:nextId++,x,y,w,h,fill,stroke,sw:1,...opts});}
function text(x,y,w,h,value,size=24,opts={}){nodes.push({id:nextId++,x,y,w,h,value,size,fill:'none',stroke:'none',color:'#101010',align:'center',...opts});}
function line(points,color=ink,width=2,head=true){edges.push({id:nextId++,points,color,width,head});}
// Large stage boundaries and original typography.
box(588,45,1170,506,'#FFFFFF','#858585',{radius:36,dash:true,sw:2.5});
box(588,597,1170,273,'#FFFFFF','#858585',{radius:34,dash:true,sw:2.5});
text(610,1,1140,38,'Stage 1: Parallel Probabilistic Prediction',31,{bold:true,align:'left'});
text(610,558,1000,37,'Stage 2: Direct Probability Tuning',30,{bold:true,align:'left'});
// Left-hand narrative matches the accepted figure.
box(44,167,26,26,'#353631','none',{ellipse:true});
box(35,198,44,33,'#353631','none',{radius:7});
box(119,128,440,165,'#E3F0D9',ink,{radius:19,sw:3});
text(139,149,405,127,'Question: What is the capital of\nAustralia?\nChoices: A Sydney · B Melbourne ·\nC Canberra · D Brisbane',26,{align:'left',leading:1.22});
line([[338,305],[338,357]],ink,7);
text(120,369,437,34,'Standard LLM',30,{bold:true});
box(120,410,439,115,'#DCE4F5',ink,{radius:17,sw:3});
text(230,430,180,35,'Decision: A',27);
box(409,429,35,35,red,'none',{ellipse:true});
text(411,429,31,34,'×',35,{color:'#FFFFFF',bold:true,font:'Arial'});
text(153,468,327,35,'p = [0.72, 0.08, 0.14, 0.06]',26,{italic:true});
box(475,466,35,35,red,'none',{ellipse:true});
text(477,466,31,34,'×',35,{color:'#FFFFFF',bold:true,font:'Arial'});
line([[338,546],[338,601]],ink,7);
text(359,553,210,36,'After fine-tuning',26,{align:'left'});
text(120,617,439,37,'JevTuner',31,{bold:true});
box(120,664,439,116,'#DCE4F5',ink,{radius:17,sw:3});
text(230,686,180,35,'Decision: C',27);
box(411,688,36,36,green,'none',{ellipse:true});
text(413,688,32,35,'✓',28,{color:'#FFFFFF',bold:true,font:'Arial'});
text(153,727,327,35,'p = [0.03, 0.04, 0.91, 0.02]',26,{italic:true});
text(477,724,47,40,'✓',43,{color:green,bold:true,font:'Arial'});
// Stage 1: tokens, model, full vocabulary and gather.
[['State',731,126],['Question',871,157],['Choices: A B C D',1038,280],['Decision:',1334,209]].forEach(([s,x,w])=>{box(x,65,w,42,'#DCDCDC');text(x,67,w,38,s,24);});
line([[1135,113],[1135,146]],ink,5);
box(732,154,844,69,'#DEE5F6',ink,{radius:12,sw:3});
text(732,165,844,45,'LLM',35,{bold:true});
text(902,230,470,27,'Full vocabulary logits',24);
const cells=[['t₁',623,48],['t₂',671,47],['…',718,62],['A',780,57],['…',837,55],['tᵢ',892,63],['…',955,61],['B',1016,57],['…',1073,54],['tⱼ',1127,63],['…',1190,55],['C',1245,57],['…',1302,54],['tₖ',1356,62],['…',1418,61],['D',1479,57],['…',1536,65],['t|V|',1601,78]];
cells.forEach(([s,x,w])=>{box(x,260,w,43,['A','B','C','D'].includes(s)?'#FFDCBC':'#EAEAEA','#8B8B8B',{sw:.9});text(x,267,w,29,s,24,{italic:s.startsWith('t')});});
line([[808,303],[808,322],[1005,322],[1005,337]],ink,2);
line([[1044,303],[1044,322],[1136,322],[1136,337]],ink,2);
line([[1273,303],[1273,322],[1192,322],[1192,337]],ink,2);
line([[1507,303],[1507,322],[1308,322],[1308,337]],ink,2);
box(920,340,471,38,'#E3E3E3');
text(930,341,451,35,'Selected candidate logits: [ z<sub>A</sub>   z<sub>B</sub>   z<sub>C</sub>   z<sub>D</sub> ]',22,{rich:true});
text(1418,334,300,54,'Select A/B/C/D from |V|\nvocabulary tokens',23,{bold:true,color:blue,align:'left',leading:1.08});
line([[1136,378],[1136,391]],ink,2);
text(885,386,410,30,'Softmax over candidate tokens',23);
// Probability bars stay vertical, with thin black baselines.
function bars(y,xs,ps,after=false){
 ['A','B','C','D'].forEach((s,i)=>{
  const x=xs[i],p=ps[i],cardH=after?89:110,baseline=y+(after?54:74);
  box(x,y,161,cardH,'#E5E5E5','none',{radius:6});
  text(x,y+3,161,30,s,26);
  const bh=Math.max(3,p*(after?26:50));
  box(x+51,baseline-bh,57,bh,'#FF8C3D','#FFFFFF',{sw:1.2});
  line([[x+17,baseline],[x+145,baseline]],ink,1.2,false);
  text(x,y+(after?57:77),161,29,after?p.toFixed(2):'p<sub>'+s+'</sub> = '+p.toFixed(2),24,{italic:!after,rich:!after});
 });
}
bars(420,[785,954,1125,1293],[.72,.08,.14,.06]);
text(1470,451,255,58,'One forward → full\nprobability distribution',24,{bold:true,color:blue,align:'left',leading:1.12});
// Stage 2: native editable supervision, formula, bars and feedback.
line([[865,530],[865,542],[1112,542],[1112,615]],ink,2);
line([[1035,530],[1035,542],[1112,542]],ink,2,false);
line([[1205,530],[1205,615]],ink,2);
line([[1373,530],[1373,561],[1298,561],[1298,615]],ink,2);
box(620,623,365,63,'#E3F0D9',ink,{radius:14,sw:2});
text(628,632,349,43,'y = [0, 0, 1, 0]  (C is correct)',26,{italic:true});
line([[985,655],[1068,655]],ink,2.5);
text(1075,616,370,35,'Tokenized Multi-class Brier Loss',25,{bold:true,color:blue});
// Split formula makes both draw.io and SVG preserve the large summation and subscript.
text(1129,650,55,42,'L =',30,{italic:true});
text(1187,639,42,58,'Σ',46);
text(1211,677,19,22,'k',17,{italic:true});
text(1230,650,156,42,'(pₖ − yₖ)²',31,{italic:true});
text(1531,620,177,32,'Backpropagate',25,{bold:true,color:blue});
line([[1458,663],[1709,663],[1709,190],[1583,190]],ink,5);
bars(704,[818,984,1143,1310],[.03,.04,.91,.02],true);
line([[891,796],[891,823]],red,4);
line([[1058,796],[1058,823]],red,4);
line([[1224,823],[1224,796]],green,4);
line([[1393,796],[1393,823]],red,4);
text(871,823,266,38,'decrease p<sub>A</sub>, p<sub>B</sub>, p<sub>D</sub>',25,{bold:true,color:blue,italic:true,rich:true});
text(1164,823,171,38,'increase p<sub>C</sub>',25,{bold:true,color:blue,italic:true,rich:true});
// Serializers: every shape, label and connector remains independently editable.
let xml='<mxfile host="app.diagrams.net"><diagram id="jevtuner-method" name="JevTuner Method"><mxGraphModel grid="1" gridSize="5" page="1" pageWidth="'+W+'" pageHeight="'+H+'"><root><mxCell id="0"/><mxCell id="1" parent="0"/>';
let svg='<svg xmlns="http://www.w3.org/2000/svg" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'"><rect width="100%" height="100%" fill="white"/><defs>';
for(const c of [ink,blue,green,red])svg+='<marker id="a'+c.slice(1)+'" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.6" markerHeight="4.6" orient="auto"><path d="M0 0L10 5L0 10z" fill="'+c+'"/></marker>';
svg+='</defs>';
for(const s of nodes){
 const font=s.font||'Times New Roman',weight=(s.bold?1:0)+(s.italic?2:0),align=s.align||'center';
 const style='shape='+(s.ellipse?'ellipse':'rectangle')+';rounded='+(s.radius?1:0)+';absoluteArcSize=1;arcSize='+(2*(s.radius||0))+';whiteSpace=wrap;html='+(s.rich?1:0)+';fillColor='+s.fill+';strokeColor='+s.stroke+';strokeWidth='+(s.sw||1)+';dashed='+(s.dash?1:0)+';dashPattern=5 4;fontFamily='+font+';fontSize='+(s.size||24)+';fontColor='+(s.color||ink)+';fontStyle='+weight+';align='+align+';verticalAlign=middle;spacing=0;';
 xml+='<mxCell id="'+s.id+'" value="'+esc(s.value||'')+'" style="'+style+'" vertex="1" parent="1"><mxGeometry x="'+s.x+'" y="'+s.y+'" width="'+s.w+'" height="'+s.h+'" as="geometry"/></mxCell>';
 if(s.ellipse)svg+='<ellipse cx="'+(s.x+s.w/2)+'" cy="'+(s.y+s.h/2)+'" rx="'+s.w/2+'" ry="'+s.h/2+'" fill="'+s.fill+'" stroke="'+s.stroke+'"/>';
 else svg+='<rect x="'+s.x+'" y="'+s.y+'" width="'+s.w+'" height="'+s.h+'" rx="'+(s.radius||0)+'" fill="'+s.fill+'" stroke="'+s.stroke+'" stroke-width="'+(s.sw||1)+'" '+(s.dash?'stroke-dasharray="10 8"':'')+'/>';
 if(s.value){const lines=s.value.split('\n');lines.forEach((l,i)=>{
 const content=s.rich?l.split(/(<sub>.*?<\/sub>)/).map(part=>part.startsWith('<sub>')?'<tspan baseline-shift="sub" font-size="70%">'+esc(part.slice(5,-6))+'</tspan>':esc(part)).join(''):esc(l);
 svg+='<text x="'+(align==='left'?s.x:s.x+s.w/2)+'" y="'+(s.y+s.h/2+(i-(lines.length-1)/2)*s.size*(s.leading||1.2))+'" dominant-baseline="central" text-anchor="'+(align==='left'?'start':'middle')+'" font-family="'+font+', serif" font-size="'+s.size+'" font-weight="'+(s.bold?700:400)+'" font-style="'+(s.italic?'italic':'normal')+'" fill="'+s.color+'">'+content+'</text>';
 });}
}
for(const e of edges){const a=e.points[0],b=e.points.at(-1);
 xml+='<mxCell id="'+e.id+'" style="edgeStyle=none;endArrow='+(e.head?'block':'none')+';endSize=9;endFill=1;strokeColor='+e.color+';strokeWidth='+e.width+';" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="'+a[0]+'" y="'+a[1]+'" as="sourcePoint"/><mxPoint x="'+b[0]+'" y="'+b[1]+'" as="targetPoint"/><Array as="points">'+e.points.slice(1,-1).map(p=>'<mxPoint x="'+p[0]+'" y="'+p[1]+'"/>').join('')+'</Array></mxGeometry></mxCell>';
 svg+='<polyline points="'+e.points.map(p=>p.join(',')).join(' ')+'" fill="none" stroke="'+e.color+'" stroke-width="'+e.width+'" '+(e.head?'marker-end="url(#a'+e.color.slice(1)+')"':'')+'/>';
}
fs.writeFileSync(new URL('jevtuner_method.drawio',import.meta.url),xml+'</root></mxGraphModel></diagram></mxfile>');
fs.writeFileSync(new URL('jevtuner_method.svg',import.meta.url),svg+'</svg>');
