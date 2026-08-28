"""Self-contained interactive 3-D viewer for the per-atom source clouds.

One HTML file, no network access required: plotly.js is inlined and every point is
embedded as base64 uint16, so it opens from disk on any machine.

What it gives over the static renders:
  * drag to rotate, scroll to zoom, and independent x/y/z aspect sliders, because the
    natural aspect (1958 s x 200 um x 3840 um) is unreadable at 1:1:1 and the useful
    stretch is different for every question;
  * one click to switch between the 64 atoms, and between uncorrected / rigid /
    nonrigid depths for the SAME points, which is the comparison the static grid of
    384 PNGs makes you do from memory;
  * the atom waveform and its prototype redraw alongside.

Storage: x and time do not change with the drift correction, so they are stored once per
atom and only the three depth arrays are duplicated. Everything is quantised to uint16
(~0.03 s, ~0.003 um, ~0.06 um resolution) and q to uint8, which is far finer than the
data warrants and keeps the file around a tenth of the float32 size.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                            # noqa: E402
from spiketensor.drift import correction                   # noqa: E402
from spiketensor.viz_centroid_basis import shuffled_palette  # noqa: E402

TLIM = (0.0, 1958.0)
# The lateral extent of the LEARNED dictionary, not the probe: candidate
# offsets reach +-176 um, so sources legitimately sit well outside the
# 0-48 um column span. Clipping to the probe width silently piled ~4% of
# sources onto the plot edges as two false rays.
XLIM = (-160.0, 200.0)
YLIM = (0.0, 3840.0)


def _q16(a, lo, hi):
    v = np.clip((np.asarray(a, np.float64) - lo) / (hi - lo), 0, 1)
    return (v * 65535.0).astype(np.uint16)


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


HTML = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>
 html,body{margin:0;height:100%;background:#0f1115;color:#dfe3ea;
   font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
 #wrap{display:grid;grid-template-columns:230px 1fr 392px;height:100vh}
 #insp{padding:10px 12px;border-left:1px solid #262a33;overflow-y:auto}
 #insp h2{font-size:12px;margin:0 0 6px;color:#c8cede}
 #inspmeta{font-size:11px;color:#9aa3b2;white-space:pre-line;margin-bottom:6px}
 #chans{background:#12151b;border:1px solid #262a33;border-radius:5px}
 .key{font-size:10px;color:#9aa3b2;margin-top:5px}
 .key i{font-style:normal;font-weight:700}
 #side{padding:10px 12px;overflow-y:auto;border-right:1px solid #262a33}
 #plot{width:100%;height:100vh}
 h1{font-size:14px;margin:2px 0 8px}
 .grp{margin:12px 0 6px;font-size:11px;color:#9aa3b2;text-transform:uppercase;
   letter-spacing:.06em}
 .atoms{display:grid;grid-template-columns:repeat(8,1fr);gap:3px}
 .atoms button{font-size:10px;padding:3px 0;background:#1a1e26;color:#c8cede;
   border:1px solid #2c313c;border-radius:3px;cursor:pointer}
 .atoms button.on{background:#2f6f4f;border-color:#3d8f66;color:#fff}
 .modes button{display:block;width:100%;margin:3px 0;padding:5px;background:#1a1e26;
   color:#c8cede;border:1px solid #2c313c;border-radius:4px;cursor:pointer}
 .modes button.on{background:#2b4d7a;border-color:#3a67a3;color:#fff}
 label{display:block;margin:7px 0 2px;font-size:11px;color:#9aa3b2}
 input[type=range]{width:100%}
 #wave{background:#fff;border-radius:4px;margin-top:8px}
 #meta{font-size:11px;color:#9aa3b2;margin-top:8px;white-space:pre-line}
</style>
<div id="wrap"><div id="side">
 <h1>__TITLE__</h1>
 <div class="grp">view</div>
 <div class="modes" id="views"></div>
 <div class="grp">colour</div>
 <div class="modes" id="cmodes"></div>
 <div class="grp">correction</div>
 <div class="modes" id="modes"></div>
 <div class="grp">atom</div>
 <div class="modes"><button id="allbtn">ALL atoms merged</button></div>
 <div class="atoms" id="atoms"></div>
 <svg id="wave" width="204" height="86"></svg>
 <div class="modes"><button id="insbtn" class="on">show inspectable spikes</button></div>
 <div class="grp">aspect ratio</div>
 <label>time <span id="lax"></span></label><input type=range id="ax" min="0.2" max="6"
   step="0.1" value="3">
 <label>lateral x <span id="lay"></span></label><input type=range id="ay" min="0.2"
   max="6" step="0.1" value="0.8">
 <label>depth y <span id="laz"></span></label><input type=range id="az" min="0.2"
   max="6" step="0.1" value="1.6">
 <div class="grp">display</div>
 <label>point size <span id="lps"></span></label><input type=range id="ps" min="0.6"
   max="6" step="0.2" value="2">
 <div class="grp">error filter</div>
 <label>min error <span id="lemin"></span></label><input type=range id="emin"
   min="0" max="100" step="1" value="0">
 <label>max error <span id="lemax"></span></label><input type=range id="emax"
   min="0" max="100" step="1" value="100">
 <div class="grp">display</div>
 <label>amplitude gamma <span id="lgam"></span></label><input type=range id="gam"
   min="0.3" max="3" step="0.1" value="1.6">
 <label>max points <span id="lcap"></span></label><input type=range id="cap"
   min="25000" max="900000" step="25000" value="300000">
 <label>depth window</label>
 <input type=range id="y0" min="0" max="3840" step="20" value="400">
 <input type=range id="y1" min="0" max="3840" step="20" value="900">
 <div id="meta"></div>
</div><div id="plot"></div>
<div id="insp"><h2>spike inspector</h2>
 <div class="modes" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px">
   <button id="prevsp">&lsaquo; prev</button><button id="randsp">random</button>
   <button id="nextsp">next &rsaquo;</button></div>
 <div class="modes" style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
   <button id="v10" class="on">10 nearest</button>
   <button id="v384">all 384 · amplitude</button></div>
 <div class="modes"><button id="vfull">all 384 · waveforms + model extrapolation</button>
 </div>
 <div id="inspmeta">click a white marker, or use the buttons</div>
 <svg id="chans" width="304" height="620"></svg>
 <div class="key"><i style="color:#e03131">— measured</i> &nbsp;
   <i style="color:#2f9e44">— model</i><br>
   10 nearest channels at their real probe positions (□ = contact), or all 384
   contacts coloured by amplitude (○ = fitted source depth).
   Clicking anywhere snaps to the nearest INSPECTABLE spike on screen; only a
   subset carries waveforms.</div>
</div></div>
<script>__PLOTLYJS__</script>
<script>
const D=__DATA__;
const dec=(s,T)=>{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);
  return T===16?new Uint16Array(u.buffer):u;};
const un=(a,lo,hi)=>{const o=new Float32Array(a.length),s=(hi-lo)/65535;
  for(let i=0;i<a.length;i++)o[i]=lo+a[i]*s;return o;};
let atom=0, mode="none", allmode=false, cmode="atom", vmode="oblique", showIns=true;
const MODES=[["none","uncorrected"],["drr","DREDge rigid"],["drn","DREDge nonrigid"]];
const mdiv=document.getElementById("modes");
MODES.forEach(([k,lab])=>{const b=document.createElement("button");b.textContent=lab;
  b.dataset.k=k;b.onclick=()=>{mode=k;paint();};mdiv.appendChild(b);});
const adiv=document.getElementById("atoms");
for(let q=0;q<D.n_atoms;q++){const b=document.createElement("button");
  b.textContent=q;b.dataset.q=q;b.onclick=()=>{atom=q;allmode=false;paint();};
  adiv.appendChild(b);}
document.getElementById("allbtn").onclick=()=>{allmode=!allmode;paint();};
document.getElementById("insbtn").onclick=function(){showIns=!showIns;
  this.classList.toggle("on",showIns);paint();};
const CMODES=[["atom","by atom colour"],["amp","by amplitude (magma)"],
              ["err","by spike error (viridis)"]];
const cdiv=document.getElementById("cmodes");
CMODES.forEach(([k,lab])=>{const b=document.createElement("button");b.textContent=lab;
  b.dataset.k=k;b.onclick=()=>{cmode=k;paint();};cdiv.appendChild(b);});
// "raster" reproduces the 2-D depth x time panels: look straight down the lateral
// axis with an orthographic camera, so the third dimension collapses exactly the way
// the histogram panels collapse it
const VIEWS=[["oblique","3-D oblique"],["raster","depth x time (raster)"],
             ["xy","depth x lateral x"]];
const vdiv=document.getElementById("views");
VIEWS.forEach(([k,lab])=>{const b=document.createElement("button");b.textContent=lab;
  b.dataset.k=k;b.onclick=()=>{vmode=k;paint();};vdiv.appendChild(b);});

// magma, sampled; matches the colour convention of the amplitude rasters
const MAGMA=[[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],
             [229,80,100],[251,135,97],[254,194,135],[252,253,191]];
function magma(v){const x=Math.max(0,Math.min(1,v))*(MAGMA.length-1);
  const i=Math.floor(x),f=x-i,a=MAGMA[i],b=MAGMA[Math.min(i+1,MAGMA.length-1)];
  return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];}

function decAtom(q){const a=D.atoms[q];
  return {t:un(dec(a.t,16),D.tlim[0],D.tlim[1]),
          x:un(dec(a.x,16),D.xlim[0],D.xlim[1]),
          y:un(dec(a["y_"+mode],16),D.ylim[0],D.ylim[1]),
          q:dec(a.q,8), a:dec(a.a,8), e:dec(a.e,8)};}
function wave(){const s=document.getElementById("wave");
  if(allmode){s.innerHTML="";return;}
  const w=D.waves[atom],p=D.protos[atom];
  const W=204,H=86,n=w.length,mx=Math.max(...w.map(Math.abs),...p.map(Math.abs))||1;
  const path=v=>v.map((y,i)=>`${i?"L":"M"}${(i/(n-1)*(W-8)+4).toFixed(1)},${(H/2-y/mx*(H/2-6)).toFixed(1)}`).join("");
  s.innerHTML=`<path d="${path(p)}" stroke="#888" stroke-width="1.2" fill="none"
    stroke-dasharray="3,2"/><path d="${path(w)}" stroke="${D.colors[atom]}"
    stroke-width="2" fill="none"/>`;}
function camera(){
  // eye on the NEGATIVE lateral axis so time runs left-to-right exactly as in the
  // 2-D depth x time panels; the +y side mirrors them
  if(vmode==="raster") return {eye:{x:0,y:-2.5,z:0},up:{x:0,y:0,z:1},
                               projection:{type:"orthographic"}};
  if(vmode==="xy")     return {eye:{x:-2.5,y:0,z:0},up:{x:0,y:0,z:1},
                               projection:{type:"orthographic"}};
  return {eye:{x:1.6,y:1.4,z:0.85},projection:{type:"perspective"}};}
// ---------------- click inspector ----------------
const I=D.inspect;
const dec16s=s=>{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Int16Array(u.buffer);};
const HASW=!!I.obs;      // some models cannot rebuild footprints; guard the panels
const I8=s=>{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Int8Array(u.buffer);};
const IT=un(dec(I.t,16),D.tlim[0],D.tlim[1]),
      IX=un(dec(I.x,16),D.xlim[0],D.xlim[1]),
      IOBS=HASW?I8(I.obs):null, IFIT=HASW?I8(I.fit):null,
      IOFF=HASW?dec16s(I.off):null;
const IY={none:un(dec(I.y_none,16),D.ylim[0],D.ylim[1]),
          drr:un(dec(I.y_drr,16),D.ylim[0],D.ylim[1]),
          drn:un(dec(I.y_drn,16),D.ylim[0],D.ylim[1])};
function nearest(t,x,y){
  // distances are normalised by each axis' full range, so "nearest" means nearest ON
  // SCREEN rather than nearest in seconds, which would ignore depth entirely
  const ys=IY[mode], st=D.tlim[1]-D.tlim[0], sx=D.xlim[1]-D.xlim[0],
        sy=D.ylim[1]-D.ylim[0];
  let best=-1,bd=Infinity;
  for(let i=0;i<I.n;i++){
    const a=(IT[i]-t)/st,b=(IX[i]-x)/sx,c=(ys[i]-y)/sy;
    const d=a*a+b*b+c*c; if(d<bd){bd=d;best=i;}}
  return best;}
const CHAN=dec16s(I.chan), PTP=I.ptp?dec(I.ptp,8):null;
function showProbe(i){
  // All 384 contacts at their real positions, coloured by this spike's peak-to-peak on
  // that channel. Stepping through spikes shows the footprint SLIDE ALONG THE PROBE as
  // the tissue drifts, which a 10-channel view cannot show because its window moves with
  // the spike.
  const W=304,H=620,pad=12;
  if(!PTP){document.getElementById("chans").innerHTML=
     '<text x="10" y="20" fill="#9aa3b2" font-size="11">no raw file at build time</text>';
   return;}
  let xs=[],ys=[];
  for(let c=0;c<384;c++){xs.push(CHAN[c*2]*I.chan_scale);ys.push(CHAN[c*2+1]*I.chan_scale);}
  const x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),
        y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys);
  const kx=(W-2*pad-26)/Math.max(x1-x0,1e-6), ky=(H-2*pad)/Math.max(y1-y0,1e-6);
  const PX=v=>pad+(v-x0)*kx, PY=v=>H-pad-(v-y0)*ky;
  let g="";
  for(let c=0;c<384;c++){
    const v=PTP[i*384+c]/255, m=magma(Math.pow(v,0.6));
    g+='<rect x="'+(PX(xs[c])-3).toFixed(1)+'" y="'+(PY(ys[c])-1.6).toFixed(1)+'"'
      +' width="6" height="3.2" fill="rgb('+Math.round(m[0])+','+Math.round(m[1])+','
      +Math.round(m[2])+')"/>';}
  (I.src_y[i]||[]).forEach(yv=>{
    g+='<circle cx="'+(PX((x0+x1)/2)).toFixed(1)+'" cy="'+PY(yv).toFixed(1)
      +'" r="4.5" fill="none" stroke="#ffffff" stroke-width="1.1"/>';});
  for(let d=0;d<=3800;d+=500){
    g+='<line x1="'+(W-24)+'" y1="'+PY(d).toFixed(1)+'" x2="'+(W-20)+'" y2="'
      +PY(d).toFixed(1)+'" stroke="#4a5160"/>'
      +'<text x="'+(W-18)+'" y="'+(PY(d)+3).toFixed(1)
      +'" fill="#7c8494" font-size="8">'+d+'</text>';}
  document.getElementById("chans").innerHTML=g;
}
const FU=I.full;
const FOBS=FU?I8(FU.obs):null, FFIT=FU?I8(FU.fit):null,
      FCH=FU?dec16s(FU.fit_chans):null;
function nearestFull(i){
  // only a subset carries full-probe pairs; fall back to the closest one that does
  if(!FU) return -1;
  if(FU.slot[i]>=0) return i;
  const ys=IY[mode]; let best=-1,bd=Infinity;
  for(let j=0;j<I.n;j++){ if(FU.slot[j]<0) continue;
    const a=(IT[j]-IT[i])/1958,b=(IX[j]-IX[i])/360,c=(ys[j]-ys[i])/3840;
    const d=a*a+b*b+c*c; if(d<bd){bd=d;best=j;}}
  return best;}
function showFull(i){
  const j=nearestFull(i);
  if(j<0){document.getElementById("chans").innerHTML=
    '<text x="8" y="16" fill="#9aa3b2" font-size="11">no full-probe data in this build</text>';
   return j;}
  const sl=FU.slot[j], T=FU.T, W=376, pad=8, base=sl*384*T;
  const ys=[],xs=[];
  for(let c=0;c<384;c++){xs.push(CHAN[c*2]*I.chan_scale);ys.push(CHAN[c*2+1]*I.chan_scale);}
  const x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),
        y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys);
  // NOT isotropic: the probe is 3,840 µm tall and ~48 µm wide, so each trace gets a
  // fixed PIXEL box and only the channel CENTRES follow the real geometry
  const rowPx=27, H=Math.round((y1-y0)/20*rowPx)+2*pad+40, TW=78, AH=11;
  const PX=v=>pad+(v-x0)/Math.max(x1-x0,1e-6)*(W-2*pad-TW);
  const PY=v=>H-pad-(v-y0)/Math.max(y1-y0,1e-6)*(H-2*pad-40);
  const isFit={}; for(let q=0;q<10;q++) isFit[FCH[sl*10+q]]=1;
  let g="";
  for(let c=0;c<384;c++){
    const px=PX(xs[c]), py=PY(ys[c]);
    const path=(arr,col,w)=>{let d="";
      for(let m=0;m<T;m++){const v=arr[base+c*T+m]/127;
        d+=(m?"L":"M")+(px+m/(T-1)*TW).toFixed(1)+","+(py-v*AH).toFixed(1);}
      return '<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="'+w+'"/>';};
    if(isFit[c]) g+='<rect x="'+(px-2).toFixed(1)+'" y="'+(py-AH-2).toFixed(1)+'" width="'
      +(TW+4)+'" height="'+(2*AH+4)+'" fill="#1b2a20" stroke="#2f9e44"'
      +' stroke-width=".6" opacity=".55"/>';
    g+=path(FOBS,"#e03131",0.9)+path(FFIT,"#2f9e44",0.9);
  }
  document.getElementById("chans").setAttribute("height",H);
  document.getElementById("chans").setAttribute("width",W);
  document.getElementById("chans").innerHTML=g;
  const sy=(I.src_y[j]||[0])[0];
  const box=document.getElementById("insp");
  box.scrollTop=Math.max(0,PY(sy)-box.clientHeight*0.45);
  return j;}
function showSpike(i){
  if(!HASW&&chanMode!=="384"){chanMode="384";
    document.getElementById("v384").classList.add("on");
    document.getElementById("v10").classList.remove("on");
    document.getElementById("vfull").classList.remove("on");}
  if(chanMode==="384"){curSpike=i;showProbe(i);showMeta(i);return;}
  if(chanMode==="full"){curSpike=i;const j=showFull(i);showMeta(j>=0?j:i);return;}
  document.getElementById("chans").setAttribute("width",304);
  document.getElementById("chans").setAttribute("height",620);
  // Channels are drawn AT THEIR POSITIONS on the probe, matching the browser's
  // reconstruction panels: each trace occupies a small time-width / amplitude-height box
  // centred on that channel's (x, y) offset. The scale is ISOTROPIC, so the staggered
  // two-column geometry stays honest instead of collapsing to a flat stack.
  const C=I.C,T=I.T,W=304,H=620,pad=10;
  const base=i*C*T, ob=i*C*2;
  // proportions follow the browser's reconstruction panels (~29 µm of width per trace
  // against ~14 µm of half-height); with an isotropic scale a tall/narrow box turns the
  // waveform into a vertical scribble instead of a readable spike
  const TW=29, AH=13;
  let xs=[],ys=[];
  for(let c=0;c<C;c++){const ox=IOFF[ob+c*2]*I.off_scale, oy=IOFF[ob+c*2+1]*I.off_scale;
    xs.push(ox,ox+TW); ys.push(oy-AH,oy+AH);}
  const x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),
        yy0=Math.min.apply(null,ys),yy1=Math.max.apply(null,ys);
  const k=Math.min((W-2*pad)/Math.max(x1-x0,1e-6),(H-2*pad)/Math.max(yy1-yy0,1e-6));
  const PX=v=>pad+(v-x0)*k, PY=v=>H-pad-(v-yy0)*k;
  let g="";
  for(let c=0;c<C;c++){
    const ox=IOFF[ob+c*2]*I.off_scale, oy=IOFF[ob+c*2+1]*I.off_scale;
    const path=(arr,col)=>{let d="";
      for(let m=0;m<T;m++){const v=arr[base+c*T+m]/127;
        d+=(m?"L":"M")+PX(ox+m/(T-1)*TW).toFixed(1)+","+PY(oy+v*AH).toFixed(1);}
      return '<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="1.1"/>';};
    g+='<line x1="'+PX(ox).toFixed(1)+'" y1="'+PY(oy).toFixed(1)+'" x2="'
      +PX(ox+TW).toFixed(1)+'" y2="'+PY(oy).toFixed(1)
      +'" stroke="#2a2f39" stroke-width=".5"/>'
      +'<rect x="'+(PX(ox)-2.5).toFixed(1)+'" y="'+(PY(oy)-2.5).toFixed(1)
      +'" width="5" height="5" fill="none" stroke="#4a5160" stroke-width=".7"/>'
      +path(IOBS,"#e03131")+path(IFIT,"#2f9e44");
  }
  curSpike=i;
  document.getElementById("chans").innerHTML=g;
  showMeta(i);
}
function showMeta(i){
  document.getElementById("inspmeta").textContent=
    "spike "+I.spike[i]+"  ·  t = "+IT[i].toFixed(1)+" s"+NLC+
    "x = "+IX[i].toFixed(1)+" µm   depth = "+IY[mode][i].toFixed(1)+" µm"+NLC+
    "atoms: "+I.atoms[i].join(", ")+NLC+
    "relative error = "+I.err[i].toFixed(3)+
    (chanMode==="384"?NLC+"all 384 contacts · colour = peak-to-peak on that channel"
     :chanMode==="full"?NLC+"all 384 channels · red = measured, green = model"+NLC
        +"green boxes = the 10 channels the model was FIT on; everything else is"+NLC
        +"extrapolation (units tied by one scalar fitted on those 10 only)"
     :"");
}
const NLC=String.fromCharCode(10);
// The panel must never be dead: 3-D clicks only register on a marker, so stepping and
// random sampling give a way in without having to aim at one.
let curSpike=0, chanMode=(I.obs?"10":"384");
if(!I.obs){["v10","vfull"].forEach(id=>{const b=document.getElementById(id);
  b.disabled=true;b.title="this model's footprints could not be rebuilt";});}
document.getElementById("v10").onclick=()=>{setMode("10");};
document.getElementById("v384").onclick=()=>{setMode("384");};
document.getElementById("vfull").onclick=()=>{setMode("full");};
function setMode(m){chanMode=m;
  [["10","v10"],["384","v384"],["full","vfull"]].forEach(([k,id])=>
    document.getElementById(id).classList.toggle("on",k===m));
  showSpike(curSpike);}
function stepSpike(d){curSpike=(curSpike+d+I.n)%I.n;showSpike(curSpike);}
document.getElementById("prevsp").onclick=()=>stepSpike(-1);
document.getElementById("nextsp").onclick=()=>stepSpike(1);
document.getElementById("randsp").onclick=()=>{
  curSpike=Math.floor(Math.random()*I.n);showSpike(curSpike);};
// Plotly only emits a 3-D click when the cursor is ON a marker; a click on empty space
// is a camera drag and carries no coordinates. So the last HOVERED point is remembered
// and a click that hits nothing commits that instead -- clicking anywhere then always
// resolves to a spike, and in both paths the position is snapped to the nearest
// INSPECTABLE spike, since only a subset carries waveforms.
let lastHover=null, hitAt=0;
function paint(){
  document.querySelectorAll("#atoms button").forEach(b=>
    b.classList.toggle("on",!allmode&&+b.dataset.q===atom));
  document.getElementById("allbtn").classList.toggle("on",allmode);
  document.querySelectorAll("#modes button[data-k]").forEach(b=>
    b.classList.toggle("on",b.dataset.k===mode));
  document.querySelectorAll("#cmodes button").forEach(b=>
    b.classList.toggle("on",b.dataset.k===cmode));
  document.querySelectorAll("#views button").forEach(b=>
    b.classList.toggle("on",b.dataset.k===vmode));
  const y0=+document.getElementById("y0").value,
        y1=+document.getElementById("y1").value,
        cap=+document.getElementById("cap").value;
  const list=allmode?Array.from({length:D.n_atoms},(_,i)=>i):[atom];
  let avail=0, stored=0;
  list.forEach(q=>{avail+=D.atoms[q].n; stored+=D.atoms[q].pts;});
  // Per-point rgba STRINGS are what made this slow, so neither colour mode builds them:
  // amplitude mode hands plotly a numeric array plus a colorscale, and atom mode splits
  // the points into a few uniform-colour traces (one per brightness level, or one per
  // atom when merged). That is what makes a few hundred thousand points interactive.
  // gamma > 1 darkens the mid range. The 2-D rasters use 0.45, but they apply it to
  // SUMMED density; here each point carries its own amplitude, so the same exponent
  // washes almost everything to the bright end.
  const gam=+document.getElementById("gam").value;
  // the filter is a property of the SPIKE, so it applies in every colour mode, not just
  // when colouring by error -- otherwise "show me only the badly fit spikes, coloured by
  // atom" would be impossible
  const e0=+document.getElementById("emin").value/100*255,
        e1=+document.getElementById("emax").value/100*255;
  const NLEV=8, traces=[];
  let kept=0, shown=0;
  const step=Math.max(1, Math.ceil(stored/cap));
  if(cmode==="amp"){
    const T=[],X=[],Y=[],V=[];
    for(const q of list){const c=decAtom(q);
      for(let i=0;i<c.t.length;i+=step){
        if(c.y[i]<y0||c.y[i]>y1||c.e[i]<e0||c.e[i]>e1) continue;
        T.push(c.t[i]);X.push(c.x[i]);Y.push(c.y[i]);
        V.push(Math.pow(c.a[i]/255,gam));shown++;}
      for(let i=0;i<c.t.length;i++)
        if(c.y[i]>=y0&&c.y[i]<=y1&&c.e[i]>=e0&&c.e[i]<=e1) kept++;}
    traces.push({type:"scatter3d",mode:"markers",x:T,y:X,z:Y,hoverinfo:"skip",
      // plotly ships Magma light-to-dark, the reverse of matplotlib's; on a dark
      // background that paints the QUIET points white, so it has to be flipped
      marker:{size:+document.getElementById("ps").value,color:V,colorscale:"Magma",
              reversescale:true,cmin:0,cmax:1,opacity:0.85,showscale:true,
              colorbar:{title:{text:"amplitude",side:"right"},thickness:10,
                        len:.45,x:1.0,tickfont:{color:"#aab"},
                        titlefont:{color:"#aab"}}}});
  } else if(cmode==="err"){
    const T=[],X=[],Y=[],V=[];
    for(const q of list){const c=decAtom(q);
      for(let i=0;i<c.t.length;i+=step){
        if(c.y[i]<y0||c.y[i]>y1||c.e[i]<e0||c.e[i]>e1) continue;
        T.push(c.t[i]);X.push(c.x[i]);Y.push(c.y[i]);
        V.push(c.e[i]/255*D.err_hi);shown++;}
      for(let i=0;i<c.t.length;i++)
        if(c.y[i]>=y0&&c.y[i]<=y1&&c.e[i]>=e0&&c.e[i]<=e1) kept++;}
    traces.push({type:"scatter3d",mode:"markers",x:T,y:X,z:Y,hoverinfo:"skip",
      marker:{size:+document.getElementById("ps").value,color:V,colorscale:"Viridis",
              cmin:e0/255*D.err_hi,cmax:e1/255*D.err_hi,opacity:0.85,showscale:true,
              colorbar:{title:{text:"rel. error",side:"right"},thickness:10,len:.45,
                        x:1.0,tickfont:{color:"#aab"},titlefont:{color:"#aab"}}}});
  } else if(allmode){
    for(const q of list){const c=decAtom(q),col=D.rgb[q];
      const T=[],X=[],Y=[];
      for(let i=0;i<c.t.length;i+=step){
        if(c.y[i]<y0||c.y[i]>y1||c.e[i]<e0||c.e[i]>e1) continue;
        T.push(c.t[i]);X.push(c.x[i]);Y.push(c.y[i]);shown++;}
      for(let i=0;i<c.t.length;i++)
        if(c.y[i]>=y0&&c.y[i]<=y1&&c.e[i]>=e0&&c.e[i]<=e1) kept++;
      if(T.length) traces.push({type:"scatter3d",mode:"markers",x:T,y:X,z:Y,
        hoverinfo:"skip",name:"q"+q,showlegend:false,
        marker:{size:+document.getElementById("ps").value,opacity:0.8,
                color:"rgb("+col[0]+","+col[1]+","+col[2]+")"}});}
  } else {
    const c=decAtom(atom),col=D.rgb[atom];
    const bins=Array.from({length:NLEV},()=>({x:[],y:[],z:[]}));
    for(let i=0;i<c.t.length;i++){
      if(c.y[i]<y0||c.y[i]>y1||c.e[i]<e0||c.e[i]>e1) continue;
      kept++;
      if(i%step) continue;
      const lev=Math.min(NLEV-1,Math.floor(c.q[i]/256*NLEV));
      bins[lev].x.push(c.t[i]);bins[lev].y.push(c.x[i]);bins[lev].z.push(c.y[i]);shown++;}
    bins.forEach((bn,l)=>{if(!bn.x.length)return;
      const b=0.14+0.86*((l+0.5)/NLEV);
      traces.push({type:"scatter3d",mode:"markers",x:bn.x,y:bn.y,z:bn.z,
        hoverinfo:"skip",showlegend:false,
        marker:{size:+document.getElementById("ps").value,opacity:b,
                color:"rgb("+Math.round(col[0]*b)+","+Math.round(col[1]*b)+","
                      +Math.round(col[2]*b)+")"}});});
  }
  const NL=String.fromCharCode(10);   // avoids escape mangling in the template
  document.getElementById("meta").textContent=
    shown.toLocaleString()+" drawn of "+kept.toLocaleString()+" stored in window"+NL+
    (allmode?("ALL "+D.n_atoms+" atoms · "+avail.toLocaleString()+" sources total")
            :("atom "+atom+" · "+D.atoms[atom].n.toLocaleString()+" sources ("
              +D.atoms[atom].pct+"%)"))+NL+
    (cmode==="amp"?("colour = source amplitude (magma, gamma "+gam+")")
     :cmode==="err"?("colour = per-spike relative error "
                     +(e0/255*D.err_hi).toFixed(3)+" .. "
                     +(e1/255*D.err_hi).toFixed(3))
     :"brightness = q (contribution share)")+NL+
    "error filter "+(e0/255*D.err_hi).toFixed(3)+" .. "
    +(e1/255*D.err_hi).toFixed(3)+" (max "+D.err_hi.toFixed(3)+")";
  // The 4,000 spikes that carry waveforms are drawn as a white overlay so they can be
  // AIMED AT: plotly only emits a 3-D click when the cursor is on a marker, so an
  // invisible pick target would make the inspector feel broken.
  if(showIns){
    const T=[],X=[],Y=[],ys=IY[mode];
    for(let i=0;i<I.n;i++){
      if(ys[i]<y0||ys[i]>y1) continue;
      const e=I.err[i]/D.err_hi*255; if(e<e0||e>e1) continue;
      T.push(IT[i]);X.push(IX[i]);Y.push(ys[i]);}
    traces.push({type:"scatter3d",mode:"markers",x:T,y:X,z:Y,name:"inspectable",
      showlegend:false,hovertemplate:"click to inspect<extra></extra>",
      marker:{size:+document.getElementById("ps").value+1.0,
              color:"rgba(255,255,255,0.30)",line:{width:0}}});
  }
  Plotly.react("plot",traces,
   {paper_bgcolor:"#0f1115",plot_bgcolor:"#0f1115",margin:{l:0,r:0,t:0,b:0},
    scene:{aspectmode:"manual",
      aspectratio:{x:+document.getElementById("ax").value,
                   y:+document.getElementById("ay").value,
                   z:+document.getElementById("az").value},
      xaxis:{title:"time (s)",range:D.tlim,color:"#aab",gridcolor:"#2a2f39",
             backgroundcolor:"#14171d",showbackground:true},
      yaxis:{title:"lateral x (µm)",range:D.xlim,color:"#aab",gridcolor:"#2a2f39",
             backgroundcolor:"#14171d",showbackground:true},
      zaxis:{title:"depth y (µm)",range:[y0,y1],color:"#aab",gridcolor:"#2a2f39",
             backgroundcolor:"#14171d",showbackground:true},
      camera:camera()}},
   {responsive:true,displaylogo:false});
  wave();
}
["ax","ay","az","ps","y0","y1","cap","gam","emin","emax"].forEach(id=>{
  const e=document.getElementById(id);
  const lab={ax:"lax",ay:"lay",az:"laz",ps:"lps",cap:"lcap",gam:"lgam",
             emin:"lemin",emax:"lemax"}[id];
  const fmt=v=>(id==="emin"||id==="emax")
    ? (v/100*D.err_hi).toFixed(3) : v;
  const upd=()=>{if(lab)document.getElementById(lab).textContent=fmt(e.value);
    paint();};
  e.oninput=upd; if(lab)document.getElementById(lab).textContent=fmt(e.value);});
// Plotly only emits a 3-D click when the cursor sits exactly on a marker, which made
// the inspector feel dead. Rather than depend on its hit test, every inspectable spike
// is projected to SCREEN space with plotly's OWN camera matrices and the nearest in
// pixels wins -- so a click anywhere in the scene snaps to a spike.
// gl-matrix is column-major, hence the 4+i / 8+i / 12+i indexing.
function mv(m,v){const o=[0,0,0,0];
  for(let i=0;i<4;i++) o[i]=m[i]*v[0]+m[4+i]*v[1]+m[8+i]*v[2]+m[12+i]*v[3];
  return o;}
function projector(){
  const sc=document.getElementById("plot")._fullLayout.scene._scene;
  const g=sc.glplot, cp=g.cameraParams, ds=sc.dataScale;
  const r=g.canvas.getBoundingClientRect();
  return {rect:r, f:(t,x,y)=>{
    let v=[t*ds[0],x*ds[1],y*ds[2],1];
    v=mv(cp.model,v); v=mv(cp.view,v); v=mv(cp.projection,v);
    if(Math.abs(v[3])<1e-9) return null;
    return [(v[0]/v[3]*0.5+0.5)*r.width,(1-(v[1]/v[3]*0.5+0.5))*r.height];}};}
function nearestOnScreen(clientX,clientY){
  const P=projector(), ys=IY[mode];
  const px=clientX-P.rect.left, py=clientY-P.rect.top;
  const y0=+document.getElementById("y0").value,
        y1=+document.getElementById("y1").value,
        e0=+document.getElementById("emin").value/100*255,
        e1=+document.getElementById("emax").value/100*255;
  let best=-1,bd=Infinity;
  for(let i=0;i<I.n;i++){
    if(ys[i]<y0||ys[i]>y1) continue;             // only consider what is actually shown
    const e=I.err[i]/D.err_hi*255; if(e<e0||e>e1) continue;
    const q=P.f(IT[i],IX[i],ys[i]); if(!q) continue;
    const a=q[0]-px,b=q[1]-py,d=a*a+b*b;
    if(d<bd){bd=d;best=i;}}
  return best;}
function hookClick(){
  const gd=document.getElementById("plot");
  // a drag is a camera move, not a pick: only count it if the pointer barely moved
  let dx0=0,dy0=0;
  gd.addEventListener("pointerdown",e=>{dx0=e.clientX;dy0=e.clientY;});
  gd.addEventListener("pointerup",e=>{
    if(Math.abs(e.clientX-dx0)+Math.abs(e.clientY-dy0)>6) return;
    const i=nearestOnScreen(e.clientX,e.clientY);
    if(i>=0) showSpike(i);
    else document.getElementById("inspmeta").textContent="no inspectable spike in view";
  });}
paint(); hookClick();
</script>
"""


def inspect_pack(state, rec, rows: np.ndarray, summary: dict, codebook: dict):
    """Observed and reconstructed 10-channel waveforms for a subset of spikes.

    The reconstruction is built with the SAME code path the reconstruction panels use
    (`viz._source_waves`, which reapplies the learned time shift), so what the inspector
    draws is what the model actually fitted -- not a re-derivation that could drift out
    of step with the fitter.

    Both traces are quantised to int8 against ONE per-spike scale, so the overlay keeps
    its relative amplitudes; 10 x 90 x 2 int8 is 1.8 kB per spike, which is what makes
    embedding a few thousand of them affordable.
    """
    import torch
    from spiketensor.fit import load_batch
    from spiketensor.source_figures import (_run_footprints, _source_waves,
                                           selected_footprints)
    # the learned dictionary is rebuilt exactly as the row's own panels rebuild it,
    # from the run's codebook, rather than reconstructed from guessed metadata
    dictionary = _run_footprints(summary, codebook, rec)
    spike = np.asarray(state["spike_index"])[rows]
    anchor_shift = np.asarray(state["anchor_shift"], np.float32)
    off = rec.channel_offsets().astype(np.float32) - anchor_shift[:, None]
    Y, batch_off = load_batch(rec, spike, off, "cpu")
    idx = torch.as_tensor(np.asarray(state["source_index"])[rows], dtype=torch.long)
    conf = torch.as_tensor(dictionary.cfg_id_by_channel[rec.spike_channels[spike]],
                           dtype=torch.long)
    coef = torch.as_tensor(np.asarray(state["source_coeff"])[rows], dtype=torch.float32)
    H = selected_footprints(torch.as_tensor(dictionary.footprints), conf, idx)
    waves = _source_waves(state, rows, coef)
    fit = (H.transpose(1, 2).unsqueeze(-1) * waves.unsqueeze(2)).sum(1)
    Yn, Fn = Y.numpy(), fit.numpy()
    scale = np.maximum(np.abs(Yn).max((1, 2)), np.abs(Fn).max((1, 2)))
    scale = np.maximum(scale, 1e-9)
    q = lambda A: np.clip(np.rint(A / scale[:, None, None] * 127), -127,
                          127).astype(np.int8)
    sse = np.square(Yn - Fn).sum((1, 2))
    return {"obs": q(Yn), "fit": q(Fn), "scale": scale.astype(np.float32),
            "sse": sse.astype(np.float32),
            # channel positions relative to the spike's anchor, so the inspector can lay
            # the traces out in the probe's real geometry instead of a flat stack
            "off": batch_off.numpy().astype(np.float32)}


def full_probe_pack(state, rec, rows, summary, codebook, raw_bin, raw_chans):
    """Observed and MODEL-EXTRAPOLATED waveforms on all 384 channels.

    The fit only ever saw the 10-channel neighbourhood, but the spatial footprint is
    analytic -- g(r) = sigma / sqrt(|r - mu|^2 + z^2 + sigma^2) -- so it can be evaluated
    anywhere on the probe. The one subtlety is normalisation: the fitted footprints are
    unit-norm OVER THE 10 FIT CHANNELS, so the extrapolation must divide by that same
    10-channel norm, not by its own. Verified: re-deriving the 10-channel footprints this
    way matches the dictionary to 6e-8, and the 384-channel version agrees with them on
    those 10 channels to 9e-8.

    Units are tied together by a single scalar fitted on the 10 FIT CHANNELS ONLY, so the
    other 374 channels stay an honest out-of-sample test of the model's spatial decay
    rather than being fitted to match.
    """
    import torch
    from scipy.signal import butter, filtfilt
    from spiketensor.source_figures import _run_footprints, _source_waves
    dic = _run_footprints(summary, codebook, rec)
    mu = np.asarray(dic.candidate_pos, np.float32)
    sig = np.asarray([float(pr[1][0]) for pr in
                      codebook["dictionary_metadata"]["profiles"]], np.float32)
    shift = np.asarray(state["anchor_shift"], np.float32)
    loc = rec.channel_locations[:, :2].astype(np.float32)
    spike = np.asarray(state["spike_index"])[rows]
    src = np.asarray(state["source_index"])[rows]
    coef = torch.as_tensor(np.asarray(state["source_coeff"])[rows], dtype=torch.float32)
    waves = _source_waves(state, rows, coef).numpy()            # (B,R,T) amp + shift
    T = waves.shape[2]

    bb, aa = butter(3, 300.0 / (rec.fs / 2.0), btype="high")
    PAD = 300
    nsamp = raw_bin.stat().st_size // (2 * raw_chans)
    mm = np.memmap(raw_bin, dtype=np.int16, mode="r", shape=(int(nsamp), raw_chans))

    obs = np.zeros((len(rows), 384, T), np.float32)
    fit = np.zeros((len(rows), 384, T), np.float32)
    for j in range(len(rows)):
        k = int(rec.spike_channels[spike[j]])
        anchor = rec.anchors[k, :2] + shift[k]
        off_all = loc - anchor[None, :]
        off_10 = rec.channel_offsets()[k] - shift[k][None, :]

        def kern(off):
            d2 = ((off[None, :, 0] - mu[:, None, 0]) ** 2
                  + (off[None, :, 1] - mu[:, None, 1]) ** 2)
            return sig[:, None] / np.sqrt(d2 + mu[:, None, 2] ** 2 + sig[:, None] ** 2)

        n10 = np.linalg.norm(kern(off_10), axis=1, keepdims=True)
        foot_all = kern(off_all) / n10                       # (512, 384), fit's scale
        for r in range(src.shape[1]):
            if src[j, r] < 0:
                continue
            fit[j] += np.outer(foot_all[int(src[j, r])], waves[j, r])

        smp = int(rec.spike_times[spike[j]])
        lo = max(smp - 30 - PAD, 0)
        w = mm[lo:lo + 90 + 2 * PAD, :384].astype(np.float32)
        f = filtfilt(bb, aa, w, axis=0)
        f -= np.median(f, axis=1, keepdims=True)
        o = int(smp) - 30 - lo
        obs[j] = f[o:o + T].T
        # one scalar, fitted on the FIT channels only
        idx10 = rec.channel_lookup[k]
        a_num = float((obs[j][idx10] * fit[j][idx10]).sum())
        a_den = float((obs[j][idx10] ** 2).sum()) + 1e-12
        obs[j] *= a_num / a_den
    scale = np.maximum(np.maximum(np.abs(obs).max((1, 2)), np.abs(fit).max((1, 2))), 1e-9)
    q = lambda A: np.clip(np.rint(A / scale[:, None, None] * 127), -127, 127).astype(np.int8)
    fitch = np.stack([rec.channel_lookup[int(rec.channel_lookup.shape[0] and
                                             rec.spike_channels[sp])] for sp in spike])
    return {"obs": q(obs), "fit": q(fit), "fit_chans": fitch.astype(np.int16), "T": T}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path,
                    default=REPO / "zncc/runs/onehot_prior/multipole_prior2_shift_M64_R4.npz")
    ap.add_argument("--tag", default="prior2_shift_M64_R4")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/onehot_prior")
    ap.add_argument("--out", type=Path,
                    default=REPO / "zncc/figures/atom_scatter3d")
    ap.add_argument("--raw-bin", type=Path,
                    default=REPO / "data/dataset1/p1_g0_t0.imec0.ap.bin")
    ap.add_argument("--raw-chans", type=int, default=385)
    ap.add_argument("--n-full", type=int, default=200,
                    help="spikes carrying FULL 384-channel observed+model waveforms "
                         "(69 kB each, so this is deliberately a small subset)")
    ap.add_argument("--n-inspect", type=int, default=8000,
                    help="spikes whose 10-channel waveforms are embedded for the "
                         "click inspector (1.8 kB each)")
    ap.add_argument("--n-per-atom", type=int, default=50000,
                    help="sources STORED per atom; the draw cap in the UI is separate")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rec = D.load("np1")
    with np.load(a.state, mmap_mode="r") as z:
        act = np.asarray(z["source_index"]) >= 0
        parent, slot = np.nonzero(act)
        pos = np.asarray(z["source_pos"])[parent, slot]
        if "source_temporal_atom" in z.files:
            atom = np.asarray(z["source_temporal_atom"])[parent, slot].astype(np.int64)
        else:
            # models without a one-hot shape use free coefficient vectors; the closest
            # equivalent grouping -- and the one every other panel uses -- is the
            # DOMINANT basis term, argmax|v|
            key = ("source_coeff" if "source_coeff" in z.files else
                   "shape_feature" if "shape_feature" in z.files else None)
            atom = (np.abs(np.asarray(z[key])[parent, slot]).argmax(1).astype(np.int64)
                    if key else np.zeros(len(parent), np.int64))
        qv = np.asarray(z["source_weight"])[parent, slot].astype(np.float32)
        amp = np.asarray(z["source_amp"])[parent, slot].astype(np.float32)
        sse = np.asarray(z["sse"]).astype(np.float64)
        spike = np.asarray(z["spike_index"])[parent]
        omega = np.asarray(z["omega"])
        protos = (np.asarray(z["prototypes"]) if "prototypes" in z.files
                  else np.zeros_like(omega[:1]))
        assign = (np.asarray(z["atom_prototype"]) if "atom_prototype" in z.files
                  else np.zeros(len(omega), int))
    M = len(omega)
    t_smp = rec.spike_times[spike]
    t = (t_smp / rec.fs).astype(np.float32)
    y = {"none": pos[:, 1].astype(np.float32)}
    have_corr = (a.figs / a.tag / "dredge_real.npz").exists()
    for mode, key in (("real-rigid", "drr"), ("real-nonrigid", "drn")):
        # a model with no canonical DREDge run simply offers the uncorrected view
        y[key] = ((pos[:, 1] - correction(a.figs, a.tag, mode, t_smp, rec.fs,
                                          y=pos[:, 1])).astype(np.float32)
                  if have_corr else y["none"])
    palette = shuffled_palette(M)
    counts = np.bincount(atom, minlength=M)
    rng = np.random.default_rng(a.seed)
    amp_hi = float(np.percentile(amp, 99.5))
    # Per-SPIKE reconstruction error, carried by each of that spike's sources. Relative
    # (unexplained energy fraction), not raw MSE: raw MSE tracks amplitude, so colouring
    # by it would mostly redraw the amplitude map -- the same reason the MSE rasters use
    # the relative form.
    from spiketensor.spike_error import spike_energy
    energy = spike_energy(rec)
    err_spike = (sse / np.maximum(energy[np.asarray(np.load(a.state, mmap_mode="r")
                                                   ["spike_index"])], 1e-12))
    err = err_spike[parent].astype(np.float32)
    err_hi = float(np.percentile(err, 99.5))
    atoms = []
    for q in range(M):
        idx = np.flatnonzero(atom == q)
        if len(idx) > a.n_per_atom:
            idx = np.sort(rng.choice(idx, a.n_per_atom, replace=False))
        qq = qv[idx]
        hi = float(np.percentile(qq, 99)) if len(qq) > 10 else 1.0
        atoms.append({
            "n": int(counts[q]),
            "pct": round(100 * counts[q] / max(counts.sum(), 1), 2),
            "pts": int(len(idx)),
            "t": _b64(_q16(t[idx], *TLIM)), "x": _b64(_q16(pos[idx, 0], *XLIM)),
            "y_none": _b64(_q16(y["none"][idx], *YLIM)),
            "y_drr": _b64(_q16(y["drr"][idx], *YLIM)),
            "y_drn": _b64(_q16(y["drn"][idx], *YLIM)),
            "q": _b64((np.clip(qq / max(hi, 1e-6), 0, 1) * 255).astype(np.uint8)),
            # amplitude is quantised against ONE global scale so the magma colouring is
            # comparable between atoms, unlike q which is normalised per atom
            "a": _b64((np.clip(amp[idx] / amp_hi, 0, 1) * 255).astype(np.uint8)),
            "e": _b64((np.clip(err[idx] / err_hi, 0, 1) * 255).astype(np.uint8)),
        })
    # ---- inspectable subset: a few thousand spikes whose waveforms travel with the
    # file, so a click can show the actual fit rather than just a position ----
    n_ins = min(a.n_inspect, len(np.unique(parent)))
    uniq = np.unique(parent)
    ins_rows = np.sort(rng.choice(uniq, n_ins, replace=False))
    with np.load(a.state, mmap_mode="r") as zz:
        keys = ("spike_index", "source_index", "source_amp", "source_pos",
                "candidate_pos", "anchor_shift", "omega", "sse")
        st = {k: np.asarray(zz[k]) for k in keys}
        for k in ("source_shift", "source_temporal_atom", "source_coeff",
                  "shape_feature"):
            if k in zz.files:
                st[k] = np.asarray(zz[k])
    if "source_coeff" not in st and "shape_feature" in st:
        st["source_coeff"] = st["shape_feature"]        # C5 is deliberately nonseparable
    if "source_temporal_atom" not in st:
        st["source_temporal_atom"] = (
            np.abs(st["source_coeff"]).argmax(2).astype(np.int16) if "source_coeff" in st
            else np.zeros(st["source_index"].shape, np.int16))
    import json as _json
    _sum = _json.loads((a.state.parent / f"summary_{a.tag}.json").read_text())
    import torch as _torch
    _cb = _torch.load(a.state.parent / f"codebook_{a.tag}.pt", map_location="cpu",
                      weights_only=False)
    _cb["anchor_shift"] = np.asarray(st["anchor_shift"])
    try:
        pack = inspect_pack(st, rec, ins_rows, _sum, _cb)
    except Exception as exc:
        # some studies' summaries point their dictionary tag at THEMSELVES rather than at
        # a lattice baseline, so the footprints cannot be rebuilt here. The cloud and the
        # all-channel amplitude view need none of that, so the viewer is still worth
        # writing; the waveform panels are disabled and say why.
        print(f"  inspector unavailable ({type(exc).__name__}: {exc}); "
              f"building without the waveform panels", flush=True)
        pack = None
    dom = np.argmax(st["source_amp"][ins_rows], 1)
    ipos = st["source_pos"][ins_rows, dom]
    it = (rec.spike_times[st["spike_index"][ins_rows]] / rec.fs).astype(np.float32)
    iy = {"none": ipos[:, 1].astype(np.float32)}
    for md, key in (("real-rigid", "drr"), ("real-nonrigid", "drn")):
        iy[key] = ((ipos[:, 1] - correction(
            a.figs, a.tag, md, rec.spike_times[st["spike_index"][ins_rows]], rec.fs,
            y=ipos[:, 1])).astype(np.float32) if have_corr else iy["none"])
    ierr = (st["sse"][ins_rows] / np.maximum(energy[st["spike_index"][ins_rows]], 1e-12))
    iat = st["source_temporal_atom"][ins_rows]
    iact = st["source_index"][ins_rows] >= 0
    # ---- all-384-channel amplitude profile for the same spikes ----
    # Full 384 x 90 waveforms would be 276 MB for 8,000 spikes, so what travels is the
    # per-channel peak-to-peak: 384 bytes per spike. That is the quantity that shows a
    # spike's footprint MOVING ALONG THE PROBE as it drifts, which is the point.
    ptp = None
    if a.raw_bin.exists():
        # The AP binary is UNFILTERED, so a raw peak-to-peak is dominated by LFP and
        # picks the wrong channel entirely -- on one test spike it peaked at y=2540 µm
        # while the fitted source sat at 1596 µm. High-passing at 300 Hz and common-
        # median-referencing recovers the right channel (158, y=1580, matching both the
        # stored peak channel and the fit). The per-channel noise floor is then removed
        # so the footprint stands out instead of sitting on a 47%-of-max background.
        from scipy.signal import butter, filtfilt
        bb, aa = butter(3, 300.0 / (rec.fs / 2.0), btype="high")
        PAD = 300                                  # room for the filter transient
        nsamp = a.raw_bin.stat().st_size // (2 * a.raw_chans)
        mm = np.memmap(a.raw_bin, dtype=np.int16, mode="r",
                       shape=(int(nsamp), a.raw_chans))
        ptp = np.empty((n_ins, 384), np.float32)
        st_smp = rec.spike_times[st["spike_index"][ins_rows]].astype(np.int64)
        t_raw = time.perf_counter()
        for j, smp in enumerate(st_smp):        # window verified against rec.waveforms
            lo = max(int(smp) - 30 - PAD, 0)
            w = mm[lo:lo + 90 + 2 * PAD, :384].astype(np.float32)
            f = filtfilt(bb, aa, w, axis=0)
            f -= np.median(f, axis=1, keepdims=True)
            off0 = int(smp) - 30 - lo
            pk = np.ptp(f[off0:off0 + 90], axis=0)
            ptp[j] = np.maximum(pk - np.median(pk), 0.0)
            if j % 2000 == 0:
                print(f"    raw profiles {j:,}/{n_ins:,} "
                      f"{time.perf_counter() - t_raw:.0f}s", flush=True)
        pscale = np.maximum(ptp.max(1), 1e-6)
        ptp_q = np.clip(ptp / pscale[:, None] * 255, 0, 255).astype(np.uint8)
        print(f"  all-channel profiles: {n_ins:,} spikes x 384 channels", flush=True)
    # ---- full 384-channel observed + extrapolated model, for a smaller subset ----
    full = None
    # the analytic extrapolation is only defined for the learned monopole dictionary;
    # adapter-based models (scale mixture, generalized g) build footprints their own way
    _profs = (_cb.get("dictionary_metadata") or {}).get("profiles")
    # one (kernel, sigma) per candidate is the LEARNED layout; a fixed lattice indexes
    # candidates as site*S+profile, where this closed-form extrapolation does not apply
    can_extrapolate = bool(_profs) and not _cb.get("adapter") and \
        len(_profs) == len(np.asarray(_cb["candidate_pos"]))
    if a.raw_bin.exists() and a.n_full > 0 and can_extrapolate and pack is not None:
        fr = ins_rows[np.linspace(0, len(ins_rows) - 1,
                                  min(a.n_full, len(ins_rows))).astype(int)]
        fp = full_probe_pack(st, rec, fr, _sum, _cb, a.raw_bin, a.raw_chans)
        f_index = {int(v): i for i, v in enumerate(fr)}
        full = {"n": int(len(fr)), "T": int(fp["T"]),
                "obs": _b64(fp["obs"]), "fit": _b64(fp["fit"]),
                "fit_chans": _b64(fp["fit_chans"]),
                # map an inspectable-spike index -> its slot here, or -1
                "slot": [f_index.get(int(v), -1) for v in ins_rows]}
        print(f"  full-probe pairs: {len(fr):,} spikes x 384 channels", flush=True)
    inspect = {
        "n": int(n_ins),
        "T": (int(pack["obs"].shape[2]) if pack is not None else 0),
        "C": (int(pack["obs"].shape[1]) if pack is not None else 0),
        "t": _b64(_q16(it, *TLIM)), "x": _b64(_q16(ipos[:, 0], *XLIM)),
        "y_none": _b64(_q16(iy["none"], *YLIM)),
        "y_drr": _b64(_q16(iy["drr"], *YLIM)), "y_drn": _b64(_q16(iy["drn"], *YLIM)),
        "obs": (_b64(pack["obs"]) if pack is not None else None),
        "fit": (_b64(pack["fit"]) if pack is not None else None),
        "off": (_b64(np.clip(np.rint(pack["off"] * 4), -32767, 32767).astype(np.int16))
                if pack is not None else None),
        "off_scale": 0.25,
        "err": [round(float(v), 4) for v in ierr],
        "atoms": [[int(q) for q, ok in zip(row, okr) if ok]
                  for row, okr in zip(iat, iact)],
        "spike": [int(v) for v in st["spike_index"][ins_rows]],
        "ptp": (_b64(ptp_q) if ptp is not None else None),
        "chan": _b64(np.clip(np.rint(rec.channel_locations[:, :2] * 4), -32767,
                             32767).astype(np.int16)),
        "chan_scale": 0.25,
        "src_y": [[round(float(v), 1) for v, ok in zip(row, okr) if ok]
                  for row, okr in zip(st["source_pos"][ins_rows][:, :, 1], iact)],
        "full": full,
    }
    data = {
        "n_atoms": M, "tlim": list(TLIM), "xlim": list(XLIM), "ylim": list(YLIM),
        "atoms": atoms,
        "waves": [np.round(omega[q], 5).tolist() for q in range(M)],
        "protos": [np.round(protos[assign[q]], 5).tolist() for q in range(M)],
        "colors": ["#%02x%02x%02x" % tuple(int(255 * c) for c in palette[q][:3])
                   for q in range(M)],
        "rgb": [[int(255 * c) for c in palette[q][:3]] for q in range(M)],
        "amp_hi": round(amp_hi, 5), "err_hi": round(err_hi, 5),
        "inspect": inspect,
    }
    import plotly.offline as po
    html = (HTML.replace("__TITLE__", a.tag)
                .replace("__PLOTLYJS__", po.get_plotlyjs())
                .replace("__DATA__", json.dumps(data)))
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / f"{a.tag}_viewer.html"
    target.write_text(html)
    print(f"wrote {target}  ({target.stat().st_size / 1e6:.1f} MB, "
          f"{M} atoms x {a.n_per_atom:,} pts)")


if __name__ == "__main__":
    main()
