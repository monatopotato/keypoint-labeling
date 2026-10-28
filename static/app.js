/* PoseFrame Studio — self-contained browser-side COCO-17 editor. */
'use strict';

const NAMES = [
  'nose','left_eye','right_eye','left_ear','right_ear',
  'left_shoulder','right_shoulder','left_elbow','right_elbow',
  'left_wrist','right_wrist','left_hip','right_hip',
  'left_knee','right_knee','left_ankle','right_ankle'
];
const EDGES_1BASED = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]];
const EDGES = EDGES_1BASED.map(([a,b]) => [a-1,b-1]);
const COLORS = ['#62d5c9','#efb37a','#b5a1ff','#80c3ff','#f089a5','#ddce81'];
const TEMPLATE = [
  [.50,.21],[.515,.20],[.485,.20],[.54,.22],[.46,.22],
  [.58,.32],[.42,.32],[.62,.43],[.38,.43],[.65,.56],[.35,.56],
  [.55,.55],[.45,.55],[.56,.71],[.44,.71],[.57,.87],[.43,.87]
];
const S = {
  frames: [], folder: '', mode: '', index: -1,
  people: [], selectedPerson: -1, selectedPoint: 0,
  width: 0, height: 0, dirty: false, revision: 0,
  navigating: false, pendingIndex: null, busy: false,
  batch: false, batchStop: false,
  zoom: 1, baseScale: 1, filter: 'all',
  showSkeleton: true, showNumbers: false, placeMode: false,
  undo: [], redo: [], dragging: null, savePromise: null, importing: false,
};
const el = id => document.getElementById(id);
const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
const deep = object => JSON.parse(JSON.stringify(object));
const frame = () => S.frames[S.index] || null;
const selectedPerson = () => S.people[S.selectedPerson] || null;
const nice = name => name.replaceAll('_',' ').replace(/\b\w/g, c=>c.toUpperCase());
const imageUrl = name => '/api/image/' + encodeURIComponent(name);
const annUrl = name => '/api/annotations/' + encodeURIComponent(name);

async function api(url, options={}) {
  const response = await fetch(url, options);
  let result = {};
  try {result = await response.json()} catch (e) { /* handle non-JSON errors */ }
  if (!response.ok) {
    const error = new Error(result.error || `Server error (${response.status})`);
    error.status = response.status; throw error;
  }
  return result;
}
function jsonOptions(method, data) {return {method, headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)}}
function toast(message, error=false) {
  const node=document.createElement('div');
  node.className='toast'+(error?' error':'');
  node.textContent=message;
  el('toastHost').appendChild(node);
  setTimeout(()=>node.remove(),Math.max(3400,Math.min(8000,message.length*70)));
}
function setDirty() {
  S.dirty=true; S.revision++;
  showSaveStatus();
}
function showSaveStatus(saving=false) {
  const box=el('saveIndicator'), span=el('saveIndicatorText');
  box.classList.toggle('unsaved', S.dirty || saving);
  span.textContent=saving?'Saving...':S.dirty?'Unsaved changes':'All changes saved';
}
function updateButtons() {
  const has=S.index>=0 && !!frame();
  el('batchAutoBtn').disabled=!has || S.busy || S.importing || S.navigating || !S.frames.some(f=>f.status==='unlabeled');
  for(const id of ['newProjectBtn','openFolderSmall','modelSelect','pointThreshold']) el(id).disabled=S.busy;
  document.querySelector('.right-scroll').inert=S.batch;
  for (const id of ['saveBtn','addPersonBtn','inspectorAddBtn','autoBtn','zoomIn','zoomOut','zoomReset']) el(id).disabled=!has || S.busy || S.importing;
  for (const id of ['prevBtn','footPrev']) el(id).disabled=!has || S.index===0 || S.busy;
  for (const id of ['nextBtn','footNext']) el(id).disabled=!has || S.index>=S.frames.length-1 || S.busy;
  el('deletePersonBtn').disabled=!selectedPerson() || S.busy;
  el('undoBtn').disabled=S.undo.length===0 || S.busy;
  el('redoBtn').disabled=S.redo.length===0 || S.busy;
  el('placeBtn').disabled=!selectedPerson() || S.busy;
  el('frameRange').disabled=!has || S.busy;
}
// The frame browser is windowed, so thousands of video frames remain responsive.
const ROW_HEIGHT = 64;
let listRenderPending = false;
function renderVirtualRows() {
  const list=el('frameList'), view=S.visibleRows||[];
  if(!view.length){
    const p=document.createElement('div');p.className='frame-list-empty';
    p.textContent=S.frames.length?'No frames match your filter.':'No frames loaded yet.';
    list.replaceChildren(p);return;
  }
  const start=clamp(Math.floor(list.scrollTop/ROW_HEIGHT)-6,0,view.length);
  const end=Math.min(view.length,start+Math.ceil(list.clientHeight/ROW_HEIGHT)+18);
  const fragment=document.createDocumentFragment();
  if(start){const spacer=document.createElement('div');spacer.className='virtual-spacer';spacer.style.height=(start*ROW_HEIGHT)+'px';fragment.append(spacer)}
  for(let n=start;n<end;n++){
    const {f,i}=view[n];
    const button=document.createElement('button');
    button.className='frame-row'+(i===S.index?' active':'');
    button.dataset.index=String(i);
    button.title=f.name+(f.auto_note?' — '+f.auto_note:'');
    const num=document.createElement('span');num.className='frame-number';num.textContent=String(i+1).padStart(3,'0');
    const text=document.createElement('span');text.className='frame-info';
    const label=document.createElement('span');label.className='frame-name';label.textContent=f.name;
    const sub=document.createElement('span');sub.className='frame-sub';
    sub.textContent=f.status==='unlabeled'?(f.auto_note||'Not annotated'):f.status==='reviewed_empty'?'Reviewed · no people':`${f.people} ${f.people===1?'person':'people'} labeled`;
    text.append(label,sub);
    const dot=document.createElement('span');dot.className='frame-status '+f.status;
    button.append(num,text,dot);fragment.append(button);
  }
  if(end<view.length){const spacer=document.createElement('div');spacer.className='virtual-spacer';spacer.style.height=((view.length-end)*ROW_HEIGHT)+'px';fragment.append(spacer)}
  list.replaceChildren(fragment);
}
function renderFrameList(resetScroll=false) {
  el('frameCount').textContent=S.frames.length.toLocaleString();
  el('unlabeledCount').textContent=S.frames.filter(f=>f.status==='unlabeled').length.toLocaleString();
  const search=el('frameSearch').value.toLowerCase().trim();
  S.visibleRows=S.frames.map((f,i)=>({f,i})).filter(({f})=>{
    if(search && !f.name.toLowerCase().includes(search))return false;
    if(S.filter==='labeled')return f.status!=='unlabeled';
    if(S.filter==='unlabeled')return f.status==='unlabeled';
    return true;
  });
  if(resetScroll)el('frameList').scrollTop=0;
  renderVirtualRows();
  const done=S.frames.filter(f=>f.status!=='unlabeled').length;
  el('progressText').textContent=`${done} / ${S.frames.length}`;
  el('progressFill').style.width=(S.frames.length?done/S.frames.length*100:0)+'%';
}
function scrollActiveIntoView() {
  const list=el('frameList');
  const position=(S.visibleRows||[]).findIndex(({i})=>i===S.index);
  if(position<0)return;
  const top=position*ROW_HEIGHT;
  if(top<list.scrollTop || top+ROW_HEIGHT>list.scrollTop+list.clientHeight){
    list.scrollTop=Math.max(0,top-list.clientHeight/2+ROW_HEIGHT/2);
  }
  renderVirtualRows();
}
function renderHeader() {
  const f=frame();
  el('currentFilename').textContent=f?.name||'Select a frame to begin';
  el('framePosition').textContent=f?`FRAME ${String(S.index+1).padStart(3,'0')} OF ${String(S.frames.length).padStart(3,'0')}`:'NO FRAME';
  el('projectTitle').textContent=S.folder?(S.folder.split(/[\\/]/).filter(Boolean).pop() || S.folder):'No project loaded';
  el('footCounter').innerHTML=f?`${S.index+1} <span>/</span> ${S.frames.length}`:'0 <span>/</span> 0';
  el('frameRange').max=String(Math.max(0,S.frames.length-1));
  el('frameRange').value=String(Math.max(0,S.index));
  el('timelineTime').textContent=f?extractTime(f.name):'—';
  el('stageMeta').textContent=f?`${S.width} × ${S.height} PX`:'IMAGE VIEWER';
  el('coordinateMeta').textContent='COCO · ORIGINAL PIXELS';
  el('emptyStage').classList.toggle('hidden',!!f);
  el('surface').classList.toggle('hidden',!f);
  el('stageHint').classList.toggle('hidden',!f);
  el('annotationSummary').textContent=S.people.length===0?'No keypoints placed':`${S.people.length} ${S.people.length===1?'person':'people'} · ${S.people.reduce((n,p)=>n+p.keypoints.filter(q=>q[2]>0).length,0)} labeled joints`;
  updateButtons();
}
function extractTime(name) {
  let m= name.match(/\d{4}[-_.]\d{1,2}[-_.]\d{1,2}[-_.](\d{1,2})[-_.](\d{1,2})[-_.](\d{1,2})/);
  return m?`${m[1].padStart(2,'0')}:${m[2].padStart(2,'0')}:${m[3].padStart(2,'0')}`:name.replace(/\.[^.]+$/,'').slice(-21);
}
async function loadProject() {
  const result=await api('/api/project');
  await adoptProject(result);
}
async function adoptProject(result) {
  S.frames=result.frames||[]; S.folder=result.folder||''; S.mode=result.mode||'';
  S.index=-1;S.people=[];S.selectedPerson=-1;S.selectedPoint=0;S.dirty=false;S.revision=0;S.undo=[];S.redo=[];S.pendingIndex=null;
  renderFrameList();renderHeader();renderPeople();renderOverlay();showSaveStatus();
  if (S.frames.length) await navigateTo(0);
  else openModal();
}
async function saveFrame(force=false) {
  if (S.index<0 || (!force && !S.dirty)) return;
  if (S.savePromise) {await S.savePromise;if (!force && !S.dirty) return;}
  const f=frame();const revision=S.revision;
  const payload={width:S.width,height:S.height,people:deep(S.people)};
  const promise=(async()=>{
    showSaveStatus(true);
    const result=await api(annUrl(f.name),jsonOptions('PUT',payload));
    f.status=result.status;f.people=result.people;f.auto_note='';
    if (revision===S.revision) S.dirty=false;
    renderFrameList();showSaveStatus();
  })();
  S.savePromise=promise;
  try {await promise;} finally {S.savePromise=null;showSaveStatus();}
}
async function requestNavigation(index) {
  index=clamp(index,0,Math.max(0,S.frames.length-1));
  if (!S.frames.length || S.busy) return;
  if (S.navigating) {S.pendingIndex=index;return;}
  await navigateTo(index);
}
async function navigateTo(index) {
  if (index<0 || index>=S.frames.length) return;
  if (S.navigating) {S.pendingIndex=index;return;}
  if (index===S.index) return;
  S.navigating=true;
  try {
    if (S.dirty) {
      // Ensure edits made while a save was in flight get persisted too.
      for(let n=0;n<3 && S.dirty;n++) await saveFrame();
      if(S.dirty) throw new Error('Changes are still saving; frame navigation was stopped.');
    }
    const name=S.frames[index].name;
    const [anns, loadedImg]=await Promise.all([api(annUrl(name)), preloadImage(imageUrl(name))]);
    S.dragging=null;S.index=index;S.people=deep(anns.people||[]);
    S.width=loadedImg.naturalWidth;S.height=loadedImg.naturalHeight;
    S.selectedPerson=S.people.length?0:-1;S.selectedPoint=0;
    S.undo=[];S.redo=[];S.placeMode=false;S.zoom=1;S.dirty=false;S.revision=0;
    el('frameImage').src=loadedImg.src;
    renderFrameList();scrollActiveIntoView();renderHeader();renderPeople();fitSurface();renderOverlay();showSaveStatus();
  }catch(e){toast(e.message,true)}finally{
    S.navigating=false;
    updateButtons();
    const pending=S.pendingIndex;S.pendingIndex=null;
    if(pending!==null && pending!==S.index) requestNavigation(pending);
  }
}
function preloadImage(src) {return new Promise((resolve,reject)=>{
  const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('Could not open image. Check whether its original file still exists.'));img.src=src;
})}
function fitSurface() {
  if (!S.width||!S.height) return;
  const stage=el('stage');
  const availW=Math.max(180,stage.clientWidth-75),availH=Math.max(140,stage.clientHeight-80);
  S.baseScale=Math.min(availW/S.width,availH/S.height);
  applyZoom();
}
function applyZoom() {
  if (!S.width||!S.height)return;
  const scale=S.baseScale*S.zoom;
  const surface=el('surface');
  surface.style.width=`${S.width*scale}px`;
  surface.style.height=`${S.height*scale}px`;
  el('poseOverlay').setAttribute('viewBox',`0 0 ${S.width} ${S.height}`);
  el('zoomText').textContent=`${Math.round(S.zoom*100)}%`;
  renderOverlay();
}
function zoomBy(factor) {S.zoom=clamp(Math.round(S.zoom*factor*100)/100,.25,10);applyZoom()}
function snapshotUndo() {
  S.undo.push({people:deep(S.people),selectedPerson:S.selectedPerson,selectedPoint:S.selectedPoint});
  if(S.undo.length>45)S.undo.shift();S.redo=[];
  updateButtons();
}
function restoreSnapshot(history,target) {
  if(!history.length||S.busy)return;
  target.push({people:deep(S.people),selectedPerson:S.selectedPerson,selectedPoint:S.selectedPoint});
  const value=history.pop();S.people=value.people;S.selectedPerson=value.selectedPerson;S.selectedPoint=value.selectedPoint;
  S.placeMode=false;setDirty();renderPeople();renderOverlay();renderHeader();
}
function createSkeleton() {
  if(S.index<0||S.busy)return;
  snapshotUndo();
  // A front-facing editable COCO-17 pose with correctly ordered left/right joints.
  // Left in COCO refers to the PERSON'S left (screen-right for front-facing poses).
  const points=TEMPLATE.map(([px,py])=>[+(S.width*px).toFixed(2),+(S.height*py).toFixed(2),2]);
  S.people.push({id:S.people.length+1,source:'manual',keypoints:points});
  S.selectedPerson=S.people.length-1;S.selectedPoint=0;S.placeMode=false;
  setDirty();renderPeople();renderOverlay();renderHeader();
  toast('Manual skeleton created. Drag its joints onto the person.');
}
function removeSelectedPerson() {
  if(!selectedPerson()||S.busy)return;
  snapshotUndo();S.people.splice(S.selectedPerson,1);
  S.people.forEach((p,i)=>p.id=i+1);
  S.selectedPerson=S.people.length?Math.min(S.selectedPerson,S.people.length-1):-1;
  S.placeMode=false;setDirty();renderPeople();renderOverlay();renderHeader();
}
function renderPeople() {
  const tabs=el('peopleTabs');tabs.replaceChildren();
  if(!S.people.length) {const d=document.createElement('div');d.className='no-people';d.textContent='No people annotated yet';tabs.append(d)}
  for (let i=0;i<S.people.length;i++) {
    const p=S.people[i],b=document.createElement('button');
    b.className='person-tab'+(i===S.selectedPerson?' active':'');b.dataset.person=String(i);
    b.textContent=`●  Person ${i+1}`;b.style.setProperty('--person-color',COLORS[i%COLORS.length]);
    if(i===S.selectedPerson)b.style.color=COLORS[i%COLORS.length];
    tabs.appendChild(b);
  }
  el('peopleCount').textContent=S.people.length;
  el('sourceTag').textContent=selectedPerson()?selectedPerson().source==='yolo26'?'✦ YOLO26 PREDICTION':'● MANUAL ANNOTATION':'NO SELECTION';
  el('deletePersonBtn').disabled=!selectedPerson()||S.busy;
  renderPoints();updateButtons();
}
function renderPoints() {
  const list=el('keypointList');list.replaceChildren();
  const person=selectedPerson();
  for(let i=0;i<NAMES.length;i++){
    const point=person?.keypoints?.[i]||[0,0,0];
    const row=document.createElement('div');row.className='point-row'+(person&&i===S.selectedPoint?' active':'');row.dataset.point=String(i);
    const n=document.createElement('span');n.className='point-index';n.textContent=String(i+1).padStart(2,'0');
    const dot=document.createElement('span');dot.className='point-color v'+point[2];
    const label=document.createElement('span');label.className='point-name';label.textContent=nice(NAMES[i]);
    const select=document.createElement('select');select.dataset.point=String(i);select.setAttribute('aria-label',`${NAMES[i]} visibility`);select.disabled=!person;
    for(const [v,s] of [['2','Visible'],['1','Hidden'],['0','Missing']]){
      const opt=document.createElement('option');opt.value=v;opt.textContent=s;select.append(opt);
    }
    select.value=String(point[2]);row.append(n,dot,label,select);list.appendChild(row);
  }
  renderSelectedInfo();
}
function renderSelectedInfo() {
  const person=selectedPerson(),p=person?.keypoints[S.selectedPoint];
  el('selectedPointName').textContent=person?nice(NAMES[S.selectedPoint]):'Select a point';
  el('selectedPointNumber').textContent=person?`POINT ${S.selectedPoint+1} / 17`:'—';
  el('selectedPointStatus').textContent=p?(p[2]===2?'Visible':p[2]===1?'Occluded':'Not labeled'):'—';
  el('editX').disabled=!p || !p[2]; el('editY').disabled=!p || !p[2];
  el('editX').value=p&&p[2]?String(p[0]):'';el('editY').value=p&&p[2]?String(p[1]):'';
  el('placeBtn').disabled=!p||S.busy;
  el('placeBtn').classList.toggle('placing',S.placeMode);
  el('placeBtnText').textContent=S.placeMode?'Click anywhere on image to place':'Place point by clicking image';
  el('poseOverlay').style.cursor=S.placeMode?'crosshair':'default';
}
function renderOverlay() {
  const svg=el('poseOverlay');if(!S.width||!S.height){svg.innerHTML='';return;}
  let out='';
  // Draw nonselected people underneath the selected one.
  const order=S.people.map((_,i)=>i).sort((a,b)=>((a===S.selectedPerson)?1:0)-((b===S.selectedPerson)?1:0));
  for (const personIndex of order) {
    const person=S.people[personIndex],active=personIndex===S.selectedPerson,color=COLORS[personIndex%COLORS.length];
    if(S.showSkeleton){
      for (const [a,b] of EDGES) {
        const p=person.keypoints[a],q=person.keypoints[b];
        if(p[2]===0||q[2]===0)continue;
        const dashed=p[2]===1||q[2]===1?'stroke-dasharray="6 5"':'';
        out+=`<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#051017" stroke-width="${active?6:5}" stroke-opacity=".65" vector-effect="non-scaling-stroke" />`;
        out+=`<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${color}" stroke-width="${active?2.8:2.1}" stroke-opacity="${active?.94:.53}" vector-effect="non-scaling-stroke" ${dashed}/>`;
      }
    }
    for(let i=0;i<17;i++){
      const [x,y,v]=person.keypoints[i];if(v===0)continue;
      const selected=active && i===S.selectedPoint;
      const radius=clamp(5.5/Math.max(.01,S.baseScale*S.zoom),3,35); // ~5.5 screen px at any zoom
      const r=selected?radius*1.34:radius;
      const fill=v===1?'#edb477':color;
      if(selected)out+=`<circle cx="${x}" cy="${y}" r="${r*2.1}" stroke="${color}" stroke-width="1.5" stroke-opacity=".85" stroke-dasharray="3 3" fill="none" vector-effect="non-scaling-stroke" pointer-events="none"/>`;
      out+=`<circle class="pose-hit" data-person="${personIndex}" data-point="${i}" cx="${x}" cy="${y}" r="${r*2.2}" fill="transparent" pointer-events="all"/>`;
      out+=`<circle class="pose-point" data-person="${personIndex}" data-point="${i}" cx="${x}" cy="${y}" r="${r}" fill="${fill}" fill-opacity="${active?'1':'.76'}" stroke="#0c1922" stroke-width="2.1" vector-effect="non-scaling-stroke"/>`;
      if(S.showNumbers){const fs=clamp(S.width*.013,12,24);out+=`<text class="pose-label" x="${x+r+4}" y="${y-r-3}" font-size="${fs}" fill="${color}">${i+1}</text>`;}
    }
  }
  svg.innerHTML=out;
}
function pointAtEvent(evt) {
  const svg=el('poseOverlay'),ctm=svg.getScreenCTM();
  if(!ctm)return null;
  const pos=svg.createSVGPoint();pos.x=evt.clientX;pos.y=evt.clientY;
  const transformed=pos.matrixTransform(ctm.inverse());
  return [+(clamp(transformed.x,0,S.width)).toFixed(2),+(clamp(transformed.y,0,S.height)).toFixed(2)];
}
function mouseDown(evt) {
  if(S.busy||S.index<0||evt.button!==0)return;
  const circle=evt.target.closest('circle[data-point]');
  if(circle){
    const pi=Number(circle.dataset.person),ki=Number(circle.dataset.point);
    if(pi!==S.selectedPerson || ki!==S.selectedPoint){S.selectedPerson=pi;S.selectedPoint=ki;S.placeMode=false;renderPeople();renderOverlay()}
    snapshotUndo();S.dragging={person:pi,point:ki};
    el('poseOverlay').setPointerCapture(evt.pointerId);
    evt.preventDefault();
  } else if(S.placeMode&&selectedPerson()) {
    const point=pointAtEvent(evt);if(!point)return;
    snapshotUndo();selectedPerson().keypoints[S.selectedPoint]=[...point,2];
    S.placeMode=false;setDirty();renderOverlay();renderPoints();renderHeader();
  }
}
function hidePointOnRightClick(evt) {
  const circle=evt.target.closest('circle[data-point]');
  if(!circle)return;
  evt.preventDefault();
  if(S.busy||S.navigating||S.dragging||S.index<0)return;
  const pi=Number(circle.dataset.person),ki=Number(circle.dataset.point);
  if(!S.people[pi]?.keypoints[ki])return;
  snapshotUndo();
  S.selectedPerson=pi;S.selectedPoint=ki;S.placeMode=false;
  S.people[pi].keypoints[ki]=[0,0,0];
  setDirty();renderPeople();renderOverlay();renderHeader();
}
function mouseMove(evt) {
  if(!S.dragging)return;
  const point=pointAtEvent(evt);if(!point)return;
  const p=S.people[S.dragging.person];if(!p)return;
  p.keypoints[S.dragging.point]=[...point,p.keypoints[S.dragging.point][2]||2];
  setDirty();renderOverlay();
}
function mouseUp(evt) {
  if(!S.dragging)return;
  S.dragging=null;renderPoints();renderHeader();
  if(el('poseOverlay').hasPointerCapture(evt.pointerId))el('poseOverlay').releasePointerCapture(evt.pointerId);
}
async function predictPose() {
  if(!frame()||S.busy)return;
  if(S.people.length && !confirm('Replace the current frame’s person annotations with YOLO predictions? You can undo afterwards.'))return;
  const current=frame().name;
  S.busy=true;el('busyOverlay').classList.remove('hidden');
  el('busyTitle').textContent='Running YOLO pose estimation';
  el('busySubtitle').textContent='First run downloads model weights. The x model is large and may be slow on CPU.';
  updateButtons();
  try {
    const result=await api('/api/predict/'+encodeURIComponent(current),jsonOptions('POST',{
      model:el('modelSelect').value,confidence:0.25,keypoint_confidence:Number(el('pointThreshold').value)
    }));
    snapshotUndo();S.people=result.people||[];
    S.selectedPerson=S.people.length?0:-1;S.selectedPoint=0;S.placeMode=false;
    setDirty();renderPeople();renderOverlay();renderHeader();
    await saveFrame();
    toast(`YOLO completed · ${S.people.length} ${S.people.length===1?'person':'people'} detected. Drag keypoints to refine.`);
  } catch(e){toast(e.message,true)}finally{
    S.busy=false;el('busyOverlay').classList.add('hidden');updateButtons();
  }
}
function showUnlabeled() {
  S.filter='unlabeled';el('frameSearch').value='';
  document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter==='unlabeled'));
  renderFrameList(true);
}
async function runAutomaticLabels() {
  if(S.busy||S.importing||S.navigating||!frame())return;
  S.batch=true;S.batchStop=false;S.busy=true;S.pendingIndex=null;
  const currentIndex=S.index;
  const settings={folder:S.folder,model:el('modelSelect').value,confidence:0.25,keypoint_confidence:Number(el('pointThreshold').value)};
  let completed=0,labeled=0,failed=0,consecutiveErrors=0,total=0;
  el('batchStopBtn').classList.remove('hidden');el('batchStopBtn').disabled=false;
  el('batchStopBtn').textContent='Stop after current frame';
  el('batchProgress').classList.remove('hidden');el('batchProgress').value=0;
  el('busyOverlay').classList.remove('hidden');el('busyTitle').textContent='Automatic labeling';
  updateButtons();
  try {
    await saveFrame();
    const queue=S.frames.filter(f=>f.status==='unlabeled');total=queue.length;
    el('batchProgress').max=Math.max(1,total);
    for(const f of queue){
      if(S.batchStop)break;
      const progress=`${completed} / ${total} processed · ${labeled} labeled · ${failed} errors`;
      el('batchStatus').textContent=progress;
      el('busySubtitle').textContent=`${progress}. Processing ${f.name}. First use may download the model.`;
      try {
        const result=await api('/api/auto-label/'+encodeURIComponent(f.name),jsonOptions('POST',settings));
        if(!result.skipped){
          f.status=result.status;f.people=result.people.length;f.auto_note=result.auto_note;
          if(result.status==='labeled')labeled++;
        }
        consecutiveErrors=0;
      } catch(e){
        failed++;consecutiveErrors++;f.auto_note=e.message;
        // Stop on connection/setup/project failures; do not repeat them for an entire dataset.
        if(!e.status || [400,409,503].includes(e.status) || consecutiveErrors>=3)throw e;
      }
      completed++;el('batchProgress').value=completed;renderFrameList();
    }
    el('batchStatus').textContent=`${S.batchStop?'Stopped':'Finished'} · ${completed} / ${total} processed · ${labeled} labeled · ${failed} errors`;
  } catch(e){
    el('batchStatus').textContent=`Stopped · ${completed} / ${total} processed. ${e.message}`;
    toast(e.message,true);
  } finally {
    S.batch=false;S.busy=false;el('batchStopBtn').classList.add('hidden');
    el('busyOverlay').classList.add('hidden');
    try {
      const result=await api('/api/project');
      if(result.folder!==S.folder || result.mode!==S.mode)await adoptProject(result);
      else {
        S.frames=result.frames;S.index=-1;
        await navigateTo(Math.min(currentIndex,S.frames.length-1));
      }
    }catch(e){toast('Could not refresh saved results: '+e.message,true)}
    showUnlabeled();updateButtons();
  }
}
function openModal() {if(S.importing||S.busy)return;el('importModal').classList.remove('hidden');el('importError').classList.add('hidden')}
function closeModal() {if(S.importing)return;el('importModal').classList.add('hidden')}
function importError(error) {el('importError').classList.remove('hidden');el('importError').textContent=error.message||String(error)}
function confirmReplace() {return !S.frames.length || confirm('Open a different dataset? This replaces the current working project and its CSV. Export the CSV first if you need to keep it.')}
function showImportProgress(done,total){
  el('importProgress').classList.remove('hidden');
  el('importProgressText').textContent=`Importing images · ${done.toLocaleString()} of ${total.toLocaleString()}`;
  const percent=total?Math.round(100*done/total):0;
  el('importProgressPercent').textContent=`${percent}%`;
  el('importProgressFill').style.width=percent+'%';
}
async function uploadFolder(files) {
  if(S.importing)return;
  const supported=Array.from(files).filter(f=>/\.(jpe?g|png|webp|bmp)$/i.test(f.name));
  el('folderInput').value='';
  if(!supported.length){importError(new Error('No supported images were selected.'));return}
  if(!confirmReplace())return;
  try {
    if(S.dirty)await saveFrame();
    S.importing=true;el('importError').classList.add('hidden');showImportProgress(0,supported.length);
    await api('/api/import/start',jsonOptions('POST',{}));
    for(let i=0;i<supported.length;i+=25){
      const batch=supported.slice(i,i+25),form=new FormData();
      for(const file of batch)form.append('files',file,file.name);
      await api('/api/import/batch',{method:'POST',body:form});
      showImportProgress(Math.min(i+batch.length,supported.length),supported.length);
    }
    const result=await api('/api/import/finish',jsonOptions('POST',{}));
    el('importModal').classList.add('hidden');
    await adoptProject(result);
    toast(`Imported ${result.count.toLocaleString()} frames in chronological order.`);
  }catch(e){importError(e);toast(`Import failed: ${e.message}`,true)}finally{
    S.importing=false;el('importProgress').classList.add('hidden');updateButtons();
  }
}
async function openPath() {
  if(S.importing)return;
  const path=el('pathInput').value.trim();if(!path){importError(new Error('Enter a folder path.'));return}
  if(!confirmReplace())return;
  try{
    if(S.dirty)await saveFrame();
    S.importing=true;el('importError').classList.add('hidden');
    el('loadPathBtn').disabled=true;el('loadPathBtn').textContent='Reading folder...';
    const result=await api('/api/import/path',jsonOptions('POST',{path}));
    el('importModal').classList.add('hidden');await adoptProject(result);
    toast(`Opened ${result.count.toLocaleString()} frames without copying them.`);
  }catch(e){importError(e);toast(e.message,true)}finally{
    S.importing=false;el('loadPathBtn').disabled=false;el('loadPathBtn').textContent='Open this folder →';updateButtons();
  }
}
function wireEvents() {
  ['newProjectBtn','openFolderSmall','emptyImportBtn'].forEach(id=>el(id).addEventListener('click',openModal));
  el('closeModal').addEventListener('click',closeModal);
  el('importModal').addEventListener('click',evt=>{if(evt.target===el('importModal') && S.frames.length)closeModal()});
  el('pickFolder').addEventListener('click',()=>el('folderInput').click());
  el('folderInput').addEventListener('change',evt=>{if(evt.target.files?.length)uploadFolder(evt.target.files)});
  el('loadPathBtn').addEventListener('click',openPath);
  el('pathInput').addEventListener('keydown',evt=>{if(evt.key==='Enter')openPath()});
  document.querySelectorAll('.modal-tabs button').forEach(b=>b.addEventListener('click',()=>{
    document.querySelectorAll('.modal-tabs button').forEach(x=>x.classList.toggle('active',x===b));
    el('uploadPanel').classList.toggle('hidden',b.dataset.tab!=='upload');
    el('pathPanel').classList.toggle('hidden',b.dataset.tab!=='path');
  }));
  el('frameList').addEventListener('click',evt=>{
    const b=evt.target.closest('.frame-row');if(b)requestNavigation(Number(b.dataset.index));
  });
  el('frameSearch').addEventListener('input',()=>renderFrameList(true));
  el('frameList').addEventListener('scroll',()=>{
    if(listRenderPending)return;listRenderPending=true;
    requestAnimationFrame(()=>{listRenderPending=false;renderVirtualRows()});
  });
  document.querySelectorAll('.filter-tabs button').forEach(b=>b.addEventListener('click',()=>{
    S.filter=b.dataset.filter;
    document.querySelectorAll('.filter-tabs button').forEach(x=>x.classList.toggle('active',x===b));
    renderFrameList(true);
  }));
  ['prevBtn','footPrev'].forEach(id=>el(id).addEventListener('click',()=>requestNavigation(S.index-1)));
  ['nextBtn','footNext'].forEach(id=>el(id).addEventListener('click',()=>requestNavigation(S.index+1)));
  el('frameRange').addEventListener('change',evt=>requestNavigation(Number(evt.target.value)));
  el('saveBtn').addEventListener('click',async()=>{try{await saveFrame(true);toast('Saved annotations to CSV.')}catch(e){toast(e.message,true)}});
  el('exportBtn').addEventListener('click',()=>{window.location.href='/api/export'});
  ['addPersonBtn','inspectorAddBtn'].forEach(id=>el(id).addEventListener('click',createSkeleton));
  el('deletePersonBtn').addEventListener('click',removeSelectedPerson);
  el('autoBtn').addEventListener('click',predictPose);
  el('batchAutoBtn').addEventListener('click',runAutomaticLabels);
  el('reviewUnlabeledBtn').addEventListener('click',showUnlabeled);
  el('batchStopBtn').addEventListener('click',()=>{
    S.batchStop=true;el('batchStopBtn').disabled=true;el('batchStopBtn').textContent='Stopping after this frame…';
  });
  el('undoBtn').addEventListener('click',()=>restoreSnapshot(S.undo,S.redo));
  el('redoBtn').addEventListener('click',()=>restoreSnapshot(S.redo,S.undo));
  el('peopleTabs').addEventListener('click',evt=>{
    const b=evt.target.closest('button[data-person]');if(!b)return;
    S.selectedPerson=Number(b.dataset.person);S.selectedPoint=0;S.placeMode=false;
    renderPeople();renderOverlay();
  });
  el('keypointList').addEventListener('click',evt=>{
    const b=evt.target.closest('.point-row');if(!b||!selectedPerson())return;
    S.selectedPoint=Number(b.dataset.point);S.placeMode=false;
    // Keep a clicked <select> in the DOM until its native change event fires.
    if(evt.target.closest('select')){
      el('keypointList').querySelectorAll('.point-row').forEach(row=>row.classList.toggle('active',row===b));
      renderSelectedInfo();renderOverlay();
    } else {renderPoints();renderOverlay();}
  });
  el('keypointList').addEventListener('change',evt=>{
    const sel=evt.target.closest('select[data-point]');if(!sel||!selectedPerson())return;
    const i=Number(sel.dataset.point),v=Number(sel.value);
    snapshotUndo();const p=selectedPerson().keypoints[i];
    // If formerly missing, give the joint a usable central location.
    if(v && p[2]===0)selectedPerson().keypoints[i]=[+(S.width*TEMPLATE[i][0]).toFixed(2),+(S.height*TEMPLATE[i][1]).toFixed(2),v];
    else if(v===0)selectedPerson().keypoints[i]=[0,0,0];
    else p[2]=v;
    S.selectedPoint=i;setDirty();renderPoints();renderOverlay();renderHeader();
  });
  for(const [id,coord,limit] of [['editX',0,()=>S.width],['editY',1,()=>S.height]]){
    el(id).addEventListener('change',evt=>{
      const p=selectedPerson()?.keypoints[S.selectedPoint];if(!p||!p[2])return;
      const value=Number(evt.target.value);if(!Number.isFinite(value)){renderSelectedInfo();return}
      snapshotUndo();p[coord]=Math.round(clamp(value,0,limit())*100)/100;
      setDirty();renderPoints();renderOverlay();
    });
  }
  el('placeBtn').addEventListener('click',()=>{S.placeMode=!S.placeMode;renderSelectedInfo()});
  el('showSkeleton').addEventListener('change',evt=>{S.showSkeleton=evt.target.checked;renderOverlay()});
  el('showNumbers').addEventListener('change',evt=>{S.showNumbers=evt.target.checked;renderOverlay()});
  el('pointThreshold').addEventListener('input',evt=>{el('thresholdValue').textContent=Number(evt.target.value).toFixed(2)});
  el('poseOverlay').addEventListener('pointerdown',mouseDown);
  el('poseOverlay').addEventListener('contextmenu',hidePointOnRightClick);
  el('poseOverlay').addEventListener('pointermove',mouseMove);
  el('poseOverlay').addEventListener('pointerup',mouseUp);
  el('poseOverlay').addEventListener('pointercancel',mouseUp);
  el('zoomIn').addEventListener('click',()=>zoomBy(1.25));
  el('zoomOut').addEventListener('click',()=>zoomBy(.8));
  el('zoomReset').addEventListener('click',()=>{S.zoom=1;applyZoom()});
  el('stage').addEventListener('wheel',evt=>{
    if(!evt.ctrlKey||S.index<0)return;
    evt.preventDefault();zoomBy(evt.deltaY<0?1.12:.89);
  },{passive:false});
  window.addEventListener('resize',fitSurface);
  document.addEventListener('keydown',async evt=>{
    const ctrl=evt.ctrlKey||evt.metaKey;
    if(ctrl && evt.key.toLowerCase()==='s'){
      if(S.batch){evt.preventDefault();return;}
      evt.preventDefault();try{await saveFrame(true);toast('Saved annotations to CSV.')}catch(e){toast(e.message,true)}return;
    }
    if(ctrl && evt.key.toLowerCase()==='z'){
      evt.preventDefault();restoreSnapshot(evt.shiftKey?S.redo:S.undo,evt.shiftKey?S.undo:S.redo);return;
    }
    if(ctrl && evt.key.toLowerCase()==='y'){
      evt.preventDefault();restoreSnapshot(S.redo,S.undo);return;
    }
    if(evt.key==='Escape'&&!el('importModal').classList.contains('hidden')){if(S.frames.length)closeModal();return}
    if(evt.target.closest('input,select,textarea')||!el('importModal').classList.contains('hidden'))return;
    if(evt.key==='ArrowLeft'){evt.preventDefault();requestNavigation((S.pendingIndex??S.index)-1)}
    if(evt.key==='ArrowRight'){evt.preventDefault();requestNavigation((S.pendingIndex??S.index)+1)}
    if(evt.key==='/' && !evt.ctrlKey){evt.preventDefault();el('frameSearch').focus()}
  });
  window.addEventListener('beforeunload',evt=>{if(S.dirty||S.batch){evt.preventDefault();evt.returnValue='';}});
}

wireEvents();
loadProject().catch(e=>{toast('Failed to connect to the local server: '+e.message,true);openModal()});
