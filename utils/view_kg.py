"""Live RDF and SPA snapshot viewer: python utils/view_kg.py g2 --watch."""

import argparse
import hashlib
import json
import threading
import webbrowser
from pathlib import Path
from urllib.parse import unquote, quote
from wsgiref.simple_server import make_server

from rdflib import BNode, Graph, Literal, RDF, URIRef
from rdflib.util import guess_format


ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[1] / "episodic_memory"
SNAPSHOTS = {"g1": "g1_sense.json", "g2": "g2_plan.json", "g3": "g3_action.json"}


def resolve_path(value):
    return (ARTIFACT_DIRECTORY / SNAPSHOTS[value] if value in SNAPSHOTS
            else Path(value).expanduser().resolve())


def missing_message(path):
    stage = next((key.upper() for key, name in SNAPSHOTS.items() if path.name == name), "Graph")
    return f"{stage} snapshot not available yet. Run demo.py through the corresponding stage first."


def snapshot_to_graph(data):
    """Visualization-only RDF adapter; never change the stored JSON representation."""
    graph = Graph()
    def node(kind, value):
        return URIRef("urn:cmoc:" + kind + ":" + quote(str(value), safe=""))
    def edge(subject, label, target):
        graph.add((subject, node("predicate", label), target))
    scene = data.get("scene_graph", data)
    for obj in scene.get("objects", []):
        entity = node("entity", obj["id"])
        graph.add((entity, RDF.type, node("type", obj.get("type", "Entity"))))
        for key, value in obj.get("qualities", {}).items():
            edge(entity, key, Literal(json.dumps(value, ensure_ascii=False)))
    for relation in scene.get("relations", []):
        edge(node("entity", relation["subject"]), relation["predicate"], node("entity", relation["object"]))
    for identifier, concept in data.get("entity_links", {}).items():
        edge(node("entity", identifier), "linkedTo", node("concept", concept))
    snapshot = node("snapshot", "Snapshot")
    for key in ("instruction", "type", "issues"):
        if key in data:
            edge(snapshot, key, Literal(data[key] if isinstance(data[key], str) else json.dumps(data[key])))
    if "frame" in data:
        frame = node("frame", "Bringing")
        edge(snapshot, "frame", frame)
        for role, value in data["frame"].items():
            edge(frame, role, Literal(value if value is not None else "unbound"))
        for role, value in data.get("bindings", {}).items():
            if value is not None:
                edge(frame, "bound" + role, node("entity", value))
        if data.get("theme_concept"):
            edge(frame, "themeConcept", node("concept", data["theme_concept"]))
    return graph


# Load a complete snapshot before replacing anything the browser can see.
def load_graph(path, rdf_format=None):
    source = path.read_bytes()
    if path.suffix.lower() == ".json" and rdf_format is None:
        data = json.loads(source)
        if data is None:
            raise ValueError("G3 action snapshot not available yet; Act is not implemented.")
        if isinstance(data, dict) and ("scene_graph" in data or "objects" in data):
            return snapshot_to_graph(data)
    graph = Graph()
    graph.parse(data=source, format=rdf_format or guess_format(str(path)) or "turtle",
                publicID=path.as_uri())
    return graph


def local_name(term):
    if isinstance(term, BNode):
        return "_:" + str(term)
    value = str(term)
    return unquote(value.rstrip("/#").rsplit("#", 1)[-1].rsplit("/", 1)[-1].rsplit(":", 1)[-1]) or value


def type_color(type_uri):
    # Hash the URI, not its position in the graph, so colors survive reloads.
    hue = int(hashlib.sha256(str(type_uri).encode()).hexdigest()[:8], 16) % 360
    return f"hsl({hue}, 55%, 65%)"


def graph_to_data(graph):
    terms = sorted(set(graph.subjects()) | set(graph.objects()), key=lambda term: term.n3())
    identifiers = {term: term.n3() for term in terms}
    nodes = []
    visible_types = set()
    for term in terms:
        types = sorted(graph.objects(term, RDF.type), key=str) if not isinstance(term, Literal) else []
        visible_types.update(types)
        kind = "literal" if isinstance(term, Literal) else "blank" if isinstance(term, BNode) else "resource"
        label = str(term) if kind == "literal" else local_name(term)
        nodes.append({
            "id": identifiers[term], "label": label[:60] + ("…" if len(label) > 60 else ""),
            "kind": kind, "uri": str(term) if kind == "resource" else None,
            "types": [str(item) for item in types],
            "value": str(term) if kind == "literal" else None,
            "language": term.language if kind == "literal" else None,
            "datatype": str(term.datatype) if kind == "literal" and term.datatype else None,
            "color": "#d9dee5" if kind == "literal" else type_color(types[0]) if types else "#b4c4d6",
        })
    edges = [{"source": identifiers[subject], "target": identifiers[obj],
              "label": local_name(predicate), "uri": str(predicate)}
             for subject, predicate, obj in sorted(graph, key=lambda triple: tuple(term.n3() for term in triple))]
    legend = [{"label": local_name(item), "uri": str(item), "color": type_color(item)}
              for item in sorted(visible_types, key=str)]
    return {"nodes": nodes, "edges": edges, "legend": legend}


# Everything the browser needs lives here; RDF text is inserted only as textContent.
def build_html():
    return r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge graph</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f7f9fc;color:#263349;font:14px system-ui,sans-serif}
header{position:absolute;top:22px;left:26px;pointer-events:none}h1{font-size:18px;margin:0 0 6px}
#status{color:#69778b;font-size:12px;max-width:65vw}svg{width:100vw;height:100vh;display:block;touch-action:none}
#legend{position:absolute;bottom:20px;left:26px;max-width:65vw;max-height:25vh;overflow:auto;display:flex;gap:8px 16px;flex-wrap:wrap;font-size:12px}
.key{display:flex;align-items:center;gap:6px}.swatch{width:11px;height:11px;border-radius:50%;display:inline-block}
#details{position:absolute;right:20px;top:20px;width:290px;max-width:40vw;max-height:80vh;overflow:auto;background:#ffffffed;border:1px solid #e1e7ef;border-radius:12px;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.6}
#details:empty{display:none}.edge{fill:none;stroke:#a6b1c0;stroke-width:1.2}.edge-label{font-size:10px;fill:#64748b;paint-order:stroke;stroke:#f7f9fc;stroke-width:4px;stroke-linejoin:round}
.node{cursor:grab}.node text{font-size:11px;fill:#263349;paint-order:stroke;stroke:#f7f9fc;stroke-width:3px;pointer-events:none}.node.selected .shape{stroke:#243b60;stroke-width:3}
</style>
<header><h1>Knowledge graph</h1><div id="status">Loading…</div></header>
<svg id="canvas" aria-label="Interactive RDF graph"><defs><marker id="arrow" viewBox="0 -4 8 8" refX="8" refY="0" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,-4L8,0L0,4" fill="#a6b1c0"/></marker></defs><g id="world"><g id="edges"></g><g id="nodes"></g></g></svg>
<div id="legend"></div><aside id="details"></aside>
<script>
const canvas=document.querySelector('#canvas'), world=document.querySelector('#world');
const nodeLayer=document.querySelector('#nodes'), edgeLayer=document.querySelector('#edges');
const details=document.querySelector('#details'), status=document.querySelector('#status');
let nodes=[], edges=[], revision=-1, selected=null, drag=null, energy=0;
let view={x:innerWidth/2,y:innerHeight/2,scale:1};
function element(tag,attributes,parent){
  const item=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attributes)) item.setAttribute(key,value);
  parent.append(item); return item;
}
function transform(){world.setAttribute('transform',`translate(${view.x},${view.y}) scale(${view.scale})`)}
function describe(node){
  details.textContent=[node.kind==='blank'?'Blank node: '+node.id:node.uri || 'Literal',
    node.types.length?'Types:\n'+node.types.join('\n'):'',
    node.value!==null?'Value: '+node.value:'',node.language?'Language: '+node.language:'',
    node.datatype?'Datatype: '+node.datatype:''].filter(Boolean).join('\n\n');
}
function render(data){
  const previous=new Map(nodes.map(node=>[node.id,node]));
  nodeLayer.replaceChildren(); edgeLayer.replaceChildren();
  nodes=data.nodes.map((node,index)=>{
    const old=previous.get(node.id), angle=index*2.39996, radius=40*Math.sqrt(index);
    return {...node,x:old?old.x:Math.cos(angle)*radius,y:old?old.y:Math.sin(angle)*radius,vx:0,vy:0};
  });
  const byId=new Map(nodes.map(node=>[node.id,node])), pairs=new Map();
  edges=data.edges.map(edge=>{
    const key=JSON.stringify([edge.source,edge.target].sort());
    const group=pairs.get(key)||[]; pairs.set(key,group); group.push(edge);
    return edge;
  });
  for(const group of pairs.values()) group.forEach((edge,index)=>{edge.bend=(index-(group.length-1)/2)*30;edge.loopSize=48+index*22});
  edges=edges.map(edge=>({...edge,source:byId.get(edge.source),target:byId.get(edge.target),
    path:element('path',{'class':'edge','marker-end':'url(#arrow)'},edgeLayer),
    text:element('text',{'class':'edge-label','text-anchor':'middle'},edgeLayer)}));
  for(const edge of edges){edge.text.textContent=edge.label;element('title',{},edge.path).textContent=edge.uri}
  for(const node of nodes){
    node.group=element('g',{'class':'node'+(selected===node.id?' selected':'')},nodeLayer);
    const tag=node.kind==='literal'?'rect':node.kind==='blank'?'path':'circle';
    const shape=node.kind==='literal'?{x:-16,y:-10,width:32,height:20,rx:4}:node.kind==='blank'?{d:'M0,-17L17,0L0,17L-17,0Z'}:{r:15};
    element(tag,{...shape,'class':'shape',fill:node.color,stroke:'#7c8da3','stroke-dasharray':node.kind==='blank'?'3 2':'none'},node.group);
    element('text',{y:31,'text-anchor':'middle'},node.group).textContent=node.label;
    element('title',{},node.group).textContent=node.uri || node.value || node.id;
    node.group.onpointerenter=()=>describe(node);
    node.group.onpointerleave=()=>{const pinned=byId.get(selected);if(pinned)describe(pinned);else details.textContent=''};
    node.group.onpointerdown=event=>{
      event.stopPropagation();selected=node.id;describe(node);
      for(const item of nodes)item.group.classList.toggle('selected',item.id===selected);
      drag={node,lastX:event.clientX,lastY:event.clientY};canvas.setPointerCapture(event.pointerId);
    };
  }
  const legend=document.querySelector('#legend');legend.replaceChildren();
  for(const entry of [...data.legend,{label:'Untyped',color:'#b4c4d6'},{label:'◇ Blank node',color:'#b4c4d6'},{label:'Literal',color:'#d9dee5'}]){
    const key=document.createElement('span');key.className='key';key.title=entry.uri||entry.label;
    const swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=entry.color;
    key.append(swatch,document.createTextNode(entry.label));legend.append(key);
  }
  if(byId.has(selected))describe(byId.get(selected));else{selected=null;details.textContent=''}
  if(revision===-1)view.scale=Math.min(1,Math.min(innerWidth,innerHeight)/(100*Math.sqrt(nodes.length||1)));
  energy=240;transform();draw();
}
function draw(){
  for(const node of nodes)node.group.setAttribute('transform',`translate(${node.x},${node.y})`);
  for(const edge of edges){
    const a=edge.source,b=edge.target;
    if(a===b){
      const size=edge.loopSize;
      edge.path.setAttribute('d',`M${a.x-12},${a.y-10}C${a.x-size},${a.y-size*1.5} ${a.x+size},${a.y-size*1.5} ${a.x+12},${a.y-10}`);
      edge.text.setAttribute('x',a.x);edge.text.setAttribute('y',a.y-size-4);continue;
    }
    // A canonical perpendicular keeps opposite-direction edges in separate lanes.
    const direction=a.id<b.id?1:-1,dx=b.x-a.x,dy=b.y-a.y,distance=Math.hypot(dx,dy)||1;
    const cx=(a.x+b.x)/2-dy/distance*edge.bend*direction,cy=(a.y+b.y)/2+dx/distance*edge.bend*direction;
    const start=Math.hypot(cx-a.x,cy-a.y)||1,end=Math.hypot(b.x-cx,b.y-cy)||1;
    edge.path.setAttribute('d',`M${a.x+(cx-a.x)*18/start},${a.y+(cy-a.y)*18/start}Q${cx},${cy} ${b.x-(b.x-cx)*19/end},${b.y-(b.y-cy)*19/end}`);
    edge.text.setAttribute('x',(a.x+2*cx+b.x)/4);edge.text.setAttribute('y',(a.y+2*cy+b.y)/4-5);
  }
}
function tick(){
  if(energy>0){
    energy--;
    for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j],dx=b.x-a.x||0.1,dy=b.y-a.y||0.1,d2=dx*dx+dy*dy+100;
      const force=1100/d2, distance=Math.sqrt(d2);
      a.vx-=dx/distance*force;a.vy-=dy/distance*force;b.vx+=dx/distance*force;b.vy+=dy/distance*force;
    }
    for(const {source:a,target:b} of edges){
      const dx=b.x-a.x,dy=b.y-a.y,distance=Math.hypot(dx,dy)||1,force=(distance-135)*0.006;
      a.vx+=dx/distance*force;a.vy+=dy/distance*force;b.vx-=dx/distance*force;b.vy-=dy/distance*force;
    }
    for(const node of nodes){
      node.vx=(node.vx-node.x*0.0004)*0.8;node.vy=(node.vy-node.y*0.0004)*0.8;
      if(drag?.node!==node){node.x+=Math.max(-12,Math.min(12,node.vx));node.y+=Math.max(-12,Math.min(12,node.vy))}
    }
    draw();
  }
  requestAnimationFrame(tick);
}
canvas.onpointerdown=event=>{drag={lastX:event.clientX,lastY:event.clientY};canvas.setPointerCapture(event.pointerId);selected=null;details.textContent='';for(const node of nodes)node.group.classList.remove('selected')};
canvas.onpointermove=event=>{
  if(!drag)return;
  const dx=event.clientX-drag.lastX,dy=event.clientY-drag.lastY;
  if(drag.node){drag.node.x+=dx/view.scale;drag.node.y+=dy/view.scale;drag.node.vx=drag.node.vy=0;energy=120;draw()}
  else{view.x+=dx;view.y+=dy;transform()}
  drag.lastX=event.clientX;drag.lastY=event.clientY;
};
canvas.onpointerup=canvas.onpointercancel=canvas.onlostpointercapture=()=>drag=null;
canvas.addEventListener('wheel',event=>{
  event.preventDefault();const scale=Math.max(0.05,Math.min(6,view.scale*Math.exp(-event.deltaY*0.001)));
  const ratio=scale/view.scale;view.x=event.clientX-(event.clientX-view.x)*ratio;view.y=event.clientY-(event.clientY-view.y)*ratio;view.scale=scale;transform();
},{passive:false});
async function refresh(){
  try{
    const response=await fetch('/graph',{cache:'no-store'});if(!response.ok)throw Error(response.status);
    const state=await response.json();
    if(state.revision!==revision){render(state.data);revision=state.revision}
    status.textContent=state.error || `${nodes.length} nodes · ${edges.length} edges · Live · Drag to move, scroll to zoom`;
  }catch(error){status.textContent='Connection lost — retrying…'}
  setTimeout(refresh,1000);
}
transform();tick();refresh();
</script></html>'''


# Publish snapshots with a lock so each request sees matching data and status.
def watch_graph(path, rdf_format, state, lock, stopped):
    last_stamp = None
    while not stopped.is_set():
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            if stamp != last_stamp:
                data = graph_to_data(load_graph(path, rdf_format))
                if path.stat().st_mtime_ns != stat.st_mtime_ns:
                    raise ValueError("file changed during parsing")
                with lock:
                    state.update(data=data, revision=state["revision"] + 1, error=None)
                last_stamp = stamp
        except Exception as error:
            last_stamp = None
            # Retry failed reads even if the writer preserves the modification time.
            with lock:
                state["error"] = (missing_message(path) if isinstance(error, FileNotFoundError) else
                                  f"Waiting for a valid graph; keeping last graph. {error}")
        stopped.wait(1)


def run_server(path, rdf_format=None):
    page = build_html().encode()
    state = {"revision": 0, "data": {"nodes": [], "edges": [], "legend": []},
             "error": "Waiting for the first valid graph…"}
    lock = threading.Lock()
    stopped = threading.Event()

    def application(environ, start_response):
        route = environ.get("PATH_INFO", "/")
        if route == "/":
            body, content_type, status = page, "text/html; charset=utf-8", "200 OK"
        elif route == "/graph":
            with lock:
                body = json.dumps(state).encode()
            content_type, status = "application/json", "200 OK"
        else:
            body, content_type, status = b"Not found", "text/plain", "404 Not Found"
        start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(body))),
                                ("Cache-Control", "no-store")])
        return [body]

    with make_server("127.0.0.1", 0, application) as server:
        watcher = threading.Thread(target=watch_graph, args=(path, rdf_format, state, lock, stopped), daemon=True)
        watcher.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Watching {path}\nViewer: {url}\nPress Ctrl+C to stop.", flush=True)
        try:
            webbrowser.open(url)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stopped.set()
            watcher.join(timeout=2)


# The module entry point keeps invocation independent of ROS or project setup.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="g1/g2/g3 shortcut, JSON snapshot, or RDF file")
    parser.add_argument("--watch", action="store_true", help="Wait for missing snapshots; live reload is always enabled")
    parser.add_argument("--format", choices=("turtle", "xml", "json-ld"), help="Override format detection")
    arguments = parser.parse_args()
    path = resolve_path(arguments.path)
    if not path.exists():
        print(missing_message(path))
        if not arguments.watch:
            return
    elif arguments.path == "g3" and json.loads(path.read_text()) is None:
        print("G3 action snapshot not available yet; Act is not implemented.")
        if not arguments.watch:
            return
    run_server(path, arguments.format)


if __name__ == "__main__":
    main()
