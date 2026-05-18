"""Phase 6: browser-based side-by-side reranker demo.

Run::

    python -m q3_reranker.web_demo [--port 8080] [--no-llm] [--no-citation]

Opens http://localhost:<port> with:
  - Preset chips for all 8 gold benchmark stages (presets only — no SerpAPI credits burned)
  - Immediate BM25 / dense / ss_default rankings (synchronous, returns in <1 s from cache)
  - LLM rerank + citation-blend columns that fill in after ~20 s (background thread + poll)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .baselines import RankedResult
from .eval import _build_rankers
from .gold import load_gold
from .retriever import Paper
from .sources import search

logger = logging.getLogger(__name__)

_LIMIT = 20  # fixed candidate pool; keeps latency and SerpAPI cost low
_FAST = frozenset({"ss_default", "bm25", "dense"})

# ---------------------------------------------------------------------------
# In-memory job store (demo-scale; no persistence needed)
# ---------------------------------------------------------------------------

# Values: None = pending | dict[ranker, list[row_dict]] = done | Exception = failed
_jobs: dict[str, dict[str, Any] | Exception | None] = {}
_jobs_lock = threading.Lock()


def _job_id(retrieval_query: str, nl_query: str) -> str:
    digest = hashlib.sha256(f"{retrieval_query}\n{nl_query}".encode()).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _enrich(r: RankedResult, papers_by_id: dict[str, Paper]) -> dict[str, Any]:
    p = papers_by_id.get(r.paper_id)
    return {
        "paper_id": r.paper_id,
        "title": r.title,
        "score": round(r.score, 4),
        "year": p.year if p else None,
        "venue": (p.venue or "") if p else "",
        "authors": (p.authors[:3] if p else []),
        "citations": p.citation_count if p else 0,
        "abstract": ((p.abstract or "")[:220]) if p else "",
    }


def _serialize(
    rankings: dict[str, list[RankedResult]],
    papers_by_id: dict[str, Paper],
) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [_enrich(r, papers_by_id) for r in ranked[:10]]
        for name, ranked in rankings.items()
    }


# ---------------------------------------------------------------------------
# Background LLM job
# ---------------------------------------------------------------------------


def _run_llm_job(
    job_id: str,
    nl_query: str,
    papers: list[Paper],
    papers_by_id: dict[str, Paper],
    llm_rankers: dict[str, Any],
) -> None:
    try:
        rankings: dict[str, list[RankedResult]] = {
            name: ranker(nl_query, papers)
            for name, ranker in llm_rankers.items()
        }
        serialized = _serialize(rankings, papers_by_id)
    except Exception as exc:
        logger.error("LLM job %s failed: %s", job_id, exc, exc_info=True)
        with _jobs_lock:
            _jobs[job_id] = exc
        return
    with _jobs_lock:
        _jobs[job_id] = serialized


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    # Injected once by main() before the first request arrives.
    _all_rankers: dict[str, Any] = {}
    _presets: list[dict[str, Any]] = []

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send_html(_HTML)
        elif self.path == "/api/presets":
            self._send_json(self._presets)
        elif self.path.startswith("/api/llm/"):
            job_id = self.path[len("/api/llm/"):]
            with _jobs_lock:
                result = _jobs.get(job_id)
            if result is None:
                self._send_json({"status": "pending"}, 202)
            elif isinstance(result, Exception):
                self._send_json({"status": "error", "error": str(result)})
            else:
                self._send_json({"status": "done", "rankings": result})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/search":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400, "Bad JSON")
            return

        idx = payload.get("preset_idx")
        if not isinstance(idx, int) or not (0 <= idx < len(self._presets)):
            self.send_error(400, "Invalid preset_idx")
            return

        preset = self._presets[idx]
        retrieval_query: str = preset["retrieval_query"]
        nl_query: str = preset["query"]

        papers = search(retrieval_query, limit=_LIMIT)
        if not papers:
            self._send_json({"error": "No papers found."}, 404)
            return

        papers_by_id = {p.paper_id: p for p in papers}

        fast_rankings: dict[str, list[RankedResult]] = {
            name: ranker(nl_query, papers)
            for name, ranker in self._all_rankers.items()
            if name in _FAST
        }
        llm_rankers = {
            k: v for k, v in self._all_rankers.items()
            if k not in _FAST
        }

        jid = _job_id(retrieval_query, nl_query)
        with _jobs_lock:
            if jid not in _jobs:
                _jobs[jid] = None
                if llm_rankers:
                    threading.Thread(
                        target=_run_llm_job,
                        args=(jid, nl_query, papers, papers_by_id, llm_rankers),
                        daemon=True,
                    ).start()
                else:
                    _jobs[jid] = {}
            existing = _jobs[jid]

        cached: dict[str, Any] | None = existing if isinstance(existing, dict) else None

        self._send_json({
            "job_id": jid,
            "query": nl_query,
            "n_candidates": len(papers),
            "fast_rankings": _serialize(fast_rankings, papers_by_id),
            "has_llm": bool(llm_rankers),
            "llm_rankings": cached,
        })


# ---------------------------------------------------------------------------
# Embedded single-page app
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reranker Demo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;background:#f8f9fa;color:#202124;font-size:14px}
.hdr{background:#fff;border-bottom:1px solid #e8eaed;padding:14px 24px;display:flex;align-items:center;gap:16px}
.hdr-logo{font-size:18px;font-weight:500;color:#5f6368}
.hdr-sub{font-size:12px;color:#9aa0a6}
.main{max-width:1280px;margin:28px auto;padding:0 20px}
.sec-title{font-size:11px;font-weight:600;letter-spacing:.8px;color:#5f6368;text-transform:uppercase;margin-bottom:10px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:28px}
.chip{background:#fff;border:1px solid #dadce0;border-radius:20px;padding:7px 14px;font-size:13px;cursor:pointer;color:#3c4043;transition:background .12s,border-color .12s;white-space:nowrap}
.chip:hover{background:#f1f3f4;border-color:#bdc1c6}
.chip.active{background:#e8f0fe;border-color:#1a73e8;color:#1a73e8;font-weight:500}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #e8eaed;border-top-color:#1a73e8;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle}
#loading{display:none;padding:40px;text-align:center;color:#5f6368}
#loading .spin{width:22px;height:22px;margin:0 auto 10px;display:block}
#results{display:none}
.res-hdr{margin-bottom:14px}
.res-query{font-size:17px;color:#202124;font-weight:400}
.res-meta{font-size:12px;color:#5f6368;margin-top:3px}
#llm-notice{font-size:12px;color:#5f6368;margin-bottom:10px;display:none;align-items:center;gap:6px}
.tbl-wrap{overflow-x:auto;background:#fff;border:1px solid #e8eaed;border-radius:8px}
table{width:100%;border-collapse:collapse}
thead{background:#f8f9fa}
th{padding:9px 11px;text-align:left;font-weight:600;font-size:12px;color:#5f6368;border-bottom:1px solid #e8eaed;white-space:nowrap}
td{padding:9px 11px;border-bottom:1px solid #f1f3f4;vertical-align:top}
tr:last-child td{border-bottom:none}
tr:hover td{background:#fafafa}
.ptitle{font-weight:500;color:#1a0dab;line-height:1.4;max-width:400px}
.pmeta{font-size:11px;color:#5f6368;margin-top:2px}
.pabs{font-size:11px;color:#3c4043;margin-top:4px;line-height:1.5;max-width:400px}
.rb{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;font-size:11px;font-weight:700}
.rb-1{background:#e6f4ea;color:#1e8e3e}
.rb-top{background:#fce8b2;color:#b06000}
.rb-ok{background:#e8f0fe;color:#1a73e8}
.rb-na{background:#f1f3f4;color:#9aa0a6;font-size:13px}
.pend{color:#9aa0a6;font-size:11px}
.err{color:#d93025;font-size:11px}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-logo">Reranker Demo</div>
  <div class="hdr-sub">Semantic Scholar &middot; BM25 &middot; Dense &middot; LLM (DeepSeek) &middot; Citation Blend</div>
</div>
<div class="main">
  <div class="sec-title">Select a benchmark query</div>
  <div class="chips" id="chips"></div>
  <div id="loading"><div class="spin"></div>Retrieving papers&hellip;</div>
  <div id="results">
    <div class="res-hdr">
      <div class="res-query" id="res-query"></div>
      <div class="res-meta" id="res-meta"></div>
    </div>
    <div id="llm-notice"></div>
    <div class="tbl-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
  </div>
</div>
<script>
(function(){
  var presets=[],allPapers={},paperRanks={},rankerOrder=[],currentJob=null,pollT=null;

  function esc(s){
    return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
  function rb(rank){
    if(!rank)return '<span class="rb rb-na">&mdash;</span>';
    var c=rank===1?'rb-1':rank<=3?'rb-top':'rb-ok';
    return '<span class="rb '+c+'">'+rank+'</span>';
  }

  fetch('/api/presets').then(function(r){return r.json();}).then(function(data){
    presets=data;
    var chips=document.getElementById('chips');
    data.forEach(function(p,i){
      var btn=document.createElement('button');
      btn.className='chip';
      btn.textContent='Stage '+p.stage+': '+p.topic;
      btn.onclick=function(){selectPreset(i,btn);};
      chips.appendChild(btn);
    });
  });

  function selectPreset(idx,btn){
    document.querySelectorAll('.chip').forEach(function(c){c.classList.remove('active');});
    btn.classList.add('active');
    if(pollT){clearInterval(pollT);pollT=null;}
    currentJob=null;allPapers={};paperRanks={};rankerOrder=[];
    document.getElementById('results').style.display='none';
    document.getElementById('loading').style.display='block';
    fetch('/api/search',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({preset_idx:idx})
    }).then(function(r){return r.json();}).then(function(d){
      document.getElementById('loading').style.display='none';
      if(d.error){alert('Error: '+d.error);return;}
      currentJob=d.job_id;
      renderTable(d.query,d.n_candidates,d.fast_rankings,d.has_llm,d.llm_rankings||null);
      if(d.has_llm&&!d.llm_rankings){
        setNotice(true);
        pollT=setInterval(pollLlm,2500);
      }
    }).catch(function(e){
      document.getElementById('loading').style.display='none';
      alert('Request failed: '+e);
    });
  }

  function pollLlm(){
    if(!currentJob)return;
    fetch('/api/llm/'+currentJob).then(function(r){return r.json();}).then(function(d){
      if(d.status==='done'){clearInterval(pollT);pollT=null;mergeLlm(d.rankings);}
      else if(d.status==='error'){clearInterval(pollT);pollT=null;showErr(d.error);}
    });
  }

  function ingest(rankings){
    Object.keys(rankings).forEach(function(ranker){
      rankings[ranker].forEach(function(p,i){
        if(!allPapers[p.paper_id])allPapers[p.paper_id]=p;
        if(!paperRanks[p.paper_id])paperRanks[p.paper_id]={};
        paperRanks[p.paper_id][ranker]=i+1;
      });
    });
  }

  function sorted(){
    var ids=Object.keys(allPapers);
    ids.sort(function(a,b){
      return (paperRanks[a]['ss_default']||999)-(paperRanks[b]['ss_default']||999);
    });
    return ids;
  }

  function renderTable(query,nCand,fastR,hasLlm,llmR){
    document.getElementById('res-query').textContent=query;
    document.getElementById('res-meta').textContent=nCand+' candidates \xb7 top 10 shown per ranker';
    rankerOrder=Object.keys(fastR);
    if(llmR)rankerOrder=rankerOrder.concat(Object.keys(llmR));
    ingest(fastR);
    if(llmR)ingest(llmR);
    document.getElementById('thead').innerHTML='<tr><th>Paper</th>'+
      rankerOrder.map(function(r){return '<th id="th-'+esc(r)+'">'+esc(r)+'</th>';}).join('')+'</tr>';
    buildRows(sorted(),hasLlm&&!llmR);
    document.getElementById('results').style.display='block';
  }

  function buildRows(ids,pending){
    document.getElementById('tbody').innerHTML=ids.map(function(pid){
      var p=allPapers[pid];
      var meta=[p.year,p.venue,p.citations>0?p.citations+' citations':null].filter(Boolean).join(' \xb7 ');
      var cells=rankerOrder.map(function(r){
        var fast=(r==='ss_default'||r==='bm25'||r==='dense');
        if(!fast&&pending)return '<td class="pend" id="c-'+esc(pid)+'-'+esc(r)+'"><span class="spin"></span></td>';
        return '<td id="c-'+esc(pid)+'-'+esc(r)+'">'+rb(paperRanks[pid]?paperRanks[pid][r]:null)+'</td>';
      }).join('');
      return '<tr><td><div class="ptitle">'+esc(p.title)+'</div>'+(meta?'<div class="pmeta">'+esc(meta)+'</div>':'')+(p.abstract?'<div class="pabs">'+esc(p.abstract)+'</div>':'')+'</td>'+cells+'</tr>';
    }).join('');
  }

  function mergeLlm(llmR){
    setNotice(false);
    // Add any new ranker columns not yet in the header
    var newR=Object.keys(llmR).filter(function(r){return rankerOrder.indexOf(r)<0;});
    newR.forEach(function(r){
      rankerOrder.push(r);
      var th=document.createElement('th');
      th.id='th-'+r;th.textContent=r;
      document.getElementById('thead').querySelector('tr').appendChild(th);
    });
    ingest(llmR);
    // Update header text (remove any spinner text)
    Object.keys(llmR).forEach(function(r){
      var th=document.getElementById('th-'+r);
      if(th)th.textContent=r;
    });
    // Update existing cells
    sorted().forEach(function(pid){
      Object.keys(llmR).forEach(function(r){
        var cell=document.getElementById('c-'+pid+'-'+r);
        if(!cell)return;
        cell.innerHTML=rb(paperRanks[pid]?paperRanks[pid][r]:null);
        cell.className='';
      });
    });
    // Append new rows for papers the LLM promotes that weren't in fast top-10
    var tbody=document.getElementById('tbody');
    var existingIds={};sorted().forEach(function(id){existingIds[id]=1;});
    Object.keys(llmR).forEach(function(r){
      llmR[r].forEach(function(p){
        if(existingIds[p.paper_id])return;
        existingIds[p.paper_id]=1;
        var meta=[p.year,p.venue,p.citations>0?p.citations+' citations':null].filter(Boolean).join(' \xb7 ');
        var cells=rankerOrder.map(function(rn){
          return '<td id="c-'+esc(p.paper_id)+'-'+esc(rn)+'">'+rb(paperRanks[p.paper_id]?paperRanks[p.paper_id][rn]:null)+'</td>';
        }).join('');
        var tr=document.createElement('tr');
        tr.innerHTML='<td><div class="ptitle">'+esc(p.title)+'</div>'+(meta?'<div class="pmeta">'+esc(meta)+'</div>':'')+(p.abstract?'<div class="pabs">'+esc(p.abstract)+'</div>':'')+'</td>'+cells;
        tbody.appendChild(tr);
      });
    });
  }

  function setNotice(on){
    var el=document.getElementById('llm-notice');
    if(on){el.innerHTML='<span class="spin"></span> LLM reranking in progress (~20 s)…';el.style.display='flex';}
    else{el.style.display='none';el.innerHTML='';}
  }
  function showErr(msg){
    setNotice(false);
    document.querySelectorAll('.pend').forEach(function(c){c.className='err';c.innerHTML='err';});
    console.warn('LLM job failed:',msg);
  }
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Q3 Reranker web demo.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM reranker")
    parser.add_argument("--no-citation", action="store_true", help="Disable citation blend")
    args = parser.parse_args()

    all_rankers = _build_rankers(
        enable_llm=not args.no_llm,
        enable_citation=not args.no_citation,
    )
    stages = load_gold()
    presets = [
        {
            "stage": s.stage,
            "topic": s.topic,
            "query": s.query,
            "retrieval_query": s.retrieval_query,
        }
        for s in stages
    ]

    _Handler._all_rankers = all_rankers
    _Handler._presets = presets

    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    logger.info("Web demo at http://127.0.0.1:%d", args.port)
    logger.info("Rankers: %s", ", ".join(all_rankers))
    logger.info("%d preset stages loaded", len(presets))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
