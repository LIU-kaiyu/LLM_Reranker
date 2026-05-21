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

from .arxiv_retriever import ArxivRateLimitError
from .baselines import RankedResult
from .eval import _build_rankers
from .gold import load_gold
from .query_expand import expand as expand_query
from .retriever import Paper
from .sources import get_source, search

logger = logging.getLogger(__name__)

_LIMIT = 20  # fixed candidate pool; keeps latency and SerpAPI cost low
_FAST = frozenset({"ss_default", "bm25", "dense"})

_SOURCE_LABELS = {
    "arxiv": "arXiv",
    "serpapi": "SerpAPI Google Scholar",
    "semantic_scholar": "Semantic Scholar",
    "ss": "Semantic Scholar",
}


def _source_label() -> str:
    return _SOURCE_LABELS.get(get_source(), get_source())

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


def _paper_url(p: Paper | None) -> str | None:
    """Best-effort canonical URL for a paper.

    Built only from validated id/DOI fields (never a raw API-supplied href)
    so the value is safe to embed as an anchor target.
    """
    if p is None:
        return None
    ext = p.external_ids or {}
    arxiv_id = ext.get("arxiv") or ext.get("ArXiv")
    if arxiv_id:
        # The arXiv Atom <id> is the abs page; versionless slug redirects
        # to the latest revision.
        return f"https://arxiv.org/abs/{arxiv_id}"
    doi = ext.get("DOI") or ext.get("doi")
    if doi:
        return f"https://doi.org/{doi}"
    if get_source() in ("semantic_scholar", "ss") and p.paper_id:
        return f"https://www.semanticscholar.org/paper/{p.paper_id}"
    return None


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
        "url": _paper_url(p),
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
            self._send_html(_HTML.replace("__SOURCE__", _source_label()))
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
        if self.path not in ("/api/search", "/api/expand_query"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400, "Bad JSON")
            return

        if self.path == "/api/expand_query":
            raw_text = payload.get("raw")
            if not isinstance(raw_text, str) or not raw_text.strip():
                self.send_error(400, "Provide a non-empty 'raw' string")
                return
            if len(raw_text) > 300:
                self.send_error(400, "Query too long (max 300 chars)")
                return
            try:
                expansion = expand_query(raw_text)
            except Exception as exc:  # noqa: BLE001 - never crash the thread
                logger.error("Query expansion failed: %s", exc, exc_info=True)
                self._send_json({"error": f"Expansion failed: {exc}"}, 503)
                return
            self._send_json({
                "kind": expansion.kind,
                "retrieval_query": expansion.retrieval_query,
                "nl_query": expansion.nl_query,
                "raw": expansion.raw,
            })
            return

        # /api/search
        idx = payload.get("preset_idx")
        raw_query = payload.get("query")
        explicit_retrieval = payload.get("retrieval_query")

        if (
            isinstance(explicit_retrieval, str) and explicit_retrieval.strip()
            and isinstance(raw_query, str) and raw_query.strip()
        ):
            # Reviewed-and-approved smart query: caller supplied both forms.
            retrieval_query = explicit_retrieval.strip()
            nl_query = raw_query.strip()
            if len(retrieval_query) > 300 or len(nl_query) > 500:
                self.send_error(400, "Query too long")
                return
        elif isinstance(raw_query, str) and raw_query.strip():
            # Free-text search: the same string drives both retrieval and the
            # natural-language reranking prompt.
            custom = raw_query.strip()
            if len(custom) > 300:
                self.send_error(400, "Query too long (max 300 chars)")
                return
            retrieval_query = custom
            nl_query = custom
        elif isinstance(idx, int) and 0 <= idx < len(self._presets):
            preset = self._presets[idx]
            retrieval_query = preset["retrieval_query"]
            nl_query = preset["query"]
        else:
            self.send_error(400, "Provide 'query', '{retrieval_query, query}', or a valid 'preset_idx'")
            return

        try:
            papers = search(retrieval_query, limit=_LIMIT)
        except ArxivRateLimitError as exc:
            self._send_json({"error": str(exc)}, 503)
            return
        except Exception as exc:  # noqa: BLE001 - demo must not crash the thread
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            self._send_json({"error": f"Retrieval failed: {exc}"}, 503)
            return

        if not papers:
            self._send_json({"error": "No papers found."}, 404)
            return

        papers_by_id = {p.paper_id: p for p in papers}

        try:
            fast_rankings: dict[str, list[RankedResult]] = {
                name: ranker(nl_query, papers)
                for name, ranker in self._all_rankers.items()
                if name in _FAST
            }
        except Exception as exc:  # noqa: BLE001 - demo must not crash the thread
            logger.error("Fast ranking failed: %s", exc, exc_info=True)
            self._send_json({"error": f"Ranking failed: {exc}"}, 503)
            return
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
.searchbox{display:flex;gap:8px;margin-bottom:24px;max-width:680px}
.searchbox input{flex:1;padding:10px 16px;font-size:14px;border:1px solid #dadce0;border-radius:24px;outline:none;font-family:inherit;color:#202124}
.searchbox input:focus{border-color:#1a73e8;box-shadow:0 1px 6px rgba(26,115,232,.2)}
.searchbox button{padding:10px 22px;font-size:14px;font-weight:500;color:#fff;background:#1a73e8;border:none;border-radius:24px;cursor:pointer}
.searchbox button:hover{background:#1765cc}
#review{display:none;background:#fff;border:1px solid #dadce0;border-radius:8px;padding:18px 18px 14px;margin-bottom:24px;max-width:680px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.rev-hdr{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:12px;font-size:13px;color:#3c4043;flex-wrap:wrap}
.rev-hdr .rev-kind-tag{background:#e8f0fe;color:#1a73e8;padding:2px 10px;border-radius:10px;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.rev-row{display:flex;flex-direction:column;gap:4px;margin-bottom:10px}
.rev-lbl{font-size:11px;color:#5f6368;font-weight:600;text-transform:uppercase;letter-spacing:.6px}
.rev-lbl .rev-sub{text-transform:none;font-weight:400;color:#9aa0a6;margin-left:4px}
.rev-row input{padding:9px 12px;font-size:14px;border:1px solid #dadce0;border-radius:6px;outline:none;font-family:inherit;color:#202124}
.rev-row input:focus{border-color:#1a73e8;box-shadow:0 1px 6px rgba(26,115,232,.2)}
.rev-btns{display:flex;justify-content:flex-end;gap:8px;margin-top:8px}
.rev-btns button{padding:8px 18px;font-size:13px;border-radius:18px;cursor:pointer;border:1px solid transparent;font-family:inherit}
#rev-cancel{background:#fff;border-color:#dadce0;color:#3c4043}
#rev-cancel:hover{background:#f1f3f4}
#rev-go{background:#1a73e8;border-color:#1a73e8;color:#fff;font-weight:500}
#rev-go:hover{background:#1765cc}
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
.ptitle a{color:inherit;text-decoration:none}
.ptitle a:hover{text-decoration:underline}
.pmeta{font-size:11px;color:#5f6368;margin-top:2px}
.pabs{font-size:11px;color:#3c4043;margin-top:4px;line-height:1.5;max-width:400px}
.rb{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;font-size:11px;font-weight:700}
.rb-1{background:#e6f4ea;color:#1e8e3e}
.rb-top{background:#fce8b2;color:#b06000}
.rb-ok{background:#e8f0fe;color:#1a73e8}
.rb-na{background:#f1f3f4;color:#9aa0a6;font-size:13px}
.pend{color:#9aa0a6;font-size:11px}
.err{color:#d93025;font-size:11px}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:#1a73e8}
th.sorted{color:#1a73e8}
.caret{font-size:9px;margin-left:4px}
.delta{font-size:10px;font-weight:700;margin-left:5px}
.d-up{color:#1e8e3e}
.d-dn{color:#d93025}
.d-eq{color:#9aa0a6}
.d-new{color:#1a73e8}
.controls{display:flex;align-items:center;gap:16px;margin:8px 0 12px;font-size:12px;color:#5f6368}
.cmp-toggle{display:inline-flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.hint{color:#9aa0a6}
#errbox{display:none;padding:14px 16px;background:#fce8e6;color:#c5221f;border:1px solid #f5c2c0;border-radius:8px;margin-bottom:14px;font-size:13px;line-height:1.5}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-logo">Reranker Demo</div>
  <div class="hdr-sub">__SOURCE__ &middot; BM25 &middot; Dense &middot; RRF &middot; Cross-encoder &middot; LLM rerank</div>
</div>
<div class="main">
  <div class="sec-title">Search any query</div>
  <form class="searchbox" id="searchform">
    <input id="qinput" type="text" maxlength="300" autocomplete="off"
           placeholder="Type a question OR keywords — the LLM will fill in the other form">
    <button type="submit">Search</button>
  </form>
  <div id="review">
    <div class="rev-hdr">
      <span>LLM read your input as: <span class="rev-kind-tag" id="rev-kind">…</span></span>
      <span class="hint">Edit either field if needed, then Run.</span>
    </div>
    <div class="rev-row">
      <span class="rev-lbl">Keyword query <span class="rev-sub">(sent to the retrieval API)</span></span>
      <input id="rev-kw" type="text" maxlength="300" autocomplete="off">
    </div>
    <div class="rev-row">
      <span class="rev-lbl">Natural-language query <span class="rev-sub">(used as the LLM reranker prompt)</span></span>
      <input id="rev-nl" type="text" maxlength="500" autocomplete="off">
    </div>
    <div class="rev-btns">
      <button id="rev-cancel" type="button">Cancel</button>
      <button id="rev-go" type="button">Run search</button>
    </div>
  </div>
  <div class="sec-title">Or select a benchmark query</div>
  <div class="chips" id="chips"></div>
  <div id="loading"><div class="spin"></div>Retrieving papers&hellip;</div>
  <div id="errbox"></div>
  <div id="results">
    <div class="res-hdr">
      <div class="res-query" id="res-query"></div>
      <div class="res-meta" id="res-meta"></div>
    </div>
    <div class="controls">
      <label class="cmp-toggle"><input type="checkbox" id="cmp-chk"> Compare vs ss_default</label>
      <span class="hint">Click a ranker column to sort &middot; click again to reverse &middot; click "Paper" to reset</span>
    </div>
    <div id="llm-notice"></div>
    <div class="tbl-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
  </div>
</div>
<script>
(function(){
  var presets=[],allPapers={},paperRanks={},rankerOrder=[],currentJob=null,pollT=null;
  var sortCol=null,sortDir=1,compareMode=false,pendingLlm=false,curQuery='',curN=0;

  function esc(s){
    return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
  function badge(rank){
    if(!rank)return '<span class="rb rb-na">&mdash;</span>';
    var c=rank===1?'rb-1':rank<=3?'rb-top':'rb-ok';
    return '<span class="rb '+c+'">'+rank+'</span>';
  }
  function deltaHtml(pid,ranker){
    if(ranker==='ss_default')return '';
    var pr=paperRanks[pid]||{};
    var base=pr['ss_default'],r=pr[ranker];
    if(!r)return '';
    if(!base)return '<span class="delta d-new">new</span>';
    var d=base-r;
    if(d>0)return '<span class="delta d-up">&#9650;'+d+'</span>';
    if(d<0)return '<span class="delta d-dn">&#9660;'+(-d)+'</span>';
    return '<span class="delta d-eq">=</span>';
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

  document.getElementById('cmp-chk').addEventListener('change',function(e){
    compareMode=e.target.checked;render();
  });

  function showErr(msg){
    var b=document.getElementById('errbox');
    b.textContent=msg;b.style.display='block';
  }
  function clearErr(){
    var b=document.getElementById('errbox');
    b.style.display='none';b.textContent='';
  }

  function selectPreset(idx,btn){
    document.querySelectorAll('.chip').forEach(function(c){c.classList.remove('active');});
    btn.classList.add('active');
    runSearch({preset_idx:idx});
  }

  function hideReview(){document.getElementById('review').style.display='none';}
  function showReview(d){
    var kind=d.kind==='nl'?'natural-language':d.kind==='keyword'?'keywords':'unclassified';
    document.getElementById('rev-kind').textContent=kind;
    document.getElementById('rev-kw').value=d.retrieval_query||'';
    document.getElementById('rev-nl').value=d.nl_query||'';
    document.getElementById('review').style.display='block';
    document.getElementById('rev-kw').focus();
  }
  function setLoading(msg){
    var el=document.getElementById('loading');
    el.innerHTML='<div class="spin"></div>'+msg;
    el.style.display='block';
  }

  document.getElementById('searchform').addEventListener('submit',function(e){
    e.preventDefault();
    var q=document.getElementById('qinput').value.trim();
    if(!q){showErr('Enter a query to search.');return;}
    document.querySelectorAll('.chip').forEach(function(c){c.classList.remove('active');});
    hideReview();
    clearErr();
    document.getElementById('results').style.display='none';
    setLoading('Understanding your query&hellip;');
    fetch('/api/expand_query',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({raw:q})
    }).then(function(r){return r.json();}).then(function(d){
      document.getElementById('loading').style.display='none';
      if(d.error){showErr(d.error);return;}
      showReview(d);
    }).catch(function(e){
      document.getElementById('loading').style.display='none';
      showErr('Expansion request failed: '+e);
    });
  });

  document.getElementById('rev-cancel').addEventListener('click',hideReview);
  document.getElementById('rev-go').addEventListener('click',function(){
    var kw=document.getElementById('rev-kw').value.trim();
    var nl=document.getElementById('rev-nl').value.trim();
    if(!kw||!nl){showErr('Both fields are required.');return;}
    hideReview();
    setLoading('Retrieving papers&hellip;');
    runSearch({retrieval_query:kw, query:nl});
  });

  function runSearch(body){
    if(pollT){clearInterval(pollT);pollT=null;}
    currentJob=null;allPapers={};paperRanks={};rankerOrder=[];
    sortCol=null;sortDir=1;pendingLlm=false;
    clearErr();
    document.getElementById('results').style.display='none';
    setLoading('Retrieving papers&hellip;');
    fetch('/api/search',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)
    }).then(function(r){return r.json();}).then(function(d){
      document.getElementById('loading').style.display='none';
      if(d.error){showErr(d.error);return;}
      currentJob=d.job_id;curQuery=d.query;curN=d.n_candidates;
      rankerOrder=Object.keys(d.fast_rankings);
      ingest(d.fast_rankings);
      if(d.llm_rankings){
        addRankers(d.llm_rankings);ingest(d.llm_rankings);pendingLlm=false;
      }else if(d.has_llm){
        pendingLlm=true;pollT=setInterval(pollLlm,2500);
      }
      render();
    }).catch(function(e){
      document.getElementById('loading').style.display='none';
      showErr('Request failed: '+e);
    });
  }

  function pollLlm(){
    if(!currentJob)return;
    fetch('/api/llm/'+currentJob).then(function(r){return r.json();}).then(function(d){
      if(d.status==='done'){
        clearInterval(pollT);pollT=null;pendingLlm=false;
        addRankers(d.rankings);ingest(d.rankings);render();
      }else if(d.status==='error'){
        clearInterval(pollT);pollT=null;pendingLlm=false;
        showErr('LLM rerank failed: '+d.error);render();
      }
    });
  }

  function addRankers(rankings){
    Object.keys(rankings).forEach(function(r){
      if(rankerOrder.indexOf(r)<0)rankerOrder.push(r);
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

  function orderedIds(){
    var key=sortCol||'ss_default';
    var ids=Object.keys(allPapers);
    ids.sort(function(a,b){
      var ra=(paperRanks[a]&&paperRanks[a][key])||9999;
      var rbk=(paperRanks[b]&&paperRanks[b][key])||9999;
      if(ra!==rbk)return (ra-rbk)*(sortCol?sortDir:1);
      var sa=(paperRanks[a]&&paperRanks[a]['ss_default'])||9999;
      var sb=(paperRanks[b]&&paperRanks[b]['ss_default'])||9999;
      return sa-sb;
    });
    return ids;
  }

  function setSort(col){
    if(col===null){sortCol=null;sortDir=1;}
    else if(sortCol===col){sortDir=-sortDir;}
    else{sortCol=col;sortDir=1;}
    render();
  }

  function setNotice(on){
    var el=document.getElementById('llm-notice');
    if(on){el.innerHTML='<span class="spin"></span> LLM reranking in progress (~20 s)…';el.style.display='flex';}
    else{el.style.display='none';el.innerHTML='';}
  }

  function render(){
    document.getElementById('res-query').textContent=curQuery;
    document.getElementById('res-meta').textContent=curN+' candidates \xb7 top 10 shown per ranker';
    setNotice(pendingLlm);
    var caret=function(col){
      if(sortCol!==col)return '';
      return '<span class="caret">'+(sortDir>0?'&#9650;':'&#9660;')+'</span>';
    };
    var pTh='<th class="sortable" data-col="__paper__">Paper'+(sortCol===null?'<span class="caret">&#9650;</span>':'')+'</th>';
    var rTh=rankerOrder.map(function(r){
      var cls='sortable'+(sortCol===r?' sorted':'');
      return '<th class="'+cls+'" data-col="'+esc(r)+'">'+esc(r)+caret(r)+'</th>';
    }).join('');
    var thead=document.getElementById('thead');
    thead.innerHTML='<tr>'+pTh+rTh+'</tr>';
    thead.querySelectorAll('th.sortable').forEach(function(th){
      th.onclick=function(){
        var c=th.getAttribute('data-col');
        setSort(c==='__paper__'?null:c);
      };
    });
    var ids=orderedIds();
    document.getElementById('tbody').innerHTML=ids.map(function(pid){
      var p=allPapers[pid];
      var meta=[p.year,p.venue,p.citations>0?p.citations+' citations':null].filter(Boolean).join(' \xb7 ');
      var cells=rankerOrder.map(function(r){
        var rk=paperRanks[pid]?paperRanks[pid][r]:null;
        var d=compareMode?deltaHtml(pid,r):'';
        return '<td>'+badge(rk)+d+'</td>';
      }).join('');
      var titleInner=p.url?'<a href="'+esc(p.url)+'" target="_blank" rel="noopener noreferrer">'+esc(p.title)+'</a>':esc(p.title);
      return '<tr><td><div class="ptitle">'+titleInner+'</div>'+(meta?'<div class="pmeta">'+esc(meta)+'</div>':'')+(p.abstract?'<div class="pabs">'+esc(p.abstract)+'</div>':'')+'</td>'+cells+'</tr>';
    }).join('');
    document.getElementById('results').style.display='block';
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
