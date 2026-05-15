.PHONY: setup retrieve baselines demo eval eval-stage1 gold clean

PYTHON ?= python3
Q ?= monocular depth estimation SOTA

setup:
	$(PYTHON) -m pip install -e .

retrieve:
	$(PYTHON) -m q3_reranker.retriever "$(Q)"

baselines:
	$(PYTHON) -m q3_reranker.baselines "$(Q)"

demo:
	$(PYTHON) -m q3_reranker.demo "$(Q)"

eval:
	$(PYTHON) -m q3_reranker.eval $(EVAL_ARGS)

eval-stage1:
	$(PYTHON) -m q3_reranker.eval --stage 1

gold:
	$(PYTHON) -m q3_reranker.gold

clean:
	rm -rf data/cache
	find . -type d -name __pycache__ -exec rm -rf {} +
