.PHONY: sim quick rq1 fig test all
sim:    ## full simulation -> results/
	python sim/run.py
quick:  ## fast smoke run into /tmp/quick (does not touch results/)
	python sim/run.py --n 50 --n_sens 10 --out /tmp/quick
rq1:    ## RQ1 tables and figure from results_rq1/raw_rq1.csv
	python -m rq1.analysis
fig:    ## architecture diagram
	python tools/fig_architecture.py
test:   ## offline RQ1 pipeline test (skipped without the private inject.py)
	python tests/test_rq1_pipeline.py
all: sim rq1 fig
