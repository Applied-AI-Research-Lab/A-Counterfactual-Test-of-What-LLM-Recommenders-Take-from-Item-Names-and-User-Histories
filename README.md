# A Counterfactual Test of What LLM Recommenders Take from Item Names and User Histories


## Article
* **Journal**: Submitted to ACM Transactions on Recommender Systems
* **Title**: Recommending from Memory? A Counterfactual Test of What LLM Recommenders Take from Item Names and User Histories

## Authors
* **Adj. Asst. Prof. Konstantinos I. Roumeliotis | University of Peloponnese**
* **Asoc. Prof. Dionisis Margaris | University of Peloponnese**
* **Asoc. Prof. Dimitris Spiliotopoulos | University of Peloponnese**
* **Prof. Costas Vassilakis | University of Peloponnese**


Code and results of the study. The test keeps users, histories, candidate lists and the prompt fixed and changes only
how items are named (C0 real titles, C1 codes, C2 swapped titles, C3 identity-free descriptions, C4 the titles'
generic words, C5 titles plus descriptions) and adds two history controls (C6, C7: another user's history).

## Setup

```
python -m venv .venv && .venv/bin/pip install -r requirements-server.txt        # vLLM, data, evaluation
python -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt   # SASRec, LoRA, readout
```

The pinned CUDA builds in `requirements-cuda.txt` target CUDA 12.x drivers; adjust them to your system.
The Gemma models are gated on Hugging Face (accept their terms first).

## Data

MovieLens-1M and Amazon Reviews 2023 (Video Games, Beauty and Personal Care, Grocery and Gourmet Food) are downloaded
by `recmem.prepare`. The datasets are not redistributed here.

## Pipeline

Per dataset (`ml-1m`, `amazon-videogames`, `amazon-beauty`, `amazon-grocery`):

```
python -m recmem.prepare    --dataset DS
python -m recmem.split      --dataset DS --n-users 2500                    # main set
python -m recmem.split      --dataset DS --n-users 500 --n-candidates 100  # robustness (a)
python -m recmem.split      --dataset DS --n-users 500 --alpha 0           # robustness (b), uniform negatives
python -m recmem.conditions --dataset DS                                   # item views C0-C3
python -m recmem.views_ext  --dataset DS                                   # item views C4, C5
python -m recmem.baselines  --dataset DS --eval-set SET                    # MostPop, EASE
python -m recmem.sasrec     --dataset DS --eval-set SET                    # SASRec
```

LLM runs use an OpenAI-compatible vLLM server (`scripts/serve.sh GPU PORT MODEL_ID`):

```
python -m recmem.run         --dataset DS --tag TAG --model MODEL_ID --base-url http://localhost:PORT/v1 --eval-set SET   # C0-C3
python -m recmem.run_ext     --dataset DS --tag TAG --model MODEL_ID --base-url http://localhost:PORT/v1 --eval-set SET   # C4-C7
python -m recmem.readout_ext --dataset DS --tag TAG --model MODEL_ID --base-url http://localhost:PORT/v1 --eval-set SET   # label-probability readout
python -m recmem.probes      --dataset DS --tag TAG --model MODEL_ID --base-url http://localhost:PORT/v1 --eval-set SET
python -m recmem.probes_ext  --dataset DS --tag TAG --model MODEL_ID --base-url http://localhost:PORT/v1 --eval-set SET
```

The readout needs the server started with `--max-logprobs 100 --logprobs-mode processed_logprobs`.

Fine-tuning and further references (`.venv-train`):

```
python -m recmem.lora          --dataset DS --model MODEL --base --eval-set SET
python -m recmem.lora          --dataset DS --model MODEL --train-condition {real,masked} --eval-set SET
python -m recmem.lora_ext      --dataset DS --model MODEL --train-condition {real,masked} --seed N --eval-set SET
python -m recmem.baselines_ext {embed,sasrec20,ease20,textsasrec} --dataset DS
```

Evaluation and statistics:

```
python -m recmem.evaluate       --dataset DS --eval-set SET
python -m recmem.evaluate_ext   --dataset DS --eval-set SET
python -m recmem.answer_quality --dataset DS --eval-set SET
python -m recmem.strata         --dataset DS --eval-set SET
python -m recmem.stats_ext      {history,readout,baselines,vsref,catonly,vsbest,cells}
```

Numbers, tables and figures of the paper (written to `paper/generated/` and `paper/figures/`):

```
python scripts/paper_numbers.py
python scripts/paper_figures.py
```

`paper_numbers.py` reads the result files, the processed data, the fine-tuning records and the run logs, so it needs
a complete run of the pipeline.

`scripts/` also contains the job scripts used for the full grid (`grid.sh`, `grid_steps.sh`, `gpu_lane.sh`), the further
runs (`ext_scheduler.py`, `rev_scheduler.py`, `rev2_scheduler.py` with `ext_vllm_job.sh` and `rev_vllm_job2.sh`), the
pilot (`pilot.sh`, `pilot_steps.sh`) and an offline check with a mock model (`preflight_mac.sh`).
Every command takes `--help`.

## Results

`results/` holds the results of the study:

- `summary/`, `ext/summary/`: NDCG@10 and the other metrics per model and condition, paired contrasts with Holm-corrected
  p-values and bootstrap intervals, per-user metrics (`*_per_user.parquet`), probe results, answer completeness,
  history controls, comparisons with the references and the LoRA seeds;
- `raw/`, `ext/baselines/`: EASE and SASRec tuning, full-catalogue metrics, and the training records of the further
  references.

Evaluation sets: `test_n2500_c20_s42` (main), `test_n500_c100_s42` (100 candidates), `test_n500_c20_s42_a0`
(uniform negatives).
