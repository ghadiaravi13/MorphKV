# MorphKV: Dynamic Token eviction for efficient KV cache management 

<p align="center">
  <img src="Figs/morphboy.png" width="300" height="300">
</p>

## [ICML'25] Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs

This repository contains the code for MorphKV, a dynamic KV cache compression technique that delivers massive memory savings compared to SOTA methods like SnapKV and $H_2O$,
while also improving upon the benchmark accuracy scores. MorphKV uses a window of recent tokens to gather information about the importance of the distant context tokens.
Hence, it uses a two-fold approach: 1. retain all the recent window tokens in the KV cache, and 2. Identify the top-K most important distant tokens and retain them in the KV cache.

**Paper Link: https://arxiv.org/pdf/2503.00979**

<p align="center">
  <img src="Figs/overview.png" width="800" height="200">
</p>

Unlike prior methods like SnapKV, MorphKV is a dynamic algorithm and performs eviction at every timestep, thereby maintaining a constant-sized KV cache throughout inference.
Further, MorphKV also accounts for GQA, thereby allowing better practical adoption, since many models today use GQA as an architecture choice.


## MorphKV - Design
<p align="center">
  <img src="Figs/design_main.png" width="800" height="400">
</p>

Fundamentally, MorphKV's design leverages two key aspects of LLM Inference: 1. retaining recent tokens for local coherence and, 2. Identifying important past tokens for distant relevance. In the example shown above, note how the relevant context dynamically shifts as token generation progresses.
By leveraging this behaviour, MorphKV allows to retain a very lightweight KV cache comprising of only the relevant information, massively reducing the memory overhead, thereby improving system-throughput.

### Usage
MorphKV is integrated within the huggingface transformer library, and hence can be used via simple monkeypatching of a few transformer classes. 

#### 1. Pre-Requisites
Currently, MorphKV is thoroughly tested with transformers 4.45.0 and hence we recommend maintaining this version of transformers for running MorphKV, particularly since the attention class has undergone major restructuring in the recent versions.

<pre>
  pip install transformers==4.45.0
</pre>

#### 2. Install morphkv

<pre>
  git clone https://github.com/ghadiaravi13/MorphKV/
  cd MorphKV/
  pip install .
</pre>

# Benchmarks

## [LongGenBench](https://github.com/mozhu621/LongGenBench/)

We test the effectiveness of MorphKV on long-response generation task: LongGenBench. The code present here is derived from the original LongGenBench repository.

### Performance


**Note:** MorphKV achieves better scores across almost all evaluation metrics — Completion Rate (CR), Accuracy Once, Accuracy Range, Accuracy Periodic, and Accuracy Average.

| Model Group | Variant     | CR (%) | Once (%) | Range (%) | Periodic (%) | Avg. (%) |
|-------------|-------------|--------|----------|------------|----------------|-----------|
| **Llama**   | H2O         | 64     | 45       | 60         | **27**         | 44        |
|             | SnapKV      | 64     | 50       | 55         | 26             | 44        |
|             | **MorphKV** | **64** | **50**   | **61**     | 24             | **45**    |
| **Mistral** | H2O         | 71.2   | 57       | 60         | 32             | 50        |
|             | SnapKV      | 71     | 55       | 57         | 36             | 49        |
|             | **MorphKV** | **71.2**| **57**   | **62**     | **36**         | **52**    |
| **Qwen**    | H2O         | **55** | **46**   | 51         | 28             | 42        |
|             | SnapKV      | 53     | 44       | 46         | 28             | 39        |
|             | **MorphKV** | 51     | 43       | **68**     | **30**         | **47**    |

### Running LongGenBench

Launching the inference on LongGenBench: The model generates response for the corresponding LongGenBench tasks such as writing a floor-plan, diary etc. and gets saved to the JSON file. Subsequently, the JSON file is used to perform eval.

<pre>
  cd LongGenBench
  python inference_hf.py --model mistral -ws 200 -mc 4000 --morph_type max_fused --input_file ../Dataset/Dataset_short.json --preds_path preds --output_file preds/Mistral.json
</pre>

### Evaluating LongGenBench
<pre>
  python eval.py --data preds/Mistral.json --csv preds/lgb_eval.csv
</pre>

## [LongBench](https://github.com/THUDM/LongBench)

We also evaluate MorphKV performance on LongBench, which is a long-context benchmark-suite with diverse benchmarks across retrieval, reasoning, and Question-Answering. The code present in this repository is derived from the original LongBench repository.

### Performance

| Model   | Variant        | 2wmqa    | drdr     | hpqa     | mnews    | mfqa\_en | mfqa\_zh | musq     | nqa      | pcnt     | prt      | qsp      | qms      | sams     | tqa      | vcs      |
| ------- | -------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Llama   | SnapKV         | **16.0** | 22.0     | 14.9     | 25.6     | 25.4     | 18.7     | **10.7** | **32.2** | **7.6**  | **98.4** | 11.7     | 23.1     | **42.9** | **91.7** | 14.2     |
|         | MorphKV (ours) | 14.9     | **22.5** | **15.9** | **26.6** | **25.7** | **19.9** | **10.7** | 31.9     | 7.5      | 97.8     | **11.9** | **23.6** | **42.9** | 91.5     | **15.2** |
|         | Full Attention | 16.5     | 30.0     | 16.7     | 26.8     | 27.4     | 20.1     | 11.4     | 32.0     | 6.9      | 97.7     | 13.2     | 23.6     | 43.7     | 91.6     | 16.1     |
| Mistral | SnapKV         | 26.6     | 23.7     | 40.5     | 26.0     | **48.8** | 41.3     | **18.3** | 25.6     | 2.5      | **88.6** | **31.0** | **23.8** | 41.9     | **86.3** | 13.5     |
|         | MorphKV (ours) | **26.7** | **23.9** | **40.8** | **26.6** | 48.4     | **43.0** | 16.7     | **26.7** | **3.0**  | 85.9     | 30.9     | 23.6     | **42.3** | **86.3** | **13.7** |
|         | Full Attention | 27.1     | 30.4     | 43.0     | 27.1     | 49.2     | 48.3     | 18.8     | 26.7     | 2.8      | 87.0     | 33.0     | 24.2     | 42.8     | 86.2     | 15.2     |
| Phi-4   | SnapKV         | 22.3     | **24.2** | **19.5** | 25.0     | 38.0     | **47.2** | 5.2      | 20.5     | **12.6** | 63.9     | **32.4** | 22.1     | 47.2     | 90.5     | 11.4     |
|         | MorphKV (ours) | **22.6** | 24.1     | 19.3     | **25.5** | **38.2** | 46.4     | **6.2**  | **21.0** | **12.6** | **64.3** | 31.2     | **22.4** | **47.6** | **90.6** | **12.3** |
|         | Full Attention | 22.2     | 29.0     | 19.6     | 25.9     | 38.2     | 48.9     | 6.0      | 20.7     | 11.6     | 63.3     | 33.3     | 22.9     | 48.2     | 90.4     | 13.4     |


### Running LongBench

<pre>
  python pred_single.py --model mistral -ws 32 -mc 2000 --morph_type sum_fused --pred_path preds
</pre>

### Evaluating LongBench
<pre>
  python eval.py --model mistral --pred_path preds
</pre>

