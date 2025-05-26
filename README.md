# MorphKV

## [ICML 2025] Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs

This repository contains the code for MorphKV, a dynamic KV cache compression technique that delivers massive memory savings compared to SOTA methods like SnapKV and $H_2O$, \
while also improving upon the benchmark accuracy scores.

**Paper Link: https://arxiv.org/pdf/2503.00979**

**ICML webpage: https://icml.cc/virtual/2025/poster/45197**

### Usage
MorphKV is integrated within the huggingface transformer library, and hence can be used via simple monkeypatching of a few transformer classes. 

#### 1. Pre-Requisites
Currently, MorphKV is thoroughly tested with transformers 4.45.0 and hence we recommend maintaining this version of transformers for running MorphKV, particularly since the attention class has undergone major restructuring in the recent versions.

<pre>
```pip install transformers==4.45.0
```
</pre>

#### 2. 

