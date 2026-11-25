# perfgate — GPU Inference Benchmarking with Statistical Regression Gating
Tech: Python, PyTorch, GPU (Apple MPS), NumPy, Linux shell, Slurm, Docker, GitHub Actions
- Built an automated PyTorch benchmark harness that sweeps GEMM, MLP, ResNet-18 and transformer workloads across CPU/GPU, fp32/fp16 and batch sizes, with Slurm job arrays and Docker.
- Cut false regression alarms from 47 to 4 across 1,512 identical-code runs with a bootstrap-CI gate, and found unsynchronized GPU timing under-reported latency by up to 235x.
