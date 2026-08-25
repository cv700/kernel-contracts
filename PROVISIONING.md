# Provisioning: first measured entries

What to rent, what to install, what to run, in order. Prices checked
August 2026; re-check before spending money, they move.

## Cost estimate

The actual compute here is tiny -- a few matmuls and softmax calls at
n<=1024, seconds of GPU time each. The cost is instance-hours for setup
and iteration, not the workload itself.

| Provider | GPU | On-demand rate | Notes |
|---|---|---|---|
| RunPod | H100 PCIe | $2.89/hr | Simplest onboarding, community templates come with torch+CUDA preinstalled |
| RunPod | H100 SXM | $3.29/hr | Not needed for this -- PCIe is fine, no multi-GPU or NVLink use here |
| TensorWave | MI300X | $1.71/hr | Cheapest MI300X on-demand found; newer provider, check current reliability |
| Hot Aisle | MI300X | $2.99/hr | More established; higher rate |
| RunPod | MI300X | $2.39/hr | Middle ground, same platform as the NVIDIA box -- one dashboard, one billing account |

Budget: 2 hours per platform (install deps, run `selftest`, run four
`measure` calls per operator, download results, tear down), doubled for a
first-time debugging buffer. **Realistic total: $15-25.** Even a fully
blown first attempt with restarts is unlikely to clear $50.

Recommended pair for the first pass, optimizing for fewest moving parts
over lowest price: **RunPod H100 PCIe + RunPod MI300X** -- one account,
one dashboard, both templates come with a working torch build so you're
not debugging ROCm installation from scratch on top of everything else.
Switch to TensorWave for the AMD side later if cost matters more than
convenience once this is routine.

## Images

- **NVIDIA:** RunPod's "PyTorch 2.x" community template (CUDA preinstalled,
  torch already matches the CUDA version). Confirm at launch time --
  templates get bumped; you want torch built against whatever CUDA the
  image ships.
- **AMD:** RunPod's ROCm-specific PyTorch template if one is current at
  launch time; otherwise a bare ROCm image plus
  `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`
  (check https://pytorch.org/get-started/locally/ for the current ROCm
  version tag -- this changes and using the wrong one is the most likely
  first failure mode).

## Steps, per box

```bash
git clone https://github.com/cv700/kernel-contracts.git
cd kernel-contracts
pip install -r requirements.txt   # numpy + jsonschema; torch is already on the image

# Confirm torch sees the device before anything else:
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Validate the harness on THIS silicon before spending more time:
./kc selftest

# One command per operator, runs both candidates, writes one file:
./kc measure matmul --device cuda --n 1024 --out matmul_nvidia.json
```

Repeat on the second box (`--out matmul_amd.json`). Download both files to
one machine (`scp` or the provider's file browser) -- `combine` needs both
at once but doesn't need a GPU itself:

```bash
./kc combine matmul_nvidia.json matmul_amd.json
```

That's it -- no flags to fill in by hand. `combine` refuses to call the
result `status: measured` unless both readings are `--device cuda` on two
different silicon arches; if either check fails it writes `illustrative`
instead rather than erroring, so you always get a file, just correctly
labeled.

## What you'll actually be looking at

For matmul (C-PRC-01), the interesting result isn't whether it passes --
it's whether NVIDIA's and AMD's own default accumulation behavior at fp32
diverges enough to matter, independent of the injected-bad candidate. The
injected-bad candidate is there to prove the contract *would* catch a
real failure; the conforming candidate's actual measured divergence
between the two vendors' default kernels is the first real corpus number.
Record both, not just the calibration soundness.

## Teardown

Terminate both instances as soon as the readings are downloaded --
nothing after that needs the GPU. `pair` runs anywhere.
