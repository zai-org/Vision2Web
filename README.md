# Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification

![Web Development](https://img.shields.io/badge/Task-Web%20Development-red)
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red)
![Vision2Web](https://img.shields.io/badge/Dataset-Vision2Web-blue)
<div align='center'>

[[🏠 Project Page](https://vision2web-bench.github.io/)] [[📖 arXiv Paper](https://arxiv.org/abs/2603.26648)] [[📊 Dataset](https://huggingface.co/datasets/zai-org/Vision2Web)] [[📮 Submit Results](https://huggingface.co/datasets/zai-org/Vision2Web-Leaderboard)]

</div>

<p align="center">
    <img src="./docs/images/vision2web-cover.png" width="85%">
</p>

Vision2Web is a comprehensive benchmark designed to evaluate multimodal coding agents on visual website development tasks spanning the full software development lifecycle.

---

## 🔥 News

* **`2026.06.15`** 🔧 Functional testing now runs on Claude Code + `playwright-cli`, and inference adds a Codex framework alongside the Claude Code CLI.
* **`2026.04.30`** 🎉 Vision2Web is accepted by ICML 2026 as a Spotlight Paper!
* **`2026.03.30`** 🌟 We released Vision2Web with comprehensive evaluation tools and leaderboard!

## 👀 Introduction to Vision2Web

Vision2Web is a hierarchical benchmark for evaluating multimodal coding agents on end-to-end visual website development, measuring their ability to integrate UI understanding, requirements reasoning, interactive logic, and full-stack implementation in long-horizon scenarios.

<p align="center">
  <img src="./docs/images/compare_bench.png" width="70%">
</p>

The benchmark is organized into three progressive levels:

- **Level 1 – Static Webpage:** Generate responsive, executable webpages from multi-device UI prototypes (desktop / tablet / mobile).  
  *Metric:* Visual Score (VS)

- **Level 2 – Interactive Frontend:** Develop multi-page interactive frontends with coherent navigation flows from multiple prototypes and textual specifications.  
  *Metrics:* Visual Score (VS) + Functional Score (FS)

- **Level 3 – Full-Stack Website:** Build complete full-stack systems from structured requirement documents and visual prototypes, handling state management and backend logic.  
  *Metrics:* Visual Score (VS) + Functional Score (FS)

Evaluation is conducted via a workflow-based agent verification paradigm that combines GUI agent verifiers for functional correctness and VLM-based judges for visual fidelity, enabling scalable and implementation-agnostic assessment across increasing levels of complexity.

---

## 📊 Benchmark Statistics

Vision2Web comprises **193 tasks** spanning 16 subcategories across 4 major domains (E-Commerce, SaaS, Content, and Public Service), supported by **918 prototype images** and **1,255 functional test cases**.

<table align="center">
<tr>
<td align="center" width="50%">
  <img src="./docs/images/task_distribution.png" width="100%"/>
</td>

<td align="center" width="50%">
  <img src="./docs/images/test_case_distribution.png" width="100%"/><br/><br/>
  <img src="./docs/images/compare_task.png" width="80%"/>
</td>
</tr>
</table>


## 📥 Dataset

### License

Vision2Web is licensed under CC-BY-NC-SA-4.0 and is intended for academic research only. Commercial use in any form is prohibited.

### Download

The dataset is organized in the following structure:

```
datasets/
├── webpage/              # Level 1: Static Webpage (100 tasks)
├── frontend/             # Level 2: Interactive Frontend (66 tasks)
└── website/              # Level 3: Full-Stack Website (27 tasks)
```

Each task directory contains:
- `prototypes/`: UI prototype images (desktop/tablet/mobile)
- `resources/`: Multimedia assets (images, icons, videos, fonts)
- `workflow.json`: Test workflow specification
- `prompt.txt`: Textual requirements (Level 2 only)
- `prd.md`: Requirement Document (Level 3 only)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Docker

### Install Vision2Web

```bash
# Clone repository
git clone https://github.com/zai-org/Vision2Web.git
cd Vision2Web

# Install package
pip install -e .
```

## 🔧 Quick Start

### Step 1: Build Docker Sandbox

The Docker sandbox provides isolated environments for running inference and evaluation:

```bash
cd docker
bash build.sh
```

This builds the `vision2web-sandbox:latest` image with all necessary dependencies.

The sandbox ships **Claude Code**, **Codex**, **OpenHands**, and **`playwright-cli`** (with its agent skills).

### Step 2: Configure Model Endpoints

**Recommended: use each agent's native API** — Claude Code with the Anthropic API, Codex with the OpenAI API — to avoid deviations introduced by cross-format conversion.

Alternatively, you can route through [LiteLLM](https://github.com/BerriAI/litellm) as a proxy for unified model routing across providers:

```bash
# Install LiteLLM
pip install litellm[proxy]

# Start LiteLLM proxy with your configuration
litellm --config litellm_config.yaml
```

### Step 3: Run Inference

Execute inference to generate project implementations:

```bash
bash scripts/run_inference.sh
```

**Key Parameters**:
- `--framework`: Agent framework (`claude_code`, `codex`, or `openhands`)
- `--model`: Model identifier (should match LiteLLM configuration)
- `--base-url`: API base URL (use LiteLLM proxy endpoint)
- `--task`: Task type filter (`webpage`, `frontend`, or `website`)
- `--projects`: Specific project names to run (optional)
- `--max-workers`: Number of concurrent inference tasks
- `--timeout`: Max seconds for a single task run before it is killed (default: 7200)

**Results Structure**:
```
results/
└── webpage|frontend|website/
    └── framework/
        └── model/
            └── project_name/
                ├── start.sh              # Deployment script
                ├── prototypes/           # Copied prototypes
                └── resources/            # Copied resources
```

### Step 4: Run Evaluation

After inference completes, run automated evaluation:

```bash
bash scripts/run_evaluation.sh
```

Evaluation has two phases:
- **Functional testing**: A Claude Code session drives the deployed app through each workflow with `playwright-cli`, checking validation criteria and recording Pass/Fail/Blocked per test case.
- **Visual scoring**: A VLM judge compares prototype images against actual page screenshots captured during the run.

Or use the CLI with separate model endpoints for the two phases:

**Key Parameters**:
- `--functional-model`: Model for functional testing via Claude Code
- `--functional-api-key`: API key (auth token) for the functional testing model
- `--functional-base-url`: API base URL for the functional testing model
- `--visual-model`: Model for visual prototype comparison
- `--visual-api-key`: API key for the visual scoring model
- `--visual-base-url`: API base URL for the visual scoring model
- `--model`: Filter for inference model results to evaluate
- `--framework`: Filter for framework results to evaluate
- `--task`: Filter for task type to evaluate

**Evaluation Outputs**:
```
project_name/
└── test_results/
    ├── workflow_i/
    │   └── test_case_i/
    │       ├── result.json
    │       └── screenshots/
    └── prototypes/
        ├── desktop_actual.jpg
        └── desktop_scores.json
```

### Step 5: Analyze Results

Generate summary statistics and visualizations:

```bash
bash scripts/run_analysis.sh
```

### Step 6: Submit Leaderboard Results

You can run evaluation locally to test your agent's performance. **Official leaderboard scores are evaluated by the maintainers** using the latest VLM Judge and GUI Agent.

To submit to the leaderboard, you only need to submit your **inference outputs**. Please follow the submission guidelines in the [leaderboard repository](https://huggingface.co/datasets/zai-org/Vision2Web-Leaderboard).

## 📊 Experimental Results
- Overall Performance
<p align="center">
    <img src="./docs/images/all_results.png" width="95%">
</p>

- Performance across Page Size
<p align="center">
    <img src="./docs/images/height_performance.png" width="55%">
</p>

- Performance across Task Categories
<p align="center">
    <img src="./docs/images/task_performance.png" width="55%">
</p>

- Performance across Test Cases
<p align="center">
    <img src="./docs/images/test_case_performance.png" width="55%">
</p>

## ✒️ Citation

If you find Vision2Web helpful for your research, please consider citing:

```bibtex
@misc{he2026vision2webhierarchicalbenchmarkvisual,
      title={Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification},
      author={Zehai He and Wenyi Hong and Zhen Yang and Ziyang Pan and Mingdao Liu and Xiaotao Gu and Jie Tang},
      year={2026},
      eprint={2603.26648},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2603.26648},
}
```
