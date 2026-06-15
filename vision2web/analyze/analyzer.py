"""Result analysis module for Vision2Web"""

import json
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict


class ResultAnalyzer:
    """Analyzes evaluation results and generates summary statistics"""

    def __init__(self, results_dir: str = "./results"):
        """
        Initialize result analyzer.

        Args:
            results_dir: Path to results directory
        """
        self.results_dir = Path(results_dir)

    def collect_results(self, datasets_dir: str = "./datasets") -> Dict[str, Any]:
        """
        Collect and summarize test results.

        Args:
            datasets_dir: Path to datasets directory

        Returns:
            Dictionary mapping project keys to test results
        """
        results = {}
        datasets_path = Path(datasets_dir)

        # Iterate through task directories
        for task_dir in self.results_dir.iterdir():
            if not task_dir.is_dir() or task_dir.name.startswith('.'):
                continue

            task_name = task_dir.name

            # Iterate through framework directories
            for framework_dir in task_dir.iterdir():
                if not framework_dir.is_dir() or framework_dir.name.startswith('.'):
                    continue

                framework_name = framework_dir.name

                # Iterate through model directories
                for model_dir in framework_dir.iterdir():
                    if not model_dir.is_dir() or model_dir.name.startswith('.'):
                        continue

                    model_name = model_dir.name

                    # Iterate through project directories
                    for project_dir in model_dir.iterdir():
                        if not project_dir.is_dir() or project_dir.name.startswith('.'):
                            continue

                        project_name = project_dir.name
                        project_key = f"{task_name}:{framework_name}:{model_name}:{project_name}"

                        # Process project
                        try:
                            result = self._process_project(
                                project_dir, task_name, project_name, datasets_path
                            )
                            results[project_key] = result
                        except Exception as e:
                            print(f"Error processing {project_key}: {e}")
                            results[project_key] = {
                                "success": False,
                                "prototypes": {},
                                "function": {},
                                "error": str(e)
                            }

        return results

    def _process_project(
        self,
        project_path: Path,
        task_name: str,
        project_name: str,
        datasets_path: Path
    ) -> Dict[str, Any]:
        """Process a single project and extract test results"""
        # A project counts as successfully evaluated only if its
        # evaluation_result.json exists AND records status == "success".
        # Deploy timeouts / errors now also write this file (with an error
        # status) so they are skipped on re-runs; those must score 0, not
        # be treated as successful, so we check the status rather than mere
        # file existence.
        evaluation_result_file = project_path / "evaluation_result.json"
        success = False
        if evaluation_result_file.exists():
            try:
                with open(evaluation_result_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                success = eval_data.get("status") == "success"
            except (json.JSONDecodeError, IOError):
                success = False

        # Get prototype scores
        prototypes = self._get_prototype_scores(
            project_path, task_name, project_name, success, datasets_path
        )

        # Get function scores
        function = self._get_function_scores(
            project_path, task_name, project_name, success, datasets_path
        )

        return {
            "success": success,
            "prototypes": prototypes,
            "function": function
        }

    def _get_prototype_scores(
        self,
        project_path: Path,
        task_name: str,
        project_name: str,
        success: bool,
        datasets_path: Path
    ) -> Dict[str, float]:
        """Get prototype scores for a project"""
        prototypes = {}

        # Get all prototype names from datasets
        prototypes_dir = datasets_path / task_name / project_name / "prototypes"
        if not prototypes_dir.exists():
            return prototypes

        for prototype_file in prototypes_dir.glob("*.jpg"):
            prototype_name = prototype_file.stem
            prototypes[prototype_name] = 0.0  # Default to 0

        # If success, get scores from test_results
        if success:
            scores_dir = project_path / "test_results" / "prototypes"
            if scores_dir.exists():
                for prototype_name in prototypes.keys():
                    scores_file = scores_dir / f"{prototype_name}_scores.json"
                    if scores_file.exists():
                        try:
                            with open(scores_file, 'r') as f:
                                scores_data = json.load(f)

                            # Calculate mean score
                            if isinstance(scores_data, list):
                                scores = [item.get("score", 0) for item in scores_data if "score" in item]
                                if scores:
                                    prototypes[prototype_name] = sum(scores) / len(scores)
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"Warning: Failed to read {scores_file}: {e}")

        return prototypes

    def _get_function_scores(
        self,
        project_path: Path,
        task_name: str,
        project_name: str,
        success: bool,
        datasets_path: Path
    ) -> Dict[str, int]:
        """Get function test scores for a project"""
        function_scores = {}

        # Read workflow.json to get all test cases with validations
        workflow_file = datasets_path / task_name / project_name / "workflow.json"
        if not workflow_file.exists():
            return function_scores

        try:
            with open(workflow_file, 'r') as f:
                workflows = json.load(f)
        except (json.JSONDecodeError, IOError):
            return function_scores

        # Get all keys from workflow.json
        for workflow_idx, workflow in enumerate(workflows):
            if "content" in workflow:
                for test_case_idx, test_case in enumerate(workflow["content"]):
                    validations = test_case.get("validations", [])
                    if validations:
                        key = f"workflow_{workflow_idx}:test_case_{test_case_idx}"
                        function_scores[key] = 0  # Default to 0

        # If success, get results from test_results
        if success:
            test_results_dir = project_path / "test_results"
            if test_results_dir.exists():
                for key in function_scores.keys():
                    parts = key.split(":")
                    workflow_part = parts[0]
                    test_case_part = parts[1]

                    result_file = test_results_dir / workflow_part / test_case_part / "result.json"
                    if result_file.exists():
                        try:
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)

                            if "Pass" in result_data.get("result", ""):
                                function_scores[key] = 1
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"Warning: Failed to read {result_file}: {e}")

        return function_scores

    def analyze(self, datasets_dir: str = "./datasets") -> Dict[str, Any]:
        """
        Analyze all results and compute statistics.

        Args:
            datasets_dir: Path to datasets directory

        Returns:
            Analysis summary with grouped statistics
        """
        # Collect results
        results = self.collect_results(datasets_dir)

        # Group by task:framework:model
        stats = defaultdict(lambda: {
            'projects': [],
            'success_count': 0,
            'total_count': 0,
            'prototype_scores': [],
            'desktop_scores': [],
            'tablet_scores': [],
            'mobile_scores': [],
            'function_scores': []
        })

        for key, value in results.items():
            parts = key.split(':')
            if len(parts) >= 4:
                task_name = parts[0]
                framework = parts[1]
                model_name = parts[2]
                project_name = ':'.join(parts[3:])

                group_key = f"{task_name}:{framework}:{model_name}"

                # Count success
                stats[group_key]['total_count'] += 1
                if value.get('success', False):
                    stats[group_key]['success_count'] += 1

                # Collect prototype scores
                if 'prototypes' in value:
                    for prototype_name, proto_score in value['prototypes'].items():
                        stats[group_key]['prototype_scores'].append(proto_score)
                        if prototype_name in ['desktop', 'tablet', 'mobile']:
                            stats[group_key][f'{prototype_name}_scores'].append(proto_score)

                # Collect function scores
                if 'function' in value:
                    for func_score in value['function'].values():
                        stats[group_key]['function_scores'].append(func_score)

                stats[group_key]['projects'].append(project_name)

        return {
            'results': results,
            'statistics': dict(stats)
        }

    def print_summary(self, analysis: Dict[str, Any]):
        """Print leaderboard-style table without params/date columns."""
        stats = analysis.get("statistics", {})

        def avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        def parse_key(group_key):
            parts = group_key.split(":")
            if len(parts) < 3:
                return None, None, None
            task = parts[0]
            framework = parts[1]
            model = ":".join(parts[2:])
            return task.lower(), framework, model

        from collections import defaultdict

        agg = defaultdict(lambda: {
            "framework": "",
            "model": "",
            "webpage": {"desktop": 0, "tablet": 0, "mobile": 0, "proto": 0},
            "frontend": {"proto": 0, "func": 0},
            "website": {"proto": 0, "func": 0}
        })

        for key, data in stats.items():
            task, framework, model = parse_key(key)
            agg[(framework, model)]["framework"] = framework
            agg[(framework, model)]["model"] = model

            proto = avg(data.get("prototype_scores", []))
            func = avg(data.get("function_scores", []))
            
            desktop = avg(data.get("desktop_scores", []))
            tablet = avg(data.get("tablet_scores", []))
            mobile = avg(data.get("mobile_scores", []))

            if task == "webpage":
                agg[(framework, model)]["webpage"] = {
                    "desktop": desktop,
                    "tablet": tablet,
                    "mobile": mobile,
                    "proto": proto
                }
            elif task == "frontend":
                agg[(framework, model)]["frontend"] = {"proto": proto, "func": func}
            elif task == "website":
                agg[(framework, model)]["website"] = {"proto": proto, "func": func}

        rows = []
        for (framework, model), d in agg.items():

            l1_desktop = d["webpage"]["desktop"] * 100
            l1_tablet = d["webpage"]["tablet"] * 100
            l1_mobile = d["webpage"]["mobile"] * 100
            l1_avg = d["webpage"]["proto"] * 100

            l2_vs = d["frontend"]["proto"] * 100
            l2_fs = d["frontend"]["func"] * 100
            l2_avg = (l2_vs + l2_fs) / 2

            l3_vs = d["website"]["proto"] * 100
            l3_fs = d["website"]["func"] * 100
            l3_avg = (l3_vs + l3_fs) / 2

            overall = (l1_avg + l2_avg + l3_avg) / 3

            rows.append({
                "name": f"{model} ({framework})",
                "overall": overall,
                "l1_desktop": l1_desktop,
                "l1_tablet": l1_tablet,
                "l1_mobile": l1_mobile,
                "l1_avg": l1_avg,
                "l2_vs": l2_vs,
                "l2_fs": l2_fs,
                "l2_avg": l2_avg,
                "l3_vs": l3_vs,
                "l3_fs": l3_fs,
                "l3_avg": l3_avg
            })

        rows.sort(key=lambda x: x["overall"], reverse=True)

        # column widths
        w_rank = 3
        w_model = 42
        w_score = 8

        total_width = (
            w_rank + w_model + w_score +
            (w_score * 4) +
            (w_score * 3) +
            (w_score * 3) + 20
        )

        sep = "=" * total_width
        print(sep)

                # group header row
        left_width = w_rank + w_model + w_score + 2
        l1_width = w_score * 4
        l2_width = w_score * 3
        l3_width = w_score * 3

        def center(text, width):
            return text.center(width)

        print(
            " " * left_width + " || " +
            center("L1: Static Webpage", l1_width) + " || " +
            center("L2: Interactive Frontend", l2_width) + " || " +
            center("L3: Full-Stack Website", l3_width)
        )

        # column header row
        print(
            f"{'#':<{w_rank}} "
            f"{'Model (Framework)':<{w_model}} "
            f"{'Overall':>{w_score}}"
            f" || "
            f"{'Desktop':>{w_score}}"
            f"{'Tablet':>{w_score}}"
            f"{'Mobile':>{w_score}}"
            f"{'Avg':>{w_score}}"
            f" || "
            f"{'VS':>{w_score}}"
            f"{'FS':>{w_score}}"
            f"{'Avg':>{w_score}}"
            f" || "
            f"{'VS':>{w_score}}"
            f"{'FS':>{w_score}}"
            f"{'Avg':>{w_score}}"
        )

        print(sep)

        for i, r in enumerate(rows, 1):

            def f(x):
                return f"{x:.1f}"

            print(
                f"{i:<{w_rank}} "
                f"{r['name']:<{w_model}} "
                f"{f(r['overall']):>{w_score}}"
                f" || "
                f"{f(r['l1_desktop']):>{w_score}}"
                f"{f(r['l1_tablet']):>{w_score}}"
                f"{f(r['l1_mobile']):>{w_score}}"
                f"{f(r['l1_avg']):>{w_score}}"
                f" || "
                f"{f(r['l2_vs']):>{w_score}}"
                f"{f(r['l2_fs']):>{w_score}}"
                f"{f(r['l2_avg']):>{w_score}}"
                f" || "
                f"{f(r['l3_vs']):>{w_score}}"
                f"{f(r['l3_fs']):>{w_score}}"
                f"{f(r['l3_avg']):>{w_score}}"
            )

        print(sep)