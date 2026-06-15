"""Command-line interface for Vision2Web"""

import sys
import asyncio
import click
from pathlib import Path

from vision2web import __version__
from vision2web.core.config import Config
from vision2web.core.logger import setup_logger
from vision2web.inference import InferenceEngine
from vision2web.evaluation import EvaluationEngine
from vision2web.analyze import ResultAnalyzer


@click.group()
@click.version_option(version=__version__)
def cli():
    """Vision2Web: Benchmark for visual web development with AI agents"""
    pass


@cli.command()
@click.option('--framework', required=True,
              type=click.Choice(['claude_code', 'openhands', 'codex']),
              help='Agent framework to use')
@click.option('--model', required=True, help='Model name to use')
@click.option('--api-key', required=True, help='API key for the agent framework')
@click.option('--base-url', help='Optional API base URL')
@click.option('--sandbox', 'sandbox_image', default='vision2web-sandbox:latest',
              help='Docker sandbox image name')
@click.option('--datasets-dir', type=click.Path(exists=True, path_type=Path),
              default='./datasets', help='Datasets directory')
@click.option('--results-dir', type=click.Path(path_type=Path),
              default='./results', help='Results output directory')
@click.option('--max-workers', type=int, default=5,
              help='Maximum concurrent workers')
@click.option('--timeout', type=int, default=7200,
              help='Max seconds for a single task run before it is killed (default: 7200)')
@click.option('--task', type=click.Choice(['webpage', 'frontend', 'website']),
              help='Task type to run (default: all)')
@click.option('--projects', multiple=True, help='Specific projects to run (default: all)')
def inference(framework, model, api_key, base_url, sandbox_image, datasets_dir,
              results_dir, max_workers, timeout, task, projects):
    """Run inference phase to generate projects from specifications"""

    logger = setup_logger('vision2web', level='INFO')

    # Create config
    config = Config()
    config.datasets_dir = datasets_dir
    config.results_dir = results_dir
    config.sandbox.image = sandbox_image
    config.inference.framework = framework
    config.inference.model = model
    config.inference.api_key = api_key
    config.inference.base_url = base_url
    config.inference.max_workers = max_workers
    config.inference.timeout = timeout
    config.inference.task = task
    config.ensure_dirs()

    # Log configuration
    logger.info("=" * 60)
    logger.info("Vision2Web Inference")
    logger.info("=" * 60)
    logger.info(f"Framework: {framework}")
    logger.info(f"Model: {model}")
    logger.info(f"Sandbox: {sandbox_image}")
    logger.info(f"Task type: {task or 'all'}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Timeout: {timeout}s")
    logger.info(f"Datasets: {datasets_dir}")
    logger.info(f"Results: {results_dir}")
    logger.info("=" * 60)

    engine = InferenceEngine(config)

    # Run inference
    try:
        project_list = list(projects) if projects else None
        results = asyncio.run(engine.run_all_projects(
            project_names=project_list,
            task_type=task
        ))

        # Print summary
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = len(results) - successful

        task_summary = {}
        for result in results:
            t = result.get('task_type', 'unknown')
            if t not in task_summary:
                task_summary[t] = {'total': 0, 'success': 0, 'failed': 0}
            task_summary[t]['total'] += 1
            if result.get('status') == 'success':
                task_summary[t]['success'] += 1
            else:
                task_summary[t]['failed'] += 1

        click.echo()
        click.echo("=" * 60)
        click.echo("INFERENCE COMPLETE")
        click.echo("=" * 60)
        click.echo(f"Total: {len(results)}")
        click.echo(f"Success: {successful}")
        click.echo(f"Failed: {failed}")
        click.echo()
        click.echo("By Task Type:")
        for task_name, stats in sorted(task_summary.items()):
            click.echo(f"  {task_name}: {stats['success']}/{stats['total']} succeeded")
        click.echo()
        click.echo(f"Results: {results_dir}")
        click.echo("=" * 60)

        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--results-dir', type=click.Path(exists=True, path_type=Path),
              required=True, help='Inference results directory')
@click.option('--datasets-dir', type=click.Path(exists=True, path_type=Path),
              required=True, help='Datasets directory')
@click.option('--sandbox', 'sandbox_image', default='vision2web-sandbox:latest',
              help='Docker sandbox image name')
@click.option('--functional-model', default='claude-sonnet-4-5-20250929',
              help='Model for functional testing via Claude Code')
@click.option('--functional-api-key', required=True,
              help='API key (auth token) for the functional testing model')
@click.option('--functional-base-url',
              help='API base URL for the functional testing model (e.g. LiteLLM proxy)')
@click.option('--visual-api-key', required=True, help='API key for visual scoring model')
@click.option('--visual-base-url', default='https://api.openai.com/v1',
              help='API base URL for visual scoring model')
@click.option('--visual-model', default='gemini-3-pro-preview',
              help='Model for visual prototype comparison')
@click.option('--max-workers', type=int, default=1,
              help='Maximum concurrent evaluations')
@click.option('--task', type=click.Choice(['webpage', 'frontend', 'website']),
              help='Task type to evaluate (default: all)')
@click.option('--framework', help='Framework to evaluate (default: all)')
@click.option('--model', 'eval_model_filter', help='Inference model to evaluate (default: all)')
def evaluate(results_dir, datasets_dir, sandbox_image, functional_model,
             functional_api_key, functional_base_url,
             visual_api_key, visual_base_url, visual_model, max_workers,
             task, framework, eval_model_filter):
    """Run evaluation phase to test generated projects"""

    logger = setup_logger('vision2web', level='INFO')

    # Create config
    config = Config()
    config.datasets_dir = datasets_dir
    config.results_dir = results_dir
    config.sandbox.image = sandbox_image
    config.evaluation.api_key = visual_api_key
    config.evaluation.base_url = visual_base_url
    config.evaluation.visual_model = visual_model
    config.evaluation.functional_model = functional_model
    config.evaluation.functional_api_key = functional_api_key
    config.evaluation.functional_base_url = functional_base_url
    config.evaluation.max_workers = max_workers
    config.evaluation.task = task
    config.evaluation.framework = framework
    config.evaluation.model = eval_model_filter

    # Log configuration
    logger.info("=" * 60)
    logger.info("Vision2Web Evaluation")
    logger.info("=" * 60)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Datasets: {datasets_dir}")
    logger.info(f"Functional model: {functional_model}")
    logger.info(f"Visual model: {visual_model}")
    logger.info(f"Task type: {task or 'all'}")
    logger.info(f"Framework: {framework or 'all'}")
    logger.info(f"Inference model filter: {eval_model_filter or 'all'}")
    logger.info(f"Max workers: {max_workers}")
    logger.info("=" * 60)

    engine = EvaluationEngine(config)

    # Run evaluation
    try:
        results = asyncio.run(engine.evaluate_all_projects(
            task_type=task,
            framework=framework,
            model=eval_model_filter
        ))

        # Print summary
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = len(results) - successful

        task_summary = {}
        for result in results:
            t = result.get('task_type', 'unknown')
            if t not in task_summary:
                task_summary[t] = {'total': 0, 'success': 0, 'failed': 0}
            task_summary[t]['total'] += 1
            if result.get('status') == 'success':
                task_summary[t]['success'] += 1
            else:
                task_summary[t]['failed'] += 1

        click.echo()
        click.echo("=" * 60)
        click.echo("EVALUATION COMPLETE")
        click.echo("=" * 60)
        click.echo(f"Total: {len(results)}")
        click.echo(f"Success: {successful}")
        click.echo(f"Failed: {failed}")
        click.echo()
        click.echo("By Task Type:")
        for task_name, stats in sorted(task_summary.items()):
            click.echo(f"  {task_name}: {stats['success']}/{stats['total']} succeeded")
        click.echo()
        click.echo(f"Results: {results_dir}")
        click.echo("=" * 60)

        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--results-dir', type=click.Path(exists=True, path_type=Path),
              default='./results', help='Results directory')
@click.option('--datasets-dir', type=click.Path(exists=True, path_type=Path),
              default='./datasets', help='Datasets directory')
@click.option('--output', type=click.Path(path_type=Path),
              help='Output JSON file (optional)')
def analyze(results_dir, datasets_dir, output):
    """Analyze evaluation results and print summary statistics"""

    click.echo("Analyzing results...")

    analyzer = ResultAnalyzer(results_dir=str(results_dir))
    analysis = analyzer.analyze(datasets_dir=str(datasets_dir))

    # Print summary
    analyzer.print_summary(analysis)

    # Save to file if requested
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
        click.echo(f"\nResults saved to: {output}")


def main():
    """Main entry point"""
    cli()


if __name__ == '__main__':
    main()
