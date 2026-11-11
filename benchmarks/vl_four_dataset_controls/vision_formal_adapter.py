"""Allow proven terminal model failures in the official 40-task quality denominator."""

import argparse
import asyncio
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import time


NO_START = 'Agent completed but start.sh was not generated'
CLI_STDERR_ERROR = re.compile(
    r'API[ _-]error|api_error|connection refused|ECONNRESET|ETIMEDOUT|'
    r'failed to create container|error running Claude Code|CUDA out of memory|'
    r'previous response had no visible output|invalid.*(?:tool|JSON)|'
    r'HTTP\s*(?:401|403|429|500|502|503)|browser.*(?:crash|failed)', re.I)


def read(path):
    return json.loads(Path(path).read_text())


def argument(command, name):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def freeze_policy(adapter, path, policy):
    if path.exists():
        if read(path) != policy:
            raise ValueError('formal coverage wrapper or frozen policy changed; preserve this run and use a new output')
        return
    adapter.save(path, policy)


def terminal_reason(record, project, model):
    strict_failure = record.get('status') != 'success'
    if record.get('project') != project or record.get('name') != project:
        return 'result_identity_mismatch'
    if record.get('model') != model or record.get('framework') != 'claude_code':
        return 'result_model_mismatch'
    history = record.get('attempts_history')
    if (type(record.get('attempt_count')) is not int or record['attempt_count'] != 1
            or type(record.get('retry_count')) is not int or record['retry_count'] != 0
            or not isinstance(history, list) or len(history) != 1):
        return 'frozen_single_attempt_protocol_mismatch'
    attempt = history[0]
    if (not isinstance(attempt, dict) or type(attempt.get('attempt')) is not int
            or attempt['attempt'] != 1 or attempt.get('status') != record.get('status')
            or 'error' not in attempt or attempt['error'] != record.get('error')):
        return 'incomplete_or_conflicting_attempt_history'
    try:
        if datetime.fromisoformat(record['end_time']) < datetime.fromisoformat(record['start_time']):
            return 'invalid_terminal_time'
    except (KeyError, TypeError, ValueError):
        return 'missing_terminal_time'
    events = record.get('conversation', [])
    endings = [event for event in events if event.get('type') == 'result']
    if len(endings) != 1 or not events or events[-1].get('type') != 'result':
        return 'missing_cli_terminal_result'
    end = endings[0]
    if (end.get('subtype') != 'success' or end.get('is_error') is not False
            or end.get('terminal_reason') != 'completed' or end.get('stop_reason') != 'end_turn'
            or end.get('api_error_status') is not None or end.get('permission_denials')):
        return 'abnormal_cli_terminal_result'
    if not end.get('result', '').strip() or end.get('usage', {}).get('output_tokens', 0) <= 0:
        return 'missing_visible_final_response'
    logs = record.get('logs', [])
    if not isinstance(logs, list) or any(not isinstance(line, str) for line in logs):
        return 'invalid_cli_stderr_record'
    if ((strict_failure and logs)
            or any(event.get('isApiErrorMessage') or event.get('error')
                   or event.get('api_error_status') is not None
                   for event in events)):
        return 'cli_error_or_stderr_requires_review'
    if any(CLI_STDERR_ERROR.search(line) for line in logs):
        return 'cli_stderr_infrastructure_error_signal'
    tools, results = {}, set()
    visible_final = False
    for event in events:
        message = event.get('message', {})
        content = message.get('content', [])
        if not isinstance(content, list):
            continue
        if event.get('type') == 'assistant':
            visible_final = any(block.get('type') == 'text' and block.get('text', '').strip()
                                for block in content)
        for block in content:
            if block.get('type') in ('thinking', 'redacted_thinking'):
                return 'unexpected_thinking_channel'
            if block.get('type') == 'text':
                if '<tool_call>' in block.get('text', ''):
                    return 'unparsed_tool_markup'
            if block.get('type') == 'tool_use' and event.get('type') == 'assistant':
                if not block.get('id') or not block.get('name') or not isinstance(block.get('input'), dict):
                    return 'invalid_tool_use'
                tools[block['id']] = block['name']
            if block.get('type') == 'tool_result' and event.get('type') == 'user':
                if block.get('tool_use_id') not in tools or not block.get('content'):
                    return 'tool_result_error_or_unmatched'
                if strict_failure and block.get('is_error'):
                    content = block['content']
                    if (tools[block['tool_use_id']] != 'Bash' or not isinstance(content, str)
                            or not re.match(r'^Exit code [1-9][0-9]*(?:\n|$)', content)):
                        return 'unclassified_tool_host_error'
                results.add(block['tool_use_id'])
    if set(tools) != results or not visible_final:
        return 'unproven_tool_roundtrip_or_final_response'
    return None


class FormalCoverage:
    def __init__(self, adapter, config, experiment_root):
        self.adapter, self.config = adapter, config
        self.root = Path(config['output'])
        self.experiment_root = Path(experiment_root)
        self.abort_reason = None
        self.blocked_before_start = []
        self.started_projects = []
        self.latest_audit = None
        rows = [json.loads(line) for line in
                (Path(experiment_root) / 'prepare/Vision2Web_Level1/evaluation.jsonl').read_text().splitlines()]
        self.expected = [row['metadata']['task'] for row in rows]
        projects = [row['name'] for row in read(self.root / 'projects.json')]
        if (len(self.expected) != 40 or len(set(self.expected)) != 40
                or config['projects'] != self.expected or projects != self.expected):
            raise ValueError('formal scoring requires the complete frozen 40-task ordered manifest')
        self.original_evaluation_coverage = adapter.evaluation_coverage

    def audit(self):
        unresolved, quality_zero, successes = {}, [], []
        command_path = self.root / 'inference_command.json'
        command = read(command_path).get('command', []) if command_path.exists() else []
        command_valid = (argument(command, '--reasoning-parser') == 'off'
            and argument(command, '--tool-call-parser') in ('auto', 'qwen25')
            and argument(command, '--served-model-name') == self.config['generation']['model']
            and argument(command, '--model') == '/data/fanya/models/Qwen3-VL-30B-A3B-Instruct')
        for project, workspace in self.adapter.workspaces(self.config):
            name = project['name']
            path = workspace / 'results' / f'{name}.json'
            if not command_valid:
                unresolved[name] = 'missing_instruct_parser_command_evidence'
                continue
            if not path.is_file():
                unresolved[name] = 'missing_real_terminal_record'
                continue
            try:
                record = read(path)
                reason = terminal_reason(record, name, self.config['generation']['model'])
            except (OSError, ValueError, TypeError, AttributeError) as error:
                unresolved[name] = f'invalid_terminal_record:{type(error).__name__}'
                continue
            if reason:
                unresolved[name] = reason
            elif record.get('status') == 'success' and not record.get('error') and (workspace / 'start.sh').is_file():
                successes.append(name)
            elif (record.get('status') == 'failed' and record.get('error') == NO_START
                    and not (workspace / 'start.sh').exists()
                    and not (workspace / 'evaluation_result.json').exists()
                    and not (workspace / 'test_results').exists()):
                quality_zero.append(name)
            else:
                unresolved[name] = 'unsupported_failure_or_conflicting_artifacts'
        receipt = dict(complete=not unresolved, expected_projects=self.expected,
            terminal_success_projects=successes, terminal_model_failure_zero_projects=quality_zero,
            unresolved=unresolved, total_terminal_projects=len(successes) + len(quality_zero),
            policy='only proven normal end_turn without start.sh can receive official zero; tool use is optional',
            frozen_attempt_count=1, frozen_retry_count=0,
            official_denominator_projects=40, official_denominator_viewports=120)
        self.adapter.save(self.root / 'formal_generation_audit.json', receipt)
        self.latest_audit = receipt
        return receipt

    def publish_progress(self, active_project=None):
        audit = self.latest_audit if self.latest_audit is not None else self.audit()
        terminal_records = [project['name'] for project, workspace in self.adapter.workspaces(self.config)
                     if (workspace / 'results' / f'{project["name"]}.json').is_file()]
        started = list(dict.fromkeys(self.started_projects + ([active_project] if active_project else [])))
        failures = {name: reason for name, reason in audit['unresolved'].items() if name in terminal_records}
        value = dict(complete=bool(audit['complete'] and not self.abort_reason), stage='inference',
            completed=audit['total_terminal_projects'], total=len(self.expected),
            attempted=len(started), attempted_projects=started,
            terminal_records=len(terminal_records), terminal_record_projects=terminal_records,
            successful_projects=audit['terminal_success_projects'],
            quality_zero_projects=audit['terminal_model_failure_zero_projects'],
            infrastructure_failures=failures, active_project=active_project,
            not_started_projects=[name for name in self.expected if name not in started],
            blocked_before_start=list(self.blocked_before_start), abort_reason=self.abort_reason,
            official_output=str(self.root), updated_unix=time.time(),
            scoring_complete=False, formal_performance_eligible=False)
        method = self.root.name
        if method not in ('fused', 'tablemoe', 'modes', 'schedule', 'table'):
            raise ValueError('formal progress output must name a supported comparison method')
        self.adapter.save(self.root / 'formal_progress.json', value)
        for folder in (self.experiment_root / 'cells/Vision2Web_Level1' / method,
                       self.experiment_root / 'vision' / method):
            self.adapter.save(folder / 'progress.json', value)
        return value

    def install_inference_guard(self, engine_class=None):
        if engine_class is None:
            from vision2web.inference.engine import InferenceEngine
            engine_class = InferenceEngine
        original = engine_class.run_single_project
        policy = self

        async def guarded(engine, project):
            if policy.abort_reason is not None:
                policy.blocked_before_start.append(project.name)
                policy.publish_progress()
                raise RuntimeError('formal inference stopped after an unresolved infrastructure failure')
            if engine.config.inference.max_workers != 1:
                policy.abort_reason = dict(project=project.name, reason='frozen_bs1_concurrency_changed')
                policy.publish_progress()
                raise ValueError('formal inference requires one project worker')
            try:
                policy.publish_progress(active_project=project.name)
                policy.started_projects.append(project.name)
                result = await original(engine, project)
                audit = policy.audit()
                reason = audit['unresolved'].get(project.name)
                if reason is not None:
                    policy.abort_reason = dict(project=project.name, reason=reason)
                    raise RuntimeError(f'formal project {project.name} has an unresolved infrastructure failure: {reason}')
                policy.publish_progress()
                return result
            except BaseException as error:
                if policy.abort_reason is None:
                    policy.abort_reason = dict(project=project.name, reason=f'project_exception:{type(error).__name__}')
                try:
                    policy.audit()
                    policy.publish_progress()
                except Exception as evidence_error:
                    engine.logger.error(f'Could not persist aborted formal progress: {evidence_error}')
                raise

        engine_class.run_single_project = guarded
        self.publish_progress()

        def restore():
            engine_class.run_single_project = original

        return restore

    def generation_coverage(self, config):
        if config != self.config:
            raise ValueError('formal coverage config changed')
        return list(self.audit()['unresolved'])

    def evaluation_coverage(self, config):
        audit = self.audit()
        zeros = set(audit['terminal_model_failure_zero_projects'])
        missing = self.original_evaluation_coverage(config)
        missing = [item for item in missing if item.split(':', 1)[0] not in zeros]
        return missing + [f'{name}:generation:{reason}' for name, reason in audit['unresolved'].items()]

    def install(self):
        self.adapter.generation_coverage = self.generation_coverage
        self.adapter.evaluation_coverage = self.evaluation_coverage

    def annotate_stage(self, stage):
        path = self.root / f'{stage}_complete.json'
        record = read(path)
        audit = self.audit()
        record.update(terminal_model_failure_zero_projects=audit['terminal_model_failure_zero_projects'],
            terminal_projects=audit['total_terminal_projects'], all_projects_successful=not audit['terminal_model_failure_zero_projects'])
        self.adapter.save(path, record)

    def report(self):
        result = self.adapter.report(self.config)
        analysis = read(self.root / 'official_analysis.json')
        key = f'webpage:claude_code:{self.config["generation"]["model"]}'
        stats = analysis.get('statistics', {}).get(key, {})
        exact = (stats.get('total_count') == 40 and set(stats.get('projects', [])) == set(self.expected)
            and len(stats.get('prototype_scores', [])) == 120
            and all(len(stats.get(f'{view}_scores', [])) == 40 for view in ('desktop', 'tablet', 'mobile')))
        audit = self.audit()
        zeros = audit['terminal_model_failure_zero_projects']
        for name in zeros:
            scores = analysis.get('results', {}).get(f'{key}:{name}', {}).get('prototypes', {})
            exact = exact and scores == dict(desktop=0, tablet=0, mobile=0)
        result.update(complete=bool(result['complete'] and exact and audit['complete']),
            frozen_denominator_verified=exact, terminal_model_failure_zero_projects=zeros,
            terminal_projects=audit['total_terminal_projects'])
        self.adapter.save(self.root / 'complete.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('inference', 'evaluate', 'report'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--experiment-root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = args.experiment_root / 'runtime_snapshot/benchmarks/bench_vision2web.py'
    spec = importlib.util.spec_from_file_location('frozen_vision_adapter', source)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    config = adapter.load_config(args.config)
    adapter.prepare(config)
    policy = FormalCoverage(adapter, config, args.experiment_root)
    policy.install()
    freeze_policy(adapter, Path(config['output']) / 'formal_coverage_policy.json', dict(
        wrapper=str(Path(__file__).resolve()), wrapper_sha256=adapter.fingerprint(__file__),
        strict_smoke_unchanged=True, expected_projects=policy.expected,
        frozen_attempt_count=1, frozen_retry_count=0,
        supported_quality_zero='normal_end_turn_without_start_sh'))
    if args.stage == 'report':
        raise SystemExit(0 if policy.report()['complete'] else 1)
    if args.stage == 'inference' and any((workspace / 'results' / f'{project["name"]}.json').exists()
            for project, workspace in adapter.workspaces(config)):
        audit = policy.audit()
        if not audit['complete']:
            raise ValueError('partial or unresolved formal inference needs preserved outputs and a fresh run')
        adapter.save(Path(config['output']) / 'inference_complete.json', dict(
            complete=True, missing=[], recovered_from_terminal_records=True))
        policy.annotate_stage('inference')
        return
    restore = policy.install_inference_guard() if args.stage == 'inference' else None
    try:
        asyncio.run(adapter.run_stage(config, args.stage))
    finally:
        if restore is not None:
            restore()
            policy.audit()
            policy.publish_progress()
    policy.annotate_stage(args.stage)


if __name__ == '__main__':
    main()
