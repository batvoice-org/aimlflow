import fnmatch
import click
import collections
import mlflow
import json
import time
import os.path

from ast import literal_eval
from tempfile import TemporaryDirectory
from tqdm import tqdm

from aim import Run, Image, Text, Audio

IMAGE_EXTENSIONS = ('jpg', 'bmp', 'jpeg', 'png', 'gif', 'svg')
HTML_EXTENSIONS = ('html',)
TEXT_EXTENSIONS = (
    'txt',
    'log',
    'py',
    'js',
    'yaml',
    'yml',
    'json',
    'csv',
    'tsv',
    'md',
    'rst',
    'jsonnet',
)

# Audio is not handled in mlflow but including here just in case
AUDIO_EXTENSIONS = (
    'flac',
    'mp3',
    'wav',
)


def get_mlflow_experiments(client, experiment):
    if experiment is None:
        # process all experiments
        experiments = client.search_experiments()
    else:
        try:
            ex = client.get_experiment(experiment)
        except mlflow.exceptions.MlflowException:
            ex = client.get_experiment_by_name(experiment)
        if not ex:
            raise RuntimeError(f'Could not find experiment with id or name "{experiment}"')
        experiments = (ex,)

    return experiments


def get_aim_run(repo_inst, run_id, run_name, experiment_name):
    query = f"run.mlflow_run_id == '{run_id}'"
    query_res = repo_inst.query_runs(query).iter_runs()

    if aim_run_info := next(query_res, None):
        aim_run = Run(
            run_hash=aim_run_info.run.hash,
            repo=repo_inst,
            system_tracking_interval=None,
            capture_terminal_logs=False,
        )
    else:
        aim_run = Run(
            repo=repo_inst,
            system_tracking_interval=None,
            capture_terminal_logs=False,
            experiment=experiment_name,
        )

    aim_run.name = run_name

    return aim_run

def get_aim_run_from_run_id(repo_inst, run_id):
    query = f"run.mlflow_run_id == '{run_id}'"
    query_res = repo_inst.query_runs(query).iter_runs()

    if aim_run_info := next(query_res, None):
        aim_run = Run(
            run_hash=aim_run_info.run.hash,
            repo=repo_inst,
            system_tracking_interval=None,
            capture_terminal_logs=False,
        )

        return aim_run

    return None


def collect_run_params(aim_run, mlflow_run):
    aim_run['mlflow_run_id'] = mlflow_run.info.run_id
    aim_run['mlflow_experiment_id'] = mlflow_run.info.experiment_id
    aim_run.description = mlflow_run.data.tags.get("mlflow.note.content")

    # Collect params & tags
    # MLflow provides "string-ified" params values and we try to revert that
    aim_run['params'] = _map_nested_dicts(_try_parse_str, mlflow_run.data.params)
    aim_run['tags'] = {
        k: v for k, v in mlflow_run.data.tags.items() if not k.startswith('mlflow')
    }


def collect_artifacts(aim_run, mlflow_run, mlflow_client, exclude_artifacts):
    if '*' in exclude_artifacts:
        return

    run_id = mlflow_run.info.run_id

    artifacts_cache_key = '_mlflow_artifacts_cache'
    artifacts_cache = aim_run.meta_run_tree.get(artifacts_cache_key) or []

    __html_warning_issued = False
    with TemporaryDirectory(prefix=f'mlflow_{run_id}_') as temp_path:
        artifact_loc_stack = [None]
        while artifact_loc_stack:
            loc = artifact_loc_stack.pop()
            artifacts = mlflow_client.list_artifacts(run_id, path=loc)

            for file_info in artifacts:
                if file_info.is_dir:
                    artifact_loc_stack.append(file_info.path)
                    continue

                if file_info.path in artifacts_cache:
                    continue
                else:
                    artifacts_cache.append(file_info.path)

                if exclude_artifacts:
                    exclude = False
                    for expr in exclude_artifacts:
                        if fnmatch.fnmatch(file_info.path, expr):
                            exclude = True
                            break
                    if exclude:
                        continue

                downloaded_path = mlflow_client.download_artifacts(run_id, file_info.path, dst_path=temp_path)
                if file_info.path.endswith(HTML_EXTENSIONS):
                    if not __html_warning_issued:
                        click.secho(
                            'Handler for html file types is not yet implemented.', fg='yellow'
                        )
                        __html_warning_issued = True
                    continue
                elif file_info.path.endswith(IMAGE_EXTENSIONS):
                    aim_object = Image
                    kwargs = dict(
                        image=downloaded_path,
                        caption=file_info.path
                    )
                    content_type = 'image'
                elif file_info.path.endswith(TEXT_EXTENSIONS):
                    with open(downloaded_path) as fh:
                        content = fh.read()
                    aim_object = Text
                    kwargs = dict(
                        text=content
                    )
                    content_type = 'text'
                elif file_info.path.endswith(AUDIO_EXTENSIONS):
                    audio_format = os.path.splitext(file_info.path)[1].lstrip('.')
                    aim_object = Audio
                    kwargs = dict(
                        data=downloaded_path,
                        caption=file_info.path,
                        format=audio_format
                    )
                    content_type = 'audio'
                else:
                    click.secho(
                        f'Unresolved or unsupported type for artifact {file_info.path}', fg='yellow'
                    )
                    continue

                try:
                    item = aim_object(**kwargs)
                except Exception as exc:
                    click.echo(
                        f'Could not convert artifact {file_info.path} into aim object - {exc}', err=True
                    )
                    continue
                aim_run.track(item, name=loc or 'root', context={'type': content_type})

            aim_run.meta_run_tree[artifacts_cache_key] = artifacts_cache


def collect_metrics(aim_run, mlflow_run, mlflow_client, timestamp=None):
    for key in mlflow_run.data.metrics.keys():
        metric_history = mlflow_client.get_metric_history(mlflow_run.info.run_id, key)
        if timestamp:
            metric_history = list(filter(lambda m: m.timestamp >= timestamp, metric_history))

        for m in metric_history:
            aim_run.track(m.value, step=m.step, name=m.key)


def _wait_forever(watcher):
    try:
        while True:
            time.sleep(24 * 60 * 60)  # sleep for a day
    except KeyboardInterrupt:
        watcher.stop()


def _map_nested_dicts(fun, tree):
    if isinstance(tree, collections.abc.Mapping):
        return {k: _map_nested_dicts(fun, subtree) for k, subtree in tree.items()}
    else:
        return fun(tree)


def _try_parse_str(s):
    assert isinstance(s, str), f'Expected a string, got {s} of type {type(s)}'
    try:
        return literal_eval(s.strip())
    except:  # noqa: E722
        return s
