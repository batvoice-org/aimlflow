import click
import os

import mlflow.entities
from click import core
from click import ClickException

from aim.sdk.repo import Repo
from aim.sdk.utils import clean_repo_path

from aimlflow.watcher import MLFlowWatcher
core._verify_python3_env = lambda: None


@click.group()
def cli_entry_point():
    pass


@cli_entry_point.command(name='sync')
@click.option('--aim-repo', required=False, type=click.Path(exists=True,
                                                            file_okay=False,
                                                            dir_okay=True,
                                                            writable=True))
@click.option('--mlflow-tracking-uri',  required=False, default=None)
@click.option('--experiment', '-e', required=False, default=None)
@click.option('--exclude-artifacts', multiple=True, required=False)
@click.option('--run_statuses', multiple=True, required=False,
              default=[mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.RUNNING)])

def sync(aim_repo, mlflow_tracking_uri, experiment, exclude_artifacts, run_statuses):

    repo_path = clean_repo_path(aim_repo) or Repo.default_repo_path()
    repo_inst = Repo.from_path(repo_path)
    run_statuses = [mlflow.entities.RunStatus.from_string(run_status) for run_status in run_statuses]


    mlflow_tracking_uri = mlflow_tracking_uri or os.environ.get('MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri:
        raise ClickException('MLFlow tracking URI must be provided either through ENV or CLI.')


    watcher = MLFlowWatcher(
        repo=repo_inst,
        tracking_uri=mlflow_tracking_uri,
        experiment=experiment,
        exclude_artifacts=exclude_artifacts,
        run_statuses=run_statuses
    )

    click.echo('Converting existing MLflow logs.')

    click.echo(f'Starting watcher on {mlflow_tracking_uri}.')
    watcher.start()
    from aimlflow.utils import _wait_forever
    _wait_forever(watcher)
