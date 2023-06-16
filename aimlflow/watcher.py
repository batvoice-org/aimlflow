import logging
import time

from threading import Thread
from typing import Union, TYPE_CHECKING, Dict, List

import mlflow.entities
from mlflow import MlflowClient
from aimlflow.utils import (
    get_mlflow_experiments,
    get_aim_run,
    collect_metrics,
    collect_artifacts,
    collect_run_params,
    get_aim_run_from_run_id
)

if TYPE_CHECKING:
    from aim import Repo, Run

logger = logging.getLogger(__name__)


class MLFlowWatcher:
    WATCH_INTERVAL_DEFAULT = 10

    def __init__(self,
                 repo: 'Repo',
                 tracking_uri: str,
                 experiment: str = None,
                 exclude_artifacts: str = None,
                 interval: Union[int, float] = WATCH_INTERVAL_DEFAULT,
                 run_statuses: List[mlflow.entities.RunStatus] = None
                 ):

        self._last_watch_time = time.time()
        self._active_aim_runs_pool: Dict[str, 'Run'] = dict()

        self._watch_interval = interval

        self._client = MlflowClient(tracking_uri)

        self._exclude_artifacts = exclude_artifacts
        self._experiment = experiment
        self._experiments = get_mlflow_experiments(self._client, self._experiment)
        self._repo = repo
        self._run_statuses = run_statuses or [mlflow.entities.RunStatus.RUNNING]

        self._th_collector = Thread(target=self._watch, daemon=True)
        self._shutdown = False
        self._started = False

    def start(self):
        if self._started:
            return

        self._started = True
        self._th_collector.start()

    def stop(self):
        if not self._started:
            return

        self._shutdown = True
        self._th_collector.join()

    def _search_experiment(self, experiment_id):
        return next((exp for exp in self._experiments if exp.experiment_id == experiment_id), None)

    def _get_current_valid_mlflow_runs(self):
        experiment_ids = [ex.experiment_id for ex in self._experiments]

        valid_runs = []
        for run_status in self._run_statuses:
            valid_runs += self._client.search_runs(
                experiment_ids=experiment_ids,
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                filter_string=f'attribute.status="{mlflow.entities.RunStatus.to_string(run_status)}"'
            )

        return valid_runs

    def _process_single_run(self, aim_run, mlflow_run):
        # Collect params and tags
        collect_run_params(aim_run, mlflow_run)

        # Collect metrics
        collect_metrics(aim_run, mlflow_run, self._client, timestamp=self._last_watch_time)

        # Collect artifacts
        collect_artifacts(aim_run, mlflow_run, self._client, self._exclude_artifacts)

    def _process_runs(self):
        try:
            # refresh experiments list
            self._experiments = get_mlflow_experiments(self._client, self._experiment)

            # process valid runs
            valid_mlflow_runs = {
                run.info.run_id: run
                for run in self._get_current_valid_mlflow_runs()
            }

            valid_aim_runs = {
                run.get("mlflow_run_id"): run
                for run in self._repo.iter_runs()
                if run.get("mlflow_run_id", None)
            }

            valid_mlflow_ids = set(valid_mlflow_runs.keys())
            valid_aim_ids = set(valid_aim_runs.keys())
            logger.info("valid_mlflow_ids", valid_mlflow_ids)
            logger.info("valid_aim_ids", valid_aim_ids)


            ids_to_remove = valid_aim_ids - valid_mlflow_ids
            ids_to_add = valid_mlflow_ids - valid_aim_ids

            logger.info("ids_to_remove", ids_to_remove)
            logger.info("ids_to_add", ids_to_add)

            # Adding new items
            for run_id in ids_to_add: #.union(ids_to_update):
                mlflow_run = valid_mlflow_runs[run_id]
                aim_run = get_aim_run(self._repo,
                                      run_id,
                                      mlflow_run.info.run_name,
                                      mlflow_run.info.experiment_id,        # TODO: Transform into name
                )

                self._process_single_run(aim_run, mlflow_run)

                del aim_run

            for run_id in ids_to_remove:
                if aim_run := get_aim_run_from_run_id(self._repo, run_id):
                    hash = aim_run.hash
                    del aim_run
                    self._repo.delete_run(hash)
        except Exception as e:
            logger.error(repr(e))
            import traceback
            traceback.print_exc()

    def _watch(self):
        self._process_runs()
        watch_interval_counter = 0
        while True:
            if self._shutdown:
                break

            time.sleep(1)
            watch_interval_counter += 1

            if watch_interval_counter > self._watch_interval:
                self._process_runs()
                watch_interval_counter = 0
