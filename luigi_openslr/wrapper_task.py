import datetime
import logging

import luigi

from luigi_openslr.neural_tasks import TrainModel

@luigi.Task.event_handler(luigi.Event.SUCCESS)
def celebrate_success(task):
    logger = logging.getLogger('training')
    logger.info('Succes for {}'.format(task.get_task_family()))

@luigi.Task.event_handler(luigi.Event.FAILURE)
def mourn_failure(task, exception):
    logger = logging.getLogger('training')
    logger.error('Task {} failed with error: {}'.format(task.get_task_family(), exception))


class PipelineSlr(luigi.WrapperTask):
    def requires(self):
        return [TrainModel()]

    def output(self):
        return luigi.LocalTarget('pipe-status')

    def run(self):
        today_str = datetime.datetime.now().strftime("%Y-%B-%d at %I:%M%p")
        with self.output().open('w') as f:
            f.write('Pipeline run successfully the {}.'.format(today_str))
