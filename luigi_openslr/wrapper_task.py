import datetime

import luigi

from luigi_openslr.neural_tasks import TrainModel


class PipelineSlr(luigi.WrapperTask):
    def requires(self):
        return [TrainModel()]

    def output(self):
        return luigi.LocalTarget('pipe-status')

    def run(self):
        today_str = datetime.datetime.now().strftime("%Y-%B-%d at %I:%M%p")
        with self.output().open('w') as f:
            f.write('Pipeline run successfully the {}.'.format(today_str))
