import logging
import luigi

from luigi_openslr.wrapper_task import PipelineSlr

if __name__ == "__main__":
    logger = logging.getLogger('training')
    logger.info('-------------------> New run')
    luigi.run(main_task_cls=PipelineSlr, local_scheduler=True)
