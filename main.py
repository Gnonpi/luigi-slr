import luigi

from luigi_openslr.wrapper_task import PipelineSlr

if __name__ == "__main__":
    luigi.build([PipelineSlr()], local_scheduler=True)
