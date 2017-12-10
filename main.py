import luigi

from luigi_openslr.wrapper_task import PipelineSlr

if __name__ == "__main__":
    # luigi.run(["--local-scheduler"], main_task_cls=PipelineSlr)
    luigi.run(main_task_cls=PipelineSlr)
