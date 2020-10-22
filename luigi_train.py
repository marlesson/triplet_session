import luigi

class ExperimentRuns(luigi.WrapperTask):
  '''
  https://luigi.readthedocs.io/en/stable/luigi_patterns.html
  '''
  sample_days = luigi.IntParameter(default=4)
  model_triplet_embs = luigi.Parameter()



  def requires(self):
      yield SomeReport(self.date)
      yield SomeOtherReport(self.date)
      yield CropReport(self.date)
      yield TPSReport(self.date)
      yield FooBarBazReport(self.date)