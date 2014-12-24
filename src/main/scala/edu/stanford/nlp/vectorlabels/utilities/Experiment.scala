package edu.stanford.nlp.vectorlabels.utilities


trait Experiment extends HasCommandLineOptions {
  def run(directory: String, args: Array[String]): Unit = {
    optionParser.parse(args, defaultOptions) map {
      params => {
        printParameters(params)
        run(directory, new Options(params))
      }
    } getOrElse {
      println("Invalid usage of " + name)
    }
  }

  def run(directory: String, params: Options): Unit

  def doc = optionParser.usage

  def validate(args: Array[String]) = optionParser.parse(args, defaultOptions) != None
}
