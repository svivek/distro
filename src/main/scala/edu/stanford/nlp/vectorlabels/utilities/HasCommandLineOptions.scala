package edu.stanford.nlp.vectorlabels.utilities


/**
 *
 *
 * @author svivek
 */
trait HasCommandLineOptions extends HasLogging {
  self =>
  def description: String

  def name: String

  type OptionValue = Either[Either[Boolean, String], Either[Int, Double]]

  private def valueOf(value: OptionValue) = {
    value match {
      case Left(Left(b)) => b
      case Left(Right(s)) => s
      case Right(Left(i)) => i
      case Right(Right(d)) => d
      case _ => throw new RuntimeException()
    }
  }


  private def makeOptionValue[T](value: T) = {
    value match {
      case b: Boolean => Left(Left(b))
      case s: String => Left(Right(s))
      case i: Int => Right(Left(i))
      case d: Double => Right(Right(d))
      case _ => throw new RuntimeException()
    }
  }


  case class OptionParameter(shortCommandLineOption: Char = '-',
                             commandLineOption: String,
                             description: String,
                             defaultValue: OptionValue,
                             valueName: String = "",
                             optional: Boolean = false)

  private val options = collection.mutable.ListBuffer[OptionParameter]()

  private val knownShortOptions = new collection.mutable.HashSet[Char]()

  def printParameters(params: Map[OptionParameter, OptionValue]) = {
    val paramInfo = options.map {
      option => option.description + ": " + valueOf(params(option))
    }

    infoBlock("Parameters", paramInfo.mkString("\n"))

    withHTMLOutput {
      htmlList("Settings",
        options.map(option => option.description + ": <b>" + valueOf(params(option)) + "</b>").toList)
    }
  }


  def addOption[T](shortCommandLineOption: Char = '-',
                   commandLineOption: String,
                   description: String,
                   defaultValue: T,
                   valueName: String = "",
                   optional: Boolean = true) = {
    val option =
      OptionParameter(shortCommandLineOption,
        commandLineOption,
        description,
        makeOptionValue(defaultValue),
        valueName,
        optional)

    if (!shortCommandLineOption.equals('-') && knownShortOptions.contains(shortCommandLineOption)) {
      throw new RuntimeException("The flag " + shortCommandLineOption + " is repeated!")
    }

    knownShortOptions += shortCommandLineOption
    options += option
    option
  }

  def defaultOptions = options.map(option => option -> option.defaultValue).toMap

  def optionParser = {
    new scopt.OptionParser[Map[OptionParameter, OptionValue]](name) {
      head(description)
      for (option <- self.options) {
        import option._
        defaultValue match {
          case Left(Left(b)) =>
            if (shortCommandLineOption == '-') {
              opt[Unit](commandLineOption) optional() action {
                (x, c) => c + (option -> makeOptionValue(true))
              } text option.description
            } else {
              opt[Unit](shortCommandLineOption, commandLineOption) optional() action {
                (x, c) => c + (option -> makeOptionValue(true))
              } text option.description
            }
          case Left(Right(s)) =>
            if (shortCommandLineOption == '-') {
              if (option.optional)
                opt[String](commandLineOption) optional() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
              else
                opt[String](commandLineOption) required() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
            } else {
              if (option.optional)
                opt[String](shortCommandLineOption, commandLineOption) optional() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
              else
                opt[String](shortCommandLineOption, commandLineOption) required() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
            }
          case Right(Left(i)) =>
            if (shortCommandLineOption == '-') {
              if (option.optional) {
                opt[Int](commandLineOption) optional() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
              } else {
                opt[Int](commandLineOption) required() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
              }
            } else {
              if (option.optional) {
                opt[Int](shortCommandLineOption, commandLineOption) optional() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
              }
              else {
                opt[Int](shortCommandLineOption, commandLineOption) required() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description

              }
            }
          case Right(Right(d)) =>
            if (shortCommandLineOption equals '-') {
              if (option.optional) {
                opt[Double](commandLineOption) optional() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
              } else {
                opt[Double](commandLineOption) required() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description

              }
            } else {
              if (option.optional) {
                opt[Double](shortCommandLineOption, commandLineOption) optional() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description
              }
              else {
                opt[Double](shortCommandLineOption, commandLineOption) required() valueName option.valueName action {
                  (x, c) => c + (option -> makeOptionValue(x))
                } text option.description

              }
            }
          case _ => throw new RuntimeException()
        }
      }
    }
  }

  class Options(private val params: Map[OptionParameter, OptionValue]) {
    def apply[T](option: OptionParameter): T =
      params.get(option) match {
        case Some(o) => valueOf(o).asInstanceOf[T]
        case None => throw new RuntimeException("Unknown option: " + option.description)
      }
  }

  def numRequiredOptions = options.filter(!_.optional).size

}


