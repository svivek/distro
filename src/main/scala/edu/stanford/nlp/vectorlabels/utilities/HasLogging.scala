package edu.stanford.nlp.vectorlabels.utilities

import org.slf4j.LoggerFactory
import ch.qos.logback.classic.{Level, Logger, LoggerContext}
import ch.qos.logback.core.FileAppender
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.spi.ILoggingEvent
import java.io.{FileWriter, BufferedWriter, PrintWriter, File}
import edu.stanford.nlp.io.IOUtils


/**
 *
 *
 * @author svivek
 */
trait HasLogging {
  private val log: Logger = LoggerFactory.getLogger(this.getClass).asInstanceOf[Logger]

  protected var htmlOutput = false

  @inline protected def debug(msg: => Any) = {
    if (log.isDebugEnabled) log.debug(msg.toString)
    if (allowHTMLOutput)
      writeHTML( s"""<span class="text-muted"><i>[DEBUG]</i> ${msg.toString}</span><br/>""")
  }

  @inline protected def info(msg: => Any) = {
    if (log.isInfoEnabled) log.info(msg.toString)
    if (allowHTMLOutput)
      writeHTML( s"""<span><b class="text-info">[INFO]</b> ${msg.toString}</span><br/>""")

  }

  @inline protected def warn(msg: => Any) = {
    if (log.isWarnEnabled) log.warn(msg.toString)

    if (allowHTMLOutput)
      writeHTML( s"""<span class="text-warning"><b>[WARN]</b> ${msg.toString}</span><br/>""")
  }

  @inline protected def error(msg: => Any) = {
    if (log.isErrorEnabled) log.error(msg.toString)

    if (allowHTMLOutput)
      writeHTML( s"""<span class="text-danger"><b>[ERROR]</b> ${msg.toString}</span><br/>""")
  }

  @inline protected def fatal(msg: => Any, e: Throwable) = error(msg)

  @inline protected def debugBlock(blockName: String, msg: => Any) = {
    if (log.isDebugEnabled) log.debug(blockMessage(blockName, msg))
  }

  @inline protected def infoBlock(blockName: String, msg: => Any) = {
    if (log.isWarnEnabled) log.info(blockMessage(blockName, msg))
  }

  @inline protected def warnBlock(blockName: String, msg: => Any) = {
    if (log.isInfoEnabled) log.warn(blockMessage(blockName, msg))
  }

  @inline protected def errorBlock(blockName: String, msg: => Any) = {
    if (log.isErrorEnabled) log.error(blockMessage(blockName, msg))
  }

  @inline protected def debugTable(tableName: String, table: Table, heatmap: Boolean = false, hLevel: String = "h3"): Unit = {
    debugBlock(tableName, table.toOrg.mkString("\n"))
    if (allowHTMLOutput && log.isDebugEnabled) {
      val tableHTML = (if (heatmap) table.toHTML("table-heatmap") else table.toHTML("")).mkString("\n")

      if (hLevel.length > 0)
        writeHTML(s"<$hLevel>$tableName</$hLevel>")

      writeHTML( s"""<div class="row"><div class="span8" style="overflow: auto">$tableHTML</div></div>""")
    }
  }


  @inline protected def debugArray[T](tableName: String,
                                      data: Array[Array[T]],
                                      heatmap: Boolean = false, hLevel: String = "h3"): Unit = {
    val table = new Table("" :: (0 until data(0).size).map(_.toString).toList: _*)
    table.setHeaderColumn(0)

    for (rowId <- 0 until data.size) {
      table += (rowId + "" :: data(rowId).map(_.toString).toList: _*)
    }

    infoTable(tableName, table, heatmap, hLevel)
  }

  @inline protected def infoTable(tableName: String, table: Table, heatmap: Boolean = false, hLevel: String = "h3"): Unit = {
    infoBlock(tableName, table.toOrg.mkString("\n"))
    if (allowHTMLOutput) {
      val tableHTML = (if (heatmap) table.toHTML("table-heatmap") else table.toHTML("")).mkString("\n")

      if (hLevel.length > 0)
        writeHTML(s"<$hLevel>$tableName</$hLevel>")

      writeHTML( s"""<div class="row"><div class="span8" style="overflow: auto">$tableHTML</div></div>""")
    }
  }


  @inline protected def infoArray[T](tableName: String,
                                     data: Array[Array[T]],
                                     heatmap: Boolean = false, hLevel: String = "h3"): Unit = {
    val table = new Table("" :: (0 until data(0).size).map(_.toString).toList: _*)
    table.setHeaderColumn(0)

    for (rowId <- 0 until data.size) {
      table += (rowId + "" :: data(rowId).map(_.toString).toList: _*)
    }

    infoTable(tableName, table, heatmap, hLevel)
  }

  @inline protected def htmlList(header: String, items: List[String]) = {
    if (allowHTMLOutput)
      writeHTML("<h3>" + header + "</h3> \n<ol><li>" + items.mkString("</li>\n<li>") + "</li>\n </ol>")
  }

  private def blockMessage(blockName: String, msg: => Any) = {
    val block = "\t" + msg.toString.trim.replaceAll("\n", s"\n\t")
    s"Start $blockName\n" + block + s"\nEnd $blockName"
  }

  private def allowHTMLOutput = LogSettings.htmlFile.length > 0 && htmlOutput

  protected def withHTMLOutput[S](f: => S): S = {

    if(LogSettings.htmlFile.length >0) {
      htmlOutput = true

      val output = f

      htmlOutput = false
      output
    } else f
  }

  protected def withTimer[S](html: Boolean, message: String)(f: => S): S = {
    val start = System.currentTimeMillis

    val output = f

    val end = System.currentTimeMillis
    val time = (end - start) / 1000

    val msg = s"$message: $time ms"

    if (html)
      withHTMLOutput(info(msg))
    else
      info(msg)
    output
  }

  protected def writeHTML(s: String) = {
    if (htmlOutput) {
      val out = new PrintWriter(new BufferedWriter(new FileWriter(LogSettings.htmlFile, true)))
      out.println(s)
      out.close()
    }
  }

  protected def finalizeHTML() = writeHTML("</div></body></html>")
}

object LogSettings {
  private var _level = Level.INFO
  private var _outputDir = ""

  private var _htmlFile = ""


  /* Ugly, possibly shameful, hack follows that gets rid of the default console appender that logback provides */
  private val rootLogger = LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME).asInstanceOf[Logger]

  /* End of ugly hack */

  def outputDirectory = _outputDir

  def outputDirectory_=(s: String) = {
    _outputDir = s
    IOUtils.ensureDir(new File(s))
    rootLogger.detachAndStopAllAppenders()
    rootLogger.addAppender(fileAppender)
  }

  def level: Level = _level

  def level_=(l: Level) = {
    _level = l
    rootLogger.setLevel(l)
  }

  def htmlFile_=(s: String) = {
    if (s.endsWith(".html"))
      _htmlFile = s
    else
      _htmlFile = s + ".html"

    // initialize the html file

    val initHTML =
      """
        |<!DOCTYPE html>
        |<html lang="en">
        |<head>
        |  <meta charset="utf-8">
        |  <meta http-equiv="X-UA-Compatible" content="IE=edge">
        |  <meta name="viewport" content="width=device-width, initial-scale=1">
        |  <title>Experiment log</title>
        |  <script src="//ajax.googleapis.com/ajax/libs/jquery/2.1.0/jquery.min.js"></script>
        |  <script type="text/javascript" src = "../js/script.js"></script>
        |
        |  <!-- Latest compiled and minified CSS -->
        |  <link rel="stylesheet" href="//netdna.bootstrapcdn.com/bootstrap/3.1.1/css/bootstrap.min.css">
        |
        |  <!-- Optional theme -->
        |  <link rel="stylesheet" href="//netdna.bootstrapcdn.com/bootstrap/3.1.1/css/bootstrap-theme.min.css">
        |
        |  <!-- Latest compiled and minified JavaScript -->
        |  <script src="//netdna.bootstrapcdn.com/bootstrap/3.1.1/js/bootstrap.min.js"></script>
        |
        |
        |  <!-- HTML5 Shim and Respond.js IE8 support of HTML5 elements and media queries -->
        |  <!-- WARNING: Respond.js doesn't work if you view the page via file:// -->
        |  <!--[if lt IE 9]>
        |    <script src="https://oss.maxcdn.com/libs/html5shiv/3.7.0/html5shiv.js"></script>
        |    <script src="https://oss.maxcdn.com/libs/respond.js/1.4.2/respond.min.js"></script>
        |  <![endif]-->
        |
        |</head>
        |<body>
        |  <div class="container container-fluid">
        |  <div class="navbar navbar-default" role="navigation">
        |      <div class="container">
        |        <div class="navbar-header">
        |          <button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
        |            <span class="sr-only">Toggle navigation</span>
        |            <span class="icon-bar"></span>
        |            <span class="icon-bar"></span>
        |            <span class="icon-bar"></span>
        |          </button>
        |          <a class="navbar-brand" href="../">Experiment Manager</a>
        |        </div>
        |        <div class="collapse navbar-collapse">
        |          <ul class="nav navbar-nav">
        |            <li><a href="../experiments">Experiments</a></li>
        |            <li><a href="../debug">Debug</a></li>
        |          </ul>
        |        </div><!--/.nav-collapse -->
        |      </div>
        |    </div>
        |
        |  <h1>Experiment log</h1>
        |
      """.stripMargin

    val out = new PrintWriter(new File(s))
    out.println(initHTML)
    out.close()

  }

  def htmlFile = _htmlFile


  val loggerContext = LoggerFactory.getILoggerFactory.asInstanceOf[LoggerContext]

  def fileAppender = {
    val fileAppender = new FileAppender[ILoggingEvent]

    fileAppender.setContext(loggerContext)
    fileAppender.setName("output")
    fileAppender.setFile(_outputDir + "/log.txt")

    val encoder = new PatternLayoutEncoder()
    encoder.setContext(loggerContext)
    encoder.setPattern("%level: %d{HH:mm:ss} [%thread] %logger{0} - %msg%n")
    encoder.start()

    fileAppender.setEncoder(encoder)
    fileAppender.start()

    fileAppender
  }

}