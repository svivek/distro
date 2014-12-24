package edu.stanford.nlp.vectorlabels.utilities

import edu.stanford.nlp.vectorlabels.Main
import java.io.{FileWriter, BufferedWriter, File, PrintWriter}
import edu.stanford.nlp.io.IOUtils
import ch.qos.logback.classic.Level

class ExperimentManager(experimentOutputDirectory: String, htmlOutputDir: String, private var logLevel: Level = Level.INFO) extends HasLogging {

  val experiments = collection.mutable.Map[String, Experiment]()

  def +=(expt: Experiment) = {
    assert(!experiments.contains(expt.name), s"Experiment ${expt.name} already defined!")

    experiments += (expt.name -> expt)
  }

  def isValid(name: String) = experiments.contains(name) || experiments.contains(removeDebugMarker(name))

  def removeDebugMarker(name: String) = name.replaceFirst("d:", "")

  def hasDebugMarker(name: String) = name.startsWith("d:")

  def doc(name: String) =
    if (experiments contains name) experiments(name).doc
    else if (experiments.contains(removeDebugMarker(name))) experiments(removeDebugMarker(name)).doc
    else assert(false)

  def run(args: Array[String]): Unit = {
    def printValidCommands() = {
      println("List of valid commands:")
      experiments.keys.toList.sorted.foreach {
        e => println(e + "\n\t" + experiments(e).description.replaceAll("\n", "\n\t"))
      }
    }

    if (args.length == 0) {
      println("No command specified. ")
      printValidCommands()
    } else if (args(0) == "help") {
      if (args.length == 2) {
        val name = args(1)
        if (isValid(name)) {
          println(doc(name))
        } else {
          println("Invalid command.")
          printValidCommands()
        }
      } else {
        println("Invalid usage of help. Usage: 'help <command-name>', where <command-name> is a valid command")
      }
    } else {
      val name = args(0)
      if (!isValid(name)) {
        println("Invalid command.")
        printValidCommands()
      } else if (args.length == 1 && experiments(removeDebugMarker(name)).numRequiredOptions > 0) {
        println(doc(name))
      } else {

        val dbg = hasDebugMarker(name)
        val expt = experiments(removeDebugMarker(name))
        run(expt, args.tail, dbg)
      }
    }
  }


  def run(expt: Experiment, args: Array[String], debug: Boolean): Unit = {
    if (debug) {
      info(msg = "Found debug marker. Redirecting output to debug directory")
      logLevel = Level.DEBUG
    }

    val dir = experimentDirectory(debug)
    println("Experiment directory: " + dir)

    // set up the log output to the log directory
    setupLog(dir, expt.name, args)

    // first copy all the parameters to the experiment directory for
    // future records
    printConfiguration(expt, args, dir)


    // one thread runs the experiment
    val mainThread = new Thread(new Runnable() {
      def run() = runExpt(dir, expt, args)
    },
      "Main-Thread")
    mainThread.start()


    // one thread looks for the poison pill file (called 'kill' in the
    // experiment directory.
    val monitorThread = new Thread(new Runnable() {
      def run() = runMonitor(dir, mainThread)
    },
      "Monitor-Thread")

    monitorThread.setDaemon(true)
    monitorThread.start()
  }

  def runMonitor(dir: String, mainThread: Thread) = {
    while (true) {
      // check if there is a file called "kill" in dir. If so,
      // interrupt the main thread

      if (new File(dir + File.separator + "kill").exists) {
        withHTMLOutput {
          warn("Received kill signal! Interrupting experiment")
        }
        finalizeHTML()

        mainThread.stop() // yeah thread.stop is deprecated. So what will you do?
      }

      // sleep for 30 seconds
      Thread.sleep(30000)
    }
  }

  def runExpt(dir: String, expt: Experiment, args: Array[String]) = {
    try {
      val running = new File(dir + File.separator + "running")
      val out = new PrintWriter(running)
      out.println("Running " + expt.name)
      out.close()

      expt.run(dir, args)

      running.delete()
      finalizeHTML()

    } catch {
      case e: Throwable =>

        withHTMLOutput {
          error(s"Experiment ${expt.name} exception: ${e.getMessage}")
          e.printStackTrace()
        }

        val out = new PrintWriter(new File(dir + File.separator + "exception"))
        out.println("Exception: " + e.getMessage)
        e.printStackTrace(out)
        out.close()

        throw e
    }
    println("Output written to " + dir)
  }

  def setupLog(dir: String, experimentName: String, args: Array[String]) = {
    IOUtils.ensureDir(new File(dir))
    LogSettings.outputDirectory = dir
    LogSettings.level = logLevel


    if (new File(htmlOutputDir).exists) {
      val id = dir.split(File.separator).last

      LogSettings.htmlFile = s"$htmlOutputDir${File.separator}$dir.html"

      val parentDir = dir.replaceAll(id, "")
      val out1 = new PrintWriter(new BufferedWriter(new FileWriter(htmlOutputDir + "/" + parentDir + "/index.html", true)))

      val formattedArgs = args.map(a => if (a.startsWith("-")) s"<code>$a</code>" else s"""<span class="text-warning">$a</span>""").mkString(" ")

      val randomSeed = s"<code>RANDOM_SEED=${Main.randomSeed}</code>"

      out1.println(s"<li><a href=$id.html>$id</a>: $randomSeed && <code>$experimentName</code> $formattedArgs</li>")
      out1.close()
    }
  }

  def printConfiguration(expt: Experiment, args: Array[String], dir: String) = {

    val timeStamp = new java.util.Date
    val config = List("Experiment: " + expt.name,
      "Command line options: " + args.mkString(" "),
      "Documentation: \n" + expt.doc,
      "Time stamp: " + timeStamp,
      "Experiment id: " + dir)

    val out = new PrintWriter(new File(dir + File.separator + "configuration.txt"))
    config.foreach(out.println)
    out.close()

    if (new File(htmlOutputDir) exists) {

      val htmlConfig =
        s"""
        |<li><strong>Experiment</strong>: <code>${expt.name}</code></li>
        |<li><strong>Time stamp</strong>: <code>$timeStamp</code></li>
        |<li><strong>Experiment id</strong>: <code>$dir</code></li>
      """.stripMargin

      withHTMLOutput(writeHTML(s"<h3>Execution details</h3> <ol>$htmlConfig </ol>"))
    }

  }

  def experimentDirectory(debug: Boolean) = {
    val outDir = if (debug) "debug" else experimentOutputDirectory

    val baseDir = new File(outDir)

    IOUtils.ensureDir(baseDir)
    val children = baseDir.listFiles.map {
      _.getName.toInt
    }.sorted

    val nextDir = if (children.size == 0) "1" else (children.last + 1).toString

    val directory = outDir + File.separator + nextDir

    IOUtils.ensureDir(new File(directory))

    directory
  }
}


