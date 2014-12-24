name := "vectorlabels"

version := "1.0"

// Only works with 2.10.0+
scalaVersion := "2.10.3"

// Scalaxy/Loops snapshots are published on the Sonatype repository.
resolvers += Resolver.sonatypeRepo("snapshots")

libraryDependencies ++= Seq(
  "net.sf.trove4j" % "trove4j" % "3.0.3",
  "com.github.scopt" %% "scopt" % "3.1.0",
  "ch.qos.logback" % "logback-classic" % "1.0.13",
  "gov.nist.math" % "jama" % "1.0.3",  
  "edu.stanford.nlp" % "stanford-corenlp" % "3.3.1",
  "com.nativelibs4java" %% "scalaxy-loops" % "0.3-SNAPSHOT" % "provided"
)



retrieveManaged := true
