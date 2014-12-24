#!/bin/bash

CP=target/scala-2.10/classes

JARS=

for file in `find lib | grep '.jar$'`; do
   CP=$CP:$file
done

for file in `find lib_managed | grep '.jar$'`; do
   CP=$CP:$file
done

MEMORY="-J-Xmx15g"

time nice scala $MEMORY -cp $CP edu.stanford.nlp.vectorlabels.Main "$@"
