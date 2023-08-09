val scala3Version = "3.3.0-RC3"

ThisBuild / scalaVersion := scala3Version
ThisBuild / organization := "com.donderom"
ThisBuild / version := "0.9.0"
ThisBuild / versionScheme := Some("early-semver")

lazy val root = project
  .in(file("."))
  .settings(
    name := "llm4s",

    scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Wunused:all"),

    libraryDependencies += "fr.hammons" %% "slinc-runtime" % "0.5.0",
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.16" % "test",
  )
