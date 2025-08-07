inThisBuild(List(
  organization := "com.donderom",
  homepage := Some(url("https://github.com/donderom/llm4s")),
  description := "Scala 3 bindings for llama.cpp",
  licenses := List("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")),
  developers := List(
    Developer(
      id = "donderom",
      name = "Roman Parykin",
      email = "github@donderom.com",
      url = url("https://donderom.com")
    )
  ),
))

val scala3Version = "3.3.0"

ThisBuild / scalaVersion := scala3Version
ThisBuild / versionScheme := Some("early-semver")

lazy val root = project
  .in(file("."))
  .settings(
    name := "llm4s",

    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Wunused:all",
      "-Xmax-inlines:64"
    ),

    libraryDependencies += "fr.hammons" %% "slinc-runtime" % "0.6.0",
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.16" % "test",
  )
