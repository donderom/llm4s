ThisBuild / organization := "com.donderom"

ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/donderom/llm4s"),
    "scm:git@github.com:donderom/llm4s.git"
  )
)

ThisBuild / developers := List(
  Developer(
    id = "donderom",
    name = "Roman Parykin",
    email = "@donderom",
    url = url("https://github.com/donderom")
  )
)


ThisBuild / description := "Scala bindings for llama.cpp"
ThisBuild / licenses := List(
  "Apache 2" -> new URL("http://www.apache.org/licenses/LICENSE-2.0.txt")
)
ThisBuild / homepage := Some(url("https://github.com/donderom/llm4s"))

// Remove all additional repository other than Maven Central from POM
ThisBuild / pomIncludeRepository := { _ => false }
ThisBuild / publishTo := {
  val nexus = "https://s01.oss.sonatype.org/"
  if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
  else Some("releases" at nexus + "service/local/staging/deploy/maven2")
}
ThisBuild / publishMavenStyle := true
