## llm4s

*Experimental* Scala 3 bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) using [Slinc](https://github.com/scala-interop/slinc).

### Setup

Add Slinc runtime and llm4s to your `build.sbt`:

```scala
libraryDependencies += "fr.hammons" %% "slinc-runtime" % "0.5.0"
libraryDependencies += "com.donderom" %% "llm4s" % "0.1.0"
```

For JDK 17 add `.jvmopts` file in the project root:

```
--add-modules=jdk.incubator.foreign
--enable-native-access=ALL-UNNAMED
```

Version compatibility:

| llm4s | Slinc runtime |     Scala |    JDK |                  llama.cpp (commit hash) |
|------:|--------------:|----------:|-------:|-----------------------------------------:|
| 0.1.* |         0.5.0 | 3.3.0-RC3 | 17, 19 | 447ccbe8c39332fcdd0d98a041b6e2ff6f06219d |


### Usage

```scala
import java.nio.file.Paths

import com.donderom.llm4s.*

val lib = Paths.get("path/to/libllama.so")
val model = Paths.get("path/to/ggml-model.bin")
val params = LlmParams(threads = 6, seed = 1337)
val llm = Llm(lib = lib, model = model, params = params)

val prompt = "Deep learning is "

// To print generation as it goes
llm(prompt, params).foreach: stream =>
  stream.foreach: token =>
    print(token)

// Or build a string
llm(prompt, params).map(_.foldLeft(new StringBuilder)(_ ++= _).toString)

llm.close()
```
