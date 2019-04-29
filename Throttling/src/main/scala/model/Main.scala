package model

object Main {

  def main(args: Array[String]): Unit = {
    import java.io.File
    val folders: Array[File] = (new File("/home/avinash/"))
      .listFiles
      .filter(_.isDirectory)  //isFile to find files
    folders.foreach(println)
  }
}
