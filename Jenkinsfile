def projectName = "maxent" /* set to app/repo name */

/* which platform to build documentation on */
def documentationPlatform = "ubuntu-clang-unstable"
/* whether to publish the results (disabled for template project) */
def publish = !env.BRANCH_NAME.startsWith("PR-") && projectName != "app4triqs"

properties([
  disableConcurrentBuilds(),
  buildDiscarder(logRotator(numToKeepStr: '10', daysToKeepStr: '30')),
  pipelineTriggers([
    upstream(
      threshold: 'SUCCESS',
      upstreamProjects: '/TRIQS/triqs/unstable,/TRIQS/triqs/master'
    )
  ])
])

/* map of all builds to run, populated below */
def platforms = [:]

/****************** linux builds (in docker) */
/* Each platform must have a cooresponding Dockerfile.PLATFORM in triqs/packaging */
def dockerPlatforms = [
  ['unstable', 'ubuntu-clang'],
  ['unstable', 'ubuntu-gcc'],
  ['master',   'ubuntu-gcc'],
  ['unstable', 'centos-gcc']
]
/* .each is currently broken in jenkins */
for (int i = 0; i < dockerPlatforms.size(); i++) {
  def triqsPlatform = dockerPlatforms[i]
  def triqsBranch = triqsPlatform[0]
  def platform = triqsPlatform[1]
  platforms["${platform}-${triqsBranch}"] = { -> node('docker') {
    stage("${platform}-${triqsBranch}") { timeout(time: 1, unit: 'HOURS') {
      checkout scm
      /* construct a Dockerfile for this base */
      sh """
      ( echo "FROM flatironinstitute/triqs:${triqsBranch}-${env.STAGE_NAME}" ; sed '0,/^FROM /d' Dockerfile ) > Dockerfile.jenkins
        mv -f Dockerfile.jenkins Dockerfile
      """
      /* build and tag */
      def img = docker.build("flatironinstitute/${projectName}:${env.BRANCH_NAME}-${env.STAGE_NAME}", "--build-arg APPNAME=${projectName} --build-arg BUILD_DOC=${env.STAGE_NAME==documentationPlatform} .")
      if (!publish || env.STAGE_NAME != documentationPlatform) {
        /* but we don't need the tag so clean it up (except for documentation) */
        sh "docker rmi --no-prune ${img.imageName()}"
      }
    } }
  } }
}

/****************** osx builds (on host) */
def osxPlatforms = [
  ['unstable', 'gcc', ['CC=gcc-7', 'CXX=g++-7']],
  ['unstable', 'clang', ['CC=$BREW/opt/llvm/bin/clang', 'CXX=$BREW/opt/llvm/bin/clang++', 'CXXFLAGS=-I$BREW/opt/llvm/include', 'LDFLAGS=-L$BREW/opt/llvm/lib']]
]
for (int i = 0; i < osxPlatforms.size(); i++) {
  def platformEnv = osxPlatforms[i]
  def triqsBranch = platformEnv[0]
  def platform = platformEnv[1]
  platforms["osx-${platform}-${triqsBranch}"] = { -> node('osx && triqs') {
    stage("osx-${platform}-${triqsBranch}") { timeout(time: 1, unit: 'HOURS') {
      def srcDir = pwd()
      def tmpDir = pwd(tmp:true)
      def buildDir = "$tmpDir/build"
      def installDir = "$tmpDir/install"
      def triqsDir = "${env.HOME}/install/triqs/${triqsBranch}/${platform}"
      dir(installDir) {
        deleteDir()
      }

      checkout scm
      dir(buildDir) { withEnv(platformEnv[2].collect { it.replace('\$BREW', env.BREW) } + [
        "PATH=$triqsDir/bin:${env.BREW}/bin:/usr/bin:/bin:/usr/sbin",
        "CPATH=$triqsDir/include:${env.BREW}/include",
        "LIBRARY_PATH=$triqsDir/lib:${env.BREW}/lib",
        "CMAKE_PREFIX_PATH=$triqsDir/share/cmake",
        "OMP_NUM_THREADS=2",
        "NUMEXPR_NUM_THREADS=2",
        "MKL_NUM_THREADS=2"]) {
        deleteDir()
        sh "cmake $srcDir -DCMAKE_INSTALL_PREFIX=$installDir -DTRIQS_ROOT=$triqsDir"
        sh "make -j2"
        try {
          sh "make test"
        } catch (exc) {
          archiveArtifacts(artifacts: 'Testing/Temporary/LastTest.log')
          throw exc
        }
        sh "make install"
      } }
    } }
  } }
}

/****************** wrap-up */
try {
  parallel platforms
  if (publish) { node("docker") {
    /* Publish results */
    stage("publish") { timeout(time: 1, unit: 'HOURS') {
      def commit = sh(returnStdout: true, script: "git rev-parse HEAD").trim()
      def workDir = pwd()
      /* Update documention on gh-pages branch */
      dir("$workDir/gh-pages") {
        def subdir = "${projectName}/${env.BRANCH_NAME}"
        git(url: "ssh://git@github.com/TRIQS/TRIQS.github.io.git", branch: "master", credentialsId: "ssh", changelog: false)
        sh "rm -rf ${subdir}"
        docker.image("flatironinstitute/${projectName}:${env.BRANCH_NAME}-${documentationPlatform}").inside() {
          sh "cp -rp \$INSTALL/share/doc/${projectName} ${subdir}"
        }
        sh "git add -A ${subdir}"
        sh """
          git commit --author='Flatiron Jenkins <jenkins@flatironinstitute.org>' --allow-empty -m 'Generated documentation for ${subdir}' -m '${env.BUILD_TAG} ${commit}'
        """
        // note: credentials used above don't work (need JENKINS-28335)
        sh "git push origin master"
      }
    } }
  } }
} catch (err) {
  /* send email on build failure (declarative pipeline's post section would work better) */
  if (env.BRANCH_NAME != "jenkins") emailext(
    subject: "\$PROJECT_NAME - Build # \$BUILD_NUMBER - FAILED",
    body: """\$PROJECT_NAME - Build # \$BUILD_NUMBER - FAILED

$err

Check console output at \$BUILD_URL to view full results.

Building \$BRANCH_NAME for \$CAUSE
\$JOB_DESCRIPTION

Chages:
\$CHANGES

End of build log:
\${BUILD_LOG,maxLines=60}
    """,
    to: 'mzingl@flatironinstitute.org, nwentzell@flatironinstitute.org, dsimon@flatironinstitute.org',
    recipientProviders: [
      [$class: 'DevelopersRecipientProvider'],
    ],
    replyTo: '$DEFAULT_REPLYTO'
  )
  throw err
}
