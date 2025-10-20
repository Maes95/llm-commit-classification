# Definitions

### Purpose of the commit

The purpose of the commit is the intention with which the code change (commit) was done. It will usually be a short text, which to some extent may be a summary of the commit message. But in the end, it is what the annotator understands as the purpose or intention of the code change.

The annotator will not always fully understand that purpose, because of technical complexities, unfamiliarity with the details mentioned, deficiencies in the commit message and associated information, or anything else.

### Definition of failure, bug and fault.

Failures are defined as erroneous behaviour of the software system, when the system deviates from its expected behavior, resulting in an inability to perform its intended functions or deliver the expected outputs. If the failure is caused by a software fault (a bug), it can be fixed only by some change to the software. Therefore, a commit that doesn't change the source code (including its default configuration or other assets that may influence in the behaviour of the system) cannot fix or prevent a bug.

Therefore, we will use "bug" and "fault" as synonyms, which will cause failures. If no failure was found, no bug was found either, so any change could maybe be preventive of bugs, but for sure it will not fix a bug. Also, if a commit does not change the behaviour of the system, it cannot fix a bug (because behaviour before and after the commit is the same). Therefore, changes which only touch comments, or which are only refactoring, cannot fix a bug.

We will also consider changes to the default configuration as changes that could change the behaviour of the system, and therefore could introduce and fix bugs.

### Bug-fixing commit (BFC)

Any commit that fixes a bug present in the source code, defined as a software fault that manifests as a failure when running in a certain way the version of the system previous to the commit.

### Bug-Preventing Commit (BPC)

Any commit that prevents a bug that could cause a failure in the future. This kind of commit doesn't fix known bugs, but possible, still undiscovered bugs, that could happen in the future. For example, they could improve the values returned by a function, in a way that is likely to prevent failures in code calling that function.

### Perfective Commit (PRC)

Any commit that improves the quality of the code (understandability, composability, performance, etc). This includes refactoring, optimizations, code style improvements, adding comments, etc. The key aspect is that perfective commits do not fix bugs or add new features.

### New Feature Commit (NFC)

Any commit that adds new functionality or capabilities to the codebase. This includes adding support for new hardware, implementing new APIs, exposing new configuration options, etc. The key aspect is that new feature commits add new code to enable new behaviors not previously possible.
