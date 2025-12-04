# Context for annotators

Please, read carefully this context information. Have them in mind when annotating all commits.

## Main intention of the annotation

For each commit, the main intention of the annotation is to discern:

* How much the annotator understands the commit, and a (very brief) description of the main purpose of it. This is to know to which extent the annotation is an "informed annotation", or maybe annotators have doubts on their understanding of it. The description will also allow us, in the future, to contrast that impression of understanding by consulting with experts in the Linux kernel.

* The relationship of the commit with bugs (defects). We are mainly interested in determining which commits are fixing a bug, but also in finding out other relationships with bugs. In any case, it is important to realize these categories are not completely independent.
  * For example, BFC could also prevent some other future bugs (different from the fixed bug), which means that it could be labeled as "I'm sure it is a BFC" and "I'm sure it is a BPC". But it is important to notice that in this case, the "fixed bug" should be different from the "prevented bugs".
  * Also, any commit preventing bugs is certainly a perfective commit. But for annotating it as both a BPC and a PRC, there should be some clear improvement besides preventing those bugs (if not, it will be labeled only as BPC).
  * In other words, if a commit is in two or more categories, please be sure that is because there are different reasons for that: if the reason is the same, label it only in the first of the considered categories: BFC before BPC, or PRC before NFC). The categories are:
    * Does the commit fix a bug? (Bug Fixing Commit, BFC)
    * Does the commit prevent a bug? (Bug Preventing Commit, BPC)
    * Does the commit improve the code somehow, for instance by improving performance, improving the overall quality of the code, making it more readable, etc.? (Perfective Commit, PRC)
    * Does the commit include code for some new feature or a part of it? (New Feature Commit, NFC)
