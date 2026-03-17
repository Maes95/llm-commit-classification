# Few-Shot Human Annotation Examples

## Example 1 - Total Agreement
- Commit hash: `d11a327ed95dbec756b99cbfef2a7fd85c9eeb09`
- Commit message:
`KVM: arm64: vgic-v3: Restrict SEIS workaround to known broken systems

Contrary to what df652bcf1136 ("KVM: arm64: vgic-v3: Work around GICv3
locally generated SErrors") was asserting, there is at least one other
system out there (Cavium ThunderX2) implementing SEIS, and not in
an obviously broken way.

So instead of imposing the M1 workaround on an innocent bystander,
let's limit it to the two known broken Apple implementations.

### Annotator A
- Understanding: `3` - `A regression seems to be reverted`
- BFC: `4` - `The prior workaround was too broad; this commit fixes the regression by restricting it to the two confirmed broken Apple implementations`
- BPC: `0` - `Not a preventive change; it fixes an already introduced regression`
- PRC: `0` - `Not a refactoring; functional behavior changes`
- NFC: `0` - `No new functionality added`
- Summary: `I do not understand very well what is being fixed, so I do not know its possible impact.`

### Annotator B
- Understanding: `4` - `Fixes an over-application of a previous workaround, refining its scope to only include known broken systems`
- BFC: `4` - `Clear bug fix: a previous workaround was incorrectly applied to hardware that does not need it; this corrects that scope`
- BPC: `0` - `No preventive aspect; the bug already exists and is directly fixed`
- PRC: `0` - `Not a refactoring`
- NFC: `0` - `Not a new feature`
- Summary: `Restricts the SEIS workaround to only the two known broken Apple implementations, avoiding unintended side effects on well-behaved hardware like Cavium ThunderX2.`

### Annotator C
- Understanding: `2` - `Difficult to understand but it seems that the commit is related to a previous workaround`
- BFC: `4` - `Seems to correct a bug introduced by df652bcf1136, which asserted incorrectly that all SEIS implementations were broken`
- BPC: `0` - `No preventive aspect`
- PRC: `0` - `Not a refactoring`
- NFC: `0` - `Not a new feature`
- Summary: `It seems it is not safety related, but I'm not totally sure.`

## Example 2 - Partial Disagreement
- Commit hash: `1eba86c096e35e3cc83de1ad2c26f2d70470211b`
- Commit message:
`mm: change page type prior to adding page table entry

Patch series "page table check", v3.

Ensure that some memory corruptions are prevented by checking at the
time of insertion of entries into user page tables that there is no
illegal sharing.

A problem existed in the kernel since 4.14 caused by broken page ref
counts, leading to memory leaking from one process into another.

This patch (of 4): change the order so that struct page type is updated
before the page table entry is inserted, in preparation for the page
table check mechanism that will verify no illegal sharing occurs at
insertion time.`

### Annotator A
- Understanding: `3` - `It is a fix that prevents memory corruption`
- BFC: `4` - `Fixes the incorrect ordering that allowed memory corruption, a known bug since kernel 4.14`
- BPC: `4` - `Also preventive: the reordering enables a page table check mechanism that will catch future illegal sharing`
- PRC: `0` - `Not a pure refactoring; correctness guarantees change`
- NFC: `0` - `No new user-visible feature`
- Summary: `Fix a memory leak/memory corruption`

### Annotator B
- Understanding: `4` - `This commit directly addresses issues related to memory corruption and refcount problems`
- BFC: `4` - `Directly fixes a memory corruption issue caused by incorrect ordering of page type updates and page table insertions`
- BPC: `0` - `Viewed as a pure bug fix; the preventive dimension of enabling the check mechanism is not recognized`
- PRC: `0` - `Not a refactoring`
- NFC: `0` - `Not a new feature`
- Summary: `This commit directly deals with memory management issues, particularly related to how memory pages are handled in the kernel.`

### Annotator C
- Understanding: `3` - `Resolved a fix to prevent memory corruption`
- BFC: `4` - `Clearly fixes a memory bug by correcting the order of page type update and page table entry insertion`
- BPC: `4` - `Also preventive: enables a future check that will prevent similar illegal sharing bugs`
- PRC: `0` - `Not a refactoring`
- NFC: `0` - `Not a new feature`
- Summary: `Clear fixing of a memory problem, pages management in the kernel.`

## Example 3 - High Disagreement
- Commit hash: `9a10064f5625d5572c3626c1516e0bebc6c9fe9b`
- Commit message:
`mm: add a field to store names for private anonymous memory

In many userspace applications, and especially in VM-based applications
like Android, multiple allocators coexist (libc malloc, stack, direct
mmap syscalls, multiple VM heaps). Each layer has its own tools to
inspect its usage, making consistent physical memory accounting (PSS/USS)
across the whole system difficult and inconsistent.

This patch adds a field to /proc/pid/maps and /proc/pid/smaps to show a
userspace-provided name for anonymous VMAs, displayed as [anon:<name>].
Userspace can set the name for a region of memory by calling
prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, start, len, name).

A regression test is also included.`

### Annotator A
- Understanding: `2` - `Commit dense. It appears that anonymous memory needs to be better managed to avoid performance issues. It seems that a test is provided to check that a regression exists.`
- BFC: `3` - `The presence of a regression test suggests an existing bug may be addressed`
- BPC: `3` - `The new memory tracking infrastructure may prevent future memory management issues`
- PRC: `1` - `Minor code quality improvement as part of the broader change`
- NFC: `0` - `Not primarily identified as a new feature due to limited understanding`
- Summary: `Probably it is from memory, but I don't have enough knowledge.`

### Annotator B
- Understanding: `4` - `Introduce new feature to the Linux kernel, specifically for managing private anonymous VMAs`
- BFC: `0` - `Not a bug fix; the code before this commit had no known defect, it simply lacked this naming capability`
- BPC: `4` - `The naming infrastructure could help prevent future debugging and memory tracking mistakes`
- PRC: `0` - `Not a refactoring; new behavior is introduced`
- NFC: `4` - `Clearly a new feature: adds prctl interface and /proc reporting for anonymous VMA naming, enabling capabilities not previously possible`
- Summary: `Introduces a new mechanism for userspace to name anonymous memory regions, improving observability and memory accounting in multi-allocator environments.`

### Annotator C
- Understanding: `1` - `I would say that is a performance enhancement, but it could be related to many things, as it is a difficult to understand commit.`
- BFC: `3` - `Could be fixing an implicit bug in memory tracking or reporting`
- BPC: `3` - `Likely prevents future memory management or tracking issues`
- PRC: `1` - `Some code quality improvement component present`
- NFC: `0` - `Not identified as a new feature due to very low understanding of the commit`
- Summary: `Could be a safety memory safety bug, as it is related to memory changes, but also related to timing and execution.`
