# The Microbots Textbook

This textbook explains the code in this repo from scratch, one chapter per phase of the [roadmap](../ROADMAP.md). Each chapter is written alongside its code, so the book always matches what's in the repo.

## How to use it

1. **Run the code first.** Each chapter starts with the commands. Seeing it work makes the explanation stick.
2. **Read the chapter with the code open beside it.** Every section names the file and function it explains.
3. **Do the "Try this" exercises.** They're small changes you make by hand, with a prediction first. This is where the learning actually happens.
4. **Answer "Check your understanding" before opening the answers.**

## Understanding gates

Before each outreach checkpoint in the roadmap, you should be able to:

- answer that phase's "Check your understanding" questions without looking, and
- make one of the chapter's changes on your own, without help.

Those are the moments when someone will ask how the project works. The gate makes sure you can answer.

## Chapters

| # | Chapter | Phase | Code |
| --- | --- | --- | --- |
| 0 | [A random walk in NumPy and JAX](00-random-walk.md) | 0 | `phase0/` |
| 1 | [Flocking with local rules](01-flocking.md) | 1 | `phase1/` |
| 2 | [Self-assembly into a shape](02-self-assembly.md) | 2 | `phase2/` |

### Side track: linking

How bots that must stay attached to each other can still move. Not tied to one phase; it feeds the hand-off between the swarm and the learned shapes.

| # | Chapter | Code |
| --- | --- | --- |
| L1 | [Moving while attached](L1-linking.md) | `linking/` |
| L2 | [From rules to a trained model](L2-from-rules-to-a-model.md) | `linking/learn/` |
