# Capture clips

Full-resolution (1080×1080) clips from each phase, saved for teasers and the launch "how we built it" thread. Not for posting before the 80% mark (see *Going public* in the roadmap).

| Clip | Phase | What it shows | Made with |
| --- | --- | --- | --- |
| `phase0_random_walk.mp4` | 0 | 1,000 bots wandering in a box, 10 s | `python phase0/animate.py --bots 1000 --size 10.8 --save …` |
| `phase1_flock.mp4` | 1 | 1,000 bots forming flocks with no leader, then gathering when the beacon switches on, 20 s | `python phase1/animate.py --size 10.8 --save …` |
| `phase2_assemble_k.mp4` | 2 | 150 robots self-assembling into the letter K, 12 s | `python phase2/animate.py --shape k --size 10.8 --save …` |
| `phase2_assemble_star.mp4` | 2 | 150 robots self-assembling into a star, 12 s | `python phase2/animate.py --shape star --size 10.8 --save …` |
| `linking_sheet.mp4` | Linking | 93 linked cube bots building a one-bot-thick wall (the "surface" test) by sliding and pivoting over each other, 10 s | `python linking/animate.py --shape sheet --save …` |
| `linking_tower.mp4` | Linking | 172 linked bots building a hollow square tower, 10 s | `python linking/animate.py --shape tower --save …` |
| `linking_arch_repair.mp4` | Linking | An arch is built, the middle of its bridge is cut out, and the spare bots climb back up and rebuild it, 21 s | `python linking/animate.py --shape arch --repair --save …` |
| `linking_sheet_repair.mp4` | Linking | The top of the wall is cut off (36 bots lost) and regrown from spares, 21 s | `python linking/animate.py --shape sheet --repair --save …` |

The linking clips are 1080×810.
