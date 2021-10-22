# Gryff-RSC

## What is Gryff-RSC?

Gryff-RSC is a variant of the Gryff key-value store presented at NSDI 2020. Gryff combines a shared register and consensus protocol
to offer linearizable reads, writes, and read-modify-writes operations.
Gryff-RSC relaxes Gryff's consistency from linearizability to regular sequential consistency, and as a result,
Gryff-RSC offers lower tail read latency. This code was used for the SOSP 2021 paper,
["Regular Sequential Serializability and Regular Sequential
Consistency."](https://dl.acm.org/doi/10.1145/3477132.3483566) It is based off
of the code originally used in the evaluation of
[Gryff](https://www.usenix.org/conference/nsdi20/presentation/burke).

This repository includes an implementation of Gryff's protocol, an
implementation of our Gryff-RSC variant, and scripts to run the experiments
presented in our paper.

## Compiling & Running
Directions coming soon!

## Authors
Jeffrey Helt, Amit Levy, Wyatt Lloyd -- Princeton University

Matthew Burke -- Cornell University
