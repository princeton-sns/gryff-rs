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

### Tool Versions

We've built and run the spanner-rss with the following compiler tools:
* make v4.1
* go v1.13
* python v3.5.2
* gnuplot v5.0

### Running experiments

Experiments for the paper were run on CloudLab using the
[gryff](https://www.cloudlab.us/p/cops/gryff) profile.
After starting an experiment,

1. Clone the experiment repository to one of the CloudLab machines. (We often use `client-0-0`.)  
   `$ git clone https://github.com/princeton-sns/gryff-rs.git`

2. Build the Go client and server.  
   `$ export PATH="$PATH:/usr/local/go/bin"`
   `$ cd gryff-rs && make`

3. Install experiment script dependencies.  
   `$ sudo apt update && sudo apt install -y python3-numpy gnuplot`

4. Update experiment config. You will likely need to update the following fields:
   - `project_name`
   - `experiment_name`
   - `base_local_exp_directory`
   - `base_remote_bin_directory_nfs`
   - `src_directory`
   - `src_commit_hash`
   - `client_host_format_str`
   - `server_host_format_str`

5. After updating the config file, you can run the experiment using a python3 script. For example,  
   `$ python3 ./scripts/run_multiple_experiments.py ./experiments/sosp2021/vary-reads-writes-5-conflict-10.json`

## Authors
Jeffrey Helt, Amit Levy, Wyatt Lloyd -- Princeton University

Matthew Burke -- Cornell University
