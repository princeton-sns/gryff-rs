package main

import (
	"flag"
  "masterlib"
  "log"
)

var portnum *int = flag.Int("port", 7087, "Port # to listen on. Defaults to 7087")
var numNodes *int = flag.Int("N", 3, "Number of replicas. Defaults to 3.")

func main() {
	flag.Parse()

	log.Printf("Master starting on port %d\n", *portnum)
	log.Printf("...waiting for %d replicas\n", *numNodes)

  master := masterlib.NewMaster(*numNodes, *portnum)
  master.Serve()
}

