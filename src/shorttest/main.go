package main

import (
  "flag"
  "clients"
  "log"
  "dlog"
  "time"
  "masterlib"
  "gryff"
  "serverlib"
  "fmt"
  "net"
  "net/http"
  "net/rpc"
)

func main() {
	flag.Parse()

  dlog.DLOG = true

  master := masterlib.NewMaster(3, 7087)

  go master.Serve()

  replicas := make([]*gryff.Replica, 3)
  rpcServers := make([]*rpc.Server, 3)
  for j := 0; j < len(replicas); j++ {
    go func(i int) {
      replicaId, nodeList := serverlib.RegisterWithMaster("", 7087 + 1 + i, fmt.Sprintf(":%d", 7087))
      replicas[replicaId] = gryff.NewReplica(replicaId, nodeList, false, true, false, true, false,
        fmt.Sprintf("replica-%d.stats", replicaId), false, true, false, false, gryff.EPAXOS, 100, 100, 100, false)
      rpcServers[replicaId] = rpc.NewServer()
      rpcServers[replicaId].Register(replicas[replicaId])
      rpcServers[replicaId].HandleHTTP(fmt.Sprintf("replica-%d-rpc", replicaId), fmt.Sprintf("replica-%d-rpc-debug", replicaId))
      rpcPort := 8087 + 1 + i
      log.Printf("Listening for RPCs on port %d\n", rpcPort)
      l, err := net.Listen("tcp", fmt.Sprintf(":%d", rpcPort))
      if err != nil {
        log.Fatal("Error listening for RPCs:", err)
      }

      go http.Serve(l, nil)
    }(j)
  }

  time.Sleep(500 * time.Millisecond)
  cs := make([]clients.Client, 3)
  for i := 0; i < len(cs); i++ {
    cs[i] = clients.NewGryffClient(int32(i), "", 7087, i, fmt.Sprintf("client-%d.stats", i),
        false, false, true, false, true, false)
    cs[i].ConnectToMaster()
    cs[i].ConnectToReplicas()
    cs[i].DetermineLeader()
  }

  cs[0].Write(0, 1)
  cs[2].Write(0, 2)
  for i := 0; i < len(cs); i++ {
    cs[i].Finish()
  }
}
