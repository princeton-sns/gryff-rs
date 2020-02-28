package clients

import (
	"bufio"
	"fmt"
	"log"
	"masterproto"
	"net"
	"net/rpc"
  "fastrpc"
  "io"
  "clientproto"
  "time"
  "stats"
  "encoding/binary"
  "dlog"
)

const CHAN_BUFFER_SIZE = 200000

const TIMEOUT_SECS = 10
const MAX_RETRIES = 5

type ReplicaReply struct {
  reply interface{}
  replica int
  err error
}

type RPCPair struct {
	Obj  fastrpc.Serializable
	Chan chan fastrpc.Serializable
}

type Client interface {
  Read(key int64) (bool, int64)
  Write(key int64, value int64) bool
  CompareAndSwap(key int64, oldValue int64,
    newValue int64) (bool, int64)
  Finish()
  ConnectToMaster()
  ConnectToReplicas()
  DetermineLeader()
  DetermineReplicaPings()
  DelayRPC(replica int, opCode uint8)
}

type AbstractClient struct {
  id int32
  masterAddr string
  masterPort int
  forceLeader int
  stats *stats.StatsMap
  statsFile string
	rpcTable map[uint8]*RPCPair
  glArgs *masterproto.GetLeaderArgs
  glReply *masterproto.GetLeaderReply
  replyChan chan *ReplicaReply
  masterRPCClient *rpc.Client
  replicaAddrs []string
  numReplicas int
	replicas []net.Conn
  readers []*bufio.Reader
  writers []*bufio.Writer
  shutdown bool
  leader int
  pingReplyChan chan fastrpc.Serializable
  replicaAlive []bool
  replicaPing []uint64
  replicasByPingRank []int32
  retries []int
  delayRPC []map[uint8]bool
  delayedRPC []map[uint8]chan fastrpc.Serializable
}

func NewAbstractClient(id int32, masterAddr string, masterPort int, forceLeader int, statsFile string) *AbstractClient {
  c := &AbstractClient{
    id,                             // id
    masterAddr,                     // masterAddr
    masterPort,                     // masterPort
    forceLeader,                    // forceLeader
    stats.NewStatsMap(),            // stats
    statsFile,                      // statsFile
    make(map[uint8]*RPCPair),       // rpcTable
    &masterproto.GetLeaderArgs{},   // glArgs
    &masterproto.GetLeaderReply{},  // glReply
    make(chan *ReplicaReply),       // replyChan
    nil,                            // masterRPCClient
    make([]string, 0),              // replicaAddrs
    -1,                             // numReplicas
    make([]net.Conn, 0),            // replicas
    make([]*bufio.Reader, 0),       // readers
    make([]*bufio.Writer, 0),       // writers
    false,                          // shutdown
    -1,                             // leader
    make(chan fastrpc.Serializable, // pingReplyChan
      CHAN_BUFFER_SIZE), 
    make([]bool, 0),                // replicasAlive
    make([]uint64, 0),              // replicaPing
    make([]int32, 0),                 // replicasByPingRank
    make([]int, 0),                 // retries
    make([]map[uint8]bool, 0),      // delayRPC
    make([]map[uint8]chan fastrpc.Serializable, 0), // delayedRPC
  }
  c.RegisterRPC(new(clientproto.PingReply), clientproto.GEN_PING_REPLY, c.pingReplyChan)

  c.ConnectToMaster()
  c.ConnectToReplicas()
  c.DetermineLeader()
  c.DetermineReplicaPings()

  return c
}

func (c *AbstractClient) Finish() {
  if !c.shutdown {
    if len(c.statsFile) > 0 {
      c.stats.Export(c.statsFile)
    }
    for _, replica := range c.replicas {
      replica.Close()
    }
    c.masterRPCClient.Close()
    c.shutdown = true
  }
}

func (c *AbstractClient) ConnectToMaster() {
  log.Printf("Dialing master at addr %s:%d\n", c.masterAddr, c.masterPort)
  var err error
	c.masterRPCClient, err = rpc.DialHTTP("tcp", fmt.Sprintf("%s:%d",
    c.masterAddr, c.masterPort))
	if err != nil {
    log.Fatalf("Error connecting to master: %v\n", err)
	}

	rlReply := new(masterproto.GetReplicaListReply)
	err = c.masterRPCClient.Call("Master.GetReplicaList",
    new(masterproto.GetReplicaListArgs), rlReply)
	if err != nil {
    log.Fatalf("Error calling GetReplicaList: %v\n", err)
	}
  log.Printf("Got replica list from master.\n")
  c.replicaAddrs = rlReply.ReplicaList
  c.numReplicas = len(c.replicaAddrs)
  c.replicaAlive = make([]bool, c.numReplicas)
  c.replicaPing = make([]uint64, c.numReplicas)
  c.replicasByPingRank = make([]int32, c.numReplicas)
  c.retries = make([]int, c.numReplicas)
  c.delayRPC = make([]map[uint8]bool, c.numReplicas)
  c.delayedRPC = make([]map[uint8]chan fastrpc.Serializable, c.numReplicas)
  for i := 0; i < c.numReplicas; i++ {
    c.delayRPC[i] = make(map[uint8]bool)
    c.delayedRPC[i] = make(map[uint8]chan fastrpc.Serializable)
  }
}

func (c *AbstractClient) ConnectToReplicas() {
  log.Printf("Connecting to replicas...\n")
  c.replicas = make([]net.Conn, c.numReplicas)
	c.readers = make([]*bufio.Reader, c.numReplicas)
	c.writers = make([]*bufio.Writer, c.numReplicas)
  for i := 0; i < c.numReplicas; i++ {
    if !c.connectToReplica(i) {
      log.Fatalf("Must connect to all replicas on startup.\n")
    }
	}
  log.Printf("Successfully connected to all %d replicas.\n", c.numReplicas)
}

func (c *AbstractClient) connectToReplica(i int) bool {
	var err error
  if c.replicas[i] != nil {
    c.replicas[i].Close()
  }
  c.retries[i]++
  log.Printf("Dialing replica %d with addr %s\n", i,
    c.replicaAddrs[i])
  c.replicas[i], err = net.Dial("tcp", c.replicaAddrs[i])
  if err != nil {
    log.Printf("Error connecting to replica %d: %v\n", i, err)
    return false
  }
  log.Printf("Connected to replica %d with connection %s\n", i, c.replicas[i].LocalAddr().String())
  c.readers[i] = bufio.NewReader(c.replicas[i])
  c.writers[i] = bufio.NewWriter(c.replicas[i])

  var idBytes [4]byte
  idBytesS := idBytes[:4]
  binary.LittleEndian.PutUint32(idBytesS, uint32(c.id))
  c.writers[i].Write(idBytesS)
  c.writers[i].Flush()

  c.replicaAlive[i] = true
  go c.replicaListener(i)
  return true
}

func (c *AbstractClient) DetermineLeader() {
  err := c.masterRPCClient.Call("Master.GetLeader", c.glArgs, c.glReply)
  if err != nil {
    log.Fatalf("Error calling GetLeader: %v\n", err)
	}
	c.leader = c.glReply.LeaderId
	log.Printf("The leader is replica %d\n", c.leader)
}

func (c *AbstractClient) DetermineReplicaPings() {
  log.Printf("Determining replica pings...\n")
  for i := 0; i < c.numReplicas; i++ {
  }

  done := make(chan bool, c.numReplicas)
  for i := 0; i < c.numReplicas; i++ {
    go c.pingReplica(i, done)
  }

  for i := 0; i < c.numReplicas; i++ {
    if ok := <- done; !ok {
      log.Fatalf("Must successfully ping all replicas on startup.\n")
    }
  }

  // mini selection sort
  for i := 0; i < c.numReplicas; i++ {
    c.replicasByPingRank[i] = int32(i)
    for j := i + 1; j < c.numReplicas; j++ {
      if c.replicaPing[j] < c.replicaPing[c.replicasByPingRank[i]] {
        c.replicasByPingRank[i] = int32(j)
      }
    }
  }
  log.Printf("Successfully pinged all replicas!\n")
}

func (c *AbstractClient) pingReplica(i int, done chan bool) {
  var err error
  log.Printf("Sending ping to replica %d\n", i)
  err = c.writers[i].WriteByte(clientproto.GEN_PING)
  if err != nil {
    log.Printf("Error writing GEN_PING opcode to replica %d: %v\n", i, err)
    done <- false
    return
  }
  ping := &clientproto.Ping{c.id, uint64(time.Now().UnixNano())}
  ping.Marshal(c.writers[i])
  err = c.writers[i].Flush()
  if err != nil {
    log.Printf("Error flushing connection to replica %d: %v\n", i, err)
    done <- false
    return
  }
  select {
    case pingReplyS := <-c.pingReplyChan:
      pingReply := pingReplyS.(*clientproto.PingReply)
      c.replicaPing[pingReply.ReplicaId] = uint64(time.Now().UnixNano())- 
        pingReply.Ts
      log.Printf("Received ping from replica %d in time %d\n",
        pingReply.ReplicaId, c.replicaPing[pingReply.ReplicaId])
      done <- true
      log.Printf("Done pinging replica %d.\n", i)
      break
    case <-time.After(TIMEOUT_SECS * time.Second):
      log.Printf("Timeout out pinging replica %d.\n", i)
      for c.retries[i] < MAX_RETRIES {
        if c.connectToReplica(i) {
          c.pingReplica(i, done)
          return
        }
      }
      done <- false
      break
  }
}

func (c *AbstractClient) RegisterRPC(msgObj fastrpc.Serializable,
    opCode uint8, notify chan fastrpc.Serializable) {
	c.rpcTable[opCode] = &RPCPair{msgObj, notify}
}

func (c *AbstractClient) DelayRPC(replica int, opCode uint8) {
  _, ok := c.delayedRPC[replica][opCode]
  if !ok {
    c.delayedRPC[replica][opCode] = make(chan fastrpc.Serializable, CHAN_BUFFER_SIZE)
  }
  c.delayRPC[replica][opCode] = true
}

func (c *AbstractClient) ShouldDelayNextRPC(replica int, opCode uint8) bool {
  delay, ok := c.delayRPC[replica][opCode]
  c.delayRPC[replica][opCode] = false
  return ok && delay
}

func (c *AbstractClient) replicaListener(replica int) {
  var msgType byte
  var err error
  var errS string
  for !c.shutdown && err == nil {
    if msgType, err = c.readers[replica].ReadByte(); err != nil {
      errS = "reading opcode"
			break
		}
    dlog.Printf("Received opcode %d from replica %d.\n", msgType, replica)

    if rpair, present := c.rpcTable[msgType]; present {
			obj := rpair.Obj.New()
			if err = obj.Unmarshal(c.readers[replica]); err != nil {
        errS = "unmarshling message"
        break
			} 
			rpair.Chan <- obj
		} else {
      log.Printf("Error: received unknown message type: %d\n", msgType)
		}
	}
  if err != nil && err != io.EOF {
    log.Printf("Error %s from replica %d: %v\n", errS, replica, err) 
    c.replicaAlive[replica] = false
  }
}
