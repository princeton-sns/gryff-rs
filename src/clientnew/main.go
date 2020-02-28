package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"time"
  "clients"
  "os"
  "os/signal"
  "runtime/pprof"
  "dlog"
)

var clientId *int = flag.Int(
  "clientId",
  0,
  "Client identifier for use in replication protocols.")

var conflicts *int = flag.Int(
  "conflicts",
  -1,
  "Percentage of conflicts. If < 0, a zipfian distribution will be used for " +
  "choosing keys.")

var conflictsDenom *int = flag.Int(
  "conflictsDenom",
  100,
  "Denominator of conflict fraction when conflicts >= 0.")

var cpuProfile *string = flag.String(
  "cpuProfile",
  "",
  "Name of file for CPU profile. If empty, no profile is created.")

var debug *bool = flag.Bool(
  "debug",
  false,
  "Enable debug output.")

var defaultReplicaOrder *bool = flag.Bool(
  "defaultReplicaOrder",
  false,
  "Use default replica order for Gryff coordination.")

var epaxosMode *bool = flag.Bool(
  "epaxosMode",
  false,
  "Run Gryff with same message pattern as EPaxos.")

var expLength *int = flag.Int(
  "expLength",
  30,
  "Length of the timed experiment (in seconds).")

var fastPaxos *bool = flag.Bool(
  "fastPaxos",
  false,
  "Send message directly to all replicas a la Fast Paxos.")

var forceLeader *int = flag.Int(
  "forceLeader",
  -1,
  "Replica ID to which leader-based operations will be sent. If < 0, an " +
  "appropriate leader is chosen by default.")

var masterAddr *string = flag.String(
  "masterAddr",
  "",
  "Master address.")

var masterPort *int = flag.Int(
  "masterPort",
  7087,
  "Master port.")

var maxProcessors *int = flag.Int(
  "maxProcessors",
  2,
  "GOMAXPROCS. Defaults to 2")

var numKeys *uint64 = flag.Uint64(
  "numKeys",
  10000,
  "Number of keys in simulated store.")

var proxy *bool = flag.Bool(
  "proxy",
  false,
  "Proxy writes at local replica.")

var rampDown *int = flag.Int(
  "rampDown",
  5,
  "Length of the cool-down period after statistics are measured (in seconds).")

var rampUp *int = flag.Int(
  "rampUp",
  5,
  "Length of the warm-up period before statistics are measured (in seconds).")

var randSleep *int = flag.Int(
  "randSleep",
  0,
  "Max number of milliseconds to sleep after operation completed.")

var randomLeader *bool = flag.Bool(
  "randomLeader",
  false,
  "Egalitarian (no leader).")

var reads *int = flag.Int(
  "reads",
  0,
  "Percentage of reads.")

var regular *bool = flag.Bool(
  "regular",
  false,
  "Perform operations with regular consistency. (only for applicable protocols)")

var replProtocol *string = flag.String(
  "replProtocol",
  "",
  "Replication protocol used by clients and servers.")

var rmws *int = flag.Int(
  "rmws",
  0,
  "Percentage of rmws.")

var sequential *bool = flag.Bool(
  "sequential",
  true,
  "Perform operations with sequential consistency. " +
  "(only for applicable protocols")

var statsFile *string = flag.String(
  "statsFile",
  "",
  "Export location for collected statistics. If empty, no file file is written.")

var tailAtScale *int = flag.Int(
  "tailAtScale",
  -1,
  "Simulate storage request fan-out by performing <tailAtScale> " +
  "requests and aggregating statistics.")

var thrifty *bool = flag.Bool(
  "thrifty",
  false,
  "Only initially send messages to nearest quorum of replicas.")

var writes *int = flag.Int(
  "writes",
  100,
  "Percentage of updates (writes).")

var zipfS = flag.Float64(
  "zipfS",
  2,
  "Zipfian s parameter. Generates values k∈ [0, numKeys] such that P(k) is " +
  "proportional to (v + k) ** (-s)")

var zipfV = flag.Float64(
  "zipfV",
  1,
  "Zipfian v parameter. Generates values k∈ [0, numKeys] such that P(k) is " +
  "proportional to (v + k) ** (-s)")

type OpType int
const (
  WRITE OpType = iota
  READ
  RMW
)

func createClient() clients.Client {
  switch *replProtocol {
    case "abd":
      return clients.NewAbdClient(int32(*clientId), *masterAddr, *masterPort, *forceLeader,
        *statsFile, *regular)
    case "gryff":
      return clients.NewGryffClient(int32(*clientId), *masterAddr, *masterPort, *forceLeader,
        *statsFile, *regular, *sequential, *proxy, *thrifty, *defaultReplicaOrder,
        *epaxosMode)
    case "epaxos":
      return clients.NewProposeClient(int32(*clientId), *masterAddr, *masterPort, *forceLeader,
        *statsFile, false, true)
    default:
      return clients.NewProposeClient(int32(*clientId), *masterAddr, *masterPort, *forceLeader,
        *statsFile, false, false)
  }
}

type RequestResult struct {
  k int64
  lat int64
  typ OpType
}

func Max(a int64, b int64) int64 {
  if a > b {
    return a
  } else {
    return b
  }
}

func main() {
	flag.Parse()

	if *conflicts > 100 {
		log.Fatalf("Conflicts percentage must be between 0 and 100.\n")
	}

  dlog.DLOG = *debug

	if *conflicts >= 0 {
		dlog.Println("Using uniform distribution")
	} else {
		dlog.Println("Using zipfian distribution")
	}

  if *writes + *reads + *rmws != 1000 {
    log.Fatalf("Writes (%d), reads (%d), and rmws (%d) must add up to 1000.\n", *writes, *reads, *rmws)
  }

	runtime.GOMAXPROCS(*maxProcessors)
  
  if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
      log.Fatalf("Error creating CPU profile file %s: %v\n", *cpuProfile, err)
		}
		pprof.StartCPUProfile(f)
		interrupt := make(chan os.Signal, 1)
		signal.Notify(interrupt)
		go catchKill(interrupt)
    defer pprof.StopCPUProfile() 
	}

  client := createClient()

	r := rand.New(rand.NewSource(int64(*clientId)))
	zipf := rand.NewZipf(r, *zipfS, *zipfV, uint64(*numKeys))

  var count int32
  count = 0
  var before time.Time
  var after time.Time

  go func(client clients.Client) {
    time.Sleep(time.Duration(*expLength + 1) * time.Second)
    client.Finish()
  }(client)

  coalescedOps := *tailAtScale
  if coalescedOps == -1 {
    coalescedOps = 1
  }

  start := time.Now()
  now := start
  currRuntime := now.Sub(start)
  for int(currRuntime.Seconds()) < *expLength {
    if *randSleep > 0 {
      time.Sleep(time.Duration(r.Intn(*randSleep * 1e6))) // randSleep ms
    }
    var opString string
    var k int64
    maxLats := []int64{-1, -1, -1, -1} // one for each {w,r,rmw,overall}
    for i := 0; i < coalescedOps; i++ {
      opTypeRoll := r.Intn(1000)
      var opType OpType
      if opTypeRoll < *reads {
        opType = READ
      } else if opTypeRoll < *reads + *writes {
        opType = WRITE
      } else {
        opType = RMW
      }
      if *conflicts >= 0 {
        if r.Intn(*conflictsDenom) < *conflicts {
          k = 0
        } else {
          k = (int64(count) << 32) | int64(*clientId)
        }
      } else {
        k = int64(zipf.Uint64())
      }

      var success bool
      before = time.Now()
      if opType == READ {
        opString = "r"
        success, _ = client.Read(k)
      } else if opType == WRITE {
        opString = "w"
        success = client.Write(k, int64(count))
      } else {
        opString = "rmw"
        success, _ = client.CompareAndSwap(k, int64(count - 1),
          int64(count))
      }
      if success {
        after = time.Now()
        lat := int64(after.Sub(before).Nanoseconds())
        maxLats[3] = Max(maxLats[3], lat) 
        maxLats[opType] = Max(maxLats[opType], lat)
        if *tailAtScale >= 0 {
          fmt.Printf("%s,%d,%d\n", opString,
            int64(after.Sub(before).Nanoseconds()), k)
        }
      } else {
        log.Printf("Failed %s(%d).\n", opString, count)
      }
      count++
      dlog.Printf("Requests attempted: %d\n", count)
    }
    if *rampUp < int(currRuntime.Seconds()) && int(currRuntime.Seconds()) < *expLength - *rampDown {
      if *tailAtScale == -1 {
        fmt.Printf("%s,%d,%d,%d\n", opString,
          int64(after.Sub(before).Nanoseconds()), k, count)
      } else {
        fmt.Printf("max,%d,maxw,%d,maxr,%d,maxrmw,%d\n",
          maxLats[3], maxLats[0], maxLats[1], maxLats[2])
      }
    }
    now = time.Now()
    currRuntime = now.Sub(start)
  }
  log.Printf("Total requests attempted: %d\n", count)
  log.Printf("Experiment over after %f seconds\n", currRuntime.Seconds())
  client.Finish()
}

func catchKill(interrupt chan os.Signal) {
	<-interrupt
	if *cpuProfile != "" {
		pprof.StopCPUProfile()
	}
	log.Printf("Caught signal and stopped CPU profile before exit.\n")
	os.Exit(0)
}
